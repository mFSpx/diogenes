#!/usr/bin/env python3
"""Standalone KrampusBrain DBSTREAM ingest.

This is the brain-map sidecar: it reuses KORPUS' deterministic sticker extractors,
feeds one document at a time to River DBSTREAM, and appends a JSONL point that can
be plotted later. No LLM calls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
import stat
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BRAIN_PATH = ROOT / "03_VAULT" / "krampus_dbstream_brain.pkl"
MAP_PATH = ROOT / "05_OUTPUTS" / "korpus_krampii" / "krampus_brain_map.jsonl"
AUDIT_PATH = Path(os.environ.get("LUCIDOTA_BRAIN_AUDIT_JSONL", str(ROOT / "05_OUTPUTS" / "korpus_krampii" / "brain_sidecar_audit.jsonl")))

# Brain sidecar is a text/concept stream, not the evidence vault.  Binary/media
# bytes are already preserved by the KORPUS/CAS path; trying to latin-1 decode a
# video into "text" can monopolize the chronological watcher for no useful gain.
BRAIN_BINARY_SKIP_SUFFIXES = {
    ".3gp",
    ".7z",
    ".aac",
    ".avi",
    ".bmp",
    ".bz2",
    ".db",
    ".dmg",
    ".doc",
    ".docx",
    ".exe",
    ".flac",
    ".gif",
    ".gz",
    ".heic",
    ".ico",
    ".iso",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".odp",
    ".ods",
    ".odt",
    ".ogg",
    ".parquet",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tif",
    ".tiff",
    ".wav",
    ".webm",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}
BRAIN_TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md", ".py", ".rst", ".tsv", ".txt", ".yaml", ".yml"}

sys.path.insert(0, str(ROOT))

from river import cluster  # type: ignore  # noqa: E402
from ALGOS.krampus_chrono import chrono_file_date  # type: ignore  # noqa: E402
from ALGOS.krampus_brainmap import (  # type: ignore  # noqa: E402
    brain_xyz,
    dbstream_features,
    extract_full_features,
    extract_master_vector,
)
from ALGOS.krampus_stickers import operator_cluster_hint  # type: ignore  # noqa: E402


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_audit(event: dict[str, Any]) -> None:
    path = Path(os.environ.get("LUCIDOTA_BRAIN_AUDIT_JSONL", str(AUDIT_PATH))).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(jdump({"at": now(), "tool": "lucidota_brain_ingest", **event}) + "\n")


def read_text(path: Path, max_bytes: int = 16 * 1024 * 1024) -> str:
    data = path.read_bytes()[:max_bytes]
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc, errors="replace")
        except (LookupError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iso_from_ts(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def should_skip_brain_sidecar(path: Path, size_bytes: int) -> str:
    if not path.is_file():
        return "not_a_regular_file"
    suffix = path.suffix.lower()
    if suffix in BRAIN_BINARY_SKIP_SUFFIXES:
        return f"binary_or_media_suffix:{suffix}"
    max_bytes = int(os.environ.get("LUCIDOTA_BRAIN_MAX_TEXT_BYTES", str(16 * 1024 * 1024)))
    if size_bytes > max_bytes:
        if suffix in BRAIN_TEXT_SUFFIXES and not env_flag("LUCIDOTA_BRAIN_ALLOW_OVERSIZE_TEXT"):
            return f"oversize_text_sidecar:{size_bytes}>{max_bytes}"
        if suffix not in BRAIN_TEXT_SUFFIXES:
            return f"oversize_non_text_sidecar:{size_bytes}>{max_bytes}"
    return ""


def file_metadata(path: Path, raw: bytes | None = None, fast_filesystem_time: bool = False) -> dict[str, Any]:
    if fast_filesystem_time:
        st = path.stat()
        ts, ts_source = st.st_mtime, "os_mtime_fast_skip"
        chrono: dict[str, Any] = {
            "confidence_bps": 2500,
            "raw": iso_from_ts(ts),
        }
    else:
        chrono = chrono_file_date(path)
        ts, ts_source = chrono["timestamp"].timestamp(), chrono["source"]
    size = len(raw) if raw is not None else path.stat().st_size
    metadata = {
        "path": str(path),
        "sha256": sha256_bytes(raw) if raw is not None else "",
        "sha256_status": "computed" if raw is not None else "not_computed_for_brain_sidecar_skip",
        "size_bytes": size,
        "file_time": iso_from_ts(ts),
        "file_time_source": ts_source,
        "file_time_confidence_bps": chrono.get("confidence_bps", 0),
        "file_time_raw": chrono.get("raw", ""),
    }
    return metadata


class KrampusDBStream:
    def __init__(self) -> None:
        self.model = cluster.DBSTREAM(clustering_threshold=0.55, fading_factor=0.01, cleanup_interval=64, minimum_weight=1.0)
        self.files_ingested = 0
        self.created_at = dt.datetime.now(dt.timezone.utc).isoformat()

    def centers_count(self) -> int:
        centers = getattr(self.model, "centers", {})
        try:
            return len(centers)
        except TypeError:
            return 0

    def ingest(self, text: str, file_metadata: dict[str, Any]) -> dict[str, Any]:
        full = extract_full_features(text)
        if not full:
            return {"status": "skipped", "reason": "No extractable features", "metadata": file_metadata}
        master = extract_master_vector(text)
        stream_vector = dbstream_features(full)
        try:
            cluster_before = self.model.predict_one(stream_vector)
        except Exception as exc:
            self.last_error = f"predict_before_failed: {exc}"
            cluster_before = None
        self.model.learn_one(stream_vector)
        self.files_ingested += 1
        try:
            assigned_cluster = self.model.predict_one(stream_vector)
        except Exception as exc:
            self.last_error = f"predict_after_failed: {exc}"
            assigned_cluster = cluster_before
        return {
            "status": "ingested",
            "assigned_cluster_id": assigned_cluster,
            "cluster_before": cluster_before,
            "total_clusters_in_brain": self.centers_count(),
            "files_ingested": self.files_ingested,
            "operator_cluster_hint": operator_cluster_hint(full),
            "xyz": brain_xyz(master),
            "features": master,
            "stream_vector": stream_vector,
            "metadata": file_metadata,
        }


def load_brain(path: Path) -> KrampusDBStream:
    if path.exists():
        assert_trusted_pickle_path(path)
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, KrampusDBStream):
            return obj
    return KrampusDBStream()


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def assert_trusted_pickle_path(path: Path) -> None:
    if env_flag("LUCIDOTA_ALLOW_UNTRUSTED_PICKLE"):
        return
    resolved = path.expanduser().resolve(strict=False)
    vault = (ROOT / "03_VAULT").resolve()
    try:
        resolved.relative_to(vault)
    except ValueError as exc:
        raise PermissionError(f"pickle state must live under {vault}; set LUCIDOTA_ALLOW_UNTRUSTED_PICKLE=1 to override") from exc
    if path.exists():
        if path.is_symlink():
            raise PermissionError("refusing symlinked pickle state; set LUCIDOTA_ALLOW_UNTRUSTED_PICKLE=1 to override")
        mode = resolved.stat().st_mode
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise PermissionError("refusing group/world-writable pickle state; set LUCIDOTA_ALLOW_UNTRUSTED_PICKLE=1 to override")


def save_brain(path: Path, brain: KrampusDBStream) -> None:
    assert_trusted_pickle_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(brain, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    path.chmod(0o600)


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-brain-ingest")
    ap.add_argument("path")
    ap.add_argument("--state-path", default=str(BRAIN_PATH))
    ap.add_argument("--map-jsonl", default=str(MAP_PATH))
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    path = Path(args.path).expanduser().resolve()
    state_path = Path(args.state_path).expanduser()
    map_path = Path(args.map_jsonl).expanduser()
    if args.reset and state_path.exists():
        assert_trusted_pickle_path(state_path)
        state_path.unlink()
    stat_size = path.stat().st_size if path.exists() else 0
    skip_reason = should_skip_brain_sidecar(path, stat_size)
    if skip_reason:
        metadata = file_metadata(path, None, fast_filesystem_time=True)
        result = {
            "status": "skipped",
            "reason": skip_reason,
            "metadata": metadata,
            "sidecar_policy": "brain_text_stream_skip_preserve_in_korpus_cas",
        }
    else:
        brain = load_brain(state_path)
        raw = path.read_bytes()
        text = read_text(path)
        metadata = file_metadata(path, raw)
        result = brain.ingest(text, metadata)
    if not args.dry_run:
        if not skip_reason:
            save_brain(state_path, brain)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with map_path.open("a", encoding="utf-8") as out:
            out.write(jdump(result) + "\n")
        append_audit(
            {
                "kind": "brain_sidecar_result",
                "status": result.get("status", "unknown"),
                "reason": result.get("reason", ""),
                "path": str(path),
                "map_jsonl": str(map_path),
                "state_path": str(state_path),
                "metadata": result.get("metadata", {}),
                "sidecar_policy": result.get("sidecar_policy", ""),
            }
        )
    if args.json:
        print(jdump({"ok": True, **result}))
    else:
        print(json.dumps({"ok": True, **result}, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
