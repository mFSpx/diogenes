#!/usr/bin/env python3
"""Validate large KRAMPUSCHEWING files that were intentionally not fully hashed.

This is a metadata/partial-fingerprint pass only:
- no archive unpacking
- no full-file hashing
- no file moves/deletes
- no graph materialization/writes
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "krampuschewing"
PARTIAL_BYTES = 1024 * 1024

ARCHIVE_EXTS = {".zip", ".7z", ".tar", ".gz", ".bz2", ".xz", ".rar", ".iso"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".tif", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
DATABASE_EXTS = {".sqlite", ".sqlite3", ".db", ".duckdb", ".mdb"}
MODEL_INDEX_EXTS = {".pkl", ".pickle", ".model", ".bin", ".index", ".faiss", ".parquet"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def utc_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_large_rows(index_path: Path, threshold: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "path": None,
                        "relative_path": f"[malformed-jsonl-line:{line_no}]",
                        "size_bytes": None,
                        "sha256_status": "malformed_json",
                        "blockers": [f"malformed_jsonl_line:{line_no}:{exc}"],
                    }
                )
                continue
            size = row.get("size_bytes")
            skipped = row.get("sha256_status") == "skipped_large_file"
            too_large = isinstance(size, int) and size > threshold
            if skipped or too_large:
                rows.append(row)
    return rows


def classify(row: dict[str, Any], path: Path | None) -> tuple[str, list[str], str]:
    relative = str(row.get("relative_path") or row.get("repo_relative_path") or row.get("path") or "")
    lower = relative.lower()
    ext = (path.suffix.lower() if path else str(row.get("extension") or "").lower())
    tags: list[str] = []

    active_db_risk = ext in DATABASE_EXTS or "chromadb" in lower or "chroma.sqlite" in lower
    if active_db_risk:
        tags.extend(["ACTIVE_RUNTIME_DB_RISK", "LARGE_DATABASE"])
        return "ACTIVE_RUNTIME_DB_RISK", tags, "do_not_touch_active_db"

    if ext in ARCHIVE_EXTS:
        primary = "LARGE_ARCHIVE"
        tags.append("LARGE_ARCHIVE")
    elif ext in VIDEO_EXTS or ext in IMAGE_EXTS or ext in AUDIO_EXTS:
        primary = "LARGE_MEDIA"
        tags.append("LARGE_MEDIA")
    elif ext in MODEL_INDEX_EXTS or "index" in lower or "model" in lower:
        primary = "LARGE_MODEL_OR_INDEX"
        tags.append("LARGE_MODEL_OR_INDEX")
    else:
        primary = "LARGE_UNKNOWN"
        tags.append("LARGE_UNKNOWN")

    case_terms = [
        "rickshaw",
        "nordley",
        "squeezecopy",
        "rtb",
        "bcfsa",
        "property",
        "case",
        "evidence",
        "screenshots",
        "pictures",
        "phone_backup",
        "northernstrike",
        "archive/corpus",
        "coop_detat",
    ]
    dev_terms = ["luci", "lucidota", "repo", "source", "src", "scripts", "docs_luci", "project"]

    if any(term in lower for term in case_terms):
        tags.append("CASE_ARCHIVE_CANDIDATE")
    if any(term in lower for term in dev_terms):
        tags.append("DEV_ARCHIVE_CANDIDATE")

    if primary in {"LARGE_ARCHIVE", "LARGE_MEDIA", "LARGE_UNKNOWN"}:
        tags.append("PROOF_KERNEL_LATER_CANDIDATE")
    tags.append("CHUNKED_HASH_LATER_CANDIDATE")

    if primary == "LARGE_MEDIA":
        action = "proof_kernel_later"
    elif primary == "LARGE_UNKNOWN":
        action = "manual_review"
    else:
        action = "chunked_hash_later"
    return primary, sorted(set(tags)), action


def partial_fingerprint(path: Path, enable: bool) -> dict[str, Any]:
    fp = {
        "enabled": False,
        "method": None,
        "first_mib_sha256": None,
        "last_mib_sha256": None,
    }
    if not enable:
        return fp
    size = path.stat().st_size
    with path.open("rb") as fh:
        first = fh.read(PARTIAL_BYTES)
        if size > PARTIAL_BYTES:
            fh.seek(max(0, size - PARTIAL_BYTES))
            last = fh.read(PARTIAL_BYTES)
        else:
            last = first
    fp.update(
        {
            "enabled": True,
            "method": "first_1MiB_sha256_plus_last_1MiB_sha256",
            "first_mib_sha256": hashlib.sha256(first).hexdigest(),
            "last_mib_sha256": hashlib.sha256(last).hexdigest(),
        }
    )
    return fp


def record_for(row: dict[str, Any], threshold: int) -> dict[str, Any]:
    blockers = list(row.get("blockers") or [])
    raw_path = row.get("path")
    path = Path(raw_path) if raw_path else None
    exists = bool(path and path.exists())
    is_file = bool(path and path.is_file())
    stat = path.stat() if path and exists and is_file else None
    primary_class, tags, action = classify(row, path)
    if not exists:
        blockers.append("large_file_missing")
    elif not is_file:
        blockers.append("large_path_not_file")

    partial_enabled = bool(exists and is_file and primary_class != "ACTIVE_RUNTIME_DB_RISK")
    if primary_class == "ACTIVE_RUNTIME_DB_RISK":
        blockers.append("active_runtime_db_risk:stat_only_no_partial_read")

    try:
        fp = partial_fingerprint(path, partial_enabled) if path and exists and is_file else partial_fingerprint(Path("/nonexistent"), False)
    except Exception as exc:
        fp = partial_fingerprint(Path("/nonexistent"), False)
        blockers.append(f"partial_fingerprint_failed:{exc}")

    stat_size = int(stat.st_size) if stat else row.get("size_bytes")
    record = {
        "path": str(path) if path else raw_path,
        "relative_path": row.get("relative_path") or row.get("repo_relative_path"),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": stat_size,
        "index_size_bytes": row.get("size_bytes"),
        "size_matches_index": (stat_size == row.get("size_bytes")) if stat and isinstance(row.get("size_bytes"), int) else None,
        "modified_time_utc": utc_from_ts(stat.st_mtime) if stat else row.get("modified_time_utc"),
        "extension": (path.suffix.lower() if path else row.get("extension")),
        "mime_guess": f"extension/{(path.suffix.lower() if path else row.get('extension') or '[none]').lstrip('.')}",
        "parent_folder": str(path.parent) if path else None,
        "kind_guess": row.get("kind_guess"),
        "lane_guess": row.get("lane_guess"),
        "large_class": primary_class,
        "classification_tags": tags,
        "full_sha256_present": bool(row.get("sha256")),
        "source_sha256_status": row.get("sha256_status"),
        "exceeded_threshold": bool(isinstance(row.get("size_bytes"), int) and int(row["size_bytes"]) > threshold),
        "partial_fingerprint": fp,
        "recommended_next_action": action,
        "blockers": sorted(set(blockers)),
    }
    return record


def validate(summary_path: Path, index_path: Path) -> tuple[dict[str, Any], Path]:
    summary = load_json(summary_path)
    threshold = int(summary.get("max_hash_bytes") or 268_435_456)
    rows = load_large_rows(index_path, threshold)
    records = [record_for(row, threshold) for row in rows]
    by_class = Counter(str(r.get("large_class")) for r in records)
    blockers: list[str] = []

    expected = summary.get("hash_skipped_large_files")
    if isinstance(expected, int) and expected != len(records):
        blockers.append(f"large_file_count_mismatch:summary={expected}:index={len(records)}")

    missing = sum(1 for r in records if not r.get("exists"))
    unreadable = sum(1 for r in records if "partial_fingerprint_failed" in " ".join(r.get("blockers", [])))
    active_db_risks = sum(1 for r in records if r.get("large_class") == "ACTIVE_RUNTIME_DB_RISK")
    partial_skipped = sum(1 for r in records if not (r.get("partial_fingerprint") or {}).get("enabled"))

    if missing:
        blockers.append(f"large_files_missing:{missing}")
    if unreadable:
        blockers.append(f"partial_fingerprint_failures:{unreadable}")
    if active_db_risks:
        blockers.append(f"active_runtime_db_risk:{active_db_risks}")
    if partial_skipped:
        blockers.append(f"partial_fingerprint_skipped:{partial_skipped}")
    for r in records:
        if not r.get("recommended_next_action"):
            blockers.append(f"missing_recommended_next_action:{r.get('relative_path')}")

    if missing or unreadable:
        verdict = "FAIL"
    elif blockers:
        verdict = "PARTIAL_FAIL"
    else:
        verdict = "PASS"

    receipt = {
        "schema": "lucidota.krampuschewing.large_file_validation.v1",
        "generated_at_utc": now(),
        "source_summary": rel(summary_path),
        "source_index": rel(index_path),
        "threshold_bytes": threshold,
        "large_files_found": len(rows),
        "large_files_validated": sum(1 for r in records if r.get("exists") and r.get("is_file")),
        "large_files_missing": missing,
        "total_large_bytes": sum(int(r.get("size_bytes") or 0) for r in records),
        "by_large_class": dict(sorted(by_class.items())),
        "records": records,
        "full_archive_unpacking": False,
        "full_hashing_performed": False,
        "files_moved": [],
        "files_deleted": [],
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "river_training_performed": False,
        "verdict": verdict,
        "blockers": sorted(set(blockers)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"krampuschewing_large_file_validation_{stamp()}.json"
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    return receipt, out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Metadata-only validation for large KRAMPUSCHEWING files")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--index", required=True)
    args = ap.parse_args()
    summary_path = (ROOT / args.summary).resolve() if not Path(args.summary).is_absolute() else Path(args.summary).resolve()
    index_path = (ROOT / args.index).resolve() if not Path(args.index).is_absolute() else Path(args.index).resolve()
    receipt, out_path = validate(summary_path, index_path)
    print("REPORT_PATH=" + rel(out_path))
    print("KRAMPUSCHEWING_LARGE_FILE_VALIDATION=" + receipt["verdict"])
    print("LARGE_FILES_FOUND=" + str(receipt["large_files_found"]))
    print("LARGE_FILES_VALIDATED=" + str(receipt["large_files_validated"]))
    print("LARGE_FILES_MISSING=" + str(receipt["large_files_missing"]))
    print("TOTAL_LARGE_BYTES=" + str(receipt["total_large_bytes"]))
    print("BY_LARGE_CLASS=" + json.dumps(receipt["by_large_class"], sort_keys=True))
    if receipt["blockers"]:
        print("BLOCKERS=" + json.dumps(receipt["blockers"], sort_keys=True))
    return 0 if receipt["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
