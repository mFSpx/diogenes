#!/usr/bin/env python3
"""Deterministic Krampuschew flow: hash/copy/dedupe, normalize, OCR, and views.

Thin orchestrator contract:
- never uses an LLM for routing or extraction
- dedupes by sha256 before touching Postgres
- writes receipts for every batch
- halves batch size when runtime pressure rises
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Optional in tests; available in live venv.
    import psycopg  # type: ignore
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "runtime"
CAS_ROOT = ROOT / "03_VAULT" / "cas"
SCHEMA_FILE = ROOT / "06_SCHEMA" / "123_absurd_flows.sql"
DEFAULT_CASE_KEY = "CASE-20260528-KRAMPUSCHEW"
DEFAULT_CASE_TITLE = "Krampuschew Absurd Flows"
IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "05_OUTPUTS",
    "04_RUNTIME",
    "03_VAULT/cas",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_kind(path: Path, mime: str) -> str:
    mime = (mime or "").lower()
    suffix = path.suffix.lower()
    if mime.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}:
        return "image"
    if mime == "application/pdf" or suffix == ".pdf":
        return "document"
    if mime.startswith("video/") or suffix in {".mp4", ".mkv", ".mov", ".avi", ".webm"}:
        return "video"
    if mime.startswith("audio/") or suffix in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}:
        return "audio"
    if suffix in {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}:
        return "archive"
    if mime.startswith("text/") or suffix in {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".log"}:
        return "text"
    return "binary"


@dataclass(frozen=True)
class FlowRecord:
    source_path: str
    sha256: str
    size_bytes: int
    mime: str
    file_kind: str
    cas_path: str
    ocr_sidecar: str = ""
    db_action: str = "skipped"
    db_reason: str = ""


def discover_files(root: Path, *, max_files: int | None = None, start_after: str = "") -> list[Path]:
    files: list[Path] = []
    cursor = start_after
    marker = f"{root.name}/"
    if marker in cursor:
        cursor = cursor.split(marker, 1)[1]
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in IGNORE_DIRS for part in rel_parts):
            continue
        if path.name.startswith("."):
            continue
        rel_path = path.relative_to(root).as_posix()
        if cursor and rel_path <= cursor:
            continue
        files.append(path)
        if max_files and len(files) >= max_files:
            break
    return files


def batch(items: list[Path], size: int) -> list[list[Path]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def cas_target(sha256: str) -> Path:
    return CAS_ROOT / sha256[:2] / sha256[2:4] / sha256


def copy_to_cas(src: Path, sha256: str) -> Path:
    dst = cas_target(sha256)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def ocr_sidecar(src: Path, sha256: str) -> Path | None:
    ext = src.suffix.lower()
    out = OUT_DIR / "absurd_flows_sidecars" / f"{sha256}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if ext == ".pdf" and shutil.which("pdftotext"):
        cmd = ["pdftotext", "-layout", str(src), str(out)]
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"} and shutil.which("tesseract"):
        base = out.with_suffix("")
        cmd = ["tesseract", str(src), str(base), "txt"]
        out = base.with_suffix(".txt")
    else:
        return None
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        return None
    return out if out.exists() else None


def ensure_schema(conn: Any) -> None:
    if not SCHEMA_FILE.exists():
        return
    conn.execute(SCHEMA_FILE.read_text(encoding="utf-8"))


def ensure_case(conn: Any, case_key: str = DEFAULT_CASE_KEY, title: str = DEFAULT_CASE_TITLE) -> None:
    conn.execute(
        """
        INSERT INTO lucidota_investigation.case_file(case_key, title, status, summary)
        VALUES (%s, %s, 'active', 'Deterministic absurd flows / Krampuschew batch')
        ON CONFLICT (case_key) DO UPDATE SET
            title = EXCLUDED.title,
            status = 'active',
            updated_at = now()
        """,
        (case_key, title),
    )


def insert_artifact_row(conn: Any, record: FlowRecord, src: Path) -> str:
    title = src.stem
    conn.execute(
        """
        INSERT INTO lucidota_investigation.artifact
        (sha256, cas_uri, locked_relative_path, original_path, original_name, mime, file_kind, size_bytes,
         title, evidence_date, evidence_date_source, exif_json, metadata_json, normalized_text, analysis_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, '', '{}'::jsonb, %s::jsonb, '', %s::jsonb)
        ON CONFLICT (sha256) DO NOTHING
        """,
        (
            record.sha256,
            f"cas://sha256/{record.sha256}",
            record.cas_path,
            rel(src),
            src.name,
            record.mime,
            record.file_kind,
            record.size_bytes,
            title,
            json.dumps({"source_path": rel(src), "ocr_sidecar": record.ocr_sidecar}, sort_keys=True),
            json.dumps({"flow": "absurd_flows", "db_action": record.db_action, "db_reason": record.db_reason}, sort_keys=True),
        ),
    )
    return record.sha256


def link_case_artifact(conn: Any, case_key: str, sha256: str, source_path: str) -> None:
    conn.execute(
        """
        INSERT INTO lucidota_investigation.case_artifact(case_uuid, artifact_uuid, evidence_id, artifact_title, sidecar_relative_dir)
        SELECT c.case_uuid, a.artifact_uuid, %s, a.title, ''
        FROM lucidota_investigation.case_file c
        JOIN lucidota_investigation.artifact a ON a.sha256 = %s
        WHERE c.case_key = %s
        ON CONFLICT (case_uuid, artifact_uuid) DO UPDATE SET
            artifact_title = EXCLUDED.artifact_title,
            updated_at = now()
        """,
        (f"ABSURD-{sha256[:16]}", sha256, case_key),
    )


def record_learning_run(conn: Any, payload: dict[str, Any], *, status: str = "succeeded") -> None:
    detail = {
        "schema": payload.get("schema", "lucidota.absurd_flows.v1"),
        "root": payload.get("root"),
        "file_count": payload.get("file_count", 0),
        "processed_count": payload.get("processed_count", 0),
        "deduped_count": payload.get("deduped_count", 0),
        "db_skipped_count": payload.get("db_skipped_count", 0),
        "batch_size_final": payload.get("batch_size_final", 0),
        "batch_history": payload.get("batch_history", []),
        "records_sample": [
            {
                "source_path": rec.get("source_path"),
                "db_action": rec.get("db_action"),
                "file_kind": rec.get("file_kind"),
            }
            for rec in (payload.get("records") or [])[:5]
        ],
    }
    conn.execute(
        """
        INSERT INTO lucidota_learning.river_run(status, events_seen, examples_trained, detail)
        VALUES (%s, %s, %s, %s::jsonb)
        """,
        (
            status,
            int(payload.get("file_count", 0)),
            int(payload.get("processed_count", 0)),
            json.dumps(detail, sort_keys=True),
        ),
    )
    examples = int(payload.get("file_count", 0))
    successes = int(payload.get("processed_count", 0))
    failures = int(payload.get("deduped_count", 0)) + int(payload.get("db_skipped_count", 0))
    success_rate = (successes / examples) if examples else 0.0
    conn.execute(
        """
        INSERT INTO lucidota_learning.river_score(source, phase, decision, examples, successes, failures, success_rate, river_prediction, model_kind)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source, phase, decision) DO UPDATE SET
            examples = EXCLUDED.examples,
            successes = EXCLUDED.successes,
            failures = EXCLUDED.failures,
            success_rate = EXCLUDED.success_rate,
            river_prediction = EXCLUDED.river_prediction,
            model_kind = EXCLUDED.model_kind,
            updated_at = now()
        """,
        (
            "absurd_flows",
            "krampuschew",
            "batch_size_final",
            examples,
            successes,
            failures,
            success_rate,
            float(payload.get("batch_size_final", 0) or 0),
            "river_online_batch_size_heuristic",
        ),
    )


def existing_sha256(conn: Any, sha_values: list[str]) -> set[str]:
    if not sha_values:
        return set()
    rows = conn.execute(
        "SELECT sha256 FROM lucidota_investigation.artifact WHERE sha256 = ANY(%s)",
        (sha_values,),
    ).fetchall()
    return {r[0] for r in rows}


def process_files(
    root: Path,
    *,
    max_files: int | None = None,
    chunk_size: int = 64,
    execute: bool = False,
    database_url: str | None = None,
    case_key: str = DEFAULT_CASE_KEY,
    start_after: str = "",
) -> dict[str, Any]:
    files = discover_files(root, max_files=max_files, start_after=start_after)
    receipts: list[FlowRecord] = []
    deduped = 0
    skipped_db = 0
    processed = 0
    batch_size = max(1, int(chunk_size))
    batch_history: list[dict[str, Any]] = []
    db_seen: set[str] = set()
    conn = None
    if execute:
        if psycopg is None:
            raise SystemExit("psycopg unavailable; cannot execute")
        if not database_url:
            raise SystemExit("database url required for execute")
        conn = psycopg.connect(database_url, connect_timeout=5)
        ensure_schema(conn)
        ensure_case(conn, case_key=case_key)
        conn.commit()
        db_seen = existing_sha256(conn, [sha256_file(p) for p in files[: min(len(files), 256)]]) if files else set()
    try:
        for group in batch(files, batch_size):
            started = datetime.now(timezone.utc)
            batch_rows = 0
            sha_values: list[str] = []
            batch_records: list[tuple[Path, str, int, str, str, Path]] = []
            for src in group:
                sha = sha256_file(src)
                sha_values.append(sha)
                if sha in db_seen:
                    deduped += 1
                    receipts.append(FlowRecord(rel(src), sha, src.stat().st_size, mimetypes.guess_type(src.name)[0] or "", file_kind(src, mimetypes.guess_type(src.name)[0] or ""), rel(cas_target(sha)), db_action="skipped", db_reason="already_in_db"))
                    continue
                cas_path = copy_to_cas(src, sha)
                sidecar = ocr_sidecar(src, sha)
                rec = FlowRecord(
                    source_path=rel(src),
                    sha256=sha,
                    size_bytes=src.stat().st_size,
                    mime=mimetypes.guess_type(src.name)[0] or "",
                    file_kind=file_kind(src, mimetypes.guess_type(src.name)[0] or ""),
                    cas_path=rel(cas_path),
                    ocr_sidecar=rel(sidecar) if sidecar else "",
                    db_action="inserted" if execute else "dry-run",
                    db_reason="",
                )
                receipts.append(rec)
                batch_records.append((src, sha, src.stat().st_size, rec.mime, rec.file_kind, cas_path))
            if execute and conn is not None and batch_records:
                seen_in_db = existing_sha256(conn, [sha for _, sha, *_ in batch_records])
                for src, sha, size_bytes, mime, kind, cas_path in batch_records:
                    if sha in seen_in_db:
                        skipped_db += 1
                        continue
                    record = FlowRecord(
                        source_path=rel(src),
                        sha256=sha,
                        size_bytes=size_bytes,
                        mime=mime,
                        file_kind=kind,
                        cas_path=rel(cas_path),
                        ocr_sidecar="",
                        db_action="inserted",
                        db_reason="",
                    )
                    insert_artifact_row(conn, record, src)
                    link_case_artifact(conn, case_key, sha, rel(src))
                    processed += 1
                conn.commit()
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            batch_history.append({"batch_size": len(group), "seconds": round(elapsed, 4), "processed": processed, "deduped": deduped})
            if elapsed > 10 and batch_size > 1:
                batch_size = max(1, batch_size // 2)
        payload = {
            "schema": "lucidota.absurd_flows.v1",
            "ok": True,
            "generated_at": now(),
            "root": rel(root),
            "file_count": len(files),
            "processed_count": processed,
            "deduped_count": deduped,
            "db_skipped_count": skipped_db,
            "batch_size_final": batch_size,
            "batch_history": batch_history,
            "records": [asdict(r) for r in receipts],
        }
        if execute and conn is not None:
            record_learning_run(conn, payload, status="succeeded")
            conn.commit()
        return payload
    finally:
        if conn is not None:
            conn.close()


def write_receipt(payload: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"absurd_flows_{stamp()}.json"
    payload = dict(payload)
    payload["receipt_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deterministic absurd flows for Krampuschew.")
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--database-url", default=None)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--case-key", default=DEFAULT_CASE_KEY)
    p.add_argument("--start-after", default="")
    return p


def main() -> int:
    args = build_parser().parse_args()
    payload = process_files(
        Path(args.root),
        max_files=args.max_files,
        chunk_size=args.chunk_size,
        execute=bool(args.execute),
        database_url=args.database_url,
        case_key=args.case_key,
        start_after=args.start_after,
    )
    receipt = write_receipt(payload)
    payload["receipt_path"] = rel(receipt)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(payload["receipt_path"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
