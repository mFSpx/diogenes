#!/usr/bin/env python3
"""Bounded top-level KRAMPUS document ingestion into corpus_chunk.

This ingests only top-level KRAMPUSCHEWING documents (.pdf/.docx/.odt/.txt/.md),
never deletes source files, and writes a receipt for every run. It is deliberately
bounded by --max-files/--file so LUCI can make visible ingestion progress without
launching an unbounded sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import psycopg

ROOT = Path("/home/mfspx/LUCIDOTA")
KRAMPUS_DIR = ROOT / "KRAMPUSCHEWING"
RECEIPT_DIR = ROOT / "05_OUTPUTS" / "receipts"
ELIGIBLE_EXTENSIONS = (".pdf", ".docx", ".odt", ".txt", ".md")
SCHEMA = "lucidota.krampus_top_level_document_ingest.v1"
EXTRACTOR = "krampus_pdf_v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def source_path_for(file_path: Path) -> str:
    try:
        return f"KRAMPUSCHEWING/{file_path.resolve(strict=False).relative_to(KRAMPUS_DIR.resolve(strict=False)).as_posix()}"
    except Exception:
        return rel(file_path)


def source_path_keys(raw_path: str) -> set[str]:
    text = str(raw_path or "").strip()
    if not text:
        return set()

    path = Path(text)
    keys = {text, path.as_posix(), path.name}
    try:
        resolved = path.resolve(strict=False)
    except Exception:
        resolved = None

    if resolved is not None:
        keys.add(resolved.as_posix())
        try:
            rel_path = resolved.relative_to(KRAMPUS_DIR.resolve(strict=False))
        except Exception:
            rel_path = None
        if rel_path is not None:
            keys.add(rel_path.as_posix())
            keys.add(KRAMPUS_DIR.name + "/" + rel_path.as_posix())

    if KRAMPUS_DIR.name in path.parts:
        idx = path.parts.index(KRAMPUS_DIR.name)
        rel_from_krampus = Path(*path.parts[idx:])
        keys.add(rel_from_krampus.as_posix())
        keys.add(rel_from_krampus.name)

    return {key for key in keys if key}


def eligible_files(krampus_dir: Path, only_file: str | None = None) -> list[Path]:
    if only_file:
        candidate = Path(only_file)
        if not candidate.exists() and not candidate.is_absolute():
            candidate = krampus_dir / candidate
        return [candidate] if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in ELIGIBLE_EXTENSIONS else []
    if not krampus_dir.exists():
        return []
    return sorted(p for p in krampus_dir.iterdir() if p.is_file() and p.suffix.lower() in ELIGIBLE_EXTENSIONS)


def chunk_text(text: str, max_chars: int = 1800) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars) if text[i : i + max_chars]]


def insert_chunks(cursor: Any, file_path: Path, source_path: str, content: str, *, dry_run: bool) -> int:
    chunks = chunk_text(content)
    inserted = 0
    for chunk_index, chunk in enumerate(chunks):
        chunk_sha = hashlib.sha256(chunk.encode("utf-8", errors="replace")).hexdigest()
        chunk_uuid = str(uuid5(NAMESPACE_URL, f"{EXTRACTOR}:{source_path}:{chunk_index}:{chunk_sha}"))
        if dry_run:
            inserted += 1
            continue
        cursor.execute(
            """
            INSERT INTO lucidota_korpus.corpus_chunk
              (chunk_uuid, file_uuid, sha256, source_path, mime, chunk_index,
               content, go25, embedding, embedding_model, extractor)
            VALUES (%s::uuid, NULL, %s, %s, %s, %s, %s, '{}'::jsonb, NULL, 'skip', %s)
            ON CONFLICT DO NOTHING
            """,
            (chunk_uuid, chunk_sha, source_path, file_path.suffix.lower(), chunk_index, chunk, EXTRACTOR),
        )
        rowcount = getattr(cursor, "rowcount", 1)
        inserted += 1 if rowcount is None or rowcount > 0 else 0
    return inserted


def run(args: argparse.Namespace) -> dict[str, Any]:
    krampus_dir = Path(args.krampus_dir)
    receipt_dir = Path(args.receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "status": "PASS",
        "dry_run": bool(args.dry_run),
        "krampus_dir": rel(krampus_dir),
        "file_filter": args.file or "",
        "max_files": args.max_files,
        "files_seen": 0,
        "files_processed": 0,
        "files_skipped": 0,
        "chunks_inserted": 0,
        "errors": [],
        "files": [],
        "canonical_graph_writes_performed": False,
        "source_files_deleted": False,
        "extractor": EXTRACTOR,
    }

    candidates = eligible_files(krampus_dir, args.file)
    if args.max_files:
        candidates = candidates[: args.max_files]

    candidate_source_keys: set[str] = set()
    for file_path in candidates:
        candidate_source_keys.update(source_path_keys(str(file_path)))
        candidate_source_keys.update(source_path_keys(source_path_for(file_path)))

    conn = psycopg.connect(args.storage_dsn)
    cursor = conn.cursor()
    try:
        if candidate_source_keys:
            cursor.execute(
                "SELECT source_path FROM lucidota_korpus.corpus_chunk WHERE source_path = ANY(%s)",
                (sorted(candidate_source_keys),),
            )
        else:
            cursor.execute("SELECT source_path FROM lucidota_korpus.corpus_chunk WHERE false")
        existing_source_keys: set[str] = set()
        for row in cursor.fetchall():
            existing_source_keys.update(source_path_keys(row[0]))

        for file_path in candidates:
            report["files_seen"] += 1
            source_path = source_path_for(file_path)

            if source_path_keys(str(file_path)) & existing_source_keys or source_path in existing_source_keys:
                report["files_skipped"] += 1
                entry = {"path": str(file_path), "source_path": source_path, "status": "skipped", "reason": "already_ingested"}
                report["files"].append(entry)
                if not args.json:
                    print(f"Skipping already ingested file: {file_path}")
                continue

            try:
                content = extract_content(file_path)
                if not content or not content.strip():
                    report["files_skipped"] += 1
                    entry = {"path": str(file_path), "source_path": source_path, "status": "skipped", "reason": "empty_or_unextractable"}
                    report["files"].append(entry)
                    if not args.json:
                        print(f"Skipping empty file: {file_path}")
                    continue

                inserted = insert_chunks(cursor, file_path, source_path, content, dry_run=args.dry_run)
                report["chunks_inserted"] += inserted
                if inserted > 0:
                    if not args.dry_run:
                        conn.commit()
                        existing_source_keys.update(source_path_keys(source_path))
                    report["files_processed"] += 1
                    entry = {"path": str(file_path), "source_path": source_path, "status": "success", "chunks": inserted}
                else:
                    report["files_skipped"] += 1
                    entry = {"path": str(file_path), "source_path": source_path, "status": "skipped", "reason": "already_ingested_or_duplicate"}
                report["files"].append(entry)
                if not args.json:
                    if inserted > 0:
                        print(f"Successfully ingested {file_path} with {inserted} chunks")
                    else:
                        print(f"Skipping duplicate-content file: {file_path}")
            except Exception as exc:
                report["status"] = "DEGRADED"
                try:
                    conn.rollback()
                except Exception:
                    pass
                entry = {"path": str(file_path), "source_path": source_path, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                report["files"].append(entry)
                report["errors"].append(entry)
                if not args.json:
                    print(f"Error processing {file_path}: {exc}")
    finally:
        cursor.close()
        conn.close()

    report["elapsed_s"] = round(time.time() - started, 3)
    receipt_key = hashlib.sha256(
        json.dumps(
            {
                "schema": SCHEMA,
                "file_filter": args.file or "",
                "max_files": args.max_files,
                "files": [(row.get("source_path"), row.get("status"), row.get("chunks")) for row in report["files"]],
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    receipt_path = receipt_dir / f"krampus_pdf_{receipt_key}.json"
    report["receipt_path"] = rel(receipt_path)
    receipt_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded ingest of top-level KRAMPUS documents into lucidota_korpus.corpus_chunk")
    parser.add_argument("--storage-dsn", default=os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage"))
    parser.add_argument("--krampus-dir", default=str(KRAMPUS_DIR))
    parser.add_argument("--receipt-dir", default=str(RECEIPT_DIR))
    parser.add_argument("--file", help="Process one top-level KRAMPUS file by name or path")
    parser.add_argument("--max-files", type=int, default=0, help="Maximum eligible files to consider; 0 means no cap")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(f"KRAMPUS_INGEST={report['status']}")
        print(f"FILES_SEEN={report['files_seen']}")
        print(f"FILES_PROCESSED={report['files_processed']}")
        print(f"FILES_SKIPPED={report['files_skipped']}")
        print(f"CHUNKS_INSERTED={report['chunks_inserted']}")
        print(f"RECEIPT_PATH={report['receipt_path']}")
    return 0 if report["status"] in {"PASS", "DEGRADED"} else 4


def extract_content(file_path: Path) -> str | None:
    if file_path.suffix.lower() == ".pdf":
        try:
            result = subprocess.run(
                ["pdftotext", "-", "-"],
                input=file_path.read_bytes(),
                capture_output=True,
                check=True,
            )
            return result.stdout.decode(errors="replace")
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    if file_path.suffix.lower() == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(file_path.read_bytes()))
        return "\n".join(p.text for p in doc.paragraphs)

    if file_path.suffix.lower() == ".odt":
        with zipfile.ZipFile(file_path) as z:
            with z.open("content.xml") as f:
                xml = f.read().decode(errors="replace")
                return re.sub("<[^>]+>", "", xml)

    if file_path.suffix.lower() in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    return None


if __name__ == "__main__":
    raise SystemExit(main())
