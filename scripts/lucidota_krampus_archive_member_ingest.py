#!/usr/bin/env python3
"""Bounded KRAMPUS archive-member ingestion into corpus_chunk.

Archive files in KRAMPUSCHEWING are containers, not exemptions. This opens
bounded zip archives, ingests extractable document/text members, preserves the
original archive bytes, and writes a receipt for every run.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import psycopg

ROOT = Path("/home/mfspx/LUCIDOTA")
KRAMPUS_DIR = ROOT / "KRAMPUSCHEWING"
RECEIPT_DIR = ROOT / "05_OUTPUTS" / "receipts"
ARCHIVE_EXTENSIONS = (".zip", ".7z")
ELIGIBLE_MEMBER_EXTENSIONS = (".pdf", ".docx", ".odt", ".txt", ".md")
SCHEMA = "lucidota.krampus_archive_member_ingest.v1"
EXTRACTOR = "krampus_archive_member_v1"
DEFAULT_MAX_MEMBER_BYTES = int(os.environ.get("LUCIDOTA_ARCHIVE_MEMBER_CAP_BYTES", str(64 * 1024 * 1024)))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def safe_receipt_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")[:120] or "archive"


def unsafe_member(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if normalized.startswith("/"):
        return "absolute_path"
    if pure.parts and re.match(r"^[A-Za-z]:", pure.parts[0]):
        return "drive_root"
    if ".." in pure.parts:
        return "path_traversal"
    if not normalized.strip("/"):
        return "empty_path"
    return None


def source_path_for(archive_path: Path, member_name: str, krampus_dir: Path) -> str:
    try:
        archive_rel = archive_path.resolve(strict=False).relative_to(krampus_dir.resolve(strict=False)).as_posix()
    except Exception:
        archive_rel = archive_path.name
    member_rel = member_name.replace("\\", "/").lstrip("/")
    return f"KRAMPUSCHEWING/{archive_rel}!{member_rel}"


def archive_prefix_for(archive_path: Path, krampus_dir: Path) -> str:
    try:
        archive_rel = archive_path.resolve(strict=False).relative_to(krampus_dir.resolve(strict=False)).as_posix()
    except Exception:
        archive_rel = archive_path.name
    return f"KRAMPUSCHEWING/{archive_rel}"


def source_path_keys(raw_path: str) -> set[str]:
    text = str(raw_path or "").strip()
    if not text:
        return set()
    keys = {text}
    if text.startswith("KRAMPUSCHEWING/"):
        keys.add(text.removeprefix("KRAMPUSCHEWING/"))
    else:
        keys.add(f"KRAMPUSCHEWING/{text}")
    return {k for k in keys if k}


def archive_already_opened(cursor: Any, archive_path: Path, krampus_dir: Path) -> bool:
    if cursor is None:
        return False
    prefixes = sorted(source_path_keys(archive_prefix_for(archive_path, krampus_dir)))
    member_prefixes = [f"{prefix}!%" for prefix in prefixes]
    cursor.execute(
        """
        SELECT source_path
        FROM lucidota_korpus.corpus_chunk
        WHERE source_path = ANY(%s) OR source_path LIKE ANY(%s)
        LIMIT 1
        """,
        (prefixes, member_prefixes),
    )
    return bool(cursor.fetchall())


def eligible_archives(krampus_dir: Path, only_archive: str | None = None) -> list[Path]:
    if only_archive:
        candidate = Path(only_archive)
        if not candidate.is_absolute():
            candidate = krampus_dir / candidate
        return [candidate] if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in ARCHIVE_EXTENSIONS else []
    if not krampus_dir.exists():
        return []
    return sorted(p for p in krampus_dir.iterdir() if p.is_file() and p.suffix.lower() in ARCHIVE_EXTENSIONS)


def eligible_member(name: str) -> bool:
    return Path(name).suffix.lower() in ELIGIBLE_MEMBER_EXTENSIONS


def archive_member(name: str) -> bool:
    return Path(name).suffix.lower() in ARCHIVE_EXTENSIONS


def seven_zip_tool() -> str | None:
    env_tool = os.environ.get("LUCIDOTA_7Z")
    if env_tool and os.access(env_tool, os.X_OK):
        return env_tool
    local_tool = ROOT / "09_STORAGE" / "tool_cache" / "apt" / "7zip_root" / "usr" / "lib" / "7zip" / "7z"
    if local_tool.exists() and os.access(local_tool, os.X_OK):
        return str(local_tool)
    for name in ("7z", "7zz"):
        found = shutil.which(name)
        if found:
            return found
    return None


def parse_7z_slt(stdout: bytes) -> list[dict[str, str]]:
    text = stdout.decode("utf-8", errors="replace")
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        current[key.strip()] = value.strip()
    if current:
        records.append(current)
    return records


def list_7z_members(archive_path: Path) -> list[dict[str, Any]]:
    tool = seven_zip_tool()
    if not tool:
        raise RuntimeError("missing_7z_tool:p7zip-full_or_7zip_required")
    proc = subprocess.run([tool, "l", "-slt", str(archive_path)], capture_output=True, check=True)
    out: list[dict[str, Any]] = []
    for rec in parse_7z_slt(proc.stdout):
        path = rec.get("Path", "")
        if not path or path == str(archive_path) or path == archive_path.name:
            continue
        folder = rec.get("Folder", "")
        attrs = rec.get("Attributes", "")
        if folder == "+" or attrs.startswith("D"):
            continue
        try:
            size = int(rec.get("Size") or 0)
        except ValueError:
            size = 0
        out.append({"path": path, "size": size})
    return out


def extract_7z_member(archive_path: Path, member_name: str) -> bytes:
    tool = seven_zip_tool()
    if not tool:
        raise RuntimeError("missing_7z_tool:p7zip-full_or_7zip_required")
    proc = subprocess.run([tool, "x", "-so", str(archive_path), "--", member_name], capture_output=True, check=True)
    return bytes(proc.stdout)


def chunk_text(text: str, max_chars: int = 1800) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars) if text[i : i + max_chars]]


def extract_member_content(member_name: str, data: bytes) -> str | None:
    suffix = Path(member_name).suffix.lower()
    if suffix == ".pdf":
        try:
            result = subprocess.run(["pdftotext", "-", "-"], input=data, capture_output=True, check=True)
            return result.stdout.decode(errors="replace")
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    if suffix == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    if suffix == ".odt":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open("content.xml") as f:
                xml = f.read().decode(errors="replace")
        return re.sub("<[^>]+>", "", xml)

    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")

    return None


def insert_chunks(cursor: Any, source_path: str, member_name: str, content: str, *, dry_run: bool) -> int:
    inserted = 0
    for chunk_index, chunk in enumerate(chunk_text(content)):
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
            (chunk_uuid, chunk_sha, source_path, Path(member_name).suffix.lower(), chunk_index, chunk, EXTRACTOR),
        )
        rowcount = getattr(cursor, "rowcount", 1)
        inserted += 1 if rowcount is None or rowcount > 0 else 0
    return inserted


def already_ingested(cursor: Any, source_path: str) -> bool:
    if cursor is None:
        return False
    cursor.execute(
        "SELECT source_path FROM lucidota_korpus.corpus_chunk WHERE source_path = ANY(%s) LIMIT 1",
        (sorted(source_path_keys(source_path)),),
    )
    return bool(cursor.fetchall())


def run(args: argparse.Namespace) -> dict[str, Any]:
    krampus_dir = Path(args.krampus_dir)
    receipt_dir = Path(args.receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    all_archives = eligible_archives(krampus_dir, args.archive)
    archives = list(all_archives)
    skipped_opened_archives: list[str] = []
    archive_selection = "explicit" if args.archive else "all"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "status": "PASS",
        "dry_run": bool(args.dry_run),
        "krampus_dir": rel(krampus_dir),
        "archive_filter": args.archive or "",
        "max_archives": args.max_archives,
        "max_members": args.max_members,
        "archives_seen": len(archives),
        "archives_opened": 0,
        "nested_archives_opened": 0,
        "members_seen": 0,
        "files_processed": 0,
        "files_skipped": 0,
        "unsafe_members_skipped": 0,
        "non_document_members_skipped": 0,
        "oversize_members_skipped": 0,
        "chunks_inserted": 0,
        "errors": [],
        "files": [],
        "canonical_graph_writes_performed": False,
        "source_files_deleted": False,
        "deleted_archives": [],
        "extractor": EXTRACTOR,
    }

    existing_source_keys: set[str] = set()
    conn = None
    cursor = None
    if not args.dry_run:
        conn = psycopg.connect(args.storage_dsn)
        cursor = conn.cursor()

    if not args.archive and not args.include_opened and cursor is not None:
        archive_selection = "pending_unopened"
        pending_archives: list[Path] = []
        for archive_path in all_archives:
            if archive_already_opened(cursor, archive_path, krampus_dir):
                skipped_opened_archives.append(rel(archive_path))
            else:
                pending_archives.append(archive_path)
        archives = pending_archives
    elif not args.archive and args.include_opened:
        archive_selection = "include_opened"
    elif not args.archive and cursor is None:
        archive_selection = "all_no_db_filter"

    if args.max_archives:
        archives = archives[: args.max_archives]

    report["archive_selection"] = archive_selection
    report["archives_available"] = len(all_archives)
    report["archives_skipped_opened"] = len(skipped_opened_archives)
    report["skipped_opened_archives"] = skipped_opened_archives
    report["archives_seen"] = len(archives)

    try:
        def at_member_cap() -> bool:
            return bool(args.max_members and int(report["members_seen"]) >= args.max_members)

        def oversize(size: int | None) -> bool:
            return bool(args.max_member_bytes and size is not None and size > args.max_member_bytes)

        def process_nested_archive_bytes(data: bytes, member_source_path: str, member_name: str, depth: int) -> None:
            suffix = Path(member_name).suffix.lower()
            if suffix == ".zip":
                with zipfile.ZipFile(io.BytesIO(data)) as nested:
                    report["nested_archives_opened"] += 1
                    report["archives_opened"] += 1
                    process_zip(nested, member_source_path, depth + 1)
                return
            if suffix == ".7z":
                with tempfile.NamedTemporaryFile(suffix=".7z") as tmp:
                    tmp.write(data)
                    tmp.flush()
                    report["nested_archives_opened"] += 1
                    report["archives_opened"] += 1
                    process_7z(Path(tmp.name), member_source_path, depth + 1)
                return
            raise ValueError(f"unsupported_nested_archive:{suffix}")

        def process_zip(zf: zipfile.ZipFile, prefix: str, depth: int) -> None:
            if depth > args.max_depth:
                report["status"] = "DEGRADED"
                report["errors"].append({"archive_prefix": prefix, "status": "error", "error": "max_depth_exceeded"})
                return
            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                if at_member_cap():
                    return
                if info.is_dir():
                    continue
                member_name = info.filename.replace("\\", "/").lstrip("/")
                reason = unsafe_member(info.filename)
                if reason:
                    report["unsafe_members_skipped"] += 1
                    continue

                member_source_path = f"{prefix}!{member_name}"
                if archive_member(member_name):
                    if oversize(int(info.file_size or 0)):
                        report["oversize_members_skipped"] += 1
                        report["files_skipped"] += 1
                        report["files"].append(
                            {
                                "member_path": member_name,
                                "source_path": member_source_path,
                                "status": "skipped",
                                "reason": "nested_archive_over_member_cap",
                            }
                        )
                        continue
                    try:
                        nested_bytes = zf.read(info)
                        process_nested_archive_bytes(nested_bytes, member_source_path, member_name, depth)
                    except Exception as exc:
                        report["status"] = "DEGRADED"
                        entry = {
                            "member_path": member_name,
                            "source_path": member_source_path,
                            "status": "error",
                            "error": f"nested_archive_open_failed:{type(exc).__name__}: {exc}",
                        }
                        report["files"].append(entry)
                        report["errors"].append(entry)
                    continue

                if not eligible_member(member_name):
                    report["non_document_members_skipped"] += 1
                    continue

                report["members_seen"] += 1
                if member_source_path in existing_source_keys or already_ingested(cursor, member_source_path):
                    existing_source_keys.update(source_path_keys(member_source_path))
                    report["files_skipped"] += 1
                    report["files"].append(
                        {
                            "member_path": member_name,
                            "source_path": member_source_path,
                            "status": "skipped",
                            "reason": "already_ingested",
                        }
                    )
                    continue
                if oversize(int(info.file_size or 0)):
                    report["oversize_members_skipped"] += 1
                    report["files_skipped"] += 1
                    report["files"].append(
                        {
                            "member_path": member_name,
                            "source_path": member_source_path,
                            "status": "skipped",
                            "reason": "member_over_cap",
                        }
                    )
                    continue
                try:
                    data = zf.read(info)
                    content = extract_member_content(member_name, data)
                    if not content or not content.strip():
                        report["files_skipped"] += 1
                        report["files"].append(
                            {
                                "member_path": member_name,
                                "source_path": member_source_path,
                                "status": "skipped",
                                "reason": "empty_or_unextractable",
                            }
                        )
                        continue
                    inserted = insert_chunks(cursor, member_source_path, member_name, content, dry_run=args.dry_run)
                    if conn is not None:
                        conn.commit()
                    existing_source_keys.update(source_path_keys(member_source_path))
                    report["chunks_inserted"] += inserted
                    report["files_processed"] += 1
                    report["files"].append(
                        {
                            "member_path": member_name,
                            "source_path": member_source_path,
                            "status": "success",
                            "chunks": inserted,
                        }
                    )
                except Exception as exc:
                    report["status"] = "DEGRADED"
                    entry = {
                        "member_path": member_name,
                        "source_path": member_source_path,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    report["files"].append(entry)
                    report["errors"].append(entry)

        def process_7z(archive_path: Path, prefix: str, depth: int) -> None:
            if depth > args.max_depth:
                report["status"] = "DEGRADED"
                report["errors"].append({"archive_prefix": prefix, "status": "error", "error": "max_depth_exceeded"})
                return
            try:
                member_records = sorted(list_7z_members(archive_path), key=lambda row: str(row.get("path") or ""))
            except Exception as exc:
                report["status"] = "DEGRADED"
                entry = {"archive": prefix, "status": "error", "error": f"7z_list_failed:{type(exc).__name__}: {exc}"}
                report["files"].append(entry)
                report["errors"].append(entry)
                return
            for rec in member_records:
                if at_member_cap():
                    return
                member_name = str(rec.get("path") or "").replace("\\", "/").lstrip("/")
                reason = unsafe_member(member_name)
                if reason:
                    report["unsafe_members_skipped"] += 1
                    continue
                member_source_path = f"{prefix}!{member_name}"
                size = int(rec.get("size") or 0)
                if archive_member(member_name):
                    if oversize(size):
                        report["oversize_members_skipped"] += 1
                        report["files_skipped"] += 1
                        report["files"].append(
                            {
                                "member_path": member_name,
                                "source_path": member_source_path,
                                "status": "skipped",
                                "reason": "nested_archive_over_member_cap",
                            }
                        )
                        continue
                    try:
                        process_nested_archive_bytes(extract_7z_member(archive_path, member_name), member_source_path, member_name, depth)
                    except Exception as exc:
                        report["status"] = "DEGRADED"
                        entry = {
                            "member_path": member_name,
                            "source_path": member_source_path,
                            "status": "error",
                            "error": f"nested_archive_open_failed:{type(exc).__name__}: {exc}",
                        }
                        report["files"].append(entry)
                        report["errors"].append(entry)
                    continue
                if not eligible_member(member_name):
                    report["non_document_members_skipped"] += 1
                    continue
                report["members_seen"] += 1
                if member_source_path in existing_source_keys or already_ingested(cursor, member_source_path):
                    existing_source_keys.update(source_path_keys(member_source_path))
                    report["files_skipped"] += 1
                    report["files"].append(
                        {
                            "member_path": member_name,
                            "source_path": member_source_path,
                            "status": "skipped",
                            "reason": "already_ingested",
                        }
                    )
                    continue
                if oversize(size):
                    report["oversize_members_skipped"] += 1
                    report["files_skipped"] += 1
                    report["files"].append(
                        {
                            "member_path": member_name,
                            "source_path": member_source_path,
                            "status": "skipped",
                            "reason": "member_over_cap",
                        }
                    )
                    continue
                try:
                    content = extract_member_content(member_name, extract_7z_member(archive_path, member_name))
                    if not content or not content.strip():
                        report["files_skipped"] += 1
                        report["files"].append(
                            {
                                "member_path": member_name,
                                "source_path": member_source_path,
                                "status": "skipped",
                                "reason": "empty_or_unextractable",
                            }
                        )
                        continue
                    inserted = insert_chunks(cursor, member_source_path, member_name, content, dry_run=args.dry_run)
                    if conn is not None:
                        conn.commit()
                    existing_source_keys.update(source_path_keys(member_source_path))
                    report["chunks_inserted"] += inserted
                    report["files_processed"] += 1
                    report["files"].append(
                        {
                            "member_path": member_name,
                            "source_path": member_source_path,
                            "status": "success",
                            "chunks": inserted,
                        }
                    )
                except Exception as exc:
                    report["status"] = "DEGRADED"
                    entry = {
                        "member_path": member_name,
                        "source_path": member_source_path,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    report["files"].append(entry)
                    report["errors"].append(entry)

        for archive_path in archives:
            if at_member_cap():
                break
            archive_error_count_before = len(report["errors"])
            try:
                prefix = f"KRAMPUSCHEWING/{archive_path.resolve(strict=False).relative_to(krampus_dir.resolve(strict=False)).as_posix()}"
            except Exception:
                prefix = f"KRAMPUSCHEWING/{archive_path.name}"
            try:
                suffix = archive_path.suffix.lower()
                if suffix == ".zip":
                    with zipfile.ZipFile(archive_path) as zf:
                        report["archives_opened"] += 1
                        process_zip(zf, prefix, 0)
                elif suffix == ".7z":
                    report["archives_opened"] += 1
                    process_7z(archive_path, prefix, 0)
                else:
                    report["status"] = "DEGRADED"
                    entry = {"archive": archive_path.name, "status": "error", "error": f"unsupported_archive_type:{suffix}"}
                    report["files"].append(entry)
                    report["errors"].append(entry)
            except Exception as exc:
                report["status"] = "DEGRADED"
                entry = {"archive": archive_path.name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                report["files"].append(entry)
                report["errors"].append(entry)
            else:
                if not args.dry_run and len(report["errors"]) == archive_error_count_before:
                    try:
                        archive_path.unlink()
                        report["source_files_deleted"] = True
                        report["deleted_archives"].append(rel(archive_path))
                    except Exception as exc:
                        report["status"] = "DEGRADED"
                        entry = {
                            "archive": archive_path.name,
                            "status": "error",
                            "error": f"delete_failed:{type(exc).__name__}: {exc}",
                        }
                        report["files"].append(entry)
                        report["errors"].append(entry)
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

    report["elapsed_s"] = round(time.time() - started, 3)
    receipt_identity = {
        "schema": SCHEMA,
        "archive_filter": args.archive or "",
        "max_archives": args.max_archives,
        "max_members": args.max_members,
        "files": [(row.get("source_path"), row.get("status"), row.get("chunks")) for row in report["files"]],
        "dry_run": args.dry_run,
        "source_files_deleted": report["source_files_deleted"],
        "deleted_archives": report["deleted_archives"],
    }
    receipt_key = hashlib.sha256(json.dumps(receipt_identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    archive_slug = safe_receipt_key(args.archive or "batch")
    receipt_path = receipt_dir / f"krampus_archive_{archive_slug}_{receipt_key}.json"
    report["receipt_path"] = rel(receipt_path)
    receipt_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded ingest of KRAMPUS archive members into lucidota_korpus.corpus_chunk")
    parser.add_argument("--storage-dsn", default=os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage"))
    parser.add_argument("--krampus-dir", default=str(KRAMPUS_DIR))
    parser.add_argument("--receipt-dir", default=str(RECEIPT_DIR))
    parser.add_argument("--archive", help="Process one top-level KRAMPUS archive by name or path")
    parser.add_argument("--max-archives", type=int, default=0, help="Maximum archives to consider; 0 means no cap")
    parser.add_argument("--max-members", type=int, default=10, help="Maximum eligible archive members to ingest; 0 means no cap")
    parser.add_argument("--max-depth", type=int, default=8, help="Maximum nested archive depth")
    parser.add_argument("--max-member-bytes", type=int, default=DEFAULT_MAX_MEMBER_BYTES, help="Maximum bytes to read per member/nested archive; 0 means no cap")
    parser.add_argument("--include-opened", action="store_true", help="Do not skip archives that already have corpus chunks")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(f"KRAMPUS_ARCHIVE_INGEST={report['status']}")
        print(f"ARCHIVES_SEEN={report['archives_seen']}")
        print(f"MEMBERS_SEEN={report['members_seen']}")
        print(f"FILES_PROCESSED={report['files_processed']}")
        print(f"FILES_SKIPPED={report['files_skipped']}")
        print(f"CHUNKS_INSERTED={report['chunks_inserted']}")
        print(f"RECEIPT_PATH={report['receipt_path']}")
    return 0 if report["status"] in {"PASS", "DEGRADED"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
