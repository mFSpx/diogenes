#!/usr/bin/env python3
"""Clean reingest for C_ARCHIVE email material.

The old C_ARCHIVE path produced many raw MIME / quoted-printable / base64 chunks.
This script parses nested .eml files with Python's email parser, extracts decoded
text/plain (or stripped text/html fallback), chunks readable text, and writes new
candidate rows without embedding inline.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import NAMESPACE_URL, uuid5

import psycopg

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "KRAMPUSCHEWING" / "C_ARCHIVE.zip"
OUT_DIR = ROOT / "05_OUTPUTS" / "ingestion_audit"
EXTRACTOR = "c_archive_email_mime_reingest_v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def html_to_text(value: str) -> str:
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?is)<br\s*/?>", "\n", value)
    value = re.sub(r"(?is)</p\s*>", "\n\n", value)
    value = re.sub(r"(?is)<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"[ \t]+", " ", value)


def _payload_text(part) -> str:
    try:
        value = part.get_content()
        if isinstance(value, str):
            return value
    except Exception:
        pass
    raw = part.get_payload(decode=True)
    if raw is None:
        payload = part.get_payload()
        if isinstance(payload, str):
            return payload
        return ""
    charset = part.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def extract_email_text(raw: bytes) -> str:
    import email
    import email.policy

    msg = email.message_from_bytes(raw, policy=email.policy.default)
    subject = str(msg.get("Subject", ""))
    from_ = str(msg.get("From", ""))
    date = str(msg.get("Date", ""))
    plain_parts: list[str] = []
    html_parts: list[str] = []

    for part in msg.walk():
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment":
            continue
        ctype = part.get_content_type()
        text = _payload_text(part)
        if not text.strip():
            continue
        if ctype == "text/plain":
            plain_parts.append(text)
        elif ctype == "text/html":
            html_parts.append(html_to_text(text))

    body = "\n".join(plain_parts).strip() or "\n".join(html_parts).strip()
    body = re.sub(r"\r\n?", "\n", body)
    body = re.sub(r"\n{4,}", "\n\n\n", body)
    if not body.strip():
        return ""
    return f"Subject: {subject}\nFrom: {from_}\nDate: {date}\n\n{body.strip()}"


def chunk_text(text: str, max_chars: int = 1800) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def iter_email_records(archive_path: Path, limit_emails: int = 0) -> Iterable[tuple[str, bytes]]:
    yielded = 0
    with zipfile.ZipFile(archive_path) as outer:
        for outer_info in outer.infolist():
            outer_name = outer_info.filename
            low_outer = outer_name.lower()
            if outer_info.is_dir():
                continue
            if low_outer.endswith(".eml"):
                yield f"{archive_path.name}!{outer_name}", outer.read(outer_info)
                yielded += 1
            elif low_outer.endswith(".zip"):
                try:
                    data = outer.read(outer_info)
                    nested = zipfile.ZipFile(io.BytesIO(data))
                except Exception:
                    continue
                with nested:
                    for inner_info in nested.infolist():
                        if inner_info.is_dir() or not inner_info.filename.lower().endswith(".eml"):
                            continue
                        yield f"{archive_path.name}!{outer_name}!{inner_info.filename}", nested.read(inner_info)
                        yielded += 1
                        if limit_emails and yielded >= limit_emails:
                            return
            if limit_emails and yielded >= limit_emails:
                return


def insert_email_chunks(conn, source_path: str, text: str) -> int:
    chunks = chunk_text(text)
    rows = []
    for idx, chunk in enumerate(chunks):
        sha = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        chunk_uuid = str(uuid5(NAMESPACE_URL, f"{EXTRACTOR}:{source_path}:{idx}:{sha}"))
        rows.append(
            (
                chunk_uuid,
                sha,
                source_path,
                "message/rfc822-clean",
                idx,
                chunk,
                json.dumps(
                    {
                        "reingest": EXTRACTOR,
                        "supersedes_extractors": ["krampus_skip_embed"],
                        "embedding_quality_status": "pass",
                        "created_by": "scripts/lucidota_c_archive_email_reingest.py",
                    },
                    sort_keys=True,
                ),
                "bge-m3",
                EXTRACTOR,
            )
        )
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO lucidota_korpus.corpus_chunk
              (chunk_uuid, file_uuid, sha256, source_path, mime, chunk_index,
               content, go25, embedding, embedding_model, extractor)
            VALUES (%s::uuid, NULL, %s, %s, %s, %s, %s, %s::jsonb, NULL, %s, %s)
            ON CONFLICT (sha256, chunk_index) DO NOTHING
            """,
            rows,
        )
        return cur.rowcount if cur.rowcount is not None else 0


def run(args: argparse.Namespace) -> dict:
    archive_path = Path(args.archive).expanduser()
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path
    dsn = args.storage_dsn
    started = time.time()
    report = {
        "schema": "lucidota.c_archive_email_mime_reingest.v1",
        "generated_at": now_iso(),
        "archive_path": rel(archive_path),
        "execute": bool(args.execute),
        "limit_emails": args.limit_emails,
        "emails_seen": 0,
        "emails_with_text": 0,
        "chunks_inserted": 0,
        "skipped_empty_or_short": 0,
        "errors": [],
        "extractor": EXTRACTOR,
    }
    conn = psycopg.connect(dsn) if args.execute else None
    try:
        for source_path, raw in iter_email_records(archive_path, args.limit_emails):
            report["emails_seen"] += 1
            try:
                text = extract_email_text(raw)
                if not text:
                    report["skipped_empty_or_short"] += 1
                    continue
                report["emails_with_text"] += 1
                if args.execute and conn is not None:
                    report["chunks_inserted"] += insert_email_chunks(conn, source_path, text)
                    if report["emails_seen"] % args.commit_every == 0:
                        conn.commit()
                else:
                    report["chunks_inserted"] += len(chunk_text(text))
            except Exception as exc:
                report["errors"].append({"source_path": source_path, "error": f"{type(exc).__name__}: {exc}"})
                if len(report["errors"]) >= 50:
                    break
        if conn is not None:
            conn.commit()
    finally:
        if conn is not None:
            conn.close()
    report["elapsed_s"] = round(time.time() - started, 2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = OUT_DIR / f"c_archive_email_mime_reingest_{stamp()}.json"
    report["receipt_path"] = rel(receipt)
    receipt.write_text(json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean reingest of C_ARCHIVE nested .eml files")
    p.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    p.add_argument("--storage-dsn", default=os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage"))
    p.add_argument("--limit-emails", type=int, default=0)
    p.add_argument("--commit-every", type=int, default=100)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    print(f"REINGEST_RECEIPT={report['receipt_path']}")
    print(f"EMAILS_SEEN={report['emails_seen']}")
    print(f"EMAILS_WITH_TEXT={report['emails_with_text']}")
    print(f"CHUNKS={'inserted' if args.execute else 'would_insert'}={report['chunks_inserted']}")
    print(f"ERRORS={len(report['errors'])}")
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
