#!/usr/bin/env python3
"""Hydra evidence bundle + text/Wayback diff.

Reads latest Hydra captures from local CAS, computes a compact text diff and
exports a JSON evidence bundle. Optional Wayback comparison fetches one archived
snapshot for current-vs-archive drift without making Wayback mandatory in the
harness.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import psycopg
import requests

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA" / "011_hydra_capture.sql"
DB = os.environ.get("LUCIDOTA_GRAPH_DATABASE_URL", "postgresql://mfspx@/lucidota_graph")
DEFAULT_VAULT = ROOT / "03_VAULT" / "cas"
DEFAULT_OUT = ROOT / "05_OUTPUTS" / "hydra_bundles"

import sys
sys.path.insert(0, str(ROOT / "scripts"))
from lucidota_hydra_capture import canonical_hashes  # noqa: E402


def cas_path(vault: Path, cas_uri: str) -> Path:
    prefix = "cas://sha256/"
    if not cas_uri.startswith(prefix):
        raise ValueError(f"unsupported cas uri: {cas_uri}")
    digest = cas_uri.removeprefix(prefix)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"invalid sha256 in cas uri: {cas_uri}")
    return vault / digest[:2] / digest[2:4] / digest


def text_from_bytes(data: bytes, mime: str) -> str:
    raw = data[:2_000_000].decode("utf-8", errors="ignore")
    # Keep this deliberately boring and dependency-free. It matches the v0
    # canonical text intent: remove code/style noise and normalize whitespace.
    raw = re.sub(r"(?is)<(script|style|noscript|template)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw) if "html" in (mime or "") or "<html" in raw[:2000].lower() else raw
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def chunk_lines(text: str, width: int = 100) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    n = 0
    for word in words:
        if current and n + 1 + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
            n = len(word)
        else:
            current.append(word)
            n += (1 if n else 0) + len(word)
    if current:
        lines.append(" ".join(current))
    return lines


def diff_text(old: str, new: str, limit: int) -> dict:
    old_lines = chunk_lines(old)
    new_lines = chunk_lines(new)
    raw = list(difflib.unified_diff(old_lines, new_lines, fromfile="old", tofile="new", lineterm=""))
    added = sum(1 for line in raw if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in raw if line.startswith("-") and not line.startswith("---"))
    return {
        "changed": old != new,
        "old_line_count": len(old_lines),
        "new_line_count": len(new_lines),
        "added_lines": added,
        "removed_lines": removed,
        "unified_diff": raw[:limit],
        "truncated": len(raw) > limit,
    }


def summarize(source: str, old_cap: dict | None, new_cap: dict, text_delta: dict, wayback: dict | None) -> str:
    if old_cap is None:
        base = f"first evidence bundle for {source}; current capture recorded"
    elif text_delta["changed"]:
        base = f"text changed for {source}: +{text_delta['added_lines']} -{text_delta['removed_lines']} diff lines"
    else:
        base = f"no text change for {source} between latest two captures"
    if wayback and wayback.get("status") == "ok":
        wb = wayback.get("text_delta", {})
        base += f"; Wayback comparison {wayback.get('timestamp')} changed={wb.get('changed')}"
    elif wayback:
        base += f"; Wayback comparison {wayback.get('status')}"
    return base


def latest_captures(conn, source: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT capture_id::text, source, capture_kind, sha256, cas_uri, size_bytes,
               mime, title, content_hash, structure_hash, visual_hash, created_at::text
        FROM lucidota_hydra.capture
        WHERE source=%s AND status='succeeded' AND cas_uri IS NOT NULL AND cas_uri <> ''
        ORDER BY created_at DESC LIMIT 2
        """,
        (source,),
    ).fetchall()
    keys = ["capture_id", "source", "capture_kind", "sha256", "cas_uri", "size_bytes", "mime", "title", "content_hash", "structure_hash", "visual_hash", "created_at"]
    return [dict(zip(keys, row)) for row in rows]


def wayback_compare(source: str, current_text: str, timeout: float, limit: int) -> dict:
    try:
        cdx = requests.get(
            "https://web.archive.org/cdx",
            params={"url": source, "output": "json", "fl": "timestamp,original,statuscode,mimetype,digest", "filter": "statuscode:200", "limit": "1", "sort": "reverse"},
            timeout=timeout,
        )
        if cdx.status_code != 200:
            return {"status": "error", "http_status": cdx.status_code}
        rows = cdx.json()
        if len(rows) < 2:
            return {"status": "none"}
        keys = rows[0]
        cap = dict(zip(keys, rows[1]))
        ts = cap["timestamp"]
        archive_url = f"https://web.archive.org/web/{ts}id_/{source}"
        archived = requests.get(archive_url, timeout=timeout)
        if archived.status_code != 200:
            return {"status": "error", "timestamp": ts, "archive_http_status": archived.status_code}
        mime = archived.headers.get("content-type", cap.get("mimetype", ""))
        arch_text = text_from_bytes(archived.content, mime)
        content_hash, structure_hash = canonical_hashes(archived.content, mime)
        return {
            "status": "ok",
            "timestamp": ts,
            "archive_url": archive_url,
            "archive_sha256": hashlib.sha256(archived.content).hexdigest(),
            "archive_content_hash": content_hash,
            "archive_structure_hash": structure_hash,
            "text_delta": diff_text(arch_text, current_text, limit),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:400]}


def safe_name(source: str) -> str:
    parsed = urlparse(source)
    stem = parsed.netloc + parsed.path if parsed.netloc else source
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "source"
    return stem[:100]


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-hydra-evidence")
    ap.add_argument("source")
    ap.add_argument("--db-url", default=DB)
    ap.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--diff-limit", type=int, default=80)
    ap.add_argument("--wayback", action="store_true", help="Also compare latest capture with newest Wayback snapshot.")
    ap.add_argument("--wayback-timeout", type=float, default=10.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    with psycopg.connect(args.db_url) as conn:
        conn.execute(SCHEMA.read_text())
        caps = latest_captures(conn, args.source)
        if not caps:
            report = {"ok": False, "error": "no successful hydra captures for source", "source": args.source}
            print(json.dumps(report, sort_keys=True) if args.json else report)
            return 1
        new = caps[0]
        old = caps[1] if len(caps) > 1 else None
        new_text = text_from_bytes(cas_path(args.vault, new["cas_uri"]).read_bytes(), new["mime"])
        old_text = text_from_bytes(cas_path(args.vault, old["cas_uri"]).read_bytes(), old["mime"]) if old else ""
        text_delta = diff_text(old_text, new_text, args.diff_limit) if old else diff_text("", new_text, args.diff_limit)
        wb = wayback_compare(args.source, new_text, args.wayback_timeout, args.diff_limit) if args.wayback else {"status": "skipped"}
        summary = summarize(args.source, old, new, text_delta, wb)
        bundle = {
            "schema": "lucidota.hydra.evidence_bundle.v0",
            "source": args.source,
            "generated_at_epoch": int(time.time()),
            "summary": summary,
            "old_capture": old,
            "new_capture": new,
            "text_delta": text_delta,
            "wayback_delta": wb,
        }
        payload = json.dumps(bundle, indent=2, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / f"{safe_name(args.source)}_{digest[:12]}.json"
        out_path.write_text(payload + "\n", encoding="utf-8")
        row = conn.execute(
            """
            INSERT INTO lucidota_hydra.evidence_bundle
              (source, old_capture_id, new_capture_id, bundle_sha256, bundle_path, summary, detail)
            VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)
            RETURNING bundle_id::text
            """,
            (args.source, old["capture_id"] if old else None, new["capture_id"], digest, str(out_path.relative_to(ROOT)), summary, json.dumps({"wayback_status": wb.get("status"), "text_changed": text_delta["changed"]})),
        ).fetchone()[0]
        # Put a compact summary back on the delta row when possible.
        conn.execute(
            """
            UPDATE lucidota_hydra.delta
            SET detail = detail || %s::jsonb
            WHERE source=%s AND new_capture_id=%s::uuid
            """,
            (json.dumps({"summary": summary, "evidence_bundle_id": row, "evidence_bundle_path": str(out_path.relative_to(ROOT))}), args.source, new["capture_id"]),
        )
        conn.commit()
    report = {"ok": True, "bundle_id": row, "bundle_path": str(out_path), "bundle_sha256": digest, "summary": summary, "wayback_status": wb.get("status"), "text_changed": text_delta["changed"]}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
