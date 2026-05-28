#!/usr/bin/env python3
"""Universal communications dump ingester.

Handles non-exhaustive personal data dumps: Facebook messages, email JSON,
SMS/text JSON/XML-ish shapes, JSONL, ZIPs, and directories. It is heuristic and
schema-tolerant by design: preserve what can be normalized, CAS-lock everything.
No LLM calls.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import email.utils
import hashlib
import io
import json
import mimetypes
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator
from collections import defaultdict

import psycopg

ROOT = Path(__file__).resolve().parents[1]
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
SCHEMA = ROOT / "06_SCHEMA" / "022_comm_dump_timeline.sql"
WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
TEXT_EXTS = {".json", ".jsonl", ".ndjson", ".txt", ".csv", ".xml"}

sys.path.insert(0, str(ROOT / "scripts"))
from lucidota_artifact_ingest import ensure_state, normalize_ws, sha256_file, store_locked_cas  # type: ignore  # noqa: E402


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA.read_text(encoding="utf-8"))
    conn.commit()


def emit_event(run_id: str, status: str, detail: dict[str, Any]) -> str:
    with psycopg.connect(STATE_DSN) as conn:
        ensure_state(conn)
        conn.execute(WORKFLOW_SCHEMA.read_text(encoding="utf-8"))
        row = conn.execute(
            """
            INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
            VALUES ('commdump-timeline-ingest',%s,'ingest',%s,'commdump_timeline',%s::jsonb)
            RETURNING event_id::text
            """,
            (run_id, status, jdump(detail)),
        ).fetchone()
        conn.commit()
        return str(row[0])


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def parse_time(value: Any) -> tuple[dt.datetime | None, str]:
    if value is None or value == "":
        return None, ""
    raw = str(value)
    if isinstance(value, dt.datetime):
        val = value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
        return val.astimezone(dt.timezone.utc), raw
    if isinstance(value, (int, float)) or re.fullmatch(r"-?\d+(?:\.\d+)?", raw.strip()):
        try:
            f = float(value)
            # ns/us/ms/seconds all normalize to UTC seconds.
            if abs(f) > 1e17:
                f /= 1_000_000_000.0
            elif abs(f) > 1e14:
                f /= 1_000_000.0
            elif abs(f) > 1e11:
                f /= 1000.0
            parsed = dt.datetime.fromtimestamp(f, tz=dt.timezone.utc)
            if 1990 <= parsed.year <= dt.datetime.now(dt.timezone.utc).year + 5:
                return parsed, raw
        except (OverflowError, OSError, ValueError):
            numeric_error = True
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError, OverflowError):
        parsed = None
    if parsed:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc), raw
    try:
        s = raw.strip().replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(s)
    except ValueError:
        return None, raw
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc), raw


def best_time(obj: dict[str, Any]) -> tuple[dt.datetime | None, str, str, int]:
    keys = [
        ("timestamp_ms", 10000), ("timestamp", 10000), ("date", 9500), ("datetime", 9500), ("created_at", 9500),
        ("time", 9000), ("sent_at", 9500), ("received_at", 9000), ("internalDate", 9000), ("created", 8500),
    ]
    for k, conf in keys:
        if k in obj:
            t, raw = parse_time(obj.get(k))
            if t is not None:
                return t, raw, k, conf
    headers = obj.get("headers") if isinstance(obj.get("headers"), dict) else {}
    for k in ("Date", "date"):
        if k in headers:
            t, raw = parse_time(headers.get(k))
            if t is not None:
                return t, raw, f"headers.{k}", 9000
    return None, "", "missing", 0


def source_kind(path: Path, member: str, obj: Any | None = None, hint: str = "auto") -> str:
    if hint != "auto":
        return hint
    hay = f"{path.name} {member}".lower()
    if any(x in hay for x in ["facebook", "meta", "messenger", "inbox/message", "messages/inbox"]):
        return "facebook"
    if any(x in hay for x in ["sms", "text", "imessage", "messages.csv", "messages.xml"]):
        return "sms" if "imessage" not in hay else "imessage"
    if any(x in hay for x in ["email", "gmail", "mail", "takeout"]):
        return "email"
    if isinstance(obj, dict):
        if "messages" in obj and ("participants" in obj or "thread_path" in obj):
            return "facebook"
        keys = set(obj.keys())
        if keys & {"from", "to", "cc", "bcc", "subject", "snippet", "body", "headers"}:
            return "email"
        if keys & {"address", "readable_date", "contact_name", "smses", "mmses"}:
            return "sms"
    return "generic"


def text_from_any(value: Any) -> tuple[str, str, list[dict[str, Any]]]:
    attachments: list[dict[str, Any]] = []
    if value is None:
        return "", "", attachments
    if isinstance(value, str):
        return value, "text", attachments
    if isinstance(value, list):
        parts = []
        for item in value:
            t, _, a = text_from_any(item)
            if t:
                parts.append(t)
            attachments.extend(a)
        return "\n".join(parts), "list", attachments
    if isinstance(value, dict):
        if any(k in value for k in ("filename", "uri", "href", "url", "mime_type")):
            attachments.append({k: value.get(k) for k in ("filename", "uri", "href", "url", "mime_type", "creation_timestamp") if k in value})
        for k in ("content", "body", "text", "snippet", "message", "plain", "html", "value"):
            if isinstance(value.get(k), str):
                return value.get(k, ""), k, attachments
            if isinstance(value.get(k), (dict, list)):
                t, kind, a = text_from_any(value.get(k))
                attachments.extend(a)
                if t:
                    return t, kind, attachments
        return normalize_ws(json.dumps(value, ensure_ascii=False, default=str))[:20000], "dict", attachments
    return str(value), type(value).__name__, attachments


def iter_json_array(fp: io.TextIOBase) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    buf = ""
    started = False
    eof = False
    while True:
        if not eof:
            chunk = fp.read(1024 * 1024)
            if chunk == "":
                eof = True
            else:
                buf += chunk
        while True:
            s = buf.lstrip()
            if not started:
                if not s:
                    break
                if s[0] != "[":
                    raise ValueError("not array")
                started = True
                buf = s[1:]
                continue
            if not s:
                break
            if s[0] == "]":
                return
            if s[0] == ",":
                buf = s[1:]
                continue
            try:
                obj, idx = decoder.raw_decode(s)
            except json.JSONDecodeError:
                if eof:
                    raise
                break
            yield obj
            buf = s[idx:]
        if eof:
            return


def json_roots(fp: io.TextIOBase, name: str) -> Iterator[Any]:
    low = name.lower()
    if low.endswith((".jsonl", ".ndjson")):
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    prefix = fp.read(4096)
    fp.seek(0)
    if prefix.lstrip().startswith("["):
        yield from iter_json_array(fp)
        return
    data = json.load(fp)
    yield data


def flatten_message_objects(obj: Any) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield (thread_meta, message_obj). Thread meta may be empty."""
    if isinstance(obj, list):
        for item in obj:
            yield from flatten_message_objects(item)
        return
    if not isinstance(obj, dict):
        return
    # Facebook thread export: root has participants/messages/title.
    if isinstance(obj.get("messages"), list):
        thread_meta = {k: obj.get(k) for k in ("title", "participants", "thread_path", "is_still_participant") if k in obj}
        for msg in obj["messages"]:
            if isinstance(msg, dict):
                yield thread_meta, msg
        return
    # Google Takeout/Gmail-ish root arrays under messages/emails.
    for key in ("emails", "mail", "smses", "mmses", "texts", "items", "data"):
        if isinstance(obj.get(key), list):
            thread_meta = {k: obj.get(k) for k in ("title", "participants", "thread_id", "conversation_id") if k in obj}
            for msg in obj[key]:
                if isinstance(msg, dict):
                    yield thread_meta, msg
            return
    # Single message object.
    if any(k in obj for k in ("sender_name", "from", "to", "address", "body", "content", "snippet", "subject", "timestamp_ms", "date")):
        yield {}, obj


def extract_message(kind: str, thread_meta: dict[str, Any], msg: dict[str, Any], member: str, seq: int) -> dict[str, Any]:
    sender = str(msg.get("sender_name") or msg.get("sender") or msg.get("from") or msg.get("address") or msg.get("contact_name") or "")
    recipients = msg.get("recipients") or msg.get("to") or msg.get("cc") or []
    if isinstance(recipients, str):
        recipients = [recipients]
    if isinstance(recipients, dict):
        recipients = [recipients]
    subject = str(msg.get("subject") or msg.get("title") or thread_meta.get("title") or "")
    text, content_kind, attachments = text_from_any(msg.get("content") if "content" in msg else msg.get("body") if "body" in msg else msg.get("text") if "text" in msg else msg.get("snippet") if "snippet" in msg else msg)
    for k in ("photos", "videos", "audio_files", "files", "share", "gifs", "sticker", "attachments"):
        val = msg.get(k)
        if val:
            _, _, a = text_from_any(val)
            attachments.extend(a or ([{"kind": k, "value": val}] if not isinstance(val, (dict, list)) else []))
    occurred, raw, time_source, conf = best_time(msg)
    provider_id = str(msg.get("id") or msg.get("message_id") or msg.get("mid") or msg.get("guid") or msg.get("_id") or seq)
    return {
        "source_kind": kind,
        "thread_id": str(thread_meta.get("thread_id") or thread_meta.get("conversation_id") or thread_meta.get("thread_path") or thread_meta.get("title") or member),
        "thread_title": str(thread_meta.get("title") or msg.get("thread_title") or subject or member),
        "participants": thread_meta.get("participants") or [],
        "provider_message_id": provider_id,
        "sender": sender,
        "recipients": recipients,
        "occurred_at": occurred,
        "occurred_at_raw": raw,
        "time_source": time_source,
        "time_confidence_bps": conf,
        "sequence_index": seq,
        "subject": subject,
        "content_text": text,
        "content_kind": content_kind,
        "attachments": attachments,
        "raw": {k: msg.get(k) for k in ("type", "is_unsent", "reactions", "ip", "service") if k in msg},
    }


def iter_sources(path: Path) -> Iterator[tuple[Path, str, int, Any]]:
    path = path.expanduser().resolve()
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and (child.suffix.lower() in TEXT_EXTS or child.suffix.lower() == ".zip"):
                yield from iter_sources(child)
        return
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        for info in zf.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix in TEXT_EXTS:
                def opener(zf=zf, name=info.filename):
                    return io.TextIOWrapper(zf.open(name, "r"), encoding="utf-8", errors="replace")
                yield path, info.filename, int(info.file_size), opener
        return
    if path.suffix.lower() in TEXT_EXTS:
        def opener(path=path):
            return path.open("r", encoding="utf-8", errors="replace")
        yield path, path.name, path.stat().st_size, opener


def directory_manifest(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    for child in sorted(x for x in path.rglob("*") if x.is_file()):
        try:
            st = child.stat()
        except OSError:
            continue
        rel = child.relative_to(path).as_posix()
        total += int(st.st_size)
        h.update(rel.encode("utf-8", errors="replace"))
        h.update(str(int(st.st_size)).encode())
        h.update(str(int(st.st_mtime_ns)).encode())
    return h.hexdigest(), total


def upsert_export(conn: psycopg.Connection, path: Path, kind: str) -> tuple[str, str]:
    if path.is_dir():
        digest, size_bytes = directory_manifest(path)
        mime = "inode/directory"
        cas_uri = f"dir-manifest://sha256/{digest}"
        rel_locked = str(path)
    else:
        digest = sha256_file(path)
        size_bytes = path.stat().st_size
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        cas_uri, locked = store_locked_cas(path, digest, mime, "commdump_timeline")
        rel_locked = str(locked.relative_to(ROOT)) if locked.is_relative_to(ROOT) else str(locked)
    row = conn.execute(
        """
        INSERT INTO lucidota_commdump.export_object(source_kind, source_path, source_sha256, size_bytes, mime, cas_uri, locked_relative_path, status, detail)
        VALUES (%s,%s,%s,%s,%s,%s,%s,'running',%s::jsonb)
        ON CONFLICT(source_sha256, source_kind) DO UPDATE SET source_path=EXCLUDED.source_path, size_bytes=EXCLUDED.size_bytes, mime=EXCLUDED.mime, cas_uri=EXCLUDED.cas_uri, locked_relative_path=EXCLUDED.locked_relative_path, status='running', detail=lucidota_commdump.export_object.detail || EXCLUDED.detail
        RETURNING export_uuid::text
        """,
        (kind, str(path), digest, size_bytes, mime, cas_uri, rel_locked, jdump({"llm_calls": 0})),
    ).fetchone()
    return str(row[0]), digest


def upsert_thread(conn: psycopg.Connection, export_uuid: str, msg: dict[str, Any], member: str) -> str:
    row = conn.execute(
        """
        INSERT INTO lucidota_commdump.thread(export_uuid, source_kind, provider_thread_id, title, participants, source_member, detail)
        VALUES (%s::uuid,%s,%s,%s,%s::jsonb,%s,%s::jsonb)
        ON CONFLICT(export_uuid, source_member, provider_thread_id) DO UPDATE SET title=EXCLUDED.title, participants=EXCLUDED.participants, detail=lucidota_commdump.thread.detail || EXCLUDED.detail
        RETURNING thread_uuid::text
        """,
        (export_uuid, msg["source_kind"], msg["thread_id"][:1000], msg["thread_title"][:1000], jdump(msg["participants"]), member, jdump({})),
    ).fetchone()
    return str(row[0])


def insert_msg(conn: psycopg.Connection, export_uuid: str, thread_uuid: str, msg: dict[str, Any]) -> dt.datetime | None:
    text = msg.get("content_text") or ""
    conn.execute(
        """
        INSERT INTO lucidota_commdump.message(thread_uuid, export_uuid, source_kind, provider_message_id, sender, recipients, occurred_at, occurred_at_raw, time_source, time_confidence_bps, sequence_index, subject, content_text, content_sha256, content_kind, attachments, raw)
        VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
        ON CONFLICT(thread_uuid, provider_message_id, sequence_index) DO UPDATE SET sender=EXCLUDED.sender, recipients=EXCLUDED.recipients, occurred_at=EXCLUDED.occurred_at, occurred_at_raw=EXCLUDED.occurred_at_raw, time_source=EXCLUDED.time_source, time_confidence_bps=EXCLUDED.time_confidence_bps, subject=EXCLUDED.subject, content_text=EXCLUDED.content_text, content_sha256=EXCLUDED.content_sha256, content_kind=EXCLUDED.content_kind, attachments=EXCLUDED.attachments, raw=EXCLUDED.raw
        """,
        (thread_uuid, export_uuid, msg["source_kind"], msg["provider_message_id"][:500], msg["sender"][:500], jdump(msg["recipients"]), msg["occurred_at"], msg["occurred_at_raw"], msg["time_source"], msg["time_confidence_bps"], msg["sequence_index"], msg["subject"][:1000], text, sha256_text(text), msg["content_kind"][:100], jdump(msg["attachments"]), jdump(msg["raw"])),
    )
    return msg["occurred_at"]


def ingest_path(conn: psycopg.Connection, path: Path, hint: str, limit: int, commit_every: int) -> dict[str, Any]:
    # Source kind for export row is best-effort from path; per-message kind can be sharper.
    kind0 = source_kind(path, path.name, None, hint)
    export_uuid, digest = upsert_export(conn, path, kind0)
    threads: set[str] = set()
    messages = errors = 0
    earliest = latest = None
    seq_by_thread: dict[str, int] = defaultdict(int)
    for outer, member, _size, opener in iter_sources(path):
        try:
            with opener() as fp:
                for root_obj in json_roots(fp, member) if member.lower().endswith((".json", ".jsonl", ".ndjson")) else []:
                    k = source_kind(outer, member, root_obj, hint)
                    for thread_meta, obj in flatten_message_objects(root_obj):
                        if limit and messages >= limit:
                            break
                        tid = str(thread_meta.get("thread_id") or thread_meta.get("conversation_id") or thread_meta.get("thread_path") or thread_meta.get("title") or member)
                        seq = seq_by_thread[tid]
                        seq_by_thread[tid] += 1
                        msg = extract_message(k, thread_meta, obj, member, seq)
                        thread_uuid = upsert_thread(conn, export_uuid, msg, member)
                        threads.add(thread_uuid)
                        t = insert_msg(conn, export_uuid, thread_uuid, msg)
                        if t:
                            earliest = t if earliest is None or t < earliest else earliest
                            latest = t if latest is None or t > latest else latest
                        messages += 1
                        if messages % commit_every == 0:
                            conn.commit()
                    if limit and messages >= limit:
                        break
        except (OSError, json.JSONDecodeError, csv.Error, UnicodeError, ValueError):
            errors += 1
            continue
    conn.execute(
        """
        UPDATE lucidota_commdump.export_object SET status=%s, thread_count=%s, message_count=%s, earliest_message_at=%s, latest_message_at=%s, detail=detail || %s::jsonb WHERE export_uuid=%s::uuid
        """,
        ("succeeded" if errors == 0 else "partial", len(threads), messages, earliest, latest, jdump({"errors": errors, "llm_calls": 0}), export_uuid),
    )
    for thread_uuid in threads:
        conn.execute("UPDATE lucidota_commdump.thread SET message_count=(SELECT count(*) FROM lucidota_commdump.message WHERE thread_uuid=%s::uuid) WHERE thread_uuid=%s::uuid", (thread_uuid, thread_uuid))
    conn.commit()
    return {"export_uuid": export_uuid, "sha256": digest, "path": str(path), "threads": len(threads), "messages": messages, "errors": errors, "earliest": earliest, "latest": latest}


def cmd_ingest(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    totals = {"exports": 0, "threads": 0, "messages": 0, "errors": 0}
    detail = []
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_schema(conn)
        for p in args.paths:
            res = ingest_path(conn, Path(p).expanduser().resolve(), args.source_kind, args.limit_messages, args.commit_every)
            totals["exports"] += 1
            totals["threads"] += res["threads"]
            totals["messages"] += res["messages"]
            totals["errors"] += res["errors"]
            detail.append(res)
    report = {**totals, "exports_detail": detail, "elapsed_seconds": round(time.time() - started, 3), "llm_calls": 0}
    event_id = emit_event("commdump", "succeeded" if totals["errors"] == 0 else "partial", report)
    return {"ok": totals["errors"] == 0, **report, "workflow_event": event_id}


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_schema(conn)
        row = conn.execute("SELECT (SELECT count(*) FROM lucidota_commdump.export_object), (SELECT count(*) FROM lucidota_commdump.thread), (SELECT count(*) FROM lucidota_commdump.message), (SELECT min(occurred_at)::text FROM lucidota_commdump.message WHERE occurred_at IS NOT NULL), (SELECT max(occurred_at)::text FROM lucidota_commdump.message WHERE occurred_at IS NOT NULL), (SELECT count(*) FROM lucidota_commdump.message WHERE time_confidence_bps>=9000), (SELECT count(*) FROM lucidota_commdump.message WHERE time_confidence_bps<9000)").fetchone()
        kinds = conn.execute("SELECT source_kind, count(*) FROM lucidota_commdump.message GROUP BY source_kind ORDER BY count(*) DESC").fetchall()
    return {"ok": True, "status": dict(zip(["exports", "threads", "messages", "earliest", "latest", "high_conf_time", "fallback_time"], row)), "source_kinds": [{"source_kind": k, "messages": n} for k, n in kinds]}


def cmd_timeline(args: argparse.Namespace) -> dict[str, Any]:
    where = ["occurred_at IS NOT NULL"]
    params: list[Any] = []
    if args.source_kind:
        where.append("source_kind=%s")
        params.append(args.source_kind)
    if args.search:
        where.append("fts_vector @@ plainto_tsquery('english', %s)")
        params.append(args.search)
    params += [args.excerpt_chars, args.limit]
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_schema(conn)
        rows = conn.execute(f"SELECT occurred_at::text, source_kind, sender, subject, left(content_text,%s), time_source, time_confidence_bps FROM lucidota_commdump.message WHERE {' AND '.join(where)} ORDER BY occurred_at ASC, sequence_index ASC LIMIT %s", params).fetchall()
    return {"ok": True, "count": len(rows), "messages": [dict(zip(["occurred_at", "source_kind", "sender", "subject", "excerpt", "time_source", "time_confidence_bps"], r)) for r in rows]}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lucidota-commdump-timeline")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("paths", nargs="+")
    p.add_argument("--source-kind", choices=["auto", "email", "facebook", "sms", "imessage", "whatsapp", "signal", "generic"], default="auto")
    p.add_argument("--limit-messages", type=int, default=0)
    p.add_argument("--commit-every", type=int, default=500)
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("timeline")
    p.add_argument("--source-kind", choices=["email", "facebook", "sms", "imessage", "whatsapp", "signal", "generic"], default="")
    p.add_argument("--search", default="")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--excerpt-chars", type=int, default=240)
    p.set_defaults(func=cmd_timeline)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
