#!/usr/bin/env python3
"""Streaming OpenAI/Claude data-export timeline ingester.

Built for very large exports (950MB+): ZIP members and top-level JSON arrays are
streamed one conversation at a time. No LLM calls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import mimetypes
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import psycopg

ROOT = Path(__file__).resolve().parents[1]
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
SCHEMA = ROOT / "06_SCHEMA" / "020_chat_dump_timeline.sql"
WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from lucidota_artifact_ingest import ensure_state, insert_graph_item, normalize_ws, sha256_file, store_locked_cas  # type: ignore  # noqa: E402

CHAT_JSON_NAMES = {
    "conversations.json", "conversation.json", "chat.json", "chats.json", "messages.json", "message.json"
}
TEXTISH_SUFFIXES = {".json", ".jsonl", ".ndjson"}


@dataclass
class SourceItem:
    outer_path: Path
    member_name: str
    provider_hint: str
    size_bytes: int
    opener: Any


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA.read_text(encoding="utf-8"))
    conn.commit()


def emit_event(workflow_id: str, run_id: str, phase: str, status: str, detail: dict[str, Any]) -> str:
    with psycopg.connect(STATE_DSN) as conn:
        ensure_state(conn)
        conn.execute(WORKFLOW_SCHEMA.read_text(encoding="utf-8"))
        row = conn.execute(
            """
            INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
            VALUES (%s,%s,%s,%s,'chatdump_timeline',%s::jsonb)
            RETURNING event_id::text
            """,
            (workflow_id, run_id, phase, status, jdump(detail)),
        ).fetchone()
        conn.commit()
        return str(row[0])


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def parse_time(value: Any) -> tuple[dt.datetime | None, str]:
    if value is None or value == "":
        return None, ""
    raw = str(value)
    try:
        if isinstance(value, (int, float)) or re.fullmatch(r"\d+(?:\.\d+)?", raw):
            f = float(value)
            if f > 1e12:
                f = f / 1000.0
            if f > 1e10:
                f = f / 1000.0
            return dt.datetime.fromtimestamp(f, tz=dt.timezone.utc), raw
    except (OSError, OverflowError, TypeError, ValueError):
        return None, raw
    try:
        s = raw.strip().replace("Z", "+00:00")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?", s):
            s += "+00:00"
        parsed = dt.datetime.fromisoformat(s)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc), raw
    except ValueError:
        return None, raw


def best_time(*candidates: tuple[Any, str, int]) -> tuple[dt.datetime | None, str, str, int]:
    for value, source, confidence in candidates:
        parsed, raw = parse_time(value)
        if parsed is not None:
            return parsed, raw, source, confidence
    return None, "", "missing", 0


def content_to_text(content: Any) -> tuple[str, str, list[dict[str, Any]]]:
    attachments: list[dict[str, Any]] = []
    if content is None:
        return "", "", attachments
    if isinstance(content, str):
        return content, "text", attachments
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            txt, kind, atts = content_to_text(part)
            if txt:
                parts.append(txt)
            attachments.extend(atts)
        return "\n".join(parts), "list", attachments
    if isinstance(content, dict):
        ctype = str(content.get("content_type") or content.get("type") or "dict")
        if "parts" in content:
            text, _, atts = content_to_text(content.get("parts"))
            attachments.extend(atts)
            return text, ctype, attachments
        for key in ("text", "value", "input", "output", "transcript"):
            val = content.get(key)
            if isinstance(val, str):
                return val, ctype, attachments
            if isinstance(val, dict) and isinstance(val.get("text"), str):
                return val.get("text", ""), ctype, attachments
        # Multimodal / file-bearing blocks: keep metadata but don't blob binary into text.
        if any(k in content for k in ("file_id", "asset_pointer", "name", "filename", "url")):
            attachments.append({k: content.get(k) for k in ("file_id", "asset_pointer", "name", "filename", "url", "mime_type") if k in content})
            return "", ctype, attachments
        return normalize_ws(json.dumps(content, ensure_ascii=False, default=str))[:20000], ctype, attachments
    return str(content), type(content).__name__, attachments


def normalize_role(value: Any) -> str:
    role = str(value or "").lower().strip()
    return {"human": "user", "user": "user", "assistant": "assistant", "claude": "assistant", "system": "system", "tool": "tool"}.get(role, role)


def stream_json_array(fp: io.TextIOBase) -> Iterator[Any]:
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
                    raise ValueError("JSON root is not a top-level array; use a provider export conversations.json/messages.json")
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
            if buf.strip() not in {"", "]"}:
                # Allow whitespace/trailing bracket remnants; otherwise surface malformed JSON.
                rem = buf.strip()[:100]
                if rem and rem != "]":
                    raise ValueError(f"Unexpected JSON remainder: {rem}")
            return


def iter_json_objects_from_text(fp: io.TextIOBase, name: str) -> Iterator[Any]:
    if name.lower().endswith((".jsonl", ".ndjson")):
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    # Top-level array streaming fast path. If it is a small object, load it and yield relevant arrays.
    pos = fp.tell() if fp.seekable() else None
    prefix = fp.read(4096)
    if pos is not None:
        fp.seek(pos)
    first = prefix.lstrip()[:1]
    if first == "[":
        yield from stream_json_array(fp)
        return
    data = json.load(fp)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        for key in ("conversations", "chats", "messages", "data"):
            if isinstance(data.get(key), list):
                yield from data[key]
                return
        yield data


def likely_provider(path: Path, member_name: str, obj: Any | None = None, hint: str = "auto") -> str:
    if hint in {"openai", "claude"}:
        return hint
    hay = f"{path.name} {member_name}".lower()
    if "openai" in hay or "chatgpt" in hay:
        return "openai"
    if "claude" in hay or "anthropic" in hay:
        return "claude"
    if isinstance(obj, dict):
        if "mapping" in obj or "current_node" in obj:
            return "openai"
        if "chat_messages" in obj or "uuid" in obj and "messages" in obj:
            return "claude"
    return "unknown"


def iter_sources(paths: list[Path], provider_hint: str) -> Iterator[SourceItem]:
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and (child.suffix.lower() in TEXTISH_SUFFIXES or child.suffix.lower() == ".zip"):
                    yield from iter_sources([child], provider_hint)
            continue
        if path.suffix.lower() == ".zip":
            zf = zipfile.ZipFile(path)
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                base = Path(name).name.lower()
                suffix = Path(name).suffix.lower()
                if base in CHAT_JSON_NAMES or suffix in {".json", ".jsonl", ".ndjson"}:
                    def opener(zf=zf, name=name):
                        return io.TextIOWrapper(zf.open(name, "r"), encoding="utf-8", errors="replace")
                    yield SourceItem(path, name, provider_hint, int(info.file_size), opener)
            continue
        if path.suffix.lower() in TEXTISH_SUFFIXES:
            def opener(path=path):
                return path.open("r", encoding="utf-8", errors="replace")
            yield SourceItem(path, path.name, provider_hint, path.stat().st_size, opener)


def openai_messages(conv: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cid = str(conv.get("id") or conv.get("conversation_id") or conv.get("uuid") or hashlib.sha256(jdump(conv).encode()).hexdigest()[:24])
    meta = {
        "provider_conversation_id": cid,
        "title": str(conv.get("title") or conv.get("name") or ""),
        "create_time": conv.get("create_time") or conv.get("created_at"),
        "update_time": conv.get("update_time") or conv.get("updated_at"),
        "detail": {k: conv.get(k) for k in ("current_node", "conversation_template_id", "gizmo_id", "is_archived") if k in conv},
    }
    out: list[dict[str, Any]] = []
    mapping = conv.get("mapping")
    if isinstance(mapping, dict):
        for node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue
            content_text, content_kind, attachments = content_to_text(msg.get("content"))
            author = msg.get("author") if isinstance(msg.get("author"), dict) else {}
            out.append({
                "provider_message_id": str(msg.get("id") or node_id),
                "parent_message_id": str(node.get("parent") or ""),
                "role": normalize_role(author.get("role")),
                "author_name": str(author.get("name") or author.get("role") or ""),
                "create_time": msg.get("create_time") or node.get("create_time") or conv.get("create_time"),
                "update_time": msg.get("update_time") or node.get("update_time") or conv.get("update_time"),
                "content_text": content_text,
                "content_kind": content_kind,
                "attachments": attachments,
                "raw": {"node_id": node_id, "status": msg.get("status"), "recipient": msg.get("recipient"), "metadata": msg.get("metadata", {})},
            })
    elif isinstance(conv.get("messages"), list):
        for idx, msg in enumerate(conv.get("messages") or []):
            if isinstance(msg, dict):
                content_text, content_kind, attachments = content_to_text(msg.get("content") or msg.get("text"))
                author = msg.get("author") if isinstance(msg.get("author"), dict) else {}
                out.append({
                    "provider_message_id": str(msg.get("id") or msg.get("message_id") or idx),
                    "parent_message_id": str(msg.get("parent") or msg.get("parent_id") or ""),
                    "role": normalize_role(msg.get("role") or author.get("role")),
                    "author_name": str(author.get("name") or msg.get("sender") or ""),
                    "create_time": msg.get("create_time") or msg.get("created_at") or conv.get("create_time"),
                    "update_time": msg.get("update_time") or msg.get("updated_at") or conv.get("update_time"),
                    "content_text": content_text,
                    "content_kind": content_kind,
                    "attachments": attachments,
                    "raw": {"metadata": msg.get("metadata", {})},
                })
    return meta, out


def claude_messages(conv: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cid = str(conv.get("uuid") or conv.get("id") or conv.get("conversation_id") or hashlib.sha256(jdump(conv).encode()).hexdigest()[:24])
    messages = conv.get("chat_messages") or conv.get("messages") or conv.get("conversation") or []
    meta = {
        "provider_conversation_id": cid,
        "title": str(conv.get("name") or conv.get("title") or conv.get("summary") or ""),
        "create_time": conv.get("created_at") or conv.get("create_time") or conv.get("createdAt"),
        "update_time": conv.get("updated_at") or conv.get("update_time") or conv.get("updatedAt"),
        "detail": {k: conv.get(k) for k in ("account", "organization_uuid", "project_uuid", "settings") if k in conv},
    }
    out: list[dict[str, Any]] = []
    if isinstance(messages, dict):
        messages = list(messages.values())
    if isinstance(messages, list):
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content_text, content_kind, attachments = content_to_text(msg.get("text") or msg.get("content") or msg.get("message"))
            extra_atts = msg.get("attachments") or msg.get("files") or []
            if isinstance(extra_atts, list):
                attachments.extend([a for a in extra_atts if isinstance(a, dict)])
            out.append({
                "provider_message_id": str(msg.get("uuid") or msg.get("id") or msg.get("message_id") or idx),
                "parent_message_id": str(msg.get("parent_uuid") or msg.get("parent_message_id") or ""),
                "role": normalize_role(msg.get("sender") or msg.get("role") or msg.get("author")),
                "author_name": str(msg.get("sender") or msg.get("author") or ""),
                "create_time": msg.get("created_at") or msg.get("create_time") or msg.get("createdAt") or conv.get("created_at"),
                "update_time": msg.get("updated_at") or msg.get("update_time") or msg.get("updatedAt") or conv.get("updated_at"),
                "content_text": content_text,
                "content_kind": content_kind,
                "attachments": attachments,
                "raw": {"metadata": msg.get("metadata", {}), "index": idx},
            })
    return meta, out


def object_to_conversation(obj: Any, provider: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(obj, dict):
        return {}, []
    prov = likely_provider(Path(""), "", obj, provider)
    if prov == "openai":
        return openai_messages(obj)
    if prov == "claude":
        return claude_messages(obj)
    # Unknown: try both, prefer one that yields messages.
    meta, msgs = openai_messages(obj)
    if msgs:
        return meta, msgs
    return claude_messages(obj)


def directory_manifest(path: Path) -> tuple[str, int]:
    """Stable enough identity for directory exports without hashing bytes twice.

    The ingest still walks and parses child sources below. This manifest only
    lets the export_object row represent a directory without calling
    sha256_file() on it.
    """
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


def upsert_export(conn: psycopg.Connection, provider: str, path: Path) -> tuple[str, str, str]:
    if path.is_dir():
        digest, size_bytes = directory_manifest(path)
        mime = "inode/directory"
        cas_uri = f"dir-manifest://sha256/{digest}"
        locked_relative_path = str(path)
    else:
        digest = sha256_file(path)
        size_bytes = path.stat().st_size
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        cas_uri, locked = store_locked_cas(path, digest, mime, "chatdump_timeline")
        locked_relative_path = str(locked.relative_to(ROOT)) if locked.is_relative_to(ROOT) else str(locked)
    row = conn.execute(
        """
        INSERT INTO lucidota_chatdump.export_object(provider, source_path, source_sha256, size_bytes, mime, cas_uri, locked_relative_path, status, detail)
        VALUES (%s,%s,%s,%s,%s,%s,%s,'running',%s::jsonb)
        ON CONFLICT(source_sha256, provider) DO UPDATE SET
          source_path=EXCLUDED.source_path, size_bytes=EXCLUDED.size_bytes, mime=EXCLUDED.mime,
          cas_uri=EXCLUDED.cas_uri, locked_relative_path=EXCLUDED.locked_relative_path,
          status='running', detail=lucidota_chatdump.export_object.detail || EXCLUDED.detail, updated_at=now()
        RETURNING export_uuid::text
        """,
        (provider, str(path), digest, size_bytes, mime, cas_uri, locked_relative_path, jdump({"llm_calls": 0})),
    ).fetchone()
    return str(row[0]), digest, cas_uri


def upsert_conversation(conn: psycopg.Connection, export_uuid: str, provider: str, source_member: str, meta: dict[str, Any]) -> str:
    ct, ctraw = parse_time(meta.get("create_time"))
    ut, utraw = parse_time(meta.get("update_time"))
    row = conn.execute(
        """
        INSERT INTO lucidota_chatdump.conversation(export_uuid, provider, provider_conversation_id, title, create_time, update_time, create_time_raw, update_time_raw, source_member, detail)
        VALUES (%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
        ON CONFLICT(provider, provider_conversation_id, export_uuid) DO UPDATE SET
          title=EXCLUDED.title, create_time=EXCLUDED.create_time, update_time=EXCLUDED.update_time,
          create_time_raw=EXCLUDED.create_time_raw, update_time_raw=EXCLUDED.update_time_raw,
          source_member=EXCLUDED.source_member, detail=lucidota_chatdump.conversation.detail || EXCLUDED.detail
        RETURNING conversation_uuid::text
        """,
        (export_uuid, provider, meta.get("provider_conversation_id", ""), meta.get("title", "")[:1000], ct, ut, ctraw, utraw, source_member, jdump(meta.get("detail", {}))),
    ).fetchone()
    return str(row[0])


def insert_message(conn: psycopg.Connection, export_uuid: str, conversation_uuid: str, provider: str, msg: dict[str, Any], idx: int, conv_meta: dict[str, Any], file_mtime: dt.datetime | None) -> dt.datetime | None:
    created, raw, source, conf = best_time(
        (msg.get("create_time"), "message_create_time", 10000),
        (msg.get("update_time"), "message_update_time", 8500),
        (conv_meta.get("create_time"), "conversation_create_time", 7500),
        (conv_meta.get("update_time"), "conversation_update_time", 6500),
        (file_mtime, "file_mtime", 3000),
    )
    updated, uraw = parse_time(msg.get("update_time"))
    text = msg.get("content_text") or ""
    content_sha = sha256_text(text)
    # Empty synthetic/system shell messages are still useful for exact structure if timed.
    raw_payload = msg.get("raw") or {}
    conn.execute(
        """
        INSERT INTO lucidota_chatdump.message(
          conversation_uuid, export_uuid, provider, provider_message_id, parent_message_id, role, author_name,
          create_time, update_time, create_time_raw, update_time_raw, time_source, time_confidence_bps,
          sequence_index, content_text, content_sha256, content_kind, token_count, attachments, raw
        ) VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
        ON CONFLICT(conversation_uuid, provider_message_id, sequence_index) DO UPDATE SET
          parent_message_id=EXCLUDED.parent_message_id, role=EXCLUDED.role, author_name=EXCLUDED.author_name,
          create_time=EXCLUDED.create_time, update_time=EXCLUDED.update_time,
          create_time_raw=EXCLUDED.create_time_raw, update_time_raw=EXCLUDED.update_time_raw,
          time_source=EXCLUDED.time_source, time_confidence_bps=EXCLUDED.time_confidence_bps,
          content_text=EXCLUDED.content_text, content_sha256=EXCLUDED.content_sha256,
          content_kind=EXCLUDED.content_kind, token_count=EXCLUDED.token_count,
          attachments=EXCLUDED.attachments, raw=EXCLUDED.raw
        """,
        (
            conversation_uuid, export_uuid, provider, str(msg.get("provider_message_id") or "")[:500], str(msg.get("parent_message_id") or "")[:500],
            msg.get("role", "")[:80], msg.get("author_name", "")[:200], created, updated, raw, uraw,
            source, conf, idx, text, content_sha, msg.get("content_kind", "")[:100], token_count(text),
            jdump(msg.get("attachments") or []), jdump(raw_payload),
        ),
    )
    return created


def source_file_mtime(path: Path) -> dt.datetime | None:
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    except OSError:
        return None


def ingest_source(conn: psycopg.Connection, export_uuid: str, provider_arg: str, source: SourceItem, limit: int, commit_every: int) -> dict[str, Any]:
    conversations = messages = errors = 0
    earliest: dt.datetime | None = None
    latest: dt.datetime | None = None
    file_mtime = source_file_mtime(source.outer_path)
    with source.opener() as fp:
        for obj in iter_json_objects_from_text(fp, source.member_name):
            if limit and conversations >= limit:
                break
            provider = likely_provider(source.outer_path, source.member_name, obj, provider_arg)
            meta, msgs = object_to_conversation(obj, provider)
            if not meta:
                errors += 1
                continue
            if provider == "unknown":
                provider = likely_provider(source.outer_path, source.member_name, obj, "auto")
            if provider == "unknown":
                provider = provider_arg if provider_arg != "auto" else "unknown"
            conversations += 1
            conv_uuid = upsert_conversation(conn, export_uuid, provider, source.member_name, meta)
            # Stable contemporaneous ordering: timestamp first, then original provider order.
            indexed = list(enumerate(msgs))
            def sort_key(pair: tuple[int, dict[str, Any]]) -> tuple[float, int]:
                t, _, _, _ = best_time((pair[1].get("create_time"), "message_create_time", 10000), (pair[1].get("update_time"), "message_update_time", 8500), (meta.get("create_time"), "conversation_create_time", 7500), (file_mtime, "file_mtime", 3000))
                return ((t.timestamp() if t else 0.0), pair[0])
            for seq, (_, msg) in enumerate(sorted(indexed, key=sort_key)):
                t = insert_message(conn, export_uuid, conv_uuid, provider, msg, seq, meta, file_mtime)
                if t:
                    earliest = t if earliest is None or t < earliest else earliest
                    latest = t if latest is None or t > latest else latest
                messages += 1
            conn.execute("UPDATE lucidota_chatdump.conversation SET message_count=%s WHERE conversation_uuid=%s::uuid", (len(msgs), conv_uuid))
            if conversations % commit_every == 0:
                conn.commit()
    return {"conversations": conversations, "messages": messages, "errors": errors, "earliest": earliest, "latest": latest, "member": source.member_name}


def cmd_inspect(args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(p) for p in args.paths]
    rows = []
    for source in iter_sources(paths, args.provider):
        rows.append({"outer_path": str(source.outer_path), "member": source.member_name, "size_bytes": source.size_bytes, "provider_hint": source.provider_hint})
    return {"ok": True, "sources": rows, "source_count": len(rows)}


def cmd_ingest(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    paths = [Path(p) for p in args.paths]
    totals = {"exports": 0, "sources": 0, "conversations": 0, "messages": 0, "errors": 0}
    earliest: dt.datetime | None = None
    latest: dt.datetime | None = None
    exports: list[dict[str, Any]] = []
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_schema(conn)
        for path in paths:
            path = path.expanduser().resolve()
            provider = likely_provider(path, path.name, None, args.provider)
            export_uuid, digest, cas_uri = upsert_export(conn, provider if provider != "auto" else "unknown", path)
            totals["exports"] += 1
            conn.commit()
            export_conv = export_msg = export_err = 0
            for source in iter_sources([path], args.provider):
                totals["sources"] += 1
                result = ingest_source(conn, export_uuid, args.provider, source, args.limit_conversations, args.commit_every)
                export_conv += result["conversations"]
                export_msg += result["messages"]
                export_err += result["errors"]
                if result.get("earliest"):
                    earliest = result["earliest"] if earliest is None or result["earliest"] < earliest else earliest
                if result.get("latest"):
                    latest = result["latest"] if latest is None or result["latest"] > latest else latest
                conn.commit()
            totals["conversations"] += export_conv
            totals["messages"] += export_msg
            totals["errors"] += export_err
            status = "succeeded" if export_err == 0 else "partial"
            conn.execute(
                """
                UPDATE lucidota_chatdump.export_object
                SET status=%s, conversation_count=%s, message_count=%s, earliest_message_at=%s, latest_message_at=%s,
                    detail=detail || %s::jsonb
                WHERE export_uuid=%s::uuid
                """,
                (status, export_conv, export_msg, earliest, latest, jdump({"elapsed_seconds": round(time.time() - started, 3), "llm_calls": 0}), export_uuid),
            )
            conn.commit()
            exports.append({"export_uuid": export_uuid, "path": str(path), "sha256": digest, "cas_uri": cas_uri, "conversations": export_conv, "messages": export_msg, "errors": export_err})
    detail = {**totals, "exports_detail": exports, "elapsed_seconds": round(time.time() - started, 3), "llm_calls": 0, "earliest": earliest, "latest": latest}
    event_id = emit_event("chatdump-timeline-ingest", "chatdump", "ingest", "succeeded" if totals["errors"] == 0 else "partial", detail)
    return {"ok": totals["errors"] == 0, **detail, "workflow_event": event_id}


def cmd_timeline(args: argparse.Namespace) -> dict[str, Any]:
    where = ["m.create_time IS NOT NULL"]
    params: list[Any] = []
    if args.provider:
        where.append("m.provider=%s")
        params.append(args.provider)
    if args.from_date:
        where.append("m.create_time >= %s::timestamptz")
        params.append(args.from_date)
    if args.to_date:
        where.append("m.create_time <= %s::timestamptz")
        params.append(args.to_date)
    if args.search:
        where.append("m.fts_vector @@ plainto_tsquery('english', %s)")
        params.append(args.search)
    params.append(args.limit)
    sql = f"""
        SELECT m.create_time::text, m.provider, c.title, m.role, left(m.content_text, %s), m.time_source, m.time_confidence_bps, c.provider_conversation_id, m.provider_message_id
        FROM lucidota_chatdump.message m
        JOIN lucidota_chatdump.conversation c ON c.conversation_uuid=m.conversation_uuid
        WHERE {' AND '.join(where)}
        ORDER BY m.create_time ASC, c.provider_conversation_id ASC, m.sequence_index ASC
        LIMIT %s
    """
    params.insert(-1, args.excerpt_chars)
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_schema(conn)
        rows = conn.execute(sql, params).fetchall()
    keys = ["create_time", "provider", "title", "role", "excerpt", "time_source", "time_confidence_bps", "conversation_id", "message_id"]
    return {"ok": True, "messages": [dict(zip(keys, r)) for r in rows], "count": len(rows)}


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_schema(conn)
        row = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM lucidota_chatdump.export_object),
              (SELECT count(*) FROM lucidota_chatdump.conversation),
              (SELECT count(*) FROM lucidota_chatdump.message),
              (SELECT min(create_time)::text FROM lucidota_chatdump.message WHERE create_time IS NOT NULL),
              (SELECT max(create_time)::text FROM lucidota_chatdump.message WHERE create_time IS NOT NULL),
              (SELECT count(*) FROM lucidota_chatdump.message WHERE time_confidence_bps >= 10000),
              (SELECT count(*) FROM lucidota_chatdump.message WHERE time_confidence_bps < 10000)
            """
        ).fetchone()
    keys = ["exports", "conversations", "messages", "earliest", "latest", "exact_message_time", "fallback_time"]
    return {"ok": True, "status": dict(zip(keys, row))}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lucidota-chatdump-timeline")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("paths", nargs="+")
    p.add_argument("--provider", choices=["auto", "openai", "claude", "unknown"], default="auto")
    p.set_defaults(func=cmd_inspect)
    p = sub.add_parser("ingest")
    p.add_argument("paths", nargs="+")
    p.add_argument("--provider", choices=["auto", "openai", "claude", "unknown"], default="auto")
    p.add_argument("--commit-every", type=int, default=100)
    p.add_argument("--limit-conversations", type=int, default=0)
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("timeline")
    p.add_argument("--provider", choices=["openai", "claude", "unknown"], default="")
    p.add_argument("--from-date", default="")
    p.add_argument("--to-date", default="")
    p.add_argument("--search", default="")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--excerpt-chars", type=int, default=240)
    p.set_defaults(func=cmd_timeline)
    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
