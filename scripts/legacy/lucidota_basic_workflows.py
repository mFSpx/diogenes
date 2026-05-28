#!/usr/bin/env python3
"""Basic DIOGENES/LUCIDOTA workflows.

Small, boring, inspectable workflows:
  audit             inventory capabilities and gaps
  clawd-route       route a CLI-style message onto GO graph + DBOS event
  lane-dispatch     materialize deterministic model-lane workflow events
  graph-search      retrieve GO graph/book chunks for a text query
  book-watch-once   process newly dropped /BOOKS files once
  lora-status       list INDY_READs LoRA cartridges
  smoke             run the basic flow end-to-end
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)

STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
CONTROL_SCHEMA = ROOT / "06_SCHEMA" / "001_lucidota_control.sql"
WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
GO_SCHEMA = ROOT / "06_SCHEMA" / "016_go_graph_core.sql"
INDY_SCHEMA = ROOT / "06_SCHEMA" / "017_indy_reads_library.sql"

sys.path.insert(0, str(ROOT / "01_REPOS" / "doggystyle"))
from kernel.mini_embeddings import INT16_MAX, hash_quantized_embedding  # type: ignore  # noqa: E402

# Import pure helpers from the local clawd runtime; it must not initialize clients.
sys.path.insert(0, str(ROOT / "scripts"))
from lucidota_clawd_runtime import route_go25  # type: ignore  # noqa: E402


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def emit_event(workflow_id: str, run_id: str, phase: str, status: str, detail: dict[str, Any]) -> str:
    with psycopg.connect(STATE_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(CONTROL_SCHEMA.read_text(encoding="utf-8"))
            cur.execute(WORKFLOW_SCHEMA.read_text(encoding="utf-8"))
            cur.execute(
                """
                INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
                VALUES (%s,%s,%s,%s,'lucidota_basic_workflows',%s::jsonb)
                RETURNING event_id::text
                """,
                (workflow_id, run_id, phase, status, jdump(detail)),
            )
            event_id = cur.fetchone()[0]
        conn.commit()
    return event_id


def scalar(dsn: str, sql: str, default: Any = 0) -> Any:
    try:
        with psycopg.connect(dsn) as conn:
            return conn.execute(sql).fetchone()[0]
    except psycopg.Error:
        return default


def vector_literal(text: str) -> str:
    return "[" + ",".join(f"{float(v) / float(INT16_MAX):.8f}" for v in hash_quantized_embedding(text)) + "]"


def run_json(cmd: list[str], *, stdin: dict[str, Any] | None = None) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        input=jdump(stdin) if stdin is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {"ok": False, "command": cmd, "returncode": proc.returncode, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": True, "command": cmd, "stdout": proc.stdout}


def cmd_audit(args: argparse.Namespace) -> dict[str, Any]:
    watcher_pid_path = ROOT / "04_RUNTIME" / "indy_reads_watcher.pid"
    watcher_pid = watcher_pid_path.read_text().strip() if watcher_pid_path.exists() else ""
    watcher_running = bool(watcher_pid and Path(f"/proc/{watcher_pid}").exists())
    graph_items = int(scalar(STORAGE_DSN, "SELECT count(*) FROM lucidota_go.graph_item", 0))
    clawd_items = int(scalar(STORAGE_DSN, "SELECT count(*) FROM lucidota_go.graph_item WHERE payload->>'source'='hard_modded_clawd'", 0))
    books = int(scalar(STORAGE_DSN, "SELECT count(*) FROM lucidota_indy.book_source", 0))
    chunks = int(scalar(STORAGE_DSN, "SELECT count(*) FROM lucidota_indy.book_chunk", 0))
    over_500 = int(scalar(STORAGE_DSN, "SELECT count(*) FROM lucidota_indy.book_chunk WHERE token_count > 500", 0))
    max_chunk = int(scalar(STORAGE_DSN, "SELECT coalesce(max(token_count),0) FROM lucidota_indy.book_chunk", 0))
    workflow_events = int(scalar(STATE_DSN, "SELECT count(*) FROM lucidota_control.workflow_event", 0))
    registered = int(scalar(STATE_DSN, "SELECT count(*) FROM lucidota_control.workflow_registry", 0))
    cartridges = int(scalar(STATE_DSN, "SELECT count(*) FROM lucidota_runtime.adapter_cartridge WHERE adapter_id LIKE 'indy_reads__%'", 0))
    loadout_slots = int(scalar(STATE_DSN, "SELECT count(*) FROM lucidota_runtime.resident_loadout_slot", 0))
    auth_clean = not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    capabilities = [
        {"capability": "CLI input to GO graph", "status": "working" if clawd_items > 0 else "needs_smoke", "evidence": {"clawd_graph_items": clawd_items}},
        {"capability": "DBOS/workflow_event routing", "status": "working" if workflow_events > 0 else "needs_smoke", "evidence": {"workflow_events": workflow_events, "registered_workflows": registered}},
        {"capability": "INDY_READs book ingestion", "status": "working" if books and chunks and over_500 == 0 else "needs_attention", "evidence": {"books": books, "chunks": chunks, "max_chunk_tokens": max_chunk, "chunks_over_500": over_500}},
        {"capability": "INDY_READs watcher", "status": "working" if watcher_running else "stopped", "evidence": {"pid": watcher_pid}},
        {"capability": "LoRA cartridge dataset generation", "status": "working" if cartridges >= books and books > 0 else "queued_or_partial", "evidence": {"cartridges": cartridges}},
        {"capability": "Resident model registry", "status": "working" if loadout_slots else "missing", "evidence": {"loadout_slots": loadout_slots}},
        {"capability": "No Anthropic credential path", "status": "clean" if auth_clean else "env_has_key_set", "evidence": {"anthropic_env_present": not auth_clean}},
    ]
    gaps = []
    if not auth_clean:
        gaps.append("Unset ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN before clawd sessions.")
    if not watcher_running:
        gaps.append("Start scripts/lucidota_start_indy_reads_watcher.sh.")
    if not os.environ.get("LUCIDOTA_LORA_BASE_MODEL"):
        gaps.append("LoRA datasets are ready; adapter weight training waits for LUCIDOTA_LORA_BASE_MODEL.")
    return {
        "ok": all(c["status"] in {"working", "clean"} for c in capabilities[:6]) and auth_clean,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "storage_dsn": STORAGE_DSN,
        "state_dsn": STATE_DSN,
        "capabilities": capabilities,
        "gaps": gaps,
    }


def cmd_clawd_route(args: argparse.Namespace) -> dict[str, Any]:
    message = args.message or " ".join(args.rest).strip() or "STATUS"
    return run_json([str(PY), str(ROOT / "scripts" / "lucidota_clawd_runtime.py"), "--json", "--message", message])


def latest_clawd_event_id() -> str:
    with psycopg.connect(STATE_DSN) as conn:
        row = conn.execute(
            """
            SELECT event_id::text
            FROM lucidota_control.workflow_event
            WHERE workflow_id='clawd-go-runtime'
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if not row:
        raise SystemExit("no clawd-go-runtime workflow_event found; run clawd-route first")
    return row[0]


def cmd_lane_dispatch(args: argparse.Namespace) -> dict[str, Any]:
    event_id = args.event_id or latest_clawd_event_id()
    with psycopg.connect(STATE_DSN) as conn:
        row = conn.execute(
            "SELECT run_id, detail FROM lucidota_control.workflow_event WHERE event_id=%s::uuid",
            (event_id,),
        ).fetchone()
    if not row:
        raise SystemExit(f"workflow_event not found: {event_id}")
    run_id, detail = row
    lanes = detail.get("model_lanes") or []
    if not lanes:
        raise SystemExit(f"workflow_event has no model_lanes: {event_id}")
    written: list[dict[str, str]] = []
    for lane in lanes:
        payload = lane.get("payload") or {}
        safe_payload = {
            "source_event_id": event_id,
            "lane": lane.get("lane"),
            "model_role": lane.get("model_role"),
            "target_model_id": lane.get("target_model_id"),
            "payload": {
                "graph_item_uuid": payload.get("graph_item_uuid"),
                "go_terms": payload.get("go_terms", []),
                "instruction": payload.get("instruction", ""),
            },
            "raw_prompt_included": False,
            "status": "materialized_basic_lane",
        }
        lane_event_id = emit_event(str(lane.get("lane") or "model-lane"), str(run_id), "model_lane", "succeeded", safe_payload)
        written.append({"lane": str(lane.get("lane")), "event_id": lane_event_id})
    return {"ok": True, "source_event_id": event_id, "lanes": written}


def cmd_graph_search(args: argparse.Namespace) -> dict[str, Any]:
    query = args.query or " ".join(args.rest).strip()
    if not query:
        raise SystemExit("graph-search needs --query or trailing text")
    routes = route_go25(query, top_n=5)
    terms = [r["term"] for r in routes]
    with psycopg.connect(STORAGE_DSN) as conn:
        conn.execute(GO_SCHEMA.read_text(encoding="utf-8"))
        conn.execute(INDY_SCHEMA.read_text(encoding="utf-8"))
        items = conn.execute(
            """
            SELECT uuid::text, term, label, location_at_on_graph, created_at::text
            FROM lucidota_go.graph_item
            WHERE term = ANY(%s)
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (terms, args.limit),
        ).fetchall()
        chunks = conn.execute(
            """
            SELECT b.title, c.chunk_index, c.token_count, c.anchor, left(c.content, 280) AS excerpt,
                   (e.embedding <=> %s::vector) AS distance
            FROM lucidota_indy.chunk_embedding e
            JOIN lucidota_indy.book_chunk c ON c.chunk_uuid=e.chunk_uuid
            JOIN lucidota_indy.book_source b ON b.book_uuid=c.book_uuid
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal(query), vector_literal(query), args.limit),
        ).fetchall()
    return {
        "ok": True,
        "query": query,
        "go_route": routes,
        "graph_items": [dict(zip(["uuid", "term", "label", "location", "created_at"], row)) for row in items],
        "book_chunks": [dict(zip(["title", "chunk_index", "token_count", "anchor", "excerpt", "distance"], row)) for row in chunks],
    }


def cmd_book_watch_once(args: argparse.Namespace) -> dict[str, Any]:
    cmd = [str(PY), str(ROOT / "scripts" / "lucidota_indy_reads_watcher.py"), "--once", "--json", "--append-lora-jsonl"]
    if args.max_tokens:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    return run_json(cmd)


def cmd_lora_status(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STATE_DSN) as conn:
        rows = conn.execute(
            """
            SELECT adapter_id, target_model_id, activation_status, local_path, source_manifest
            FROM lucidota_runtime.adapter_cartridge
            WHERE adapter_id LIKE 'indy_reads__%%'
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (args.limit,),
        ).fetchall()
    return {
        "ok": True,
        "cartridges": [
            {
                "adapter_id": r[0],
                "target_model_id": r[1],
                "activation_status": r[2],
                "local_path": r[3],
                "train_count": (r[4] or {}).get("train_count"),
                "validation_count": (r[4] or {}).get("validation_count"),
            }
            for r in rows
        ],
    }


def cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    route = cmd_clawd_route(argparse.Namespace(message=args.message, rest=[]))
    event_id = ((route.get("workflow") or {}).get("event_id")) or latest_clawd_event_id()
    lanes = cmd_lane_dispatch(argparse.Namespace(event_id=event_id))
    search = cmd_graph_search(argparse.Namespace(query=args.message, rest=[], limit=3))
    watch = cmd_book_watch_once(argparse.Namespace(max_tokens=500))
    lora = cmd_lora_status(argparse.Namespace(limit=3))
    audit = cmd_audit(argparse.Namespace())
    return {"ok": route.get("ok") and lanes.get("ok") and search.get("ok") and watch.get("ok") and lora.get("ok"), "route": route, "lanes": lanes, "search": search, "watch": watch, "lora": lora, "audit": audit}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("audit").set_defaults(func=cmd_audit)

    p = sub.add_parser("clawd-route")
    p.add_argument("--message", default="")
    p.add_argument("rest", nargs="*")
    p.set_defaults(func=cmd_clawd_route)

    p = sub.add_parser("lane-dispatch")
    p.add_argument("--event-id", default="")
    p.set_defaults(func=cmd_lane_dispatch)

    p = sub.add_parser("graph-search")
    p.add_argument("--query", default="")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("rest", nargs="*")
    p.set_defaults(func=cmd_graph_search)

    p = sub.add_parser("book-watch-once")
    p.add_argument("--max-tokens", type=int, default=500)
    p.set_defaults(func=cmd_book_watch_once)

    p = sub.add_parser("lora-status")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_lora_status)

    p = sub.add_parser("smoke")
    p.add_argument("--message", default="basic workflow smoke")
    p.set_defaults(func=cmd_smoke)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
