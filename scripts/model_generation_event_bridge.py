#!/usr/bin/env python3
"""Stage model generation receipts into the targeted async PG/ABSURD event lane."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "model_generation_events"
SCHEMAS = [
    ROOT / "06_SCHEMA" / "035_absurd_queue_spine.sql",
    ROOT / "06_SCHEMA" / "043_absurd_remaining_worker_contracts.sql",
    ROOT / "06_SCHEMA" / "111_model_generation_event_lane.sql",
]
GRAPH_TABLES = ("lucidota_go.graph_item", "lucidota_go.graph_edge", "lucidota_go.graph_journal")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str, *, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def db_url(args: argparse.Namespace | None = None) -> str:
    return (
        (getattr(args, "database_url", None) if args is not None else None)
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def redacted(url: str) -> str:
    if url.startswith("postgresql:///"):
        return "postgresql:///<database>"
    if "@" in url:
        return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "set_redacted"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def write_report(action: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"model_generation_event_bridge_{action}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    return path


def resolve_receipt(path: Path | str, *, root: Path = ROOT) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def load_generation_event(receipt_path: Path | str, *, root: Path = ROOT) -> dict[str, Any]:
    path = resolve_receipt(receipt_path, root=root)
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("receipt_json_must_be_object")
    trace = data.get("generation_trace")
    if not isinstance(trace, dict) or trace.get("schema") != "lucidota.model_generation_trace.v1":
        raise ValueError("generation_trace_missing")
    target = str(trace.get("target") or "").strip()
    model_name = str(trace.get("model_name") or "").strip()
    if not target:
        raise ValueError("target_required")
    if not model_name:
        raise ValueError("model_name_required")
    raw_output = trace.get("raw_output", "")
    if not isinstance(raw_output, str):
        raw_output = str(raw_output)
    event = {
        "schema": "lucidota.model_generation_event.v1",
        "receipt_path": rel(path, root=root),
        "receipt_sha256": sha256_bytes(raw),
        "receipt_generated_at": data.get("generated_at"),
        "provider": str(data.get("provider") or ""),
        "mode": str(data.get("mode") or ""),
        "status": str(data.get("status") or ""),
        "target": target,
        "model_name": model_name,
        "payload_size_bytes": int(trace.get("payload_size_bytes") or 0),
        "payload_size_chars": int(trace.get("payload_size_chars") or 0),
        "latency_ms": float(trace.get("latency_ms") or 0.0),
        "raw_output": raw_output,
        "raw_output_chars": int(trace.get("raw_output_chars") or len(raw_output)),
        "raw_output_sha256": sha256_text(raw_output),
        "raw_response_present": bool(trace.get("raw_response_present")),
        "raw_response": data.get("raw_response"),
        "execute_performed": bool(trace.get("execute_performed")),
        "detail": {
            "receipt_schema": data.get("schema"),
            "generation_trace_schema": trace.get("schema"),
            "raw_response_keys": trace.get("raw_response_keys") or [],
            "lane": data.get("lane"),
            "endpoint": data.get("endpoint"),
            "report_path": data.get("report_path"),
        },
    }
    if event["payload_size_bytes"] < 0 or event["payload_size_chars"] < 0 or event["latency_ms"] < 0:
        raise ValueError("negative_generation_metric")
    return event


def latest_receipts(root: Path = ROOT, limit: int = 12) -> list[Path]:
    inv = root / "05_OUTPUTS" / "model_invocations"
    if not inv.exists():
        return []
    files = [p for p in inv.glob("*.json") if p.is_file()]
    files.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return files[: max(0, int(limit))]


def graph_counts(cur: Any) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for table in GRAPH_TABLES:
        cur.execute("SELECT to_regclass(%s)", (table,))
        if cur.fetchone()[0] is None:
            out[table] = None
            continue
        cur.execute(f"SELECT count(*) FROM {table}")
        out[table] = int(cur.fetchone()[0])
    return out


def insert_event(cur: Any, event: dict[str, Any]) -> dict[str, Any]:
    cur.execute(
        """
        INSERT INTO lucidota_control.model_generation_event(
          receipt_path, receipt_sha256, receipt_generated_at, provider, mode, status,
          target, model_name, payload_size_bytes, payload_size_chars, latency_ms,
          raw_output, raw_output_chars, raw_output_sha256, raw_response_present,
          raw_response, execute_performed, detail
        )
        VALUES (
          %s,%s,%s::timestamptz,%s,%s,%s,
          %s,%s,%s,%s,%s,
          %s,%s,%s,%s,
          %s::jsonb,%s,%s::jsonb
        )
        ON CONFLICT(receipt_path) DO UPDATE SET
          receipt_sha256=EXCLUDED.receipt_sha256,
          receipt_generated_at=EXCLUDED.receipt_generated_at,
          provider=EXCLUDED.provider,
          mode=EXCLUDED.mode,
          status=EXCLUDED.status,
          target=EXCLUDED.target,
          model_name=EXCLUDED.model_name,
          payload_size_bytes=EXCLUDED.payload_size_bytes,
          payload_size_chars=EXCLUDED.payload_size_chars,
          latency_ms=EXCLUDED.latency_ms,
          raw_output=EXCLUDED.raw_output,
          raw_output_chars=EXCLUDED.raw_output_chars,
          raw_output_sha256=EXCLUDED.raw_output_sha256,
          raw_response_present=EXCLUDED.raw_response_present,
          raw_response=EXCLUDED.raw_response,
          execute_performed=EXCLUDED.execute_performed,
          detail=EXCLUDED.detail,
          updated_at=now()
        RETURNING event_uuid::text, queue_event_uuid::text
        """,
        (
            event["receipt_path"],
            event["receipt_sha256"],
            event.get("receipt_generated_at"),
            event["provider"],
            event["mode"],
            event["status"],
            event["target"],
            event["model_name"],
            event["payload_size_bytes"],
            event["payload_size_chars"],
            event["latency_ms"],
            event["raw_output"],
            event["raw_output_chars"],
            event["raw_output_sha256"],
            event["raw_response_present"],
            json.dumps(event.get("raw_response"), default=str) if event.get("raw_response") is not None else None,
            event["execute_performed"],
            json.dumps(event["detail"], default=str),
        ),
    )
    event_uuid, queue_event_uuid = cur.fetchone()
    if not queue_event_uuid:
        detail = {
            "model_generation_event_uuid": event_uuid,
            "receipt_path": event["receipt_path"],
            "receipt_sha256": event["receipt_sha256"],
            "target": event["target"],
            "model_name": event["model_name"],
            "payload_size_bytes": event["payload_size_bytes"],
            "latency_ms": event["latency_ms"],
            "raw_output_sha256": event["raw_output_sha256"],
            "execute_performed": event["execute_performed"],
        }
        cur.execute(
            """
            INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, event_source, detail)
            VALUES (NULL,'model_generation','audit','model_generation_event_bridge',%s::jsonb)
            RETURNING queue_event_uuid::text
            """,
            (json.dumps(detail, default=str),),
        )
        queue_event_uuid = cur.fetchone()[0]
        cur.execute(
            "UPDATE lucidota_control.model_generation_event SET queue_event_uuid=%s::uuid, updated_at=now() WHERE event_uuid=%s::uuid",
            (queue_event_uuid, event_uuid),
        )
    return {"event_uuid": event_uuid, "queue_event_uuid": queue_event_uuid, "receipt_path": event["receipt_path"]}


def stage_receipts(receipts: list[Path | str], *, root: Path = ROOT, execute: bool, database_url: str | None = None) -> dict[str, Any]:
    url = database_url or db_url(None)
    report: dict[str, Any] = {
        "schema": "lucidota.model_generation_event_bridge.report.v1",
        "generated_at": now(),
        "database_url": redacted(url),
        "execute_performed": bool(execute),
        "receipts_seen": 0,
        "events_valid": 0,
        "events_staged": 0,
        "would_stage": [],
        "staged": [],
        "errors": [],
        "canonical_graph_writes_performed": False,
    }
    events: list[dict[str, Any]] = []
    for receipt in receipts:
        report["receipts_seen"] += 1
        try:
            event = load_generation_event(receipt, root=root)
            events.append(event)
            report["events_valid"] += 1
            if not execute:
                report["would_stage"].append(event["receipt_path"])
        except Exception as exc:
            report["errors"].append({"receipt_path": rel(resolve_receipt(receipt, root=root), root=root), "error": f"{type(exc).__name__}:{exc}"})
    if not execute or not events:
        report["status"] = "PASS" if not report["errors"] else "FAIL"
        return report

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            before_graph = graph_counts(cur)
            for event in events:
                staged = insert_event(cur, event)
                report["staged"].append(staged)
                report["events_staged"] += 1
            after_graph = graph_counts(cur)
            report["canonical_graph_counts_before"] = before_graph
            report["canonical_graph_counts_after"] = after_graph
            report["canonical_graph_writes_performed"] = before_graph != after_graph
        if report["canonical_graph_writes_performed"]:
            conn.rollback()
            report["status"] = "FAIL"
            report["errors"].append({"error": "canonical_graph_counts_changed"})
        else:
            conn.commit()
            report["status"] = "PASS" if not report["errors"] else "FAIL"
    return report


def init_schema(args: argparse.Namespace) -> dict[str, Any]:
    report = {
        "schema": "lucidota.model_generation_event_bridge.init_schema.v1",
        "generated_at": now(),
        "database_url": redacted(db_url(args)),
        "schema_paths": [rel(p) for p in SCHEMAS],
        "execute_performed": bool(args.execute),
    }
    missing = [rel(p) for p in SCHEMAS if not p.exists()]
    if missing:
        report["status"] = "FAIL"
        report["errors"] = [{"error": "schema_missing", "paths": missing}]
        return report
    if not args.execute:
        report["sql_bytes"] = sum(p.stat().st_size for p in SCHEMAS)
        report["status"] = "PASS"
        return report
    with psycopg.connect(db_url(args)) as conn:
        with conn.cursor() as cur:
            for schema in SCHEMAS:
                cur.execute(schema.read_text(encoding="utf-8"))
        conn.commit()
    report["status"] = "PASS"
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage model generation receipts into PG/ABSURD telemetry lane.")
    ap.add_argument("--database-url")
    sub = ap.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init-schema")
    init.add_argument("--execute", action="store_true")
    stage = sub.add_parser("stage")
    stage.add_argument("--receipt", action="append", default=[])
    stage.add_argument("--latest", type=int, default=0, help="Stage the latest N receipts when no --receipt is provided.")
    stage.add_argument("--execute", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "init-schema":
        report = init_schema(args)
        write_report("init_schema_execute" if args.execute else "init_schema_dry_run", report)
        print("MODEL_GENERATION_EVENT_SCHEMA=" + report["status"])
        return 0 if report["status"] == "PASS" else 4
    receipts = [Path(p) for p in args.receipt] if args.receipt else latest_receipts(ROOT, args.latest or 12)
    report = stage_receipts(receipts, root=ROOT, execute=bool(args.execute), database_url=db_url(args))
    write_report("stage_execute" if args.execute else "stage_dry_run", report)
    print("MODEL_GENERATION_EVENTS_STAGED=" + str(report.get("events_staged", 0)))
    print("MODEL_GENERATION_EVENT_BRIDGE=" + report.get("status", "FAIL"))
    return 0 if report.get("status") == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
