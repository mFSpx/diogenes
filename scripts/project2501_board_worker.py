#!/usr/bin/env python3
"""Project 2501 bounded board worker: claim one slow/audit work_order, emit receipt/River/watch_metric."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "project2501_board_worker"
CORE_SCHEMA = ROOT / "06_SCHEMA" / "112_project2501_core_board.sql"
WORKER_SCHEMA = ROOT / "06_SCHEMA" / "114_project2501_board_worker.sql"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def db_url(args: argparse.Namespace | None = None) -> str:
    return (
        (getattr(args, "database_url", None) if args is not None else None)
        or os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
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


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def write_report(payload: dict[str, Any], prefix: str = "project2501_board_worker") -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{prefix}_{stamp()}.json"
    payload["report_path"] = rel(path)
    if isinstance(payload.get("work_receipt"), dict):
        payload["work_receipt"]["receipt_path"] = rel(path)
    if isinstance(payload.get("watch_metric"), dict):
        payload["watch_metric"]["source_receipt"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    return path


def apply_schema(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(CORE_SCHEMA.read_text(encoding="utf-8"))
            cur.execute(WORKER_SCHEMA.read_text(encoding="utf-8"))
        conn.commit()


def work_order_from_row(row: tuple[Any, ...]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    (
        work_order_uuid,
        event_id,
        decision_uuid,
        lane,
        work_kind,
        status,
        payload,
        idempotency_key,
        actor,
        source,
        raw_ref,
        text,
        board_features,
        route_key,
        expected_gain,
        confidence,
        graph_write_mode,
        cost,
    ) = row
    work_order = {
        "work_order_uuid": str(work_order_uuid),
        "event_id": event_id,
        "decision_uuid": str(decision_uuid) if decision_uuid else None,
        "lane": lane,
        "work_kind": work_kind,
        "status": status,
        "payload": as_dict(payload),
        "idempotency_key": idempotency_key,
    }
    event = {
        "event_id": event_id,
        "actor": actor,
        "source": source,
        "raw_ref": raw_ref,
        "text": text or "",
        "board_features": as_dict(board_features),
    }
    decision = {
        "decision_uuid": str(decision_uuid) if decision_uuid else None,
        "lane": lane,
        "route_key": route_key or f"{lane}:{source}:{actor}",
        "expected_gain": float(expected_gain or as_dict(board_features).get("expected_gain", 0)),
        "confidence": float(confidence or 0),
        "graph_write_mode": graph_write_mode or ("staged_only" if as_dict(board_features).get("needs_graph_write") else "none"),
        "cost": as_dict(cost),
    }
    return work_order, event, decision


def validate_localized_step(work_order: dict[str, Any], event: dict[str, Any], decision: dict[str, Any]) -> tuple[bool, list[str]]:
    checks: list[str] = []
    if work_order.get("event_id") == event.get("event_id"):
        checks.append("event_id_match")
    if work_order.get("lane") in {"slow", "audit"}:
        checks.append("bounded_lane")
    if decision.get("graph_write_mode") in {"none", "staged_only", "materialization_gate"}:
        checks.append("graph_write_mode_known")
    if event.get("raw_ref") or work_order.get("payload", {}).get("raw_ref"):
        checks.append("evidence_ref_present")
    return len(checks) == 4, checks


def build_worker_result(
    *,
    work_order: dict[str, Any],
    event: dict[str, Any],
    decision: dict[str, Any],
    worker_id: str,
    execute: bool,
    latency_ms: float,
    receipt_path: str = "",
) -> dict[str, Any]:
    ok, checks = validate_localized_step(work_order, event, decision)
    verdict = "win" if execute and ok else ("loss" if execute else "stall")
    features = as_dict(event.get("board_features"))
    tokens_in = int(features.get("token_count") or decision.get("cost", {}).get("tokens") or 0)
    graph_mode = str(decision.get("graph_write_mode") or ("staged_only" if features.get("needs_graph_write") else "none"))
    reduced_entropy = round(float(decision.get("expected_gain") or features.get("expected_gain") or 0) * (1.0 if verdict == "win" else 0.25), 3)
    receipt = {
        "schema": "lucidota.project2501.board_worker.work_receipt.v1",
        "event_id": event["event_id"],
        "decision_uuid": decision.get("decision_uuid") or work_order.get("decision_uuid"),
        "work_order_uuid": work_order["work_order_uuid"],
        "receipt_path": receipt_path,
        "verdict": verdict,
        "cost": {**as_dict(decision.get("cost")), "latency_ms": round(latency_ms, 3), "model_calls": 0},
        "gain": {
            "proof": 1 if verdict == "win" else 0,
            "routing_accuracy": float(decision.get("confidence") or 0),
            "artifact": "bounded_worker_run+work_receipt+river_training_row+watch_metric",
            "fixed_code": False,
            "reduced_entropy": reduced_entropy,
            "checks": checks,
        },
        "artifact_refs": [event.get("raw_ref") or work_order.get("payload", {}).get("raw_ref", ""), "06_SCHEMA/114_project2501_board_worker.sql"],
        "canonical_graph_writes_performed": False,
        "graph_write_mode": graph_mode,
        "detail": {
            "bounded_step": "validate_and_receipt",
            "worker_id": worker_id,
            "operator_feature_authority_required": True,
            "ui_tile_source": "receipt_or_db_only",
            "checks": checks,
        },
    }
    river = {
        "schema": "lucidota.project2501.board_worker.river_training_row.v1",
        "event_id": event["event_id"],
        "decision_uuid": receipt["decision_uuid"],
        "route_chosen": work_order["lane"],
        "model_used": "none",
        "estimated_gain": float(decision.get("expected_gain") or 0),
        "actual_gain": reduced_entropy,
        "latency_ms": round(latency_ms, 3),
        "tokens_in": tokens_in,
        "tokens_out": 0,
        "verdict": verdict,
        "human_override": False,
        "features": features,
        "label": {"route_verdict": verdict, "lane": work_order["lane"], "bounded_step": "validate_and_receipt"},
    }
    metric = {
        "schema": "lucidota.project2501.board_worker.watch_metric.v1",
        "metric_key": "project2501_board_worker_once",
        "metric_value": {
            "event_id": event["event_id"],
            "work_order_uuid": work_order["work_order_uuid"],
            "lane": work_order["lane"],
            "verdict": verdict,
            "bounded_step": "validate_and_receipt",
            "canonical_graph_writes_performed": False,
        },
        "source_receipt": receipt_path,
        "source_db_ref": f"lucidota_control.work_order:{work_order['work_order_uuid']}",
        "operator_requested": True,
        "operator_feature_authority_required": True,
    }
    run = {
        "schema": "lucidota.project2501.board_worker.run.v1",
        "worker_id": worker_id,
        "mode": "worker_once",
        "status": "succeeded" if verdict == "win" else ("failed" if execute else "dry_run"),
        "work_order_uuid": work_order["work_order_uuid"],
        "event_id": event["event_id"],
        "decision_uuid": receipt["decision_uuid"],
        "lane": work_order["lane"],
        "work_kind": work_order["work_kind"],
        "bounded_step": "validate_and_receipt",
        "source_receipt": receipt_path,
        "latency_ms": round(latency_ms, 3),
        "canonical_graph_writes_performed": False,
        "operator_feature_authority_required": True,
        "detail": {"checks": checks, "route_key": decision.get("route_key")},
    }
    return {
        "schema": "lucidota.project2501.board_worker.result.v1",
        "status": "PASS" if verdict in {"win", "stall"} else "FAIL",
        "work_order": work_order,
        "event": event,
        "route_decision": decision,
        "work_receipt": receipt,
        "river_training_row": river,
        "watch_metric": metric,
        "board_worker_run": run,
        "canonical_graph_writes_performed": False,
    }


def claim_sql(lanes: list[str]) -> tuple[str, list[Any]]:
    placeholders = ",".join(["%s"] * len(lanes))
    sql = f"""
        WITH claimed AS (
          SELECT wo.work_order_uuid
          FROM lucidota_control.work_order wo
          WHERE wo.status IN ('created','queued')
            AND wo.lane IN ({placeholders})
          ORDER BY wo.created_at ASC
          FOR UPDATE SKIP LOCKED
          LIMIT 1
        )
        SELECT
          wo.work_order_uuid::text, wo.event_id, wo.decision_uuid::text, wo.lane, wo.work_kind,
          wo.status, wo.payload, wo.idempotency_key,
          ee.actor, ee.source, ee.raw_ref, ee.text, ee.board_features,
          rd.route_key, rd.expected_gain, rd.confidence, rd.graph_write_mode, rd.cost
        FROM claimed c
        JOIN lucidota_control.work_order wo ON wo.work_order_uuid = c.work_order_uuid
        JOIN lucidota_control.event_envelope ee ON ee.event_id = wo.event_id
        LEFT JOIN lucidota_control.route_decision rd ON rd.decision_uuid = wo.decision_uuid
    """
    return sql, lanes


def peek_sql(lanes: list[str]) -> tuple[str, list[Any]]:
    sql, params = claim_sql(lanes)
    return sql.replace("FOR UPDATE SKIP LOCKED", ""), params


def parse_lanes(args: argparse.Namespace) -> list[str]:
    raw = args.lanes or "audit,slow"
    lanes = [x.strip() for x in raw.split(",") if x.strip()]
    return [x for x in lanes if x in {"slow", "audit"}] or ["audit", "slow"]


def latest_event_and_decision(cur: psycopg.Cursor, event_id: str | None) -> tuple[str, str | None, str]:
    if event_id and event_id != "latest":
        cur.execute(
            """
            SELECT ee.event_id, rd.decision_uuid::text, COALESCE(rd.lane,'audit')
            FROM lucidota_control.event_envelope ee
            LEFT JOIN LATERAL (
              SELECT decision_uuid, lane
              FROM lucidota_control.route_decision
              WHERE event_id=ee.event_id
              ORDER BY created_at DESC
              LIMIT 1
            ) rd ON true
            WHERE ee.event_id=%s
            """,
            (event_id,),
        )
    else:
        cur.execute(
            """
            SELECT ee.event_id, rd.decision_uuid::text, COALESCE(rd.lane,'audit')
            FROM lucidota_control.event_envelope ee
            LEFT JOIN LATERAL (
              SELECT decision_uuid, lane
              FROM lucidota_control.route_decision
              WHERE event_id=ee.event_id
              ORDER BY created_at DESC
              LIMIT 1
            ) rd ON true
            ORDER BY ee.created_at DESC
            LIMIT 1
            """
        )
    row = cur.fetchone()
    if not row:
        raise RuntimeError("no_event_envelope_available")
    return str(row[0]), str(row[1]) if row[1] else None, str(row[2] or "audit")


def enqueue(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    url = db_url(args)
    if not args.execute:
        payload = {
            "schema": "lucidota.project2501.board_worker.enqueue.v1",
            "status": "PASS",
            "execute_performed": False,
            "would_enqueue": {"event_id": args.event_id, "lane": args.lane, "work_kind": args.work_kind},
            "canonical_graph_writes_performed": False,
            "database_url": redacted(url),
        }
        write_report(payload, prefix="project2501_board_worker_enqueue")
        return payload
    apply_schema(url)
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            event_id, decision_uuid, decision_lane = latest_event_and_decision(cur, args.event_id)
            lane = args.lane or (decision_lane if decision_lane in {"slow", "audit"} else "audit")
            idem = sha256_text(stable_json({"event_id": event_id, "decision_uuid": decision_uuid, "lane": lane, "work_kind": args.work_kind, "source": "project2501_board_worker_enqueue"}))
            cur.execute(
                """
                INSERT INTO lucidota_control.work_order(event_id, decision_uuid, lane, work_kind, status, payload, idempotency_key)
                VALUES (%s,%s::uuid,%s,%s,'queued',%s::jsonb,%s)
                ON CONFLICT(idempotency_key) DO UPDATE SET status='queued', payload=EXCLUDED.payload, updated_at=now()
                RETURNING work_order_uuid::text
                """,
                (
                    event_id,
                    decision_uuid,
                    lane,
                    args.work_kind,
                    json.dumps({"enqueued_by": "project2501_board_worker", "bounded_step": "validate_and_receipt"}),
                    idem,
                ),
            )
            work_order_uuid = cur.fetchone()[0]
        conn.commit()
    payload = {
        "schema": "lucidota.project2501.board_worker.enqueue.v1",
        "status": "PASS",
        "execute_performed": True,
        "database_url": redacted(url),
        "event_id": event_id,
        "decision_uuid": decision_uuid,
        "work_order_uuid": work_order_uuid,
        "lane": lane,
        "work_kind": args.work_kind,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "canonical_graph_writes_performed": False,
    }
    write_report(payload, prefix="project2501_board_worker_enqueue")
    return payload


def persist_worker_result(cur: psycopg.Cursor, result: dict[str, Any]) -> dict[str, str]:
    r = result["work_receipt"]
    row = result["river_training_row"]
    run = result["board_worker_run"]
    metric = result["watch_metric"]
    work_order_uuid = result["work_order"]["work_order_uuid"]
    status = "succeeded" if r["verdict"] == "win" else "failed"
    cur.execute("UPDATE lucidota_control.work_order SET status='running', updated_at=now() WHERE work_order_uuid=%s::uuid", (work_order_uuid,))
    cur.execute(
        """
        INSERT INTO lucidota_control.work_receipt(event_id, decision_uuid, work_order_uuid, receipt_path, receipt_sha256, verdict, cost, gain, artifact_refs, canonical_graph_writes_performed, graph_write_mode, detail)
        VALUES (%s,%s::uuid,%s::uuid,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,false,%s,%s::jsonb)
        RETURNING work_receipt_uuid::text
        """,
        (
            r["event_id"],
            r["decision_uuid"],
            r["work_order_uuid"],
            r["receipt_path"],
            sha256_text(stable_json(result)),
            r["verdict"],
            json.dumps(r["cost"]),
            json.dumps(r["gain"]),
            json.dumps(r["artifact_refs"]),
            r["graph_write_mode"],
            json.dumps(r["detail"]),
        ),
    )
    receipt_uuid = cur.fetchone()[0]
    cur.execute(
        """
        INSERT INTO lucidota_control.river_training_row(event_id, decision_uuid, work_receipt_uuid, route_chosen, model_used, estimated_gain, actual_gain, latency_ms, tokens_in, tokens_out, verdict, human_override, features, label)
        VALUES (%s,%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,false,%s::jsonb,%s::jsonb)
        RETURNING training_uuid::text
        """,
        (
            row["event_id"],
            row["decision_uuid"],
            receipt_uuid,
            row["route_chosen"],
            row["model_used"],
            row["estimated_gain"],
            row["actual_gain"],
            row["latency_ms"],
            row["tokens_in"],
            row["tokens_out"],
            row["verdict"],
            json.dumps(row["features"]),
            json.dumps(row["label"]),
        ),
    )
    training_uuid = cur.fetchone()[0]
    cur.execute(
        """
        INSERT INTO lucidota_control.board_worker_run(worker_id, mode, status, work_order_uuid, event_id, decision_uuid, lane, work_kind, bounded_step, receipt_uuid, source_receipt, latency_ms, canonical_graph_writes_performed, operator_feature_authority_required, detail)
        VALUES (%s,%s,%s,%s::uuid,%s,%s::uuid,%s,%s,%s,%s::uuid,%s,%s,false,true,%s::jsonb)
        RETURNING board_worker_run_uuid::text
        """,
        (
            run["worker_id"],
            run["mode"],
            run["status"],
            run["work_order_uuid"],
            run["event_id"],
            run["decision_uuid"],
            run["lane"],
            run["work_kind"],
            run["bounded_step"],
            receipt_uuid,
            run["source_receipt"],
            run["latency_ms"],
            json.dumps(run["detail"]),
        ),
    )
    run_uuid = cur.fetchone()[0]
    cur.execute(
        """
        INSERT INTO lucidota_control.watch_metric(metric_key, metric_value, source_receipt, source_db_ref, operator_requested, operator_feature_authority_required)
        VALUES (%s,%s::jsonb,%s,%s,%s,%s)
        RETURNING metric_uuid::text
        """,
        (
            metric["metric_key"],
            json.dumps(metric["metric_value"]),
            metric["source_receipt"],
            metric["source_db_ref"],
            metric["operator_requested"],
            metric["operator_feature_authority_required"],
        ),
    )
    metric_uuid = cur.fetchone()[0]
    cur.execute("UPDATE lucidota_control.work_order SET status=%s, updated_at=now() WHERE work_order_uuid=%s::uuid", (status, work_order_uuid))
    return {"work_receipt_uuid": receipt_uuid, "training_uuid": training_uuid, "board_worker_run_uuid": run_uuid, "metric_uuid": metric_uuid}


def worker_once(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    url = db_url(args)
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"
    lanes = parse_lanes(args)
    base = {
        "schema": "lucidota.project2501.board_worker.report.v1",
        "status": "PASS",
        "generated_at": now(),
        "database_url": redacted(url),
        "worker_id": worker_id,
        "lanes": lanes,
        "execute_performed": bool(args.execute),
        "canonical_graph_writes_performed": False,
    }
    if not args.execute:
        try:
            with psycopg.connect(url) as conn:
                with conn.cursor() as cur:
                    sql, params = peek_sql(lanes)
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    base["would_process"] = None
                    if row:
                        work_order, event, decision = work_order_from_row(row)
                        base.update(
                            build_worker_result(
                                work_order=work_order,
                                event=event,
                                decision=decision,
                                worker_id=worker_id,
                                execute=False,
                                latency_ms=(time.perf_counter() - started) * 1000,
                            )
                        )
                    else:
                        base["no_work_order_available"] = True
        except Exception as exc:
            base["dry_run_db_probe"] = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
            base["no_work_order_available"] = True
        write_report(base)
        return base
    apply_schema(url)
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            sql, params = claim_sql(lanes)
            cur.execute(sql, params)
            row = cur.fetchone()
            if not row:
                base.update({"no_work_order_available": True})
                cur.execute(
                    """
                    INSERT INTO lucidota_control.board_worker_run(worker_id, mode, status, lane, work_kind, canonical_graph_writes_performed, operator_feature_authority_required, detail)
                    VALUES (%s,'worker_once','no_work',%s,'none',false,true,%s::jsonb)
                    RETURNING board_worker_run_uuid::text
                    """,
                    (worker_id, lanes[0], json.dumps({"lanes": lanes})),
                )
                base["db_rows"] = {"board_worker_run_uuid": cur.fetchone()[0]}
                conn.commit()
                write_report(base)
                return base
            work_order, event, decision = work_order_from_row(row)
            result = build_worker_result(
                work_order=work_order,
                event=event,
                decision=decision,
                worker_id=worker_id,
                execute=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            base.update(result)
            path = write_report(base)
            # Keep DB source_receipt equal to the actual receipt path.
            base["work_receipt"]["receipt_path"] = rel(path)
            base["watch_metric"]["source_receipt"] = rel(path)
            base["board_worker_run"]["source_receipt"] = rel(path)
            db_rows = persist_worker_result(cur, base)
            base["db_rows"] = db_rows
            conn.commit()
            path.write_text(json.dumps(base, indent=2, sort_keys=True, default=str), encoding="utf-8")
            return base


def init_schema(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "schema": "lucidota.project2501.board_worker.init_schema.v1",
        "status": "PASS",
        "generated_at": now(),
        "execute_performed": bool(args.execute),
        "database_url": redacted(db_url(args)),
        "schema_paths": [rel(CORE_SCHEMA), rel(WORKER_SCHEMA)],
        "canonical_graph_writes_performed": False,
    }
    if args.execute:
        apply_schema(db_url(args))
    write_report(payload, prefix="project2501_board_worker_init_schema")
    return payload


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Project 2501 bounded slow/audit board worker.")
    ap.add_argument("--database-url")
    sub = ap.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init-schema")
    init.add_argument("--execute", action="store_true")
    init.add_argument("--json", action="store_true")
    enq = sub.add_parser("enqueue")
    enq.add_argument("--execute", action="store_true")
    enq.add_argument("--event-id", default="latest")
    enq.add_argument("--lane", choices=["slow", "audit"], default="audit")
    enq.add_argument("--work-kind", default="bounded_audit")
    enq.add_argument("--json", action="store_true")
    once = sub.add_parser("worker-once")
    once.add_argument("--execute", action="store_true")
    once.add_argument("--worker-id")
    once.add_argument("--lanes", default="audit,slow")
    once.add_argument("--json", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "init-schema":
        payload = init_schema(args)
    elif args.cmd == "enqueue":
        payload = enqueue(args)
    else:
        payload = worker_once(args)
    if getattr(args, "json", False):
        print(json.dumps(payload, sort_keys=True, default=str))
    print("PROJECT2501_BOARD_WORKER=" + payload["status"])
    return 0 if payload["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
