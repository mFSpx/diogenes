#!/usr/bin/env python3
"""Bytewax-compatible stream over Project 2501 board-move tables."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "project2501_board_stream"
SCHEMAS = [
    ROOT / "06_SCHEMA" / "112_project2501_core_board.sql",
    ROOT / "06_SCHEMA" / "113_project2501_bytewax_board_stream.sql",
    ROOT / "06_SCHEMA" / "116_operator_feedback_signal.sql",
]
TERNARY_VALUES = {-1, 0, 1}
NEGATIVE_VERDICTS = {"blocked", "dead_letter", "error", "fail", "failed", "loss", "rejected", "stall"}
POSITIVE_VERDICTS = {"accepted", "pass", "passed", "succeeded", "success", "win"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


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


def write_report(action: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"project2501_bytewax_board_stream_{action}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    return path


def bytewax_probe() -> dict[str, Any]:
    try:
        import bytewax  # type: ignore

        return {"available": True, "module": "bytewax", "version": getattr(bytewax, "__version__", None)}
    except Exception as exc:
        return {"available": False, "module": "bytewax", "reason": f"{type(exc).__name__}:{exc}", "fallback": "in_process_map"}


def apply_schema(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for schema in SCHEMAS:
                cur.execute(schema.read_text(encoding="utf-8"))
        conn.commit()


def coerce_ternary_valency(value: Any) -> int | None:
    try:
        val = int(value)
    except (TypeError, ValueError):
        return None
    return val if val in TERNARY_VALUES else None


def infer_ternary_valency(row: dict[str, Any]) -> int:
    features = row.get("board_features") if isinstance(row.get("board_features"), dict) else {}
    cost = row.get("route_cost") if isinstance(row.get("route_cost"), dict) else {}
    gain = row.get("gain") if isinstance(row.get("gain"), dict) else {}
    for source in (row, features, cost, gain):
        explicit = coerce_ternary_valency(source.get("ternary_valency")) if isinstance(source, dict) else None
        if explicit is not None:
            return explicit
    lane = str(row.get("lane") or "")
    verdict = str(row.get("verdict") or "").lower()
    if lane == "dead_letter" or verdict in NEGATIVE_VERDICTS:
        return -1
    if verdict in POSITIVE_VERDICTS and str(row.get("receipt") or ""):
        return 1
    return 0


def kleene_k3_collapse(values: list[int]) -> int:
    if any(v == -1 for v in values):
        return -1
    if any(v == 0 for v in values):
        return 0
    return 1


def corridor_state(net_spatial_polarity: int) -> str:
    if net_spatial_polarity > 0:
        return "signal"
    if net_spatial_polarity < 0:
        return "friction"
    return "stasis"


def annotate_neighborhood_polarity(hints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sums: dict[str, int] = {}
    k3_inputs: dict[str, list[int]] = {}
    for hint_row in hints:
        key = str(hint_row.get("route_key") or hint_row.get("lane") or "__global__")
        detail = hint_row.setdefault("detail", {})
        logic = detail.setdefault("ternary_logic", {})
        valency = int(logic.get("valency") or 0)
        sums[key] = sums.get(key, 0) + valency
        k3_inputs.setdefault(key, []).append(valency)
        net = sums[key]
        k3_state = kleene_k3_collapse(k3_inputs[key])
        state = corridor_state(net)
        logic.update(
            {
                "corridor_key": key,
                "net_spatial_polarity": net,
                "corridor_state": state,
                "kleene_k3_state": k3_state,
                "notification_suppressed": state == "stasis",
                "operator_product_rules": {"+1": "preserve", "0": "mask_to_vacuum", "-1": "invert"},
            }
        )
        hint_row["hint"] = f"{hint_row['hint']}:ternary={valency}:net={net}:{state}"
    return hints


def hint(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("board_features") if isinstance(row.get("board_features"), dict) else {}
    cost = row.get("route_cost") if isinstance(row.get("route_cost"), dict) else {}
    gain = row.get("gain") if isinstance(row.get("gain"), dict) else {}
    lane = str(row.get("lane") or "dead_letter")
    verdict = str(row.get("verdict") or "stall")
    token_count = int(features.get("token_count") or cost.get("tokens") or 0)
    risk = float(features.get("risk_of_slop") or cost.get("risk") or 0.0)
    entropy_gain = float(gain.get("reduced_entropy") or 0.0)
    routing_accuracy = float(gain.get("routing_accuracy") or 0.0)
    ternary_valency = infer_ternary_valency(row)
    lane_weight = {"fast": 20, "slow": 45, "audit": 70, "external": 65, "dead_letter": 90}.get(lane, 50)
    score = max(0, min(100, round(lane_weight + risk * 20 + entropy_gain * 15 + routing_accuracy * 10 + min(token_count, 1000) / 100)))
    treelite = {
        "gate": "board_pressure_inline",
        "model": "deterministic_bytewax_stream_score_v001",
        "score": score,
        "features": {
            "token_count": token_count,
            "risk_of_slop": risk,
            "reduced_entropy": entropy_gain,
                "routing_accuracy": routing_accuracy,
                "lane": lane,
                "ternary_valency": ternary_valency,
            },
        }
    return {
        "schema": "lucidota.project2501.board_stream_hint.v1",
        "event_id": str(row.get("event_id") or ""),
        "source": str(row.get("source") or ""),
        "actor": str(row.get("actor") or "unknown"),
        "lane": lane,
        "route_key": str(row.get("route_key") or ""),
        "verdict": verdict,
        "hint_kind": "board_pressure",
        "hint": f"{lane}:{row.get('actor')}:{verdict}:tokens={token_count}:risk={risk:.2f}",
        "score": score,
        "canonical_graph_writes_performed": False,
        "source_receipt": str(row.get("receipt") or ""),
        "detail": {
            "bytewax_flow": "project2501_board_stream",
            "treelite": treelite,
            "ternary_logic": {
                "valency": ternary_valency,
                "local_state": corridor_state(ternary_valency),
                "shift_accumulate": "linear_integer_sum",
                "kleene_k3_gate": "any_-1_collapses_else_any_0_neutral_else_+1",
            },
            "event_created_at": str(row.get("event_created_at") or ""),
            "canonical_graph_writes_performed_in_source": bool(row.get("canonical_graph_writes_performed")),
        },
    }


def flow(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from bytewax.dataflow import Dataflow
        import bytewax.operators as op
        from bytewax.testing import TestingSink, TestingSource, run_main

        dataflow = Dataflow("project2501-board-stream")
        inp = op.input("board_rows", dataflow, TestingSource(events))
        out: list[dict[str, Any]] = []
        op.output("hints", op.map("hint", inp, hint), TestingSink(out))
        run_main(dataflow)
        return annotate_neighborhood_polarity(out)
    except Exception:
        return annotate_neighborhood_polarity([hint(row) for row in events])


def rows(conn: Any, limit: int, live_cursor: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limit = max(1, min(1000, int(limit)))
    meta: dict[str, Any] = {"cursor_lock": True}
    where = ""
    params: list[Any] = []
    if live_cursor:
        locked = bool(conn.execute("SELECT pg_try_advisory_xact_lock(hashtext('project2501_board_stream'))").fetchone()[0])
        meta["cursor_lock"] = locked
        if not locked:
            return [], meta
        conn.execute("INSERT INTO lucidota_control.board_stream_cursor(cursor_name) VALUES ('project2501_board_stream') ON CONFLICT(cursor_name) DO NOTHING")
        where = """
        WHERE (e.created_at, e.event_id) > (
          SELECT last_event_created_at, COALESCE(last_event_id, '')
          FROM lucidota_control.board_stream_cursor
          WHERE cursor_name='project2501_board_stream'
        )
        """
    sql = f"""
    SELECT
      e.event_id,
      e.actor,
      e.source,
      COALESCE(d.lane, bm.lane, 'dead_letter') AS lane,
      COALESCE(d.route_key, '') AS route_key,
      COALESCE(wr.verdict, bm.verdict, 'stall') AS verdict,
      e.created_at AS event_created_at,
      e.board_features,
      COALESCE(d.cost, '{{}}'::jsonb) AS route_cost,
      COALESCE(wr.gain, bm.gain, '{{}}'::jsonb) AS gain,
      COALESCE(wr.receipt_path, bm.receipt, '') AS receipt,
      COALESCE(wr.canonical_graph_writes_performed, false) AS canonical_graph_writes_performed
    FROM lucidota_control.event_envelope e
    LEFT JOIN LATERAL (
      SELECT * FROM lucidota_control.route_decision d
      WHERE d.event_id=e.event_id
      ORDER BY d.created_at DESC
      LIMIT 1
    ) d ON true
    LEFT JOIN LATERAL (
      SELECT * FROM lucidota_control.work_receipt wr
      WHERE wr.event_id=e.event_id
      ORDER BY wr.created_at DESC
      LIMIT 1
    ) wr ON true
    LEFT JOIN LATERAL (
      SELECT * FROM lucidota_control.board_move bm
      WHERE bm.event_id=e.event_id
      ORDER BY bm.created_at DESC
      LIMIT 1
    ) bm ON true
    {where}
    ORDER BY e.created_at ASC, e.event_id ASC
    LIMIT %s
    """
    params.append(limit)
    got = conn.execute(sql, params).fetchall()
    keys = ["event_id", "actor", "source", "lane", "route_key", "verdict", "event_created_at", "board_features", "route_cost", "gain", "receipt", "canonical_graph_writes_performed"]
    return [dict(zip(keys, row)) for row in got], meta


def persist(conn: Any, events: list[dict[str, Any]], hints: list[dict[str, Any]], *, mode: str, meta: dict[str, Any], probe: dict[str, Any]) -> str:
    cur = conn.execute(
        """
        INSERT INTO lucidota_control.board_stream_run(mode,status,events_in,hints_out,cursor_lock,bytewax_available,canonical_graph_writes_performed,detail)
        VALUES (%s,'succeeded',%s,%s,%s,%s,false,%s::jsonb)
        RETURNING run_uuid::text
        """,
        (mode, len(events), len(hints), bool(meta.get("cursor_lock", True)), bool(probe.get("available")), json.dumps({"bytewax_probe": probe, **meta}, default=str)),
    )
    run_uuid = cur.fetchone()[0]
    for h in hints:
        conn.execute(
            """
            INSERT INTO lucidota_control.board_stream_hint(event_id,source,actor,lane,route_key,verdict,hint_kind,hint,score,canonical_graph_writes_performed,source_receipt,run_uuid,detail)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,false,%s,%s::uuid,%s::jsonb)
            ON CONFLICT(event_id,hint_kind) DO UPDATE SET
              source=EXCLUDED.source,
              actor=EXCLUDED.actor,
              lane=EXCLUDED.lane,
              route_key=EXCLUDED.route_key,
              verdict=EXCLUDED.verdict,
              hint=EXCLUDED.hint,
              score=EXCLUDED.score,
              source_receipt=EXCLUDED.source_receipt,
              run_uuid=EXCLUDED.run_uuid,
              detail=EXCLUDED.detail,
              created_at=now()
            """,
            (h["event_id"], h["source"], h["actor"], h["lane"], h["route_key"], h["verdict"], h["hint_kind"], h["hint"], h["score"], h["source_receipt"], run_uuid, json.dumps(h["detail"], default=str)),
        )
    if events and mode == "live_cursor":
        last = events[-1]
        conn.execute(
            """
            INSERT INTO lucidota_control.board_stream_cursor(cursor_name,last_event_created_at,last_event_id,updated_at,detail)
            VALUES ('project2501_board_stream',%s,%s,now(),%s::jsonb)
            ON CONFLICT(cursor_name) DO UPDATE SET
              last_event_created_at=EXCLUDED.last_event_created_at,
              last_event_id=EXCLUDED.last_event_id,
              updated_at=now(),
              detail=EXCLUDED.detail
            """,
            (last["event_created_at"], last["event_id"], json.dumps({"run_uuid": run_uuid, "mode": mode})),
        )
    feedback_training = train_operator_feedback(conn)
    conn.execute(
        """
        UPDATE lucidota_control.board_stream_run
        SET detail = detail || %s::jsonb
        WHERE run_uuid=%s::uuid
        """,
        (json.dumps({"operator_feedback_training": feedback_training}, default=str), run_uuid),
    )
    conn.execute(
        """
        INSERT INTO lucidota_control.watch_metric(metric_key,metric_value,source_receipt,source_db_ref,operator_requested)
        VALUES ('project2501_board_stream',%s::jsonb,'',%s,true)
        """,
        (json.dumps({"run_uuid": run_uuid, "events_in": len(events), "hints_out": len(hints), "mode": mode, "operator_feedback_training": feedback_training}, default=str), f"lucidota_control.board_stream_run:{run_uuid}"),
    )
    return run_uuid


def train_operator_feedback(conn: Any, max_rows: int = 500) -> dict[str, Any]:
    exists = conn.execute("SELECT to_regprocedure('lucidota_learning.fn_train_operator_feedback_batch(integer)')").fetchone()[0]
    if exists is None:
        return {"status": "operator_feedback_schema_missing", "events_seen": 0, "examples_trained": 0}
    row = conn.execute("SELECT lucidota_learning.fn_train_operator_feedback_batch(%s)::jsonb", (max_rows,)).fetchone()
    payload = row[0] if row else {}
    return payload if isinstance(payload, dict) else {"status": "unknown_feedback_training_result", "raw": str(payload)}


def once(args: argparse.Namespace) -> int:
    probe = bytewax_probe()
    mode = "live_cursor" if args.live_cursor else "latest_window"
    events: list[dict[str, Any]] = []
    hints: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"cursor_lock": True}
    run_uuid = None
    blockers: list[str] = []
    if args.execute:
        apply_schema(db_url(args))
    try:
        with psycopg.connect(db_url(args)) as conn:
            if args.execute:
                events, meta = rows(conn, args.limit, args.live_cursor)
                hints = flow(events)
                run_uuid = persist(conn, events, hints, mode=mode, meta=meta, probe=probe)
                conn.commit()
            else:
                # Read-only dry run: no schema apply, no cursor writes, no DB metrics.
                with conn.cursor() as cur:
                    cur.execute("SET TRANSACTION READ ONLY")
                    cur.execute("SELECT to_regclass('lucidota_control.event_envelope')")
                    if cur.fetchone()[0] is not None:
                        events, meta = rows(conn, args.limit, False)
                        hints = flow(events)
    except Exception as exc:
        blockers.append(f"{type(exc).__name__}:{str(exc)[:300]}")
    report = {
        "schema": "lucidota.project2501.bytewax_board_stream.report.v1",
        "action": "once",
        "mode": mode,
        "execute_performed": bool(args.execute),
        "database_url": redacted(db_url(args)),
        "bytewax_probe": probe,
        "events_in": len(events),
        "hints_out": len(hints),
        "sample_hints": hints[:5],
        "run_uuid": run_uuid,
        "canonical_graph_writes_performed": False,
        "blockers": blockers,
        "status": "PASS" if not blockers else "FAIL",
        **meta,
    }
    write_report("once_execute" if args.execute else "once_dry_run", report)
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    print("PROJECT2501_BYTEWAX_BOARD_STREAM=" + report["status"])
    return 0 if report["status"] == "PASS" else 4


def init_schema(args: argparse.Namespace) -> int:
    blockers: list[str] = []
    if args.execute:
        try:
            apply_schema(db_url(args))
        except Exception as exc:
            blockers.append(f"{type(exc).__name__}:{str(exc)[:300]}")
    report = {
        "schema": "lucidota.project2501.bytewax_board_stream.init_schema.v1",
        "action": "init_schema",
        "execute_performed": bool(args.execute),
        "database_url": redacted(db_url(args)),
        "schema_paths": [rel(p) for p in SCHEMAS],
        "canonical_graph_writes_performed": False,
        "blockers": blockers,
        "status": "PASS" if not blockers else "FAIL",
    }
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", report)
    print("PROJECT2501_BYTEWAX_BOARD_SCHEMA=" + report["status"])
    return 0 if report["status"] == "PASS" else 4


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Bytewax-compatible Project 2501 board stream.")
    ap.add_argument("--database-url")
    sub = ap.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init-schema")
    init.add_argument("--execute", action="store_true")
    once_p = sub.add_parser("once")
    once_p.add_argument("--limit", type=int, default=25)
    once_p.add_argument("--live-cursor", action="store_true")
    once_p.add_argument("--execute", action="store_true")
    once_p.add_argument("--json", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "init-schema":
        return init_schema(args)
    return once(args)


if __name__ == "__main__":
    raise SystemExit(main())
