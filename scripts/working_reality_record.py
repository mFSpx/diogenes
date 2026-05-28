#!/usr/bin/env python3
"""Working Reality Law recorder: evidence -> hypothesis -> selected action frame."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "working_reality"
OBS = ROOT / "04_RUNTIME" / "observation_center" / "working_reality_latest.json"
SCHEMA = ROOT / "06_SCHEMA" / "117_working_reality_law.sql"
RESULTS = {"PASS", "FAIL", "CONFLICT", "OPEN", "STALE", "SUPERSEDED"}


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
        or os.environ.get("LUCIDOTA_WORKING_REALITY_DATABASE_URL")
        or os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def build_working_reality_record(
    *,
    evidence: list[str],
    hypothesis: str,
    working_reality: str,
    move: str,
    result: str,
    contradiction_refs: list[str] | None = None,
    rejected_hypotheses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = result.upper().strip()
    return {
        "schema": "lucidota.working_reality.record.v1",
        "generated_at": now(),
        "layers": {
            "reality": {"claim": "exists_beyond_system_possession"},
            "evidence": {"refs": evidence},
            "hypothesis": {"text": hypothesis, "truth_status": "provisional"},
            "working_reality": {"text": working_reality, "selected_for_action": True},
        },
        "evidence": evidence,
        "hypothesis": hypothesis,
        "working_reality": working_reality,
        "move": move,
        "result": result,
        "contradiction_refs": contradiction_refs or [],
        "rejected_hypotheses": rejected_hypotheses or [],
        "record_for_future": True,
        "canonical_graph_writes_performed": False,
    }


def validate_working_reality_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema") != "lucidota.working_reality.record.v1":
        errors.append("schema_invalid")
    for key in ("evidence", "hypothesis", "working_reality", "move", "result"):
        if key not in record:
            errors.append(f"missing:{key}")
    if not isinstance(record.get("evidence"), list) or not record.get("evidence"):
        errors.append("evidence_refs_required")
    if str(record.get("result", "")).upper() not in RESULTS:
        errors.append("result_invalid")
    if record.get("record_for_future") is not True:
        errors.append("record_for_future_must_be_true")
    if record.get("canonical_graph_writes_performed") is not False:
        errors.append("canonical_graph_writes_forbidden")
    layers = record.get("layers") or {}
    for key in ("reality", "evidence", "hypothesis", "working_reality"):
        if key not in layers:
            errors.append(f"missing_layer:{key}")
    return errors


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_observation(receipt: dict[str, Any], *, root: Path = ROOT) -> None:
    summary = {
        "schema": "lucidota.observation_center.working_reality.v1",
        "generated_at": receipt["generated_at"],
        "receipt_path": receipt["receipt_path"],
        "record": receipt["record"],
        "execute_performed": receipt["execute_performed"],
        "canonical_graph_writes_performed": False,
    }
    obs = root / "04_RUNTIME" / "observation_center" / "working_reality_latest.json"
    write_json(obs, summary)
    big_path = root / "05_OUTPUTS" / "big_board.json"
    try:
        big = json.loads(big_path.read_text(encoding="utf-8")) if big_path.exists() else {}
    except Exception:
        big = {}
    big.setdefault("observation_center", {})["working_reality"] = summary
    counters = big.setdefault("counters", {})
    counters["working_reality_record_for_future"] = True
    counters["working_reality_last_result"] = receipt["record"]["result"]
    write_json(big_path, big)


def insert_record(conn: psycopg.Connection[Any], record: dict[str, Any], receipt_path: str) -> str:
    with conn.cursor() as cur:
        cur.execute(SCHEMA.read_text(encoding="utf-8"))
        cur.execute(
            """
            INSERT INTO lucidota_working_reality.working_reality_move(
              evidence_refs, hypothesis, working_reality, move, result, record_for_future,
              contradiction_refs, rejected_hypotheses, layer_snapshot, receipt_path,
              canonical_graph_writes_performed
            )
            VALUES (%s,%s,%s,%s,%s,true,%s,%s::jsonb,%s::jsonb,%s,false)
            RETURNING move_uuid::text
            """,
            (
                record["evidence"],
                record["hypothesis"],
                record["working_reality"],
                record["move"],
                record["result"],
                record["contradiction_refs"],
                json.dumps(record["rejected_hypotheses"], default=str),
                json.dumps(record["layers"], default=str),
                receipt_path,
            ),
        )
        return str(cur.fetchone()[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Record a Working Reality Law board move.")
    ap.add_argument("--evidence", action="append", required=True)
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--working-reality", required=True)
    ap.add_argument("--move", required=True)
    ap.add_argument("--result", default="OPEN")
    ap.add_argument("--contradiction-ref", action="append", default=[])
    ap.add_argument("--rejected-hypothesis-json", action="append", default=[])
    ap.add_argument("--database-url")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rejected = [json.loads(item) for item in args.rejected_hypothesis_json]
    record = build_working_reality_record(
        evidence=args.evidence,
        hypothesis=args.hypothesis,
        working_reality=args.working_reality,
        move=args.move,
        result=args.result,
        contradiction_refs=args.contradiction_ref,
        rejected_hypotheses=rejected,
    )
    errors = validate_working_reality_record(record)
    receipt_path = OUT / f"working_reality_record_{stamp()}.json"
    move_uuid = None
    if args.execute and not errors:
        with psycopg.connect(db_url(args)) as conn:
            move_uuid = insert_record(conn, record, rel(receipt_path))
            conn.commit()
    receipt = {
        "schema": "lucidota.working_reality.receipt.v1",
        "generated_at": now(),
        "execute_performed": bool(args.execute),
        "record": record,
        "validation_errors": errors,
        "move_uuid": move_uuid,
        "receipt_path": rel(receipt_path),
        "canonical_graph_writes_performed": False,
    }
    write_json(receipt_path, receipt)
    write_observation(receipt)
    print("RECEIPT_PATH=" + rel(receipt_path))
    print("WORKING_REALITY_RECORD=" + ("PASS" if not errors else "FAIL"))
    if args.json:
        print(json.dumps(receipt, sort_keys=True, default=str))
    return 0 if not errors else 4


if __name__ == "__main__":
    raise SystemExit(main())
