#!/usr/bin/env python3
"""LUCI learning slice: board-state map -> source/artifact trial -> score -> receipt.

This is a reusable class-handler for operator-learning tasks. It studies one
current source or internal artifact, generates a bounded improvement candidate,
runs a cheap probe, scores the result, and writes a receipt plus DB-backed work
records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "scripts" / "dev_journey_decision_points.py"
OUT = ROOT / "05_OUTPUTS" / "luci_learning"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def db_url(value: str | None = None) -> str:
    return (
        value
        or os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
        or os.environ.get("LUCIDOTA_GO_STORAGE_DSN")
        or "postgresql:///lucidota_state"
    )


@dataclass
class BoardState:
    actors: list[str]
    resources: list[str]
    constraints: list[str]
    timing: list[str]
    leverage: list[str]
    friction: list[str]
    inertia: list[str]
    visibility: list[str]
    incentives: list[str]
    terrain: list[str]
    available_moves: list[str]
    expected_counter_moves: list[str]
    cheapest_probes: list[str]
    highest_gain_pivots: list[str]


def _pick(text: str, options: dict[str, list[str]]) -> list[str]:
    low = text.lower()
    picked: list[str] = []
    for label, keys in options.items():
        if any(k in low for k in keys):
            picked.append(label)
    return picked


def map_board_state(text: str, artifact: Path) -> BoardState:
    low = text.lower()
    resources = _pick(low, {
        "postgres receipts": ["db", "postgres", "receipt"],
        "treelite routers": ["treelite", "xgboost", "router"],
        "source artifact": ["source", "artifact"],
        "promptflow sidecar": ["promptflow"],
        "vibes/groq worker lane": ["groq", "vibes"],
    })
    artifact_name = artifact.name.lower()
    if "decision_points" in artifact_name or "treelite" in artifact_name or "xgboost" in artifact_name:
        if "treelite routers" not in resources:
            resources.append("treelite routers")
    return BoardState(
        actors=["operator", "Indy_READs", "LUCI", artifact.name],
        resources=resources,
        constraints=_pick(low, {
            "receipt law": ["receipt", "proof"],
            "sidecar-only promptflow": ["sidecar"],
            "rust/db-first": ["rust", "db", "postgres"],
            "bounded probe budget": ["cheap", "small", "bounded"],
        }),
        timing=_pick(low, {
            "now": ["now", "today", "immediately"],
            "iterative": ["iterate", "retry", "mutate", "learn"],
            "asynchronous": ["async", "queue", "receipt"],
        }),
        leverage=_pick(low, {
            "existing harness": ["harness", "test", "probe"],
            "existing treelite": ["treelite", "xgboost"],
            "existing ingestion rails": ["ingest", "source", "artifact"],
        }),
        friction=_pick(low, {
            "shell-wrapper leakage": ["shell", "wrapper"],
            "one-off script risk": ["one-off", "script"],
            "stale artifact drift": ["stale", "drift"],
        }),
        inertia=_pick(low, {
            "legacy names": ["claw", "dbos"],
            "script inertia": ["script", "glue"],
            "missing adapters": ["adapter", "source"],
        }),
        visibility=_pick(low, {
            "receipt-backed": ["receipt"],
            "db-verifiable": ["db", "postgres"],
            "operator-visible": ["operator", "Indy_READs", "luci"],
        }),
        incentives=_pick(low, {
            "faster routing": ["fast", "speed", "quick"],
            "learn by doing": ["learn", "improve", "study"],
            "generalize class-handler": ["class", "reusable", "handler"],
        }),
        terrain=[artifact.suffix.lower().lstrip(".") or "unknown", artifact.parent.name],
        available_moves=[
            "inspect artifact",
            "compile decision points",
            "train treelite/xgboost candidate",
            "run cheap probe",
            "write receipt",
        ],
        expected_counter_moves=[
            "vanished file",
            "probe failure",
            "overgeneralization",
            "db write failure",
        ],
        cheapest_probes=[
            "python -m py_compile <artifact>",
            "dev_journey_decision_points.py --source <artifact> --max-points 16",
        ],
        highest_gain_pivots=[
            "convert reusable source insight into a Treelite feature",
            "promote a probe into a reusable LUCI class-handler",
        ],
    )


def classify_candidate(artifact: Path, text: str, candidate_kind: str | None = None) -> dict[str, Any]:
    name = artifact.name.lower()
    low = text.lower()
    if candidate_kind:
        explicit = candidate_kind.lower().strip()
        if explicit == "algorithm":
            kind = "algorithm_trial_harness"
        elif explicit == "source":
            kind = "current_world_source_adapter"
        elif explicit == "delegate":
            kind = "delegate_provider_class"
        elif explicit == "model":
            kind = "model_runtime_class"
        elif explicit in {"archive", "archive-class", "ingestion"}:
            kind = "archive_ingestion_class"
        else:
            kind = "operator_learning_class"
    elif "treelite" in name or "xgboost" in name or "router" in low or "decision_points" in name or "treelite" in low:
        kind = "algorithm_trial_harness"
    elif "archive" in low or "krampus" in low or "archive" in name:
        kind = "archive_ingestion_class"
    elif "ingest" in name or "source" in low or "artifact" in low:
        kind = "current_world_source_adapter"
    elif "delegate" in low or "groq" in low or "vibes" in low:
        kind = "delegate_provider_class"
    elif "model" in low or "admission" in low or "provider" in low:
        kind = "model_runtime_class"
    else:
        kind = "operator_learning_class"
    return {
        "candidate_kind": kind,
        "candidate_name": f"luci_{kind}",
        "feature_hypothesis": [
            "board-state classification",
            "receipt-backed probe",
            "treelite-compatible weak labels",
        ],
    }


def probe_candidate(artifact: Path, text: str, candidate_kind: str) -> dict[str, Any]:
    if candidate_kind == "current_world_source_adapter":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "luci_source_slice.py"),
            "--text",
            text,
            "--source",
            "github",
            "--limit",
            "1",
            "--json",
        ]
    elif candidate_kind == "archive_ingestion_class":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "luci_ingestion_status.py"),
            "--json",
        ]
    elif candidate_kind == "delegate_provider_class":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "luci_delegate_slice.py"),
            "--kind",
            "review",
            "--provider",
            "groq",
            "--text",
            text,
            "--json",
        ]
    elif candidate_kind == "model_runtime_class":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "lucidota_strict_model_stack_admission.py"),
            "--run-diogenes-gate",
            "--json",
        ]
    else:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "dev_journey_decision_points.py"),
            "--source",
            str(artifact),
            "--max-points",
            "16",
            "--train",
            "--json",
        ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    stdout_lines = proc.stdout.splitlines()
    json_blob = ""
    for i, line in enumerate(stdout_lines):
        if line.strip().startswith("{"):
            json_blob = "\n".join(stdout_lines[i:])
            break
    payload: dict[str, Any] = {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2400:],
        "stderr_tail": proc.stderr[-2400:],
        "passed": proc.returncode == 0,
    }
    if json_blob:
        try:
            payload["probe_result"] = json.loads(json_blob)
        except json.JSONDecodeError:
            payload["probe_result"] = {"parse_error": True}
    return payload


def score_attempt(board: BoardState, candidate: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    gain = 0.2
    if probe.get("passed"):
        gain += 0.5
    if (probe.get("probe_result") or {}).get("tree_artifacts", {}).get("training_performed"):
        gain += 0.2
    if board.resources:
        gain += 0.05
    cost = 0.1 if probe.get("passed") else 0.35
    risk = 0.05 if candidate["candidate_kind"] != "source_adapter_class" else 0.08
    score = round(max(0.0, gain - cost - risk), 3)
    return {
        "gain": round(gain, 3),
        "cost": round(cost, 3),
        "risk": round(risk, 3),
        "reversibility": "high",
        "score": score,
        "verdict": "promote" if probe.get("passed") else "archive",
    }


def write_db_rows(conn: psycopg.Connection, *, run_id: str, text: str, artifact: Path, board: BoardState, candidate: dict[str, Any], probe: dict[str, Any], score: dict[str, Any], receipt_path: str) -> dict[str, str]:
    with conn.cursor(row_factory=dict_row) as cur:
        candidate_kind = candidate["candidate_kind"]
        event_id = sha256_text(json.dumps({"run_id": run_id, "artifact": rel(artifact), "text": text, "candidate_kind": candidate_kind}, sort_keys=True))
        raw_ref = f"inline://luci-learning/{sha256_text(text)[:16]}/{sha256_text(rel(artifact))[:16]}/{candidate_kind}/{run_id}"
        raw_row = cur.execute(
            """
            INSERT INTO lucidota_control.raw_artifact(raw_ref, raw_sha256, hash_algo, source, actor, byte_count, char_count, mime_type, storage_hint, detail)
            VALUES (%s, %s, 'sha256', 'luci_learning_slice', 'operator', %s, %s, 'application/json', 'receipt_or_artifact', %s::jsonb)
            ON CONFLICT (raw_ref) DO UPDATE SET
              raw_sha256 = EXCLUDED.raw_sha256,
              hash_algo = EXCLUDED.hash_algo,
              source = EXCLUDED.source,
              actor = EXCLUDED.actor,
              byte_count = EXCLUDED.byte_count,
              char_count = EXCLUDED.char_count,
              mime_type = EXCLUDED.mime_type,
              storage_hint = EXCLUDED.storage_hint,
              detail = EXCLUDED.detail
            RETURNING raw_artifact_uuid::text
            """,
            (
                raw_ref,
                sha256_text(text + "\n" + rel(artifact)),
                len(text.encode("utf-8", errors="replace")),
                len(text),
                json.dumps({"artifact": rel(artifact), "board": asdict(board), "candidate": candidate, "probe": probe, "score": score}),
            ),
        ).fetchone()
        raw_artifact_uuid = raw_row["raw_artifact_uuid"] if isinstance(raw_row, dict) else raw_row[0]
        cur.execute(
            """
            INSERT INTO lucidota_control.event_envelope(event_id, ts, source, actor, raw_ref, raw_artifact_uuid, verbatim_hash, hash_algo, text, entities, claims, actions_requested, artifacts_referenced, risk_flags, route_candidates, board_features, embedding_ref, detail)
            VALUES (%s, now(), 'luci_learning_slice', 'operator', %s, %s::uuid, %s, 'sha256', %s, '[]'::jsonb, '[]'::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, NULL, %s::jsonb)
            ON CONFLICT (event_id) DO UPDATE SET
              ts = EXCLUDED.ts,
              source = EXCLUDED.source,
              actor = EXCLUDED.actor,
              raw_ref = EXCLUDED.raw_ref,
              raw_artifact_uuid = EXCLUDED.raw_artifact_uuid,
              verbatim_hash = EXCLUDED.verbatim_hash,
              hash_algo = EXCLUDED.hash_algo,
              text = EXCLUDED.text,
              actions_requested = EXCLUDED.actions_requested,
              artifacts_referenced = EXCLUDED.artifacts_referenced,
              risk_flags = EXCLUDED.risk_flags,
              route_candidates = EXCLUDED.route_candidates,
              board_features = EXCLUDED.board_features,
              detail = EXCLUDED.detail
            """,
            (
                event_id,
                raw_ref,
                raw_artifact_uuid,
                sha256_text(text),
                text,
                json.dumps(candidate["feature_hypothesis"]),
                json.dumps([rel(artifact)]),
                json.dumps([f"probe_passed={probe.get('passed')}", candidate["candidate_kind"]]),
                json.dumps([candidate["candidate_kind"], "treelite_router_feature"]),
                json.dumps(asdict(board)),
                json.dumps({"artifact": rel(artifact), "probe": probe.get("command"), "score": score}),
            ),
        )
        work_order_row = cur.execute(
            """
            INSERT INTO lucidota_control.work_order(event_id, lane, work_kind, status, payload, idempotency_key)
            VALUES (%s, 'audit', %s, %s, %s::jsonb, %s)
            ON CONFLICT (idempotency_key) DO UPDATE SET
              event_id = EXCLUDED.event_id,
              lane = EXCLUDED.lane,
              work_kind = EXCLUDED.work_kind,
              status = EXCLUDED.status,
              payload = EXCLUDED.payload,
              updated_at = now()
            RETURNING work_order_uuid::text
            """,
            (
                event_id,
                "luci_learning_slice",
                "succeeded" if probe.get("passed") else "failed",
                json.dumps({"artifact": rel(artifact), "board": asdict(board), "candidate": candidate, "probe": probe, "score": score}),
                f"luci-learning:{run_id}:{candidate_kind}:{sha256_text(text)[:16]}:{sha256_text(rel(artifact))[:16]}",
            ),
        ).fetchone()
        work_order_uuid = work_order_row["work_order_uuid"] if isinstance(work_order_row, dict) else work_order_row[0]
        receipt_row = cur.execute(
            """
            SELECT work_receipt_uuid::text
            FROM lucidota_control.work_receipt
            WHERE work_order_uuid = %s::uuid AND receipt_path = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (work_order_uuid, receipt_path),
        ).fetchone()
        if receipt_row:
            work_receipt_uuid = receipt_row["work_receipt_uuid"] if isinstance(receipt_row, dict) else receipt_row[0]
        else:
            receipt_row = cur.execute(
                """
                INSERT INTO lucidota_control.work_receipt(event_id, work_order_uuid, receipt_path, receipt_sha256, verdict, cost, gain, artifact_refs, canonical_graph_writes_performed, graph_write_mode, detail)
                VALUES (%s, %s::uuid, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, false, 'staged_only', %s::jsonb)
                RETURNING work_receipt_uuid::text
                """,
                (
                    event_id,
                    work_order_uuid,
                    receipt_path,
                    sha256_text(json.dumps({"run_id": run_id, "artifact": rel(artifact), "candidate": candidate, "score": score}, sort_keys=True)),
                    score["verdict"],
                    json.dumps({"cost": score["cost"], "risk": score["risk"]}),
                    json.dumps({"gain": score["gain"], "score": score["score"]}),
                    json.dumps([raw_ref, rel(artifact)]),
                    json.dumps({
                        "artifact": rel(artifact),
                        "candidate": candidate,
                        "probe": probe,
                        "board_state": asdict(board),
                    }),
                ),
            ).fetchone()
            work_receipt_uuid = receipt_row["work_receipt_uuid"] if isinstance(receipt_row, dict) else receipt_row[0]
        conn.commit()
    return {
        "event_id": event_id,
        "raw_artifact_uuid": raw_artifact_uuid,
        "work_order_uuid": work_order_uuid,
        "work_receipt_uuid": work_receipt_uuid,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    text = args.text or ""
    artifact = Path(args.artifact or DEFAULT_ARTIFACT)
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if not artifact.exists():
        raise FileNotFoundError(f"artifact not found: {artifact}")
    candidate_kind_arg = getattr(args, "candidate_kind", None)
    run_id = args.run_id or "luci-learning:" + sha256_text(json.dumps({"text": text, "artifact": rel(artifact), "candidate_kind": candidate_kind_arg}, sort_keys=True))[:24]
    board = map_board_state(text, artifact)
    candidate = classify_candidate(artifact, text, candidate_kind_arg)
    probe = probe_candidate(artifact, text, candidate["candidate_kind"])
    score = score_attempt(board, candidate, probe)
    learning_cmd = probe.get("command") or ""
    receipt_dir = Path(args.receipt_dir or OUT)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_key = sha256_text(
        json.dumps(
            {
                "run_id": run_id,
                "text": text,
                "artifact": rel(artifact),
                "candidate_kind": candidate["candidate_kind"],
            },
            sort_keys=True,
        )
    )[:24]
    receipt_path = receipt_dir / f"luci_learning_{receipt_key}.json"
    receipt = {
        "schema": "lucidota.luci.learning_slice.receipt.v1",
        "generated_at": now(),
        "run_id": run_id,
        "artifact": rel(artifact),
        "board_state": asdict(board),
        "candidate": candidate,
        "probe": probe,
        "score": score,
        "promotion_decision": score["verdict"],
        "receipt_path": rel(receipt_path),
        "status": "PASS" if probe.get("passed") else "DEGRADED",
        "learning_loop": {
            "slice": "luci_learning_slice",
            "board_state": asdict(board),
            "candidate": candidate,
            "probe": probe,
            "score": score,
            "promotion_decision": score["verdict"],
            "receipt_path": rel(receipt_path),
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    db_ids = {}
    with psycopg.connect(db_url(args.database_url), row_factory=dict_row) as conn:
        db_ids = write_db_rows(conn, run_id=run_id, text=text, artifact=artifact, board=board, candidate=candidate, probe=probe, score=score, receipt_path=receipt["receipt_path"])
    receipt["db_write"] = db_ids
    receipt["learning_loop"]["work_order_id"] = db_ids["work_order_uuid"]
    receipt["learning_loop"]["work_receipt_id"] = db_ids["work_receipt_uuid"]
    receipt["learning_loop"]["raw_artifact_id"] = db_ids["raw_artifact_uuid"]
    receipt["visible_response"] = {
        "summary": f"Indy_READs: studied {artifact.name}, extracted {candidate['candidate_kind']}, ran a Treelite-backed decision-points probe, and wrote the ledger.",
        "work_order_id": db_ids["work_order_uuid"],
        "work_receipt_id": db_ids["work_receipt_uuid"],
        "attempt_id": db_ids["work_order_uuid"],
        "raw_artifact_id": db_ids["raw_artifact_uuid"],
        "artifact": rel(artifact),
        "probe": learning_cmd,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LUCI learning slice / board-state trial harness.")
    ap.add_argument("--text", default="")
    ap.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    ap.add_argument("--candidate-kind", choices=["source", "delegate", "model", "algorithm", "archive", "archive-class", "ingestion"])
    ap.add_argument("--database-url")
    ap.add_argument("--run-id")
    ap.add_argument("--receipt-dir", default=str(OUT))
    ap.add_argument("--json", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    receipt = run(args)
    if args.json:
        print(json.dumps(receipt, sort_keys=True, default=str))
    else:
        print("LUCI_LEARNING=PASS" if receipt["status"] == "PASS" else "LUCI_LEARNING=DEGRADED")
        print("WORK_ORDER_ID=" + receipt["visible_response"]["work_order_id"])
        print("ATTEMPT_ID=" + receipt["visible_response"]["attempt_id"])
        print("ARTIFACT=" + receipt["visible_response"]["artifact"])
        print("PROMOTION_DECISION=" + receipt["promotion_decision"])
        print("RECEIPT_PATH=" + receipt["receipt_path"])
    return 0 if receipt["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
