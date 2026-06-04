#!/usr/bin/env python3
"""Plan a DB-backed recursive fanout runner using existing preflight/packet helpers."""
from __future__ import annotations

import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

from scripts import codex_context_preflight as preflight
from scripts import goal_agent_packet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "goals"
PY = sys.executable
TARGET_FILES = ["scripts/agent_fanout_orchestrator.py", "tests/test_agent_fanout_orchestrator.py"]
REQUIRED_KEYS = ["status", "result", "next_action", "receipt_path", "evidence_refs", "decision_pairs"]
LANES = [
    ("api_truth", "freeze DB/API task truth before any fanout"),
    ("mini_spawn", "spawn lane mini-orchestrators with exact worker families"),
    ("worker_packets", "reuse goal_agent_packet and goal_swarm_dispatch surfaces"),
    ("json_contract", "emit one exact structured JSON result"),
    ("worker_rejection", "reject commentary-only worker returns"),
    ("db_receipt", "record DB-backed orchestrator receipt or block cleanly"),
]


def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def rel(path: Path | str) -> str:
    try: return str(Path(path).resolve().relative_to(ROOT))
    except Exception: return str(path)

def stable_json(obj: Any) -> str: return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
def sha256_text(text: str) -> str: return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
def db_url() -> str: return os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL") or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def read_current_task_truth() -> dict[str, Any]:
    openapi_status, openapi, openapi_error = preflight.fetch_json("", None)
    openapi_paths = openapi.get("paths", {}) if isinstance(openapi, dict) else {}
    active_route = preflight.route_status("active_goal", openapi_paths)
    receipts_route = preflight.route_status("flow_receipts", openapi_paths)
    active_status, active_body, active_error = preflight.fetch_json("active_goal", {"limit": "1"})
    current_task = active_body[0] if active_status == 200 and isinstance(active_body, list) and active_body else {
        "title": "normalized recursive fanout runner lane",
        "source": "operator_supplied_fallback; /active_goal unavailable_or_empty",
        "active_goal_route_status": active_route.get("http_status"),
    }
    blockers = []
    if openapi_status != 200: blockers.append("db_api_truth_unavailable")
    if active_route.get("http_status") != 200: blockers.append("active_goal_route_unavailable")
    return {
        "schema": "lucidota.agent_fanout_preflight.v1",
        "postgrest_base_url": preflight.BASE_URL,
        "openapi_status": openapi_status,
        "openapi_error": openapi_error,
        "active_goal_status": active_status,
        "active_goal_error": active_error,
        "current_task": current_task,
        "route_findings": [active_route, receipts_route],
        "blockers": blockers,
    }


def worker_return_check(raw: str | dict[str, Any]) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else None
    if data is None:
        try: data = json.loads(raw)
        except Exception: return {"accepted": False, "reason": "commentary_only_worker_return", "detail": "non_json_text"}
    if not isinstance(data, dict): return {"accepted": False, "reason": "commentary_only_worker_return", "detail": "non_object_json"}
    if any(k not in data for k in REQUIRED_KEYS): return {"accepted": False, "reason": "commentary_only_worker_return", "detail": "missing_required_keys"}
    if not isinstance(data.get("result"), dict): return {"accepted": False, "reason": "commentary_only_worker_return", "detail": "result_must_be_object"}
    if not isinstance(data.get("decision_pairs"), list) or len(data["decision_pairs"]) < 2: return {"accepted": False, "reason": "commentary_only_worker_return", "detail": "decision_pairs_too_small"}
    if not data.get("receipt_path") and not data.get("evidence_refs"): return {"accepted": False, "reason": "commentary_only_worker_return", "detail": "missing_receipt_and_evidence"}
    return {"accepted": True, "reason": "accepted", "detail": "structured_worker_envelope"}


def assess_worker_returns(worker_returns: list[str | dict[str, Any]] | None) -> dict[str, Any]:
    accepted, rejected = [], []
    for idx, item in enumerate(worker_returns or []):
        check = worker_return_check(item)
        bucket = accepted if check["accepted"] else rejected
        bucket.append({"worker_index": idx, **check})
    return {"accepted": accepted, "rejected": rejected}


def build_worker(*, lane_id: str, slot: int, family: str, kind: str, task: str) -> dict[str, Any]:
    packet = goal_agent_packet.build_packet(target=family, task=task, files=TARGET_FILES, complexity="standard" if family == "vibe" else "integration", checks=["run focused tests", "return JSON only", "name receipt or blocker path"])
    cmd = [PY, "scripts/goal_swarm_dispatch.py", "--target", family, "--task", task, "--file", TARGET_FILES[0], "--file", TARGET_FILES[1], "--complexity", "standard", "--jobs", "1", "--json"]
    if family == "groq": cmd += ["--command", PY, "scripts/groq_goal_delegate.py", "--task", task, "--kind", kind, "--json"]
    return {"worker_id": f"{lane_id}:{family}-{slot}", "family": family, "kind": kind, "task": task, "packet": packet, "dispatch_cmd": cmd}


def build_mini_orchestrator(current_task: dict[str, Any], lane_id: str, focus: str) -> dict[str, Any]:
    title = current_task.get("title") or "current task"
    tasks = [
        ("vibe", "code", f"{title}: {focus}; reuse existing preflight/dispatch helpers before adding code."),
        ("vibe", "code", f"{title}: {focus}; implement the smallest runner slice in scripts/agent_fanout_orchestrator.py."),
        ("vibe", "code", f"{title}: {focus}; harden the exact top-level JSON contract and worker payload shape."),
        ("vibe", "code", f"{title}: {focus}; add only the minimal focused pytest coverage in tests/test_agent_fanout_orchestrator.py."),
        ("groq", "review", f"Review {focus} for receipt gaps, commentary-only loopholes, and missing blockers."),
        ("groq", "review", f"Adversarially review {focus} for overbuild; keep only the smallest runner/orchestration bundle."),
    ]
    workers = [build_worker(lane_id=lane_id, slot=i + 1, family=family, kind=kind, task=task) for i, (family, kind, task) in enumerate(tasks)]
    return {
        "lane_id": lane_id,
        "lane_owner": "runner/orchestration surface",
        "focus": focus,
        "spawn_contract": {"worker_count": 6, "vibe_count": 4, "groq_count": 2, "selection_rule": "choose_best_minimal_bundle", "commentary_only_policy": "reject"},
        "workers": workers,
    }


def build_plan(current_task: dict[str, Any]) -> dict[str, Any]:
    minis = [build_mini_orchestrator(current_task, lane_id=f"mini-{i+1:02d}-{lane}", focus=focus) for i, (lane, focus) in enumerate(LANES)]
    return {
        "mini_orchestrators": minis,
        "mini_orchestrator_count": len(minis),
        "worker_count": sum(len(m["workers"]) for m in minis),
        "per_lane_worker_counts": [len(m["workers"]) for m in minis],
    }


def write_db_receipt(payload: dict[str, Any]) -> dict[str, str]:
    event_id = f"agent_fanout:{sha256_text(stable_json({k: payload[k] for k in ('schema','generated_at','status','report_path')}))[:32]}"
    raw_ref = f"inline://agent_fanout_orchestrator/{sha256_text(event_id)[:16]}"
    with psycopg.connect(db_url()) as conn, conn.cursor() as cur:
        raw = cur.execute(
            """
            INSERT INTO lucidota_control.raw_artifact(raw_ref, raw_sha256, hash_algo, source, actor, byte_count, char_count, mime_type, storage_hint, detail)
            VALUES (%s,%s,'sha256','agent_fanout_orchestrator','worker',%s,%s,'application/json','inline_or_receipt',%s::jsonb)
            ON CONFLICT (raw_ref) DO UPDATE SET detail = lucidota_control.raw_artifact.detail || EXCLUDED.detail
            RETURNING raw_artifact_uuid::text
            """,
            (raw_ref, sha256_text(stable_json(payload)), len(stable_json(payload).encode()), len(stable_json(payload)), json.dumps({"report_path": payload["report_path"]})),
        ).fetchone()
        raw_uuid = raw["raw_artifact_uuid"] if isinstance(raw, dict) else raw[0]
        cur.execute(
            """
            INSERT INTO lucidota_control.event_envelope(event_id, ts, source, actor, raw_ref, raw_artifact_uuid, verbatim_hash, hash_algo, text, entities, claims, actions_requested, artifacts_referenced, risk_flags, route_candidates, board_features, embedding_ref, detail)
            VALUES (%s, now(), 'agent_fanout_orchestrator', 'worker', %s, %s::uuid, %s, 'sha256', %s, '[]'::jsonb, '[]'::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, '[]'::jsonb, %s::jsonb, NULL, %s::jsonb)
            ON CONFLICT (event_id) DO UPDATE SET detail = lucidota_control.event_envelope.detail || EXCLUDED.detail
            """,
            (event_id, raw_ref, raw_uuid, sha256_text(payload["current_task_title"]), stable_json({"title": payload["current_task_title"], "status": payload["status"]}), json.dumps(["agent_fanout_orchestrator"]), json.dumps(TARGET_FILES), json.dumps(payload.get("blockers", [])), json.dumps({"mini_orchestrator_count": payload["mini_orchestrator_count"]}), json.dumps({"report_path": payload["report_path"]})),
        )
        work = cur.execute(
            """
            INSERT INTO lucidota_control.work_order(event_id, lane, work_kind, status, payload, idempotency_key)
            VALUES (%s, 'external', 'agent_fanout_orchestrator', %s, %s::jsonb, %s)
            ON CONFLICT (idempotency_key) DO UPDATE SET status = EXCLUDED.status, payload = EXCLUDED.payload, updated_at = now()
            RETURNING work_order_uuid::text
            """,
            (event_id, 'succeeded' if payload['status'] == 'ready' else 'blocked', json.dumps({"report_path": payload["report_path"], "mini_orchestrator_count": payload["mini_orchestrator_count"]}), event_id),
        ).fetchone()
        work_uuid = work["work_order_uuid"] if isinstance(work, dict) else work[0]
        receipt = cur.execute(
            """
            INSERT INTO lucidota_control.work_receipt(event_id, work_order_uuid, receipt_path, receipt_sha256, verdict, cost, gain, artifact_refs, canonical_graph_writes_performed, graph_write_mode, detail)
            VALUES (%s, %s::uuid, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, false, 'staged_only', %s::jsonb)
            RETURNING work_receipt_uuid::text
            """,
            (event_id, work_uuid, payload["report_path"], sha256_text(stable_json({"event_id": event_id, "report_path": payload["report_path"]})), 'promote' if payload['status'] == 'ready' else 'retry', json.dumps({"workers": payload["worker_count"]}), json.dumps({"accepted_worker_returns": len(payload["worker_return_checks"]["accepted"])}), json.dumps([raw_ref, *TARGET_FILES]), json.dumps({"blockers": payload.get("blockers", [])})),
        ).fetchone()
        conn.commit()
    return {"event_id": event_id, "raw_artifact_uuid": raw_uuid, "work_order_uuid": work_uuid, "work_receipt_uuid": receipt["work_receipt_uuid"] if isinstance(receipt, dict) else receipt[0]}


def run_orchestrator(*, worker_returns: list[str | dict[str, Any]] | None = None, receipt_path: Path | None = None) -> dict[str, Any]:
    truth = read_current_task_truth()
    checks = assess_worker_returns(worker_returns)
    plan = build_plan(truth["current_task"])
    payload = {
        "schema": "lucidota.agent_fanout_orchestrator.v1",
        "generated_at": now(),
        "lane_owner": "runner/orchestration surface",
        "status": "ready" if not truth["blockers"] and not checks["rejected"] else "blocked",
        "current_task_title": truth["current_task"].get("title", "normalized recursive fanout runner lane"),
        "current_task": truth["current_task"],
        "preflight": truth,
        "worker_return_contract": {"schema": "lucidota.worker_order.v1", "required_output": REQUIRED_KEYS, "commentary_only_policy": "reject", "result_must_be_object": True},
        "worker_return_checks": checks,
        "blockers": [*truth["blockers"], *(["commentary_only_worker_return"] if checks["rejected"] else [])],
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
        **plan,
    }
    out = receipt_path or (OUT / f"agent_fanout_orchestrator_{stamp()}.json")
    OUT.mkdir(parents=True, exist_ok=True)
    payload["report_path"] = rel(out)
    try:
        payload["db_receipt"] = write_db_receipt(payload)
    except Exception as exc:
        payload["status"] = "blocked"
        if "DB_BLOCKED" not in payload["blockers"]: payload["blockers"].append("DB_BLOCKED")
        payload["db_error"] = f"{type(exc).__name__}: {exc}"
    Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the recursive fanout runner plan and record a DB-backed receipt.")
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    payload = run_orchestrator(receipt_path=args.receipt)
    print("REPORT_PATH=" + payload["report_path"])
    print("AGENT_FANOUT_ORCHESTRATOR=" + ("PASS" if payload["status"] == "ready" else "BLOCKED"))
    if args.json: print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
