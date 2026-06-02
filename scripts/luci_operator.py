#!/usr/bin/env python3
"""LUCI front-door operator command: ontology route -> MOA route -> DB event."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import claw_moa_router, language_router  # noqa: E402
from scripts.luci_response_composer import compose_response  # noqa: E402
from scripts.spine_job_adapter import ABSURDJobAdapter  # noqa: E402
from scripts.spine_common import now, rel, sha256_json  # noqa: E402

OUT = ROOT / "05_OUTPUTS" / "luci"
INGRESS_CACHE = ROOT / "04_RUNTIME" / "luci" / "operator_ingress.jsonl"
ATTEMPT_ENGINE_ROOT = ROOT / "09_STORAGE" / "luci" / "attempt_engine"
SCHEMA = "lucidota.luci.operator_frontdoor.v1"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def read_text(text: str | None, text_file: str | None) -> str:
    if text_file:
        path = Path(text_file)
        if not path.is_absolute():
            path = ROOT / path
        return path.read_text(encoding="utf-8", errors="replace")
    return text or ""


def emit_workflow_event(database_url: str, detail: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    try:
        import psycopg  # type: ignore

        with psycopg.connect(database_url, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                RETURNING event_id
                """,
                (
                    "luci_operator_frontdoor",
                    run_id,
                    "route_complete",
                    "succeeded",
                    "luci",
                    json.dumps(detail, sort_keys=True),
                ),
            )
            row = cur.fetchone()
            return {"performed": True, "event_id": str(row[0]), "database_url": "redacted"}
    except Exception as exc:
        return {"performed": False, "error": type(exc).__name__, "detail": str(exc)[:240]}


def write_receipt(payload: dict[str, Any]) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"luci_operator_{stamp()}.json"
    payload["receipt_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload["receipt_path"]


def append_ingress_cache(entry: dict[str, Any]) -> str:
    INGRESS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with INGRESS_CACHE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    return rel(INGRESS_CACHE)


def run_attempt_engine(text: str, *, run_id: str, database_url: str, queue_name: str = "luci_operator") -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "luci_attempt_engine.py"),
        "--database-url",
        database_url,
        "--queue-name",
        queue_name,
        "--synthetic",
        "--text",
        text,
        "--run-id",
        run_id,
        "--receipt-dir",
        str(ROOT / "05_OUTPUTS" / "luci_attempt_engine"),
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    payload: dict[str, Any] = {"returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:], "command": " ".join(cmd)}
    if stdout_lines:
        try:
            payload.update(json.loads(stdout_lines[0]))
        except json.JSONDecodeError:
            payload["stdout_json_parse"] = "failed"
    payload["receipt_path"] = payload.get("receipt_path") or ""
    payload["passed"] = proc.returncode == 0
    return payload


def is_learning_prompt(text: str) -> bool:
    low = text.lower()
    return any(
        token in low
        for token in (
            "study ",
            "study one",
            "extract one reusable improvement",
            "reusable improvement",
            "algorithm",
            "model candidate",
            "delegate candidate",
            "source candidate",
            "current-world candidate",
            "provider candidate",
            "source or internal artifact",
            "board state",
        )
    )


def is_source_prompt(text: str) -> bool:
    low = text.lower()
    return any(
        token in low
        for token in (
            "live world",
            "current world",
            "hacker news",
            "arxiv",
            "reddit",
            "github trending",
            "github releases",
            "github issues",
            "source adapter",
        )
    )


def infer_learning_candidate_kind(text: str) -> str | None:
    low = text.lower()
    if "archive candidate" in low or "archive-class" in low or "krampus archive" in low or "ingestion candidate" in low:
        return "archive"
    if "delegate candidate" in low or "groq" in low or "vibes" in low or "provider candidate" in low:
        return "delegate"
    if "model candidate" in low or "model runtime" in low or "admission" in low:
        return "model"
    if "source candidate" in low or "current-world candidate" in low or "live source" in low or "current world" in low:
        return "source"
    if "algorithm" in low or "treelite" in low or "xgboost" in low:
        return "algorithm"
    return None


def run_source_slice(text: str, *, run_id: str, database_url: str, source: str = "auto") -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "luci_source_slice.py"),
        "--database-url",
        database_url,
        "--text",
        text,
        "--source",
        source,
        "--run-id",
        run_id,
        "--receipt-dir",
        str(ROOT / "05_OUTPUTS" / "luci_source"),
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    payload: dict[str, Any] = {"returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:], "command": " ".join(cmd)}
    if stdout_lines:
        try:
            payload.update(json.loads(stdout_lines[0]))
        except json.JSONDecodeError:
            payload["stdout_json_parse"] = "failed"
    payload["receipt_path"] = payload.get("receipt_path") or ""
    payload["passed"] = proc.returncode == 0
    return payload


def run_learning_slice(text: str, *, run_id: str, database_url: str, artifact: str | None = None, candidate_kind: str | None = None) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "luci_learning_slice.py"),
        "--database-url",
        database_url,
        "--text",
        text,
        "--run-id",
        run_id,
        "--receipt-dir",
        str(ROOT / "05_OUTPUTS" / "luci_learning"),
        "--json",
    ]
    if candidate_kind:
        cmd.extend(["--candidate-kind", candidate_kind])
    if artifact:
        cmd.extend(["--artifact", artifact])
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    payload: dict[str, Any] = {"returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:], "command": " ".join(cmd)}
    if stdout_lines:
        try:
            payload.update(json.loads(stdout_lines[0]))
        except json.JSONDecodeError:
            payload["stdout_json_parse"] = "failed"
    payload["receipt_path"] = payload.get("receipt_path") or ""
    payload["passed"] = proc.returncode == 0
    return payload


def visible_response_field(task: dict[str, Any], key: str) -> str:
    visible = task.get("visible_response")
    if isinstance(visible, dict):
        return str(visible.get(key) or "")
    return ""


def create_attempt_task(
    *,
    text: str,
    run_id: str,
    language: dict[str, Any],
    moa: dict[str, Any],
) -> dict[str, Any]:
    adapter = ABSURDJobAdapter(ATTEMPT_ENGINE_ROOT)
    idempotency_key = sha256_json(
        {
            "schema": "lucidota.luci.attempt_task.v1",
            "run_id": run_id,
            "text_sha256": sha256_json({"text": text}),
            "lane": moa["input_route"]["lane"],
        }
    )
    payload = {
        "schema": "lucidota.luci.attempt_task.v1",
        "run_id": run_id,
        "source": "luci_frontdoor",
        "text_sha256": sha256_json({"text": text}),
        "text_preview": text[:160],
        "text_verbatim": text,
        "intent": language.get("intent"),
        "ontology_terms": language.get("ontology_terms", []),
        "lane": moa["input_route"]["lane"],
        "route_reason": moa["input_route"].get("route_reason", []),
        "moa_receipt": moa.get("receipt_path"),
        "promptflow_role": "sidecar_only_not_live_gate",
    }
    job = adapter.create_job(
        lane=f"luci.{moa['input_route']['lane'].lower()}",
        payload=payload,
        idempotency_key=idempotency_key,
    )
    if job["state"] == "CREATED":
        job = adapter.transition(job["job_id"], "QUEUED")
    receipt_path = rel(adapter.jobs_path)
    return {
        "schema": "lucidota.luci.attempt_engine_task.v1",
        "job_id": job["job_id"],
        "state": job["state"],
        "attempt_count": job["attempt_count"],
        "max_attempts": job["max_attempts"],
        "receipt_path": receipt_path,
        "adapter_root": rel(adapter.root),
    }


def operate(
    text: str,
    *,
    database_url: str,
    run_id: str | None = None,
    execute_groq: bool = False,
    enqueue_slow: bool = True,
    json_out: bool = False,
) -> dict[str, Any]:
    run_id = run_id or "luci:" + sha256_json({"text": text, "at": now()})[:24]
    language = language_router.write(language_router.route_text(text, channel="operator", verbosity="brief"))
    moa = claw_moa_router.orchestrate_text(
        text,
        metadata={
            "luci_frontdoor": True,
            "language_router_report": language.get("report_path"),
            "intent": language.get("intent"),
            "ontology_terms": language.get("ontology_terms", []),
        },
        cache_key="luci_operator",
        execute_groq=execute_groq,
        enqueue_chain=enqueue_slow and language["lane"]["lane"] == "SLOWLANE",
        absurd_dir=ROOT / "09_STORAGE" / "absurd" / "luci_operator",
        database_url=database_url,
        db_queue_name="luci_operator",
        emit_receipt=not json_out,
    )
    learning_mode = is_learning_prompt(text)
    source_mode = is_source_prompt(text)
    if source_mode:
        source_task = run_source_slice(text, run_id=run_id, database_url=database_url)
        attempt_task = source_task
    elif learning_mode:
        artifact = str((ROOT / "scripts" / "dev_journey_decision_points.py").relative_to(ROOT))
        candidate_kind = infer_learning_candidate_kind(text)
        attempt_task = run_learning_slice(text, run_id=run_id, database_url=database_url, artifact=artifact, candidate_kind=candidate_kind)
    else:
        attempt_task = run_attempt_engine(text, run_id=run_id, database_url=database_url)
    learning_loop = None
    if learning_mode:
        learning_loop = {
            "slice": "luci_learning_slice",
            "board_state": attempt_task.get("board_state"),
            "candidate": attempt_task.get("candidate"),
            "probe": attempt_task.get("probe"),
            "score": attempt_task.get("score"),
            "promotion_decision": attempt_task.get("promotion_decision"),
            "receipt_path": attempt_task.get("receipt_path"),
        }
    detail = {
        "run_id": run_id,
        "text_sha256": sha256_json({"text": text}),
        "text_verbatim": text,
        "language_router_report": language.get("report_path"),
        "moa_receipt": moa.get("receipt_path"),
        "intent": language.get("intent"),
        "ontology_terms": language.get("ontology_terms", []),
        "lane": moa["input_route"]["lane"],
        "model_calls_performed": moa.get("model_calls_performed", False),
        "model_lane_status": (
            moa.get("model_synthesis")
            or {"performed": False, "reason": "no model requested; deterministic algorithms/workflow lanes executed"}
        ),
        "algorithm_lanes": ["language_router", "fast_slow_lane_gate", "language_membrane", "fairyfuse_smoothing"],
        "task_chain": moa.get("task_chain", {}),
        "route_targets": moa.get("route_targets", []),
        "provider_lanes": moa.get("lane_plan", {}).get("provider_lanes", {}),
        "local_model_admission": moa.get("lane_plan", {}).get("local_model_admission", {}),
        "attempt_engine_task": attempt_task,
        "learning_mode": learning_mode,
        "learning_loop": learning_loop,
    }
    db_event = emit_workflow_event(database_url, detail, run_id=run_id)
    ingress_cache_path = append_ingress_cache(
        {
            "schema": "lucidota.luci.operator_ingress.v1",
            "generated_at": now(),
            "run_id": run_id,
            "text_sha256": detail["text_sha256"],
            "text_verbatim": text,
            "intent": language.get("intent"),
            "lane": moa["input_route"]["lane"],
            "route_reason": moa["input_route"].get("route_reason", []),
            "ontology_terms": language.get("ontology_terms", []),
            "language_router_report": language.get("report_path"),
            "moa_receipt": moa.get("receipt_path"),
            "attempt_engine_task": attempt_task,
        }
    )
    work_order_id = (
        (attempt_task.get("visible_response", {}) if isinstance(attempt_task.get("visible_response"), dict) else {}).get("work_order_id")
        or attempt_task.get("db_write", {}).get("work_order_uuid")
        or ""
    )
    work_receipt_id = (
        (attempt_task.get("visible_response", {}) if isinstance(attempt_task.get("visible_response"), dict) else {}).get("work_receipt_id")
        or attempt_task.get("db_write", {}).get("work_receipt_uuid")
        or ""
    )
    attempt_id = (
        (attempt_task.get("visible_response", {}) if isinstance(attempt_task.get("visible_response"), dict) else {}).get("attempt_id")
        or attempt_task.get("db_write", {}).get("work_order_uuid")
        or attempt_task.get("db_write", {}).get("work_receipt_uuid")
        or ""
    )
    composition = compose_response(
        {
            "text": text,
            "intent": language.get("intent"),
            "lane": moa["input_route"]["lane"],
            "operator": os.environ.get("USER", "operator"),
            "text_chars": len(text),
            "word_count": len([w for w in text.split() if w]),
            "ontology_terms": language.get("ontology_terms", []),
            "language_rendered": language.get("rendered"),
            "provider_lanes": detail["provider_lanes"],
            "work_order_id": work_order_id,
            "attempt_id": attempt_id,
            "work_receipt_id": work_receipt_id,
            "database_url": database_url,
            "artifact": (visible_response_field(attempt_task, "artifact") or attempt_task.get("artifact") or ""),
            "receipt_path": attempt_task.get("receipt_path") or "",
            "next_hint": "receipt written; slow work queued" if detail["lane"] == "SLOWLANE" else "fast route completed",
        }
    )
    payload = {
        "schema": SCHEMA,
        "generated_at": now(),
        "run_id": run_id,
        "input": {
            "text_sha256": detail["text_sha256"],
            "text_chars": len(text),
            "preview": text[:160],
            "text": text,
            "ingress_cache_path": ingress_cache_path,
        },
        "ontology_packet": {
            "report_path": language.get("report_path"),
            "intent": language.get("intent"),
            "terms": language.get("ontology_terms", []),
            "work_order_id": language.get("work_order", {}).get("work_order_id"),
        },
        "routing": {
            "lane": moa["input_route"]["lane"],
            "reason": moa["input_route"].get("route_reason", []),
            "moa_receipt": moa.get("receipt_path"),
        },
        "attempt_engine": attempt_task,
        "learning_slice": attempt_task if learning_mode else None,
        "learning_loop": learning_loop,
        "source_slice": attempt_task if source_mode else None,
        "routing_fabric": {
            "route_targets": detail["route_targets"],
            "provider_lanes": detail["provider_lanes"],
            "local_model_admission": detail["local_model_admission"],
        },
        "workflow": {
            "async_capable": True,
            "chain_enqueue": moa.get("task_chain", {}).get("enqueue", {}),
        },
        "model_or_algorithm_lane": detail["model_lane_status"],
        "algorithm_lanes_invoked": detail["algorithm_lanes"],
        "postgres_workflow_event": db_event,
        "promptflow_role": "sidecar_only_not_live_gate",
        "composition": composition,
        "visible_response": composition["visible_response"],
        "verdict": "PASS" if moa.get("verdict") == "PASS" and db_event.get("performed") and attempt_task.get("passed") else "DEGRADED",
        "blockers": [] if db_event.get("performed") and attempt_task.get("passed") else [b for b, ok in [("postgres_workflow_event_not_written", db_event.get("performed")), ("attempt_engine_failed", attempt_task.get("passed"))] if not ok],
        "model_calls_performed": moa.get("model_calls_performed", False),
        "network_calls_performed": moa.get("network_calls_performed", False),
        "canonical_graph_writes_performed": False,
    }
    write_receipt(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LUCI operator front door: route command, write receipt, log workflow event.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--text-file")
    parser.add_argument("--database-url", default=os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL", "postgresql:///lucidota_state"))
    parser.add_argument("--run-id")
    parser.add_argument("--execute-groq", action="store_true")
    parser.add_argument("--no-enqueue-slow", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = operate(
        read_text(args.text, args.text_file),
        database_url=args.database_url,
        run_id=args.run_id,
        execute_groq=args.execute_groq,
        enqueue_slow=not args.no_enqueue_slow,
        json_out=args.json,
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    if not args.json:
        print("LUCI=" + payload["verdict"])
        print("INTENT=" + payload["ontology_packet"]["intent"])
        print("LANE=" + payload["routing"]["lane"])
        print(payload["visible_response"]["summary"])
        print("NEXT=" + payload["visible_response"]["next"])
        event_id = payload["postgres_workflow_event"].get("event_id") or ""
        print("WORKFLOW_EVENT_ID=" + event_id)
        print("REPORT_PATH=" + payload["receipt_path"])
        if payload.get("source_slice"):
            task_key = "source_slice"
        elif payload.get("learning_slice"):
            task_key = "learning_slice"
        else:
            task_key = "attempt_engine"
        task = payload.get(task_key) or payload["attempt_engine"]
        task_id = (
            task.get("job_id")
            or task.get("job_uuid")
            or task.get("work_order_uuid")
            or task.get("db_write", {}).get("work_order_uuid")
            or task.get("db_write", {}).get("raw_artifact_uuid")
            or ""
        )
        print("ATTEMPT_ENGINE_JOB_ID=" + task_id)
        if payload.get("source_slice"):
            print("SOURCE=PASS")
            print("WORK_ORDER_ID=" + (task.get("db_write", {}).get("work_order_uuid") or visible_response_field(task, "work_order_id") or ""))
            print("WORK_RECEIPT_ID=" + (task.get("db_write", {}).get("work_receipt_uuid") or visible_response_field(task, "work_receipt_id") or ""))
            print("ATTEMPT_ID=" + (task.get("db_write", {}).get("work_order_uuid") or visible_response_field(task, "attempt_id") or ""))
            print("ARTIFACT=" + (visible_response_field(task, "artifact") or task.get("artifact") or ""))
            print("PROMOTION_DECISION=" + (task.get("promotion_decision") or task.get("score", {}).get("verdict", "")))
            print("SOURCE_RECEIPT_PATH=" + task["receipt_path"])
        if payload.get("learning_slice"):
            print("LEARNING=PASS")
            print("WORK_ORDER_ID=" + (task.get("db_write", {}).get("work_order_uuid") or visible_response_field(task, "work_order_id") or ""))
            print("WORK_RECEIPT_ID=" + (task.get("db_write", {}).get("work_receipt_uuid") or visible_response_field(task, "work_receipt_id") or ""))
            print("ATTEMPT_ID=" + (task.get("db_write", {}).get("work_order_uuid") or visible_response_field(task, "attempt_id") or ""))
            print("ARTIFACT=" + (visible_response_field(task, "artifact") or task.get("artifact") or ""))
            print("PROMOTION_DECISION=" + (task.get("promotion_decision") or task.get("score", {}).get("verdict", "")))
            print("LEARNING_RECEIPT_PATH=" + task["receipt_path"])
        else:
            print("ATTEMPT_ENGINE_RECEIPT_PATH=" + payload["attempt_engine"]["receipt_path"])
    return 0 if payload["verdict"] in {"PASS", "DEGRADED"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
