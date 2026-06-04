#!/usr/bin/env python3
"""Build a hierarchical agent fanout, emit packet receipts, and enqueue durable ABSURD jobs.

This is a conductor-side batching shim:
- top priority goes to RunPod/Talkie/LoRA and manual canon
- every major system gets a packet receipt
- every packet also becomes a durable queue row
- no graph/canonical writes
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "conductor_hierarchy"
QUEUE = "conductor_hierarchy"
DB = os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
PY = sys.executable


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


PM = "Mistral/Vibe PM"
BUILDER = "Spark builder"
BREAKER = "Spark breaker"
GROQ = [
    "Groq audit",
    "Groq extract",
    "Groq contradict",
    "Groq verify",
    "Groq reconcile",
    "Groq inventory",
    "Groq blocker-check",
]


SYSTEMS: list[dict[str, Any]] = [
    {
        "system_id": "runpod_talkie_lora_training",
        "title": "RunPod/Talkie/LoRA/Training",
        "priority": 1,
        "files": [
            "scripts/runpod_talkie_control.py",
            "GOALS/RUNPOD_TALKIE_LORA_INGEST_MASTER_PLAN.md",
            "05_OUTPUTS/runpod/talkie_book_lora/runpod_talkie_control_latest.json",
        ],
        "checks": [
            "remote SSH_OK",
            "remote custody receipt present",
            "bootstrap log advances",
        ],
        "next_command": "python3 scripts/runpod_talkie_control.py --force-after-auth-change probe --json",
        "subsystems": ["ssh auth gate", "remote custody", "talkie bootstrap", "lora train lane", "runpod upload"],
    },
    {
        "system_id": "manuals_api_canon",
        "title": "API/manual canon",
        "priority": 2,
        "files": [
            "GOALS/OPERATION_ROOT_ROTOR_SENDABLE_PROMPT.md",
            "GOALS/OPERATION_CANON_FORGE_VERBATIM.md",
            "GOALS/MULTI_MODEL_ALGORITHMIC_ROUTING_LEDGER.md",
            "scripts/root_rotor_manual_queue.py",
        ],
        "checks": [
            "manual receipts exist",
            "API canon matches workflow truth",
            "no markdown-only truth",
        ],
        "next_command": "python3 scripts/goal_agent_packet.py --target codex --task \"compile API/manual canon receipts for the conductor hierarchy\" --file GOALS/OPERATION_ROOT_ROTOR_SENDABLE_PROMPT.md --complexity architecture --json",
        "subsystems": ["sendable prompt", "canon forge", "manual queue", "operator-facing API"],
    },
    {
        "system_id": "model_fabric_scheduler",
        "title": "Model fabric scheduler",
        "priority": 3,
        "files": [
            "scripts/goal_model_fabric_control.py",
            "scripts/goal_model_fabric_orchestrate.py",
            "scripts/lucidota_model_governor.py",
            "scripts/lucidota_model_registry.py",
        ],
        "checks": [
            "model ledger queried",
            "governor decision recorded",
            "needles/router slots fit current budget",
        ],
        "next_command": ".venv/bin/python scripts/lucidota_model_governor.py --json",
        "subsystems": ["admit/open/call/release wrappers", "resident loadout registry", "governor decision row", "needles routing swarm"],
    },
    {
        "system_id": "postgres_control_plane",
        "title": "Postgres/control plane",
        "priority": 4,
        "files": [
            "06_SCHEMA/035_absurd_queue_spine.sql",
            "scripts/absurd_queue_spine.py",
            "scripts/absurd_consume_one.py",
            "scripts/boring_beast.py",
        ],
        "checks": [
            "queue registry visible",
            "job counts nonzero",
            "canonical graph unchanged",
        ],
        "next_command": ".venv/bin/python scripts/absurd_queue_spine.py --action audit --json",
        "subsystems": ["absurd_queue_job", "absurd_queue_event", "absurd_queue_dead_letter", "workflow_event bridge"],
    },
    {
        "system_id": "absurd_workflows",
        "title": "ABSURD workflows",
        "priority": 5,
        "files": [
            "scripts/conversation_command_accept_worker.py",
            "scripts/surface_instruction_compile_dry_run.py",
            "scripts/absurd_river_worker.py",
            "scripts/absurd_intake_worker.py",
        ],
        "checks": [
            "accept staged command",
            "queue one job",
            "worker returns receipt",
        ],
        "next_command": ".venv/bin/python scripts/absurd_queue_spine.py --action worker-once --queue control --execute",
        "subsystems": ["surface instruction fan-in", "conversation command accept", "river worker", "intake worker"],
    },
    {
        "system_id": "ingestion_graph_case_rebuild",
        "title": "Ingestion/graph/case rebuild",
        "priority": 6,
        "files": [
            "scripts/lucidota_indy_library_ingest.py",
            "scripts/graph_promotion_gate.py",
            "scripts/lucidota_sheet_workflow_smoke.py",
            "GOALS/EDGE_GRAIL_EXECUTION_QUEUE.md",
        ],
        "checks": [
            "sheet workflow dry-run passes",
            "graph promotion stays gated",
            "case rebuild rows preserved",
        ],
        "next_command": "python3 scripts/lucidota_sheet_workflow_smoke.py --json",
        "subsystems": ["sheet workflow spine", "graph promotion gate", "case timeline rebuild", "ingest labels"],
    },
    {
        "system_id": "indy_reads_desk",
        "title": "Indy_READs desk",
        "priority": 7,
        "files": [
            "scripts/indy_reads.py",
            "scripts/lucidota_indy_reads_watcher.py",
            "00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD/ROLE_MODES.json",
            "00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD/ARCHITECTURE.md",
        ],
        "checks": [
            "persona config loadable",
            "role modes are chainable",
            "page-locked reading remains bounded",
        ],
        "next_command": "python3 scripts/lucidota_indy_reads_watcher.py --help",
        "subsystems": ["role modes registry", "page-locked reader", "glow hunter", "watcher"],
    },
    {
        "system_id": "percyphon_identity_router",
        "title": "Percyphon/Doggystyle identity-router",
        "priority": 8,
        "files": [
            "scripts/lucidota_model_router.py",
            "scripts/lucidota_model_registry.py",
            "scripts/lucidota_model_governor.py",
            "GOALS/MULTI_MODEL_ALGORITHMIC_ROUTING_LEDGER.md",
        ],
        "checks": [
            "resident loadout rows print",
            "role fit matches ledger",
            "budget defer decision stays truthful",
        ],
        "next_command": ".venv/bin/python scripts/lucidota_model_registry.py",
        "subsystems": ["model registry", "governor decision", "resident loadout slots", "router policy ledger"],
    },
    {
        "system_id": "provider_rate_conductor",
        "title": "Provider-rate conductor",
        "priority": 9,
        "files": [
            "scripts/provider_rate_conductor.py",
            "scripts/provider_rate_conductor.sh",
            "GOALS/69.md",
        ],
        "checks": [
            "token buckets respected",
            "429 retry-after honored",
            "provider receipts written",
        ],
        "next_command": "python3 scripts/provider_rate_conductor.py",
        "subsystems": ["ABBA63 end-cycle hook", "provider buckets", "Groq audit fanout"],
    },
    {
        "system_id": "contradiction_red_team_hardening",
        "title": "Contradiction/red-team/hardening",
        "priority": 10,
        "files": [
            "scripts/recovery_matrix.py",
            "scripts/root_rotor_red_team_audit.py",
            "scripts/root_rotor_sidecar_anomaly_audit.py",
            "scripts/slop_audit_law.py",
        ],
        "checks": [
            "contradictions surfaced",
            "no fake PASS",
            "blockers become receipts",
        ],
        "next_command": "python3 scripts/recovery_matrix.py --json",
        "subsystems": ["recovery matrix", "sidecar anomaly audit", "slop audit law"],
    },
    {
        "system_id": "model_fabric_followup",
        "title": "Model fabric follow-up",
        "priority": 11,
        "files": [
            "scripts/goal_model_fabric_control.py",
            "scripts/goal_model_fabric_orchestrate.py",
            "scripts/lucidota_model_governor.py",
        ],
        "checks": [
            "status receipt current",
            "loadout decision truthful",
            "no duplicate heavy load",
        ],
        "next_command": ".venv/bin/python scripts/goal_model_fabric_control.py status --json",
        "subsystems": ["admission replay", "needles status", "loadout decision follow-up"],
    },
]


def goal_packet(system: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    task = (
        f"Top-level hierarchy fanout for {system['title']}. "
        f"Assign 1 PM ({PM}), 1 builder ({BUILDER}), 1 breaker ({BREAKER}), and 7 Groq workers ({', '.join(GROQ)}). "
        f"Do not narrate. Inspect, patch, test, receipt, and report blockers only. "
        f"Major subsystems: {', '.join(system['subsystems'])}. "
        f"System goal: {system['next_command']}. "
        f"Return changed files, commands run, tests, receipts, next command."
    )
    cmd = [PY, "scripts/goal_agent_packet.py", "--target", "codex", "--task", task, "--complexity", "architecture"]
    for f in system["files"]:
        cmd += ["--file", f]
    for chk in system["checks"]:
        cmd += ["--check", chk]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"goal_agent_packet_failed:{system['system_id']}:{proc.stderr[-1000:]}")
    report_path = next((line.split("=", 1)[1].strip() for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH=")), None)
    if not report_path:
        raise RuntimeError(f"missing_report_path:{system['system_id']}")
    pkt = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
    return report_path, pkt


def enqueue_job(conn: psycopg.Connection, system: dict[str, Any], report_path: str, packet_cmd: list[str]) -> dict[str, Any]:
    payload = {
        "handler": "external_command",
        "command": packet_cmd,
        "system_id": system["system_id"],
        "title": system["title"],
        "agent_assignments": {
            "pm": PM,
            "builder": BUILDER,
            "breaker": BREAKER,
            "groq_workers": GROQ,
        },
        "packet_report_path": report_path,
    }
    idem = sha({"queue": QUEUE, "system_id": system["system_id"], "command": packet_cmd})
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO lucidota_control.absurd_queue_job
              (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
            VALUES (%s, %s, 'external_command', %s, %s::jsonb, %s, 3, %s::jsonb)
            ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at = now()
            RETURNING job_uuid::text, (xmax = 0) AS inserted_new
            """,
            (QUEUE, f"hierarchy.{system['system_id']}", idem, json.dumps(payload), system["priority"], json.dumps({"source": "conductor_hierarchy_fanout"})),
        )
        row = cur.fetchone()
        job_uuid = row[0]
        inserted_new = bool(row[1])
        if inserted_new:
            cur.execute(
                "INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, detail) VALUES (%s,%s,'enqueued',%s::jsonb)",
                (job_uuid, QUEUE, json.dumps({"system_id": system["system_id"], "packet_report_path": report_path})),
            )
    return {"system_id": system["system_id"], "job_uuid": job_uuid, "inserted_new": inserted_new, "idempotency_key": idem}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, max_attempts, notes)
                VALUES (%s, 'Codex conductor hierarchy', 3, 'Top-level fanout queue: packet receipts first, no narration')
                ON CONFLICT (queue_name) DO UPDATE SET updated_at = now(), notes = EXCLUDED.notes
                """,
                (QUEUE,),
            )
        conn.commit()

    manifest = {
        "schema": "lucidota.conductor_hierarchy.fanout.v1",
        "generated_at": now_z(),
        "queue": QUEUE,
        "systems": [],
        "blockers": [],
    }
    packet_reports: list[str] = []
    queue_rows: list[dict[str, Any]] = []

    with psycopg.connect(DB) as conn:
        for system in SYSTEMS:
            try:
                report_path, pkt = goal_packet(system)
                packet_reports.append(report_path)
                packet_cmd = [PY, "scripts/goal_agent_packet.py", "--target", "codex", "--task", pkt["task"], "--complexity", "architecture"]
                for f in system["files"]:
                    packet_cmd += ["--file", f]
                for chk in system["checks"]:
                    packet_cmd += ["--check", chk]
                queue_rows.append(enqueue_job(conn, system, report_path, packet_cmd))
                manifest["systems"].append(
                    {
                        "system_id": system["system_id"],
                        "title": system["title"],
                        "priority": system["priority"],
                        "evidence": system["files"],
                        "next_command": system["next_command"],
                        "subsystems": system["subsystems"],
                        "agent_assignments": {"pm": PM, "builder": BUILDER, "breaker": BREAKER, "groq_workers": GROQ},
                        "packet_report_path": report_path,
                    }
                )
                conn.commit()
            except Exception as exc:
                manifest["blockers"].append({"system_id": system["system_id"], "error": str(exc)})
                conn.rollback()

    json_path = OUT / f"conductor_hierarchy_{stamp()}.json"
    jsonl_path = OUT / f"conductor_hierarchy_{stamp()}.jsonl"
    receipt_path = OUT / f"conductor_hierarchy_receipt_{stamp()}.json"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in manifest["systems"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    receipt = {
        "schema": "lucidota.conductor_hierarchy.receipt.v1",
        "generated_at": manifest["generated_at"],
        "queue": QUEUE,
        "status": "PASS" if not manifest["blockers"] else "FAIL",
        "systems_assigned": [s["system_id"] for s in manifest["systems"]],
        "system_count": len(manifest["systems"]),
        "packet_reports": packet_reports,
        "queue_rows": queue_rows,
        "manifest_path": rel(json_path),
        "jsonl_path": rel(jsonl_path),
        "blockers": manifest["blockers"],
    }
    receipt["receipt_path"] = rel(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("REPORT_PATH=" + rel(json_path))
    print("JSONL_PATH=" + rel(jsonl_path))
    print("RECEIPT_PATH=" + rel(receipt_path))
    print(json.dumps(receipt, sort_keys=True))
    return 0 if not manifest["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
