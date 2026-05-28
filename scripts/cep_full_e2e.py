#!/usr/bin/env python3
"""CEP full E2E: surface instruction -> conversation_command -> ABSURD work order.

Default mode is a graph-safe golden-path dry-run.  Execute mode preserves the
existing write path for staged conversation_command + ABSURD queue creation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS/absurd"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(p: str | Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def runc(cmd: list[str]) -> dict[str, Any]:
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    d: dict[str, Any] = {
        "cmd": " ".join(cmd),
        "rc": p.returncode,
        "stdout_tail": p.stdout[-4000:],
        "stderr_tail": p.stderr[-4000:],
    }
    for line in p.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


def script_exists(path: str) -> bool:
    p = ROOT / path
    return p.exists() and p.is_file()


def _step_ok(step: dict[str, Any], allow_nonzero: bool = False) -> bool:
    return allow_nonzero or int(step.get("rc", 1)) == 0


def _kernel_auth_payload() -> str:
    return json.dumps(
        {
            "bridge_version": "v3",
            "lane": "golden_path",
            "source_path": "fixtures/golden_path",
            "idempotency_key": "golden-path-dry-run",
            "canonical_mutation_allowed": True,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_lineage_id(operator_instruction: str) -> str:
    """Stable lineage key for dry-run probes before a DB command_uuid exists."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "lucidota:golden_path:" + operator_instruction))


def collect_receipt_refs(steps: list[dict[str, Any]]) -> list[str]:
    """Collect REPORT_PATH refs emitted by child scripts for one golden-path lineage."""
    refs: list[str] = []
    for step in steps:
        ref = step.get("REPORT_PATH")
        if isinstance(ref, str) and ref and ref not in refs:
            refs.append(ref)
        for line in str(step.get("stdout_tail") or "").splitlines():
            if line.startswith("REPORT_PATH="):
                ref = line.split("=", 1)[1]
                if ref and ref not in refs:
                    refs.append(ref)
    return refs


def run_dry_run(
    runner: Callable[[list[str]], dict[str, Any]] = runc,
    *,
    operator_instruction: str = "Build one dry-run golden path proof without canonical graph materialization.",
    lineage_id: str | None = None,
) -> dict[str, Any]:
    """Run the narrow golden path without DB/graph mutations where scripts expose it."""
    blockers: list[str] = []
    steps: list[dict[str, Any]] = []
    lineage_id = lineage_id or stable_lineage_id(operator_instruction)

    compile_step = runner(
        [
            sys.executable,
            "scripts/surface_instruction_compile_dry_run.py",
            "--surface-id",
            "cep_full_e2e",
            "--operator-action",
            "refined",
            "--target-ref",
            "workflow:surface_cep",
            "--evidence-refs",
            "scripts/cep_full_e2e.py",
            "--artifact-refs",
            "07_SURFACES/generated/marrow_loop_status.html",
            "--current-object-state",
            json.dumps({"round": "golden_path_dry_run", "lineage_id": lineage_id, "operator_instruction": operator_instruction}, sort_keys=True),
            "--allowed-effect",
            "dry-run ABSURD work order only",
            "--dry-run",
        ]
    )
    compile_step["link"] = "operator_instruction_to_command_envelope"
    steps.append(compile_step)
    if not _step_ok(compile_step):
        blockers.append("surface_instruction_compile_dry_run_failed")

    accept_step = runner(
        [
            sys.executable,
            "scripts/conversation_command_accept_worker.py",
            "accept",
            "--limit",
            "1",
            "--worker-id",
            "cep_full_e2e_dry_run",
        ]
    )
    accept_step["link"] = "conversation_command_accept_status_worker"
    steps.append(accept_step)
    if not _step_ok(accept_step):
        blockers.append("conversation_command_accept_worker_dry_run_failed")

    absurd_step = runner(
        [
            sys.executable,
            "scripts/absurd_queue_spine.py",
            "--action",
            "audit",
            "--dry-run",
        ]
    )
    absurd_step["link"] = "absurd_queue_job"
    steps.append(absurd_step)
    if not _step_ok(absurd_step):
        blockers.append("absurd_queue_spine_audit_failed")

    auth_step = runner(
        [
            sys.executable,
            "scripts/absurd_kernel_authorization.py",
            "validate",
            "--queue-name",
            "korpus",
            "--job-kind",
            "korpus_lane_job",
            "--payload-json",
            _kernel_auth_payload(),
        ]
    )
    auth_step["link"] = "kernel_authz_decision"
    steps.append(auth_step)
    if int(auth_step.get("rc", 0)) == 0:
        blockers.append("kernel_authorization_unexpectedly_allowed_without_packet")
    elif "missing_kernel_authorization" not in (auth_step.get("stdout_tail") or ""):
        blockers.append("kernel_authorization_denial_not_structured")

    graph_gate_step = runner(
        [
            sys.executable,
            "scripts/graph_promotion_gate.py",
            "gate",
            "--dry-run",
            "--candidate-kind",
            "other",
            "--candidate-payload-json",
            '{"kind":"golden_path_probe","source":"scripts/cep_full_e2e.py"}',
            "--evidence-ref",
            "scripts/cep_full_e2e.py",
            "--authority-class",
            "operator_authored_assertion",
            "--decision",
            "defer",
            "--rationale",
            "golden path dry-run staging probe",
        ]
    )
    graph_gate_step["link"] = "graph_promotion_staging"
    steps.append(graph_gate_step)
    if not _step_ok(graph_gate_step):
        report_ref = str(graph_gate_step.get("REPORT_PATH") or "")
        if "graph_promotion_gate_blocked_" not in report_ref:
            blockers.append("graph_promotion_gate_dry_run_failed")

    graph_barrier_step = runner([sys.executable, "scripts/graph_canonical_mutation_detector.py"])
    graph_barrier_step["link"] = "canonical_graph_write_barrier"
    steps.append(graph_barrier_step)
    if not _step_ok(graph_barrier_step):
        blockers.append("graph_canonical_mutation_detector_failed")

    ledger_step = runner([sys.executable, "scripts/lucidota_status_ledger.py", "--check"])
    ledger_step["link"] = "status_ledger_check"
    steps.append(ledger_step)
    if not _step_ok(ledger_step):
        blockers.append("status_ledger_check_failed")

    link_status = {
        "operator_instruction_enters_conversation_command": "CLAIMED_DRY_RUN",
        "conversation_command_accepted_statused": "CLAIMED_DRY_RUN",
        "absurd_queue_job_created": "REAL_SCHEMA_AUDITED",
        "kernel_authz_decision_exists": "REAL_DENIAL",
        "worker_executes": "CLAIMED_DRY_RUN",
        "receipt_written": "REAL",
        "status_ledger_updated_checks": "REAL_CHECK",
        "graph_promotion_staging_exists": "REAL_DRY_RUN" if "graph_promotion_gate_dry_run_failed" not in blockers else "BLOCKED",
        "canonical_graph_write_barrier_holds": "REAL" if "graph_canonical_mutation_detector_failed" not in blockers else "BLOCKED",
    }
    return {
        "mode": "dry_run",
        "lineage_id": lineage_id,
        "operator_instruction": operator_instruction,
        "execute_performed": False,
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "canonical_graph_writes_performed": False,
        "steps": steps,
        "receipt_refs": collect_receipt_refs(steps),
        "link_status": link_status,
        "blockers": blockers,
        "status": "PASS_DRY_RUN" if not blockers else "FAIL",
    }


def run_execute(
    runner: Callable[[list[str]], dict[str, Any]] = runc,
    *,
    operator_instruction: str = "Build one executable golden path proof without canonical graph materialization.",
    lineage_id: str | None = None,
    run_worker_once: bool = False,
) -> dict[str, Any]:
    blockers: list[str] = []
    steps: list[dict[str, Any]] = []
    lineage_id = lineage_id or stable_lineage_id(operator_instruction)
    comp = runner(
        [
            sys.executable,
            "scripts/surface_instruction_compile_dry_run.py",
            "--surface-id",
            "cep_full_e2e",
            "--operator-action",
            "refined",
            "--target-ref",
            "workflow:surface_cep",
            "--evidence-refs",
            "scripts/cep_full_e2e.py",
            "--artifact-refs",
            "07_SURFACES/generated/marrow_loop_status.html",
            "--current-object-state",
            json.dumps({"round": "golden_path_execute", "lineage_id": lineage_id, "operator_instruction": operator_instruction}, sort_keys=True),
            "--allowed-effect",
            "queue ABSURD work order only",
            "--execute",
            "--queue-absurd",
        ]
    )
    steps.append(comp)
    cmd = comp.get("COMMAND_UUID")
    job = comp.get("ABSURD_JOB_UUID")
    if not cmd:
        blockers.append("command_uuid_missing")
    if not job:
        blockers.append("absurd_job_uuid_missing")
    if cmd and script_exists("scripts/conversation_command_accept_worker.py"):
        accept_step = runner(
            [
                sys.executable,
                "scripts/conversation_command_accept_worker.py",
                "accept",
                "--command-uuid",
                cmd,
                "--limit",
                "1",
                "--worker-id",
                "cep_full_e2e_execute",
                "--execute",
            ]
        )
        steps.append(accept_step)
        if accept_step["rc"] != 0:
            blockers.append("conversation_command_accept_execute_failed")
    if run_worker_once and job and script_exists("scripts/spine_surface_cep_worker.py"):
        worker_step = runner(
            [
                sys.executable,
                "scripts/spine_surface_cep_worker.py",
                "worker-once",
                "--execute",
                "--job-uuid",
                job,
                "--worker-id",
                "cep_full_e2e_execute",
            ]
        )
        steps.append(worker_step)
        if worker_step["rc"] != 0:
            blockers.append("surface_cep_worker_once_failed")
    return {
        "mode": "execute",
        "lineage_id": lineage_id,
        "operator_instruction": operator_instruction,
        "execute_performed": True,
        "db_writes_performed": bool(not blockers),
        "graph_writes_performed": False,
        "canonical_graph_writes_performed": False,
        "steps": steps,
        "receipt_refs": collect_receipt_refs(steps),
        "conversation_command_uuid": steps[0].get("COMMAND_UUID") if steps else None,
        "absurd_job_uuid": steps[0].get("ABSURD_JOB_UUID") if steps else None,
        "blockers": blockers,
        "status": "PASS" if not blockers else "FAIL",
    }


def write_payload(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"cep_full_e2e_{payload['mode']}_{stamp()}.json"
    payload["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(p))
    print("CEP_FULL_E2E=" + payload["status"])
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="prove the golden path without DB/graph mutation; default")
    mode.add_argument("--execute", action="store_true", help="write conversation_command + ABSURD work order only")
    ap.add_argument("--operator-instruction", default="Build one executable golden path proof without canonical graph materialization.")
    ap.add_argument("--lineage-id")
    ap.add_argument("--run-worker-once", action="store_true", help="execute one surface CEP queue worker pass in execute mode")
    a = ap.parse_args()
    payload = (
        run_execute(operator_instruction=a.operator_instruction, lineage_id=a.lineage_id, run_worker_once=bool(a.run_worker_once))
        if a.execute
        else run_dry_run(operator_instruction=a.operator_instruction, lineage_id=a.lineage_id)
    )
    write_payload(payload)
    return 0 if not payload["blockers"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
