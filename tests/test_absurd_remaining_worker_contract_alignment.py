from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import absurd_remaining_worker_contract_alignment_check as alignment_check  # noqa: E402


def _run_check():
    return subprocess.run(
        [sys.executable, "scripts/absurd_remaining_worker_contract_alignment_check.py", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )


def _receipt(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("RECEIPT_PATH="):
            return json.loads((ROOT / line.split("=", 1)[1]).read_text(encoding="utf-8"))
    raise AssertionError(stdout)


def test_remaining_workers_contract_alignment_check_passes_for_four_blocker_workers():
    proc = _run_check()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ABSURD_REMAINING_WORKER_CONTRACT_ALIGNMENT=PASS" in proc.stdout
    data = _receipt(proc.stdout)
    assert data["schema"] == "lucidota.absurd_remaining_worker_contract_alignment.v1"
    assert data["worker_count"] == 4
    assert data["errors"] == []


def test_remaining_workers_use_reusable_validate_and_rejection_helper():
    proc = _run_check()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = {row["worker_key"]: row for row in _receipt(proc.stdout)["workers"]}
    for worker_key in ["chrono_worker", "surface_cep_worker", "document_parse_worker", "krampus_worker"]:
        row = rows[worker_key]
        assert row["uses_validate_worker_contract"] is True
        assert row["uses_record_worker_contract_rejection"] is True
        assert row["contract_checked_before_running_status"] is True


def test_contract_order_check_ignores_imports_and_requires_worker_body_order():
    import_only_false_positive = """
from absurd_worker_contracts import record_worker_contract_rejection, validate_worker_contract
def worker_once(args):
    cur.execute("UPDATE lucidota_control.absurd_queue_job SET status='running' WHERE job_uuid=%s", (job_uuid,))
"""
    assert alignment_check.checked_before_running(import_only_false_positive) is False

    body_order_true = """
def worker_once(args):
    contract = validate_worker_contract(cur, queue_name=QUEUE, job_kind=job_kind, worker_key=WORKER_KEY)
    if not contract.allowed:
        record_worker_contract_rejection(cur, job_uuid=job, queue_name=QUEUE, payload=payload, contract=contract)
        return
    cur.execute("UPDATE lucidota_control.absurd_queue_job SET status='running' WHERE job_uuid=%s", (job,))
"""
    assert alignment_check.checked_before_running(body_order_true) is True


def test_schema_082_matches_actual_remaining_worker_job_kinds():
    proc = _run_check()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = {row["worker_key"]: row for row in _receipt(proc.stdout)["workers"]}
    assert rows["chrono_worker"]["schema_job_kinds"] == ["chrono_health_check"]
    assert rows["surface_cep_worker"]["schema_job_kinds"] == [
        "conversation_command_work_order",
        "surface_cep_health_check",
        "surface_instruction_compiled_command",
        "surface_instruction_fan_in",
    ]
    assert rows["document_parse_worker"]["schema_job_kinds"] == ["document_parse_ingest"]
    assert rows["krampus_worker"]["schema_job_kinds"] == ["korpus_componentize", "krampus_health_check"]
    for row in rows.values():
        assert row["schema_job_kinds"] == row["source_job_kinds"]
