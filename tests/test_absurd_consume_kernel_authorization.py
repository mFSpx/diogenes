#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import absurd_consume_one
from kernel_control_packet import absurd_enqueue_packet

DB = os.environ.get("ABSURD_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state")


def run(cmd, *, expect: int = 0):
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode == expect, (cmd, proc.returncode, proc.stdout, proc.stderr)
    return proc


def report_from(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return json.loads((ROOT / line.split("=", 1)[1]).read_text(encoding="utf-8"))
    raise AssertionError(stdout)


def insert_job(payload: dict, *, idem: str, max_attempts: int = 1, job_kind: str = "korpus_componentize") -> str:
    with psycopg.connect(DB) as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO lucidota_control.absurd_queue(queue_name,owner_subsystem,notes) VALUES ('korpus','pytest','pytest') ON CONFLICT(queue_name) DO NOTHING")
            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status='dead_lettered', error_kind='pytest_reset', error_message='pytest_reset', completed_at=now(), updated_at=now()
                WHERE queue_name='korpus' AND status='queued'
                """
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload,status,priority,max_attempts)
                VALUES ('korpus','pytest-kernel-auth',%s,%s,%s::jsonb,'queued',-100000,%s)
                ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET payload=EXCLUDED.payload,status='queued',priority=-100000,attempt_count=0,max_attempts=EXCLUDED.max_attempts,run_after=now(),updated_at=now()
                RETURNING job_uuid::text
                """,
                (job_kind, idem, json.dumps(payload), max_attempts),
            )
            job_uuid = cur.fetchone()[0]
        conn.commit()
    return job_uuid


def fetch_job(job_uuid: str) -> dict:
    with psycopg.connect(DB) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status::text,result,error_kind,error_message,attempt_count FROM lucidota_control.absurd_queue_job WHERE job_uuid=%s", (job_uuid,))
            row = cur.fetchone()
    return {"status": row[0], "result": row[1], "error_kind": row[2], "error_message": row[3], "attempt_count": row[4]}


def test_external_command_receipt_is_shell_quoted(monkeypatch):
    class Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(absurd_consume_one.subprocess, "run", lambda *_args, **_kwargs: Proc())
    ok, result = absurd_consume_one.handle(
        {
            "handler": "external_command",
            "command": [
                sys.executable,
                "scripts/chrono_queue_event_bridge.py",
                "--case-id",
                "space case",
            ],
        }
    )
    assert ok
    assert "'space case'" in result["command"]


def test_external_command_can_target_goal_agent_packet(monkeypatch):
    class Proc:
        returncode = 0
        stdout = "REPORT_PATH=05_OUTPUTS/goals/fake.json\n"
        stderr = ""

    monkeypatch.setattr(absurd_consume_one.subprocess, "run", lambda *_args, **_kwargs: Proc())
    ok, result = absurd_consume_one.handle(
        {
            "handler": "external_command",
            "command": [
                sys.executable,
                "scripts/goal_agent_packet.py",
                "--target",
                "generic",
                "--task",
                "recovery",
                "--complexity",
                "simple",
                "--json",
            ],
        }
    )
    assert ok
    assert "goal_agent_packet.py" in result["command"]


def test_consume_one_dead_letters_korpus_job_missing_kernel_authorization():
    idem = f"pytest-consume-missing-auth-{uuid.uuid4()}"
    job_uuid = insert_job({"lane": "manifest_inventory", "bridge_version": "v2", "handler": "noop"}, idem=idem)
    proc = run([sys.executable, "scripts/absurd_consume_one.py", "--queue-name", "korpus", "--execute"], expect=2)
    report = report_from(proc.stdout)
    job = fetch_job(job_uuid)
    assert report["job_uuid"] == job_uuid
    assert report["status"] == "dead_lettered"
    assert job["status"] == "dead_lettered"
    assert job["error_kind"] == "missing_kernel_authorization"
    assert job["result"]["error"] == "kernel_authorization_rejected"


def test_consume_one_succeeds_korpus_job_with_valid_kernel_authorization():
    idem = f"pytest-consume-valid-auth-{uuid.uuid4()}"
    packet = absurd_enqueue_packet(queue_name="korpus", lane="manifest_inventory", source_path="README.md", idempotency_key=idem, authorized_by="operator_cli")
    payload = {"lane": "manifest_inventory", "source_path": "README.md", "bridge_version": "v2", "handler": "noop", "idempotency_key": idem, "kernel_authorization": packet}
    job_uuid = insert_job(payload, idem=idem)
    proc = run([sys.executable, "scripts/absurd_consume_one.py", "--queue-name", "korpus", "--execute"])
    report = report_from(proc.stdout)
    job = fetch_job(job_uuid)
    assert report["job_uuid"] == job_uuid
    assert report["status"] == "succeeded"
    assert job["status"] == "succeeded"
    assert job["error_kind"] == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)


def test_usecase_proof_is_consume_allowlisted():
    assert "scripts/lucidota_usecase_proof.py" in absurd_consume_one.ALLOWED_EXTERNAL_COMMANDS
