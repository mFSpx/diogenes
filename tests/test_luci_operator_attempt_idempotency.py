from __future__ import annotations

import importlib
import json
import subprocess
import uuid
from pathlib import Path


class _FakeJobAdapter:
    def __init__(self, root):
        self.root = root
        self.jobs_path = root / "jobs.jsonl"
        self.calls = []

    def create_job(self, *, lane, payload, idempotency_key):
        self.calls.append({"lane": lane, "payload": payload, "idempotency_key": idempotency_key})
        return {
            "job_id": f"job::{idempotency_key}",
            "state": "CREATED",
            "attempt_count": 0,
            "max_attempts": 2,
        }

    def transition(self, job_id, state):
        return {
            "job_id": job_id,
            "state": state,
            "attempt_count": 1,
            "max_attempts": 2,
        }


def test_create_attempt_task_idempotency_depends_on_text_and_run_id(monkeypatch):
    op = importlib.import_module("scripts.luci_operator")
    fake = _FakeJobAdapter(op.ATTEMPT_ENGINE_ROOT)
    monkeypatch.setattr(op, "ABSURDJobAdapter", lambda root: fake)

    language = {"intent": "route", "ontology_terms": ["ingest"]}
    moa = {"input_route": {"lane": "FASTLANE", "route_reason": ["test"]}}

    same_a = op.create_attempt_task(text="hello world", run_id="run-1", language=language, moa=moa)
    same_b = op.create_attempt_task(text="hello world", run_id="run-1", language=language, moa=moa)
    diff_run = op.create_attempt_task(text="hello world", run_id="run-2", language=language, moa=moa)
    diff_text = op.create_attempt_task(text="hello world 2", run_id="run-1", language=language, moa=moa)

    assert same_a["job_id"] == same_b["job_id"]
    assert same_a["job_id"] != diff_run["job_id"]
    assert same_a["job_id"] != diff_text["job_id"]
    assert fake.calls[0]["idempotency_key"] == fake.calls[1]["idempotency_key"]
    assert fake.calls[0]["idempotency_key"] != fake.calls[2]["idempotency_key"]
    assert fake.calls[0]["idempotency_key"] != fake.calls[3]["idempotency_key"]


def test_luci_operate_json_attempt_idempotency_depends_on_text_and_run_id():
    base = [
        str(Path(__file__).resolve().parents[1] / "luci"),
        "operate",
        "--text",
        "fix one small broken thing and prove it",
        "--json",
    ]

    root = Path(__file__).resolve().parents[1]

    same_a = subprocess.run(base + ["--run-id", "pytest-operate-idem"], cwd=root, text=True, capture_output=True, check=True)
    same_b = subprocess.run(base + ["--run-id", "pytest-operate-idem"], cwd=root, text=True, capture_output=True, check=True)
    diff_run = subprocess.run(base + ["--run-id", "pytest-operate-idem-2"], cwd=root, text=True, capture_output=True, check=True)
    diff_text = subprocess.run(
        [
            str(root / "luci"),
            "operate",
            "--text",
            "fix one different thing and prove it",
            "--run-id",
            "pytest-operate-idem",
            "--json",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )

    payload_same_a = json.loads(same_a.stdout)
    payload_same_b = json.loads(same_b.stdout)
    payload_diff_run = json.loads(diff_run.stdout)
    payload_diff_text = json.loads(diff_text.stdout)

    assert payload_same_a["attempt_engine"]["job_uuid"] == payload_same_b["attempt_engine"]["job_uuid"]
    assert payload_same_a["attempt_engine"]["job_uuid"] != payload_diff_run["attempt_engine"]["job_uuid"]
    assert payload_same_a["attempt_engine"]["job_uuid"] != payload_diff_text["attempt_engine"]["job_uuid"]
    assert payload_same_a["attempt_engine"]["db_write"]["work_order_uuid"] == payload_same_b["attempt_engine"]["db_write"]["work_order_uuid"]
    assert payload_same_a["attempt_engine"]["db_write"]["work_receipt_uuid"] == payload_same_b["attempt_engine"]["db_write"]["work_receipt_uuid"]
    assert payload_same_a["attempt_engine"]["db_write"]["work_order_uuid"] != payload_diff_run["attempt_engine"]["db_write"]["work_order_uuid"]
    assert payload_same_a["attempt_engine"]["db_write"]["work_receipt_uuid"] != payload_diff_run["attempt_engine"]["db_write"]["work_receipt_uuid"]
    assert payload_same_a["attempt_engine"]["db_write"]["work_order_uuid"] != payload_diff_text["attempt_engine"]["db_write"]["work_order_uuid"]
    assert payload_same_a["attempt_engine"]["db_write"]["work_receipt_uuid"] != payload_diff_text["attempt_engine"]["db_write"]["work_receipt_uuid"]
    assert same_a.stdout.lstrip().startswith("{")
    assert same_a.stdout.rstrip().endswith("}")
    assert same_a.stderr == "" or "WARNING:" in same_a.stderr


def test_luci_operate_runtime_closure_records_real_worker_loop():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            str(root / "luci"),
            "operate",
            "--text",
            "runtime closure smoke: read manual, check daemon, emit receipt",
            "--run-id",
            f"pytest-runtime-closure-real-loop-{uuid.uuid4()}",
            "--json",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    attempt = payload["attempt_engine"]
    db_write = attempt["db_write"]

    assert payload["verdict"] == "PASS"
    assert payload["canonical_graph_writes_performed"] is False
    assert payload["real_work_loop"]["worker_executed"] is True
    assert payload["real_work_loop"]["dead_letter_count"] == 0
    assert payload["db_write"]["work_order_uuid"] == db_write["work_order_uuid"]
    assert payload["db_write"]["workload_audit_uuid"] == db_write["workload_audit_uuid"]
    assert attempt["real_work_loop"]["worker_executed"] is True
    assert attempt["real_work_loop"]["dead_letter_count"] == 0
    assert db_write["work_order_uuid"]
    assert db_write["work_order_attempt_uuid"]
    assert db_write["work_receipt_uuid"]
    assert db_write["workload_audit_uuid"]
    assert db_write["worker_id"] == "luci_attempt_engine"
    assert payload["visible_response"]["summary"].startswith("Indy_READs:")
