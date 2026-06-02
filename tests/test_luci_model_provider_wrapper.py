#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_wrapper_exposes_model_provider_class():
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "model",
            "provider",
            "groq-chat",
            "--prompt",
            "say hello in one short sentence",
            "--max-tokens",
            "16",
            "--execute",
            "--run-id",
            "pytest-model-provider-json",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["provider"] == "groq"
    assert payload["api_key_env_used"] == "GROQ_API_KEY"
    assert payload["report_path"]
    assert payload["db_write"]["model_invocation_uuid"]
    assert payload["db_write"]["work_order_uuid"]
    assert payload["db_write"]["work_receipt_uuid"]
    assert payload["visible_response"]["model_invocation_id"] == payload["db_write"]["model_invocation_uuid"]
    assert payload["visible_response"]["work_order_id"] == payload["db_write"]["work_order_uuid"]
    assert payload["visible_response"]["work_receipt_id"] == payload["db_write"]["work_receipt_uuid"]
    assert payload["visible_response"]["attempt_id"] == payload["db_write"]["work_order_uuid"]
    assert payload["visible_response"]["raw_artifact_id"] == payload["db_write"]["raw_artifact_uuid"]
    assert "RECEIPT_PATH=" not in proc.stdout
    assert proc.stderr == "" or proc.stderr.strip()


def test_luci_shell_wrapper_model_provider_same_run_id_is_idempotent():
    base = [
        str(ROOT / "luci"),
        "model",
        "provider",
        "groq-chat",
        "--prompt",
        "say hello in one short sentence",
        "--max-tokens",
        "16",
        "--execute",
        "--json",
    ]
    same_a = subprocess.run(base + ["--run-id", "pytest-model-provider-idem"], cwd=ROOT, text=True, capture_output=True, check=True)
    same_b = subprocess.run(base + ["--run-id", "pytest-model-provider-idem"], cwd=ROOT, text=True, capture_output=True, check=True)
    diff_run = subprocess.run(base + ["--run-id", "pytest-model-provider-idem-2"], cwd=ROOT, text=True, capture_output=True, check=True)

    payload_same_a = json.loads(same_a.stdout)
    payload_same_b = json.loads(same_b.stdout)
    payload_diff_run = json.loads(diff_run.stdout)

    assert payload_same_a["db_write"]["model_invocation_uuid"] == payload_same_b["db_write"]["model_invocation_uuid"]
    assert payload_same_a["db_write"]["work_order_uuid"] == payload_same_b["db_write"]["work_order_uuid"]
    assert payload_same_a["db_write"]["work_receipt_uuid"] == payload_same_b["db_write"]["work_receipt_uuid"]
    assert payload_same_a["db_write"]["model_invocation_uuid"] != payload_diff_run["db_write"]["model_invocation_uuid"]
    assert payload_same_a["db_write"]["work_order_uuid"] != payload_diff_run["db_write"]["work_order_uuid"]
    assert payload_same_a["db_write"]["work_receipt_uuid"] != payload_diff_run["db_write"]["work_receipt_uuid"]
    assert payload_same_a["visible_response"]["work_order_id"] == payload_same_a["db_write"]["work_order_uuid"]
    assert payload_same_a["visible_response"]["work_receipt_id"] == payload_same_a["db_write"]["work_receipt_uuid"]
