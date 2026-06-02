#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(kind: str, text: str) -> dict:
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "learn",
            "--candidate-kind",
            kind,
            "--text",
            text,
            "--artifact",
            "scripts/dev_journey_decision_points.py",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert proc.stderr == "" or "WARNING:" in proc.stderr
    return json.loads(proc.stdout)


def test_luci_learn_explicit_candidate_classes_select_distinct_probes():
    cases = {
        "algorithm": ("algorithm_trial_harness", "dev_journey_decision_points.py"),
        "source": ("current_world_source_adapter", "luci_source_slice.py"),
        "delegate": ("delegate_provider_class", "luci_delegate_slice.py"),
        "model": ("model_runtime_class", "lucidota_strict_model_stack_admission.py"),
        "archive-class": ("archive_ingestion_class", "luci_ingestion_status.py"),
    }

    for kind, (expected_candidate, probe_hint) in cases.items():
        payload = _run(kind, f"study {kind} candidate and prove it")
        loop = payload["learning_loop"]
        assert loop["slice"] == "luci_learning_slice"
        assert loop["candidate"]["candidate_kind"] == expected_candidate
        assert probe_hint in loop["probe"]["command"]
        assert loop["promotion_decision"] in {"promote", "archive"}
        assert loop["receipt_path"]
        assert payload["status"] in {"PASS", "DEGRADED"}
        if kind == "delegate":
            assert "db_write_error" not in loop["probe"]["probe_result"]


def test_luci_operate_routes_candidate_prompts_into_learning_loop():
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "operate",
            "--text",
            "study model candidate and prove it",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    loop = payload["learning_loop"]
    assert payload["verdict"] in {"PASS", "DEGRADED"}
    assert loop["slice"] == "luci_learning_slice"
    assert loop["candidate"]["candidate_kind"] == "model_runtime_class"
    assert "lucidota_strict_model_stack_admission.py" in loop["probe"]["command"]
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert proc.stderr == "" or "WARNING:" in proc.stderr


def test_luci_operate_routes_all_candidate_classes_into_learning_loop():
    cases = {
        "source": ("study source candidate and prove it", "current_world_source_adapter", "luci_source_slice.py"),
        "delegate": ("study delegate candidate and prove it", "delegate_provider_class", "luci_delegate_slice.py"),
        "algorithm": ("study algorithm candidate and prove it", "algorithm_trial_harness", "dev_journey_decision_points.py"),
        "archive": ("study archive candidate and prove it", "archive_ingestion_class", "luci_ingestion_status.py"),
    }

    for kind, (text, expected_candidate, probe_hint) in cases.items():
        proc = subprocess.run(
            [
                str(ROOT / "luci"),
                "operate",
                "--text",
                text,
                "--json",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        payload = json.loads(proc.stdout)
        loop = payload["learning_loop"]
        assert payload["verdict"] in {"PASS", "DEGRADED"}
        assert loop["slice"] == "luci_learning_slice"
        assert loop["candidate"]["candidate_kind"] == expected_candidate
        assert probe_hint in loop["probe"]["command"]
        assert loop["promotion_decision"] in {"promote", "archive"}
        assert loop["receipt_path"]
        assert proc.stdout.lstrip().startswith("{")
        assert proc.stdout.rstrip().endswith("}")
        assert proc.stderr == "" or "WARNING:" in proc.stderr

    algorithm_proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "operate",
            "--text",
            "study algorithm candidate and prove it",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    algorithm_payload = json.loads(algorithm_proc.stdout)
    assert algorithm_payload["learning_loop"]["probe"]["probe_result"]["truth_status"] == "training_candidates_only"


def test_luci_learn_same_run_id_is_idempotent_for_same_candidate_class():
    base = [
        str(ROOT / "luci"),
        "learn",
        "--candidate-kind",
        "source",
        "--text",
        "study source candidate and prove it",
        "--artifact",
        "scripts/dev_journey_decision_points.py",
        "--json",
    ]

    same_a = subprocess.run(base + ["--run-id", "pytest-learn-idem"], cwd=ROOT, text=True, capture_output=True, check=True)
    same_b = subprocess.run(base + ["--run-id", "pytest-learn-idem"], cwd=ROOT, text=True, capture_output=True, check=True)
    diff_run = subprocess.run(base + ["--run-id", "pytest-learn-idem-2"], cwd=ROOT, text=True, capture_output=True, check=True)
    diff_text = subprocess.run(
        [
            str(ROOT / "luci"),
            "learn",
            "--candidate-kind",
            "source",
            "--text",
            "study a different source candidate and prove it",
            "--artifact",
            "scripts/dev_journey_decision_points.py",
            "--run-id",
            "pytest-learn-idem",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload_same_a = json.loads(same_a.stdout)
    payload_same_b = json.loads(same_b.stdout)
    payload_diff_run = json.loads(diff_run.stdout)
    payload_diff_text = json.loads(diff_text.stdout)

    assert payload_same_a["db_write"]["work_order_uuid"] == payload_same_b["db_write"]["work_order_uuid"]
    assert payload_same_a["db_write"]["work_receipt_uuid"] == payload_same_b["db_write"]["work_receipt_uuid"]
    assert payload_same_a["db_write"]["work_order_uuid"] != payload_diff_run["db_write"]["work_order_uuid"]
    assert payload_same_a["db_write"]["work_receipt_uuid"] != payload_diff_run["db_write"]["work_receipt_uuid"]
    assert payload_same_a["db_write"]["work_order_uuid"] != payload_diff_text["db_write"]["work_order_uuid"]
    assert payload_same_a["db_write"]["work_receipt_uuid"] != payload_diff_text["db_write"]["work_receipt_uuid"]
