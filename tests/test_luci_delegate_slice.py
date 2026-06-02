from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import luci_delegate_slice as delegate  # noqa: E402


def test_build_prompt_mentions_kind_provider_and_text() -> None:
    prompt = delegate.build_prompt(
        "audit the delegate rail",
        kind="review",
        provider="groq",
        context={"ontology": {"official_ontology": "GO-25"}},
    )

    assert "Delegate kind: review" in prompt
    assert "Provider lane: groq" in prompt
    assert "audit the delegate rail" in prompt
    assert "GO-25" in prompt


def test_write_vibes_prompt_creates_a_prompt_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(delegate, "RUNTIME", tmp_path / "runtime")
    vibes = delegate.write_vibes_prompt("review this", kind="plan")

    assert vibes["prompt_path"].endswith(".prompt")
    assert "vibe -p" in vibes["execute_hint"]
    assert (tmp_path / "runtime").exists()


def test_run_groq_delegate_uses_wrapper_report_subreceipt(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "groq_goal_delegate.json"
    report_path.write_text(
        json.dumps(
            {
                "subreceipt_path": "05_OUTPUTS/model_invocations/groq_chat_execute_test.json",
                "usage": {"total_tokens": 7},
                "text": "{\"summary\":\"reviewed\"}",
                "blockers": [],
            }
        )
    )

    class P:
        returncode = 0
        stderr = ""
        stdout = f"REPORT_PATH={report_path}\nGROQ_GOAL_DELEGATE=PASS\n"

    monkeypatch.setattr(delegate.subprocess, "run", lambda *args, **kwargs: P())

    payload = delegate.run_groq_delegate("review this", kind="review")

    assert payload["subreceipt_path"] == "05_OUTPUTS/model_invocations/groq_chat_execute_test.json"
    assert payload["usage"] == {"total_tokens": 7}
    assert payload["text"] == "{\"summary\":\"reviewed\"}"


def test_delegate_json_stdout_is_pure_json() -> None:
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "delegate",
            "--kind",
            "review",
            "--text",
            "review JSON purity of delegate output",
            "--provider",
            "both",
            "--run-id",
            "pytest-delegate-json",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    json.loads(proc.stdout)
    assert proc.stderr == "" or "WARNING:" in proc.stderr or "Problem" in proc.stderr


def test_delegate_visible_response_exposes_db_ids_and_is_idempotent_for_same_run_id() -> None:
    base = [
        str(ROOT / "luci"),
        "delegate",
        "--kind",
        "review",
        "--text",
        "review the delegate rail and prove it",
        "--provider",
        "both",
        "--json",
    ]

    same_a = subprocess.run(base + ["--run-id", "pytest-delegate-idem"], cwd=ROOT, text=True, capture_output=True, check=True)
    same_b = subprocess.run(base + ["--run-id", "pytest-delegate-idem"], cwd=ROOT, text=True, capture_output=True, check=True)
    diff_run = subprocess.run(base + ["--run-id", "pytest-delegate-idem-2"], cwd=ROOT, text=True, capture_output=True, check=True)

    payload_same_a = json.loads(same_a.stdout)
    payload_same_b = json.loads(same_b.stdout)
    payload_diff_run = json.loads(diff_run.stdout)

    assert payload_same_a["db_write"]["work_order_uuid"] == payload_same_b["db_write"]["work_order_uuid"]
    assert payload_same_a["db_write"]["work_receipt_uuid"] == payload_same_b["db_write"]["work_receipt_uuid"]
    assert payload_same_a["db_write"]["work_order_uuid"] != payload_diff_run["db_write"]["work_order_uuid"]
    assert payload_same_a["db_write"]["work_receipt_uuid"] != payload_diff_run["db_write"]["work_receipt_uuid"]
    assert payload_same_a["visible_response"]["work_order_id"] == payload_same_a["db_write"]["work_order_uuid"]
    assert payload_same_a["visible_response"]["work_receipt_id"] == payload_same_a["db_write"]["work_receipt_uuid"]
    assert payload_same_a["visible_response"]["attempt_id"] == payload_same_a["db_write"]["work_order_uuid"]
    assert payload_same_a["visible_response"]["raw_artifact_id"] == payload_same_a["db_write"]["raw_artifact_uuid"]
