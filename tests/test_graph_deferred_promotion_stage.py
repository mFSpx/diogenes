from __future__ import annotations

import json
from pathlib import Path
from importlib import import_module

mod = import_module("scripts.graph_deferred_promotion_stage")


class FakeCompleted:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def test_stage_deferred_packets_invokes_graph_gate_execute_without_materialize(tmp_path: Path):
    packets = tmp_path / "packets.jsonl"
    gate_report = tmp_path / "gate_report.json"
    gate_report.write_text(json.dumps({"db_writes_performed": True, "canonical_graph_writes_performed": False}), encoding="utf-8")
    packets.write_text(
        json.dumps(
            {
                "candidate_kind": "node",
                "candidate_payload": {"label": "one"},
                "evidence_refs": ["evidence-one"],
                "authority_class": "operator_authored_assertion",
                "decision": "defer",
                "rationale": "stage only",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return FakeCompleted(f"REPORT_PATH={gate_report}\nGRAPH_GATE_ALLOWED=true\n")

    receipt = mod.stage_deferred_packets(
        packets_path=packets,
        receipt_path=tmp_path / "receipt.json",
        max_packets=1,
        dry_run=False,
        run_cmd=fake_run,
    )

    assert receipt["status"] == "PASS"
    assert receipt["packets_seen"] == 1
    assert receipt["packets_staged"] == 1
    assert receipt["canonical_graph_writes_performed"] is False
    assert calls
    cmd = calls[0]
    assert "--execute" in cmd
    assert "--materialize" not in cmd
    assert "--candidate-payload-json" in cmd
    assert "--evidence-ref" in cmd


def test_stage_deferred_packets_uses_explicit_python_bin_for_gate(tmp_path: Path):
    packets = tmp_path / "packets.jsonl"
    packets.write_text(json.dumps({"candidate_payload": {"label": "one"}}) + "\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return FakeCompleted("REPORT_PATH=gate.json\nGRAPH_GATE_ALLOWED=true\n")

    receipt = mod.stage_deferred_packets(
        packets_path=packets,
        receipt_path=tmp_path / "receipt.json",
        max_packets=1,
        dry_run=False,
        python_bin="/custom/venv/python",
        run_cmd=fake_run,
    )

    assert receipt["status"] == "PASS"
    assert calls
    assert calls[0][0] == "/custom/venv/python"


def test_stage_deferred_packets_dry_run_has_no_subprocess_calls(tmp_path: Path):
    packets = tmp_path / "packets.jsonl"
    packets.write_text(json.dumps({"candidate_payload": {"label": "one"}, "evidence_refs": ["e"]}) + "\n", encoding="utf-8")
    called = False

    def fake_run(cmd, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("should not run")

    receipt = mod.stage_deferred_packets(
        packets_path=packets,
        receipt_path=tmp_path / "receipt.json",
        max_packets=1,
        dry_run=True,
        run_cmd=fake_run,
    )

    assert receipt["status"] == "PASS"
    assert receipt["dry_run"] is True
    assert receipt["packets_staged"] == 0
    assert called is False


def test_stage_deferred_packets_can_start_after_prior_batch(tmp_path: Path):
    packets = tmp_path / "packets.jsonl"
    packets.write_text(
        "\n".join(
            [
                json.dumps({"candidate_payload": {"label": "one"}}),
                json.dumps({"candidate_payload": {"label": "two"}}),
                json.dumps({"candidate_payload": {"label": "three"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    payload_labels: list[str] = []

    def fake_run(cmd, **kwargs):
        payload_path = Path(cmd[cmd.index("--candidate-payload-json") + 1])
        payload_labels.append(json.loads(payload_path.read_text(encoding="utf-8"))["label"])
        return FakeCompleted("REPORT_PATH=gate.json\nGRAPH_GATE_ALLOWED=true\n")

    receipt = mod.stage_deferred_packets(
        packets_path=packets,
        receipt_path=tmp_path / "receipt.json",
        start_index=2,
        max_packets=2,
        dry_run=False,
        python_bin="/custom/venv/python",
        run_cmd=fake_run,
    )

    assert receipt["status"] == "PASS"
    assert receipt["start_index"] == 2
    assert receipt["packets_seen"] == 2
    assert payload_labels == ["two", "three"]
