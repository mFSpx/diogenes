from __future__ import annotations

import json


def test_promptflow_eval_runner_summarizes_outputs_jsonl(tmp_path):
    import scripts.promptflow_eval_runner as runner

    outputs = tmp_path / "outputs.jsonl"
    outputs.write_text(
        "\n".join(
            [
                json.dumps({"outputs": {"score": 0.9}}),
                json.dumps({"outputs": {"verdict": "pass"}}),
                json.dumps({"outputs": {"score": 0.2}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = runner.summarize_outputs(outputs)

    assert summary["total_rows"] == 3
    assert summary["pass_count"] == 2
    assert summary["pass_rate"] == 2 / 3


def test_promptflow_run_outputs_path_points_at_local_run_dir(tmp_path):
    import scripts.promptflow_eval_runner as runner

    run_id = "run123"
    expected = tmp_path / ".promptflow" / ".runs" / run_id / "outputs.jsonl"
    expected.parent.mkdir(parents=True)
    expected.write_text("{}\n", encoding="utf-8")

    assert runner.promptflow_run_outputs_path(run_id, home=tmp_path) == expected


def test_promptflow_eval_runner_nonblocking_on_promptflow_error(monkeypatch, tmp_path, capsys):
    import scripts.promptflow_eval_runner as runner
    import sys

    outputs_dir = tmp_path / "promptflow_out"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "outputs.jsonl").write_text('{"outputs":{"score":0.1}}\n', encoding="utf-8")

    def fake_run_promptflow(flow, data, run_id, output_dir):
        return {
            "available": True,
            "exit_code": 42,
            "stdout": "promptflow command failed",
            "stderr": "provider returned 500",
        }

    monkeypatch.setattr(runner, "run_promptflow", fake_run_promptflow)
    monkeypatch.setattr(sys, "argv", [
        "promptflow_eval_runner.py",
        "--flow",
        str(tmp_path / "smoke_flow"),
        "--run-id",
        "graceful_guard",
        "--output-dir",
        str(outputs_dir),
    ])

    assert runner.main() == 0
    output = capsys.readouterr().out
    assert "run_id=graceful_guard" in output
    assert "pass_rate=" in output

    receipt_line = next(line for line in output.splitlines() if line.startswith("run_id="))
    receipt_path = runner.ROOT / receipt_line.split("receipt=")[1]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["promptflow"]["exit_code"] == 42
    assert receipt["runtime_status"] == "alive"
    assert receipt["eval_quality"] == "unproven"


def test_promptflow_eval_runner_json_mode_is_pure_and_honest(monkeypatch, tmp_path, capsys):
    import scripts.promptflow_eval_runner as runner
    import sys

    outputs_dir = tmp_path / "promptflow_out_json"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "outputs.jsonl").write_text('{"outputs":{"score":0.8}}\n', encoding="utf-8")

    def fake_run_promptflow(flow, data, run_id, output_dir):
        return {
            "available": True,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(runner, "run_promptflow", fake_run_promptflow)
    monkeypatch.setattr(sys, "argv", [
        "promptflow_eval_runner.py",
        "--flow",
        str(tmp_path / "smoke_flow"),
        "--run-id",
        "json_guard",
        "--output-dir",
        str(outputs_dir),
        "--json",
    ])

    assert runner.main() == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["runtime_status"] == "alive"
    assert payload["eval_quality"] == "unproven"
    assert output.lstrip().startswith("{")
    assert output.rstrip().endswith("}")
    assert "run_id=json_guard" not in output
    assert "pass_rate=" not in output
