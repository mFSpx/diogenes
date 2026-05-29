import json
from pathlib import Path

import pytest

from scripts.diogenes_memory_gate import (
    MEMORY_CEILING_KB,
    build_cases,
    evaluate_case,
    parse_time_max_rss_kb,
    run_gate,
)


def test_parse_time_max_rss_kb():
    stderr = "\tMaximum resident set size (kbytes): 45672\n"
    assert parse_time_max_rss_kb(stderr) == 45672
    assert parse_time_max_rss_kb("no rss here") is None


def test_evaluate_case_uses_worst_reported_rss():
    case = {
        "name": "packed",
        "stdout_json": {"peak_rss_kb": 28784},
        "time_max_rss_kb": 45672,
        "returncode": 0,
    }
    evaluated = evaluate_case(case, ceiling_kb=MEMORY_CEILING_KB)
    assert evaluated["effective_peak_rss_kb"] == 45672
    assert evaluated["passed"] is True
    assert evaluated["blockers"] == []

    failed = evaluate_case({**case, "time_max_rss_kb": 70000}, ceiling_kb=MEMORY_CEILING_KB)
    assert failed["passed"] is False
    assert "rss_over_ceiling" in failed["blockers"]


def test_build_cases_keeps_heavy_engines_outside_kernel_scope(tmp_path):
    cases = build_cases(tmp_path / "doggystyle", python_executable="python3")
    names = {case.name for case in cases}
    assert {"ckdog1_packed_5000x88_retain_packed", "ckdog1_soulless_50000", "percyphon_5000_villagers_88_fluid"} <= names
    all_commands = "\n".join(" ".join(case.command) for case in cases)
    assert "ollama" not in all_commands.lower()
    assert "deepseek" not in all_commands.lower()


def test_run_gate_writes_receipt_with_pass_fail(tmp_path):
    calls = []

    def fake_runner(case):
        calls.append(case.name)
        payload = {"peak_rss_kb": 30000, "ok": True}
        return {
            "name": case.name,
            "purpose": case.purpose,
            "command": case.command,
            "returncode": 0,
            "stdout": json.dumps(payload),
            "stderr": "Maximum resident set size (kbytes): 31000\n",
            "stdout_json": payload,
            "time_max_rss_kb": 31000,
        }

    receipt = run_gate(
        doggystyle_root=tmp_path / "doggystyle",
        python_executable="python3",
        receipt_dir=tmp_path,
        ceiling_kb=MEMORY_CEILING_KB,
        runner=fake_runner,
    )
    assert receipt["passed"] is True
    assert receipt["receipt_path"].startswith(str(tmp_path))
    assert Path(receipt["receipt_path"]).exists()
    assert len(receipt["cases"]) == len(calls)


def test_run_gate_fails_if_any_case_exceeds_ceiling(tmp_path):
    def fake_runner(case):
        rss = 60000 if case.name == "ckdog1_soulless_50000" else 30000
        payload = {"peak_rss_kb": rss}
        return {
            "name": case.name,
            "purpose": case.purpose,
            "command": case.command,
            "returncode": 0,
            "stdout": json.dumps(payload),
            "stderr": f"Maximum resident set size (kbytes): {rss}\n",
            "stdout_json": payload,
            "time_max_rss_kb": rss,
        }

    receipt = run_gate(
        doggystyle_root=tmp_path / "doggystyle",
        python_executable="python3",
        receipt_dir=tmp_path,
        ceiling_kb=MEMORY_CEILING_KB,
        runner=fake_runner,
    )
    assert receipt["passed"] is False
    failed = [case for case in receipt["cases"] if not case["passed"]]
    assert failed[0]["name"] == "ckdog1_soulless_50000"
    assert "rss_over_ceiling" in failed[0]["blockers"]
