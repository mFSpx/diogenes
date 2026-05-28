#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lucidota_synthesis_pass as lsp
from lucidota_synthesis_pass import blood_receipt, build_commands, cognitive_audit, phase_grid, render_markdown, run_command


def test_synthesis_pass_plan_reuses_existing_entrypoints():
    commands = build_commands(full_ci=False, case_id="case-x", acceptance_base_dir="05_OUTPUTS/x")
    flat = [" ".join(cmd) for cmd in commands]
    compile_step = flat[0]
    assert "py_compile" in compile_step
    assert "scripts/lucidota_ouroboros_loop.py" in compile_step
    assert "scripts/lucidota_synthesis_pass.py" in compile_step
    assert "scripts/script_bucket_manifest.py" in compile_step
    assert "scripts/script_survival_audit.py" in compile_step
    assert any("tickletrunk_scan.py --check" in cmd for cmd in flat)
    assert any("knowledge_library_check.py --check" in cmd for cmd in flat)
    assert any("lucidota_pipeline_synthesis.py" in cmd for cmd in flat)
    assert any("lucidota_acceptance.py --self-fixture" in cmd for cmd in flat)
    assert not any("lucidota_ci_gate.py --timeout-sec" in cmd for cmd in flat)


def test_synthesis_pass_plan_can_include_full_ci():
    commands = build_commands(full_ci=True, case_id="case-x", acceptance_base_dir="05_OUTPUTS/x")
    assert any("lucidota_ci_gate.py" in " ".join(cmd) for cmd in commands)


def test_shell_wrapper_has_strict_error_handling():
    text = (ROOT / "scripts" / "lucidota_synthesis_pass.sh").read_text(encoding="utf-8")
    assert text.splitlines()[1] == "set -euo pipefail"


def test_synthesis_pass_list_steps_cli():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/lucidota_synthesis_pass.py"), "--list-steps"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "lucidota.synthesis_pass.plan.v3" in proc.stdout
    assert "cognitive_audit" in proc.stdout
    assert "blood_receipt" in proc.stdout
    assert "lucidota_acceptance.py" in proc.stdout


def test_synthesis_pass_receipts_have_phase_cognitive_and_blood_tables():
    steps = [
        {"cmd": "python -m py_compile scripts/lucidota_synthesis_pass.py", "rc": 0, "passed": True},
        {"cmd": "python scripts/tickletrunk_scan.py --check", "rc": 0, "passed": True},
        {"cmd": "python scripts/knowledge_library_check.py --check", "rc": 0, "passed": True},
        {"cmd": "python scripts/script_bucket_manifest.py --no-write", "rc": 0, "passed": True},
        {
            "cmd": "python scripts/lucidota_pipeline_synthesis.py --case-id case-x",
            "rc": 0,
            "passed": True,
            "report_path": "05_OUTPUTS/synthesis_pass/pipeline_synthesis/case-x.json",
        },
        {
            "cmd": "python scripts/lucidota_acceptance.py --self-fixture --case-id case-x",
            "rc": 0,
            "passed": True,
            "report_path": "05_OUTPUTS/synthesis_pass/acceptance_cases/case-x/acceptance_result.json",
        },
    ]
    phases = phase_grid(steps, full_ci=False)
    cognitive = cognitive_audit(steps, full_ci=False)
    blood = blood_receipt(steps, full_ci=False)
    assert [row["phase"] for row in phases] == ["Survey", "Refine", "Expand"]
    assert [row["cognitive_phase"] for row in cognitive] == ["The Map", "The Cut", "The Trap"]
    assert [row["skirmish_id"] for row in blood] == ["V2-OS-RED", "V2-OS-FIT"]
    markdown = render_markdown(
        {
            "status": "PASS",
            "generated_at": "2026-05-21T00:00:00Z",
            "case_id": "case-x",
            "full_ci_requested": False,
            "phase_grid": phases,
            "cognitive_audit": cognitive,
            "blood_receipt": blood,
            "steps": steps,
        }
    )
    assert "| Phase | Action Taken | System Delta | Cathartic Resolution |" in markdown
    assert "| Cognitive Phase | Validation Target | Desired Outcome |" in markdown
    assert "| Skirmish ID | Exploits Dreamed | Code-Base Casualties | Hardening Resolution |" in markdown
    assert "The Trap" in markdown
    assert "V2-OS-RED" in markdown


def test_run_command_captures_receipt_path_aliases():
    result = run_command(
        [
            sys.executable,
            "-c",
            "print('RECEIPT_PATH=05_OUTPUTS/example.json'); print('MARKDOWN_PATH=05_OUTPUTS/example.md')",
        ],
        timeout_sec=30,
    )
    assert result["passed"]
    assert result["report_path"] == "05_OUTPUTS/example.json"
    assert result["receipt_path"] == "05_OUTPUTS/example.json"
    assert result["markdown_path"] == "05_OUTPUTS/example.md"


def test_run_command_receipt_command_is_shell_quoted():
    result = run_command([sys.executable, "-c", "print('ok')"], timeout_sec=30)
    assert result["passed"]
    assert "'print('\"'\"'ok'\"'\"')'" in result["cmd"]


def test_run_command_routes_timeout_as_failed_result(monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["python", "slow.py"],
            timeout=2,
            output="REPORT_PATH=05_OUTPUTS/partial.json\npartial",
            stderr="still running",
        )

    monkeypatch.setattr(lsp.subprocess, "run", fake_run)
    result = run_command([sys.executable, "slow.py"], timeout_sec=2)
    assert not result["passed"]
    assert result["timed_out"]
    assert result["rc"] == 124
    assert result["report_path"] == "05_OUTPUTS/partial.json"
    assert "TIMEOUT after 2 seconds" in result["stderr_tail"]
