#!/usr/bin/env python3
"""Guardrail-preserving autonomous synthesis pass wrapper.

This does not replace the existing suite. It wraps native checks and the existing
acceptance harness into one bounded, receipt-backed execution pass.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "synthesis_pass"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def build_commands(*, full_ci: bool, case_id: str, acceptance_base_dir: str) -> list[list[str]]:
    commands: list[list[str]] = [
        [
            sys.executable,
            "-m",
            "py_compile",
            "scripts/lucidota_ouroboros_loop.py",
            "scripts/lucidota_synthesis_pass.py",
            "scripts/lucidota_pipeline_synthesis.py",
            "scripts/lucidota_acceptance.py",
            "scripts/lucidota_ci_gate.py",
            "scripts/tickletrunk_scan.py",
            "scripts/knowledge_library_check.py",
            "scripts/script_bucket_manifest.py",
            "scripts/script_survival_audit.py",
        ],
        [sys.executable, "scripts/tickletrunk_scan.py", "--check"],
        [sys.executable, "scripts/knowledge_library_check.py", "--check"],
        [sys.executable, "scripts/script_bucket_manifest.py", "--no-write"],
        [
            sys.executable,
            "scripts/lucidota_pipeline_synthesis.py",
            "--case-id",
            case_id,
            "--source-folder",
            f"{acceptance_base_dir}/_fixtures/{case_id}",
            "--base-dir",
            acceptance_base_dir,
            "--out-dir",
            "05_OUTPUTS/synthesis_pass/pipeline_synthesis",
        ],
        [
            sys.executable,
            "scripts/lucidota_acceptance.py",
            "--self-fixture",
            "--base-dir",
            acceptance_base_dir,
            "--case-id",
            case_id,
            "--json",
        ],
    ]
    if full_ci:
        commands.append([sys.executable, "scripts/lucidota_ci_gate.py", "--timeout-sec", "240"])
    return commands


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _receipt_markers(stdout: str) -> dict[str, str | None]:
    markers: dict[str, str | None] = {
        "report_path": None,
        "receipt_path": None,
        "markdown_path": None,
    }
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            markers["report_path"] = line.split("=", 1)[1]
        elif line.startswith("RECEIPT_PATH="):
            markers["receipt_path"] = line.split("=", 1)[1]
        elif line.startswith("MARKDOWN_PATH="):
            markers["markdown_path"] = line.split("=", 1)[1]
    return markers


def run_command(cmd: list[str], timeout_sec: int) -> dict[str, Any]:
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=None if timeout_sec <= 0 else timeout_sec,
        )
        stdout = _text(proc.stdout)
        stderr = _text(proc.stderr)
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _text(exc.stdout)
        stderr = (_text(exc.stderr) + f"\nTIMEOUT after {timeout_sec} seconds").strip()
        rc = 124
    markers = _receipt_markers(stdout)
    return {
        "cmd": shlex.join(cmd),
        "rc": rc,
        "report_path": markers["report_path"] or markers["receipt_path"],
        "receipt_path": markers["receipt_path"],
        "markdown_path": markers["markdown_path"],
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "timed_out": timed_out,
        "timeout_sec": timeout_sec,
        "passed": rc == 0,
    }


def _status_for(steps: list[dict[str, Any]], needles: list[str]) -> str:
    matched = [step for step in steps if any(needle in step.get("cmd", "") for needle in needles)]
    if not matched:
        return "NOT_RUN"
    failed = [step.get("cmd", "<unknown>") for step in matched if not step.get("passed")]
    if failed:
        return "FAIL: " + "; ".join(failed)
    return "PASS"


def phase_grid(steps: list[dict[str, Any]], *, full_ci: bool) -> list[dict[str, str]]:
    """Summarize the bounded pass in the user-facing Survey/Refine/Expand grid."""
    expand_needles = ["lucidota_pipeline_synthesis.py", "lucidota_acceptance.py"]
    if full_ci:
        expand_needles.append("lucidota_ci_gate.py --timeout-sec")
    child_reports = [rel(step["report_path"]) for step in steps if step.get("report_path")]
    report_delta = ", ".join(child_reports[-4:]) if child_reports else "child receipts not reported"
    return [
        {
            "phase": "Survey",
            "action_taken": "Mapped native readiness and guardrail surfaces through compile, TICKLETRUNK, and knowledge checks.",
            "system_delta": _status_for(
                steps,
                ["py_compile", "tickletrunk_scan.py", "knowledge_library_check.py"],
            ),
            "cathartic_resolution": "Infrastructure respected; guardrail configs were not modified by this wrapper.",
        },
        {
            "phase": "Refine",
            "action_taken": "Ran non-destructive script bucket manifest classification as a no-write impurity/friction check.",
            "system_delta": _status_for(steps, ["script_bucket_manifest.py"]),
            "cathartic_resolution": "No jungle paving or deletion performed; reuse pressure stayed higher than reinvention pressure.",
        },
        {
            "phase": "Expand",
            "action_taken": "Executed native pipeline synthesis and acceptance self-fixture proof, with full CI only when explicitly requested.",
            "system_delta": f"{_status_for(steps, expand_needles)}; child receipts: {report_delta}",
            "cathartic_resolution": "System embiggened by receipt-backed orchestration around existing suite entrypoints.",
        },
    ]


def blood_receipt(steps: list[dict[str, Any]], *, full_ci: bool) -> list[dict[str, str]]:
    """Summarize the bounded adversarial hardening pass without unbounded self-mutation."""
    blockers = [step.get("cmd", "<unknown>") for step in steps if not step.get("passed")]
    casualties = "No code-base casualties detected." if not blockers else f"{len(blockers)} blocker(s): " + "; ".join(blockers)
    ci_clause = " Full CI was included." if full_ci else " Full CI remained an explicit opt-in heavy gate."
    return [
        {
            "skirmish_id": "V2-OS-RED",
            "exploits_dreamed": "Command failure, timeout, missing child reports, guardrail drift, and unsafe write amplification.",
            "code_base_casualties": casualties,
            "hardening_resolution": (
                "Every native command is isolated, return-code checked, tail-logged, timeout-bounded, and wrapped with no DB or "
                "canonical graph writes requested."
            ),
        },
        {
            "skirmish_id": "V2-OS-FIT",
            "exploits_dreamed": "Operator-facing receipt gaps, scattered child outputs, JSON-only opacity, and accidental pipeline reinvention.",
            "code_base_casualties": "No destructive liquidation; friction was consolidated into structured phase and blood receipts.",
            "hardening_resolution": (
                "Human-readable markdown and machine-readable JSON now share the same payload while preserving native entrypoints."
                + ci_clause
            ),
        },
    ]


def cognitive_audit(steps: list[dict[str, Any]], *, full_ci: bool) -> list[dict[str, str]]:
    """Machine-readable version of the Map/Cut/Trap mental audit matrix."""
    command_count = len(steps)
    blockers = [step.get("cmd", "<unknown>") for step in steps if not step.get("passed")]
    timeouts = [step.get("cmd", "<unknown>") for step in steps if step.get("timed_out")]
    timeout_delta = "none" if not timeouts else "; ".join(timeouts)
    return [
        {
            "cognitive_phase": "The Map",
            "validation_target": "Scope & Boundaries",
            "desired_outcome": (
                f"{command_count} native command(s) bounded; runtime writes limited to 05_OUTPUTS receipts; "
                f"full_ci_requested={bool(full_ci)}."
            ),
        },
        {
            "cognitive_phase": "The Cut",
            "validation_target": "Redundancy Sweep",
            "desired_outcome": (
                "Existing tools reused: TICKLETRUNK, knowledge library check, script bucket manifest, pipeline synthesis, "
                "and acceptance harness. No replacement pipeline introduced."
            ),
        },
        {
            "cognitive_phase": "The Trap",
            "validation_target": "Edge Case Fuzzing",
            "desired_outcome": (
                f"Blockers routed to receipt list: {len(blockers)}; timeout traps: {timeout_delta}; "
                "REPORT_PATH/RECEIPT_PATH/MARKDOWN_PATH aliases captured."
            ),
        },
    ]


def _md(value: Any) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LUCIDOTA Synthesis Pass Receipt",
        "",
        f"- Status: `{_md(payload.get('status', 'UNKNOWN'))}`",
        f"- Generated: `{_md(payload.get('generated_at', 'UNKNOWN'))}`",
        f"- Case ID: `{_md(payload.get('case_id', 'UNKNOWN'))}`",
        f"- Full CI requested: `{_md(payload.get('full_ci_requested', False))}`",
        "",
        "## Execution Receipt",
        "",
        "| Phase | Action Taken | System Delta | Cathartic Resolution |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload.get("phase_grid", []):
        lines.append(
            "| {phase} | {action} | {delta} | {resolution} |".format(
                phase=_md(row.get("phase", "")),
                action=_md(row.get("action_taken", "")),
                delta=_md(row.get("system_delta", "")),
                resolution=_md(row.get("cathartic_resolution", "")),
            )
        )
    lines += [
        "",
        "## Mental Auditing Matrix",
        "",
        "| Cognitive Phase | Validation Target | Desired Outcome |",
        "| --- | --- | --- |",
    ]
    for row in payload.get("cognitive_audit", []):
        lines.append(
            "| {phase} | {target} | {outcome} |".format(
                phase=_md(row.get("cognitive_phase", "")),
                target=_md(row.get("validation_target", "")),
                outcome=_md(row.get("desired_outcome", "")),
            )
        )
    lines += [
        "",
        "## Blood-Receipt",
        "",
        "| Skirmish ID | Exploits Dreamed | Code-Base Casualties | Hardening Resolution |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload.get("blood_receipt", []):
        lines.append(
            "| {skirmish} | {exploits} | {casualties} | {resolution} |".format(
                skirmish=_md(row.get("skirmish_id", "")),
                exploits=_md(row.get("exploits_dreamed", "")),
                casualties=_md(row.get("code_base_casualties", "")),
                resolution=_md(row.get("hardening_resolution", "")),
            )
        )
    lines += [
        "",
        "## Step Results",
        "",
        "| Command | Status | Child Receipt |",
        "| --- | --- | --- |",
    ]
    for step in payload.get("steps", []):
        lines.append(
            "| {cmd} | {status} | {report} |".format(
                cmd=_md(step.get("cmd", "")),
                status="PASS" if step.get("passed") else f"FAIL rc={_md(step.get('rc', '?'))}",
                report=_md(step.get("report_path") or ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_receipt(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / f"lucidota_synthesis_pass_{stamp()}"
    json_path = base.with_suffix(".json")
    markdown_path = base.with_suffix(".md")
    payload["report_path"] = rel(json_path)
    payload["markdown_path"] = rel(markdown_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path


def _plan_payload(commands: list[list[str]]) -> dict[str, Any]:
    return {
        "schema": "lucidota.synthesis_pass.plan.v3",
        "steps": [shlex.join(c) for c in commands],
        "receipt_tables": ["phase_grid", "cognitive_audit", "blood_receipt"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded guardrail-preserving LUCIDOTA synthesis pass.")
    parser.add_argument("--case-id", default="synthesis-pass")
    parser.add_argument("--acceptance-base-dir", default="05_OUTPUTS/synthesis_pass/acceptance_cases")
    parser.add_argument("--full-ci", action="store_true", help="Append the heavier native CI gate after the bounded pass.")
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--list-steps", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    commands = build_commands(full_ci=args.full_ci, case_id=args.case_id, acceptance_base_dir=args.acceptance_base_dir)
    if args.list_steps:
        print(json.dumps(_plan_payload(commands), indent=2))
        return 0
    steps = [run_command(cmd, args.timeout_sec) for cmd in commands]
    blockers = [s["cmd"] for s in steps if s["rc"] != 0]
    payload = {
        "schema": "lucidota.synthesis_pass.v3",
        "action": "guardrail_preserving_synthesis_pass",
        "generated_at": now(),
        "case_id": args.case_id,
        "acceptance_base_dir": args.acceptance_base_dir,
        "full_ci_requested": bool(args.full_ci),
        "guardrail_configs_modified_by_this_wrapper": False,
        "canonical_graph_writes_requested": False,
        "db_writes_requested": False,
        "steps": steps,
        "blockers": blockers,
        "phase_grid": phase_grid(steps, full_ci=args.full_ci),
        "cognitive_audit": cognitive_audit(steps, full_ci=args.full_ci),
        "blood_receipt": blood_receipt(steps, full_ci=args.full_ci),
        "status": "PASS" if not blockers else "FAIL",
    }
    path = write_receipt(payload)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    print("REPORT_PATH=" + rel(path))
    print("LUCIDOTA_SYNTHESIS_PASS=" + payload["status"])
    return 0 if not blockers else 4


if __name__ == "__main__":
    raise SystemExit(main())
