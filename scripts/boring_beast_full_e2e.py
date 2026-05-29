#!/usr/bin/env python3
"""Full Boring Beast E2E verifier.

Runs the core Boring Beast E2E, proves duplicate suppression, verifies direct
graph writes stay blocked, derives status ledger facts, and emits one compact
machine report.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS/boring_beast"
STEP_MAP: dict[str, list[str]] = {
    "boring_beast_e2e": [sys.executable, "scripts/boring_beast.py", "e2e", "--execute"],
    "graph_write_blocker_probe": [sys.executable, "scripts/graph_write_blocker_probe.py"],
    "runtime_facts_refresh": [sys.executable, "scripts/system_runtime_facts_refresh.py", "--execute"],
    "status_ledger_check": [sys.executable, "scripts/lucidota_status_ledger.py", "--check"],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def safe_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("LUCIDOTA_OUTBOUND_STATE", "draft_only")
    env.setdefault("LUCIDOTA_EXTERNAL_WRITES", "draft_only")
    return env


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout, env=safe_env())
        return {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
            "passed": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(cmd),
            "returncode": 124,
            "stdout_tail": (exc.stdout or "")[-3000:],
            "stderr_tail": (exc.stderr or "")[-3000:],
            "passed": False,
        }


def command_exists(cmd: list[str]) -> bool:
    if len(cmd) < 2:
        return False
    target = ROOT / cmd[1]
    return target.exists() and target.is_file()


def main() -> int:
    ap = argparse.ArgumentParser(description="Bounded Boring Beast E2E verifier")
    ap.add_argument("--timeout-seconds", type=int, default=180)
    ap.add_argument("--steps", default="all", help="comma list of step names or 'all'")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()
    timeout = max(30, min(int(args.timeout_seconds), 900))
    selected_steps = list(STEP_MAP.keys()) if args.steps == "all" else [x.strip() for x in args.steps.split(",") if x.strip()]
    unknown = [x for x in selected_steps if x not in STEP_MAP]
    if unknown:
        print("UNKNOWN_STEPS=" + ",".join(unknown))
        return 2
    command_specs = [(name, STEP_MAP[name]) for name in selected_steps]
    commands: list[dict[str, Any]] = []
    per_step_timeout = max(30, timeout // max(1, len(command_specs)))
    for step_name, spec in command_specs:
        if not command_exists(spec):
            commands.append(
                {
                    "step": step_name,
                    "command": " ".join(spec),
                    "returncode": 127,
                    "stdout_tail": "",
                    "stderr_tail": "missing_script",
                    "passed": False,
                }
            )
            continue
        result = run(spec, timeout=per_step_timeout)
        result["step"] = step_name
        result["timeout_seconds"] = per_step_timeout
        commands.append(result)
        if args.fail_fast and not result["passed"]:
            break
    passed = all(c["passed"] for c in commands)
    report = {
        "schema": "lucidota.boring_beast.full_e2e.v1",
        "generated_at": now(),
        "execute_performed": True,
        "selected_steps": selected_steps,
        "fail_fast": bool(args.fail_fast),
        "timeout_seconds": timeout,
        "db_writes_performed": True,
        "graph_writes_performed": False,
        "script_sha256": sha256_file(Path(__file__)),
        "commands": commands,
        "pass": passed,
        "blockers": [] if passed else ["one_or_more_full_e2e_commands_failed"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"boring_beast_full_e2e_{stamp()}.json"
    report["report_path"] = rel(out)
    out.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(out)}")
    print("BORING_BEAST_FULL_E2E=" + ("PASS" if passed else "FAIL"))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
