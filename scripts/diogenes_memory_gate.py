#!/usr/bin/env python3
"""DIOGENES / CKDOG1 low-RAM gate.

Runs the local doggystyle CKDOG1 hot-path benches plus PercyphonAI's
zero-VRAM procedural scaffold and writes one compact receipt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
MEMORY_CEILING_KB = 50 * 1024
DEFAULT_DOGGYSTYLE_ROOT = ROOT / "01_REPOS" / "doggystyle"
DEFAULT_RECEIPT_DIR = ROOT / "05_OUTPUTS" / "diogenes"


@dataclass(frozen=True)
class GateCase:
    name: str
    purpose: str
    command: list[str]
    cwd: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def parse_time_max_rss_kb(stderr: str) -> int | None:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    return int(match.group(1)) if match else None


def _json_from_stdout(stdout: str) -> Any | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _percyphon_probe_code() -> str:
    return r'''
import json
import resource
from ALGOS.percyphon import procedural_entity_generator
villagers = [f"villager-{i:04d}" for i in range(5000)]
result = procedural_entity_generator(villagers, fluid_slots=88)
print(json.dumps({
    "ok": True,
    "schema": result["schema"],
    "zero_vram": result["zero_vram"],
    "source_count": result["source_count"],
    "slot_count": result["slot_count"],
    "fluid_slot_count": result["fluid_slot_count"],
    "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
}, sort_keys=True))
'''.strip()


def build_cases(doggystyle_root: Path, python_executable: str) -> list[GateCase]:
    doggystyle_root = Path(doggystyle_root)
    dog_python = doggystyle_root / ".venv" / "bin" / "python"
    dog_python_text = str(dog_python) if dog_python.exists() else python_executable
    return [
        GateCase(
            name="ckdog1_packed_5000x88_retain_packed",
            purpose="Full 5,000 soul x 88 slot packed-pointer load; retains packed bytes; no SQLite persistence.",
            command=[
                dog_python_text,
                "-m",
                "kernel.cli",
                "bench-packed",
                "--souls",
                "5000",
                "--slots-per-soul",
                "88",
                "--retain",
                "packed",
            ],
            cwd=doggystyle_root,
        ),
        GateCase(
            name="ckdog1_soulless_50000",
            purpose="Generate 50,000 background soulless nodes without persisting them as souls.",
            command=[
                dog_python_text,
                "-m",
                "kernel.cli",
                "bench-soulless",
                "--count",
                "50000",
                "--retain",
                "none",
            ],
            cwd=doggystyle_root,
        ),
        GateCase(
            name="percyphon_5000_villagers_88_fluid",
            purpose="PercyphonAI deterministic procedural scaffold: 5,000 villager seed refs, 12 core slots, 88 fluid slots, zero model weights.",
            command=[python_executable, "-c", _percyphon_probe_code()],
            cwd=ROOT,
        ),
    ]


def run_case(case: GateCase, *, timeout_seconds: int = 60) -> dict[str, Any]:
    time_bin = shutil.which("/usr/bin/time") or shutil.which("time")
    if time_bin:
        command = [time_bin, "-v", *case.command]
    else:
        command = case.command
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT))
    proc = subprocess.run(
        command,
        cwd=case.cwd,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=env,
    )
    stdout_json = _json_from_stdout(proc.stdout)
    return {
        "name": case.name,
        "purpose": case.purpose,
        "command": case.command,
        "cwd": str(case.cwd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_json": stdout_json,
        "time_max_rss_kb": parse_time_max_rss_kb(proc.stderr),
    }


def _internal_peak_rss_kb(stdout_json: Any) -> int | None:
    if isinstance(stdout_json, dict) and isinstance(stdout_json.get("peak_rss_kb"), int):
        return int(stdout_json["peak_rss_kb"])
    return None


def evaluate_case(case: dict[str, Any], *, ceiling_kb: int) -> dict[str, Any]:
    blockers: list[str] = []
    if case.get("returncode") != 0:
        blockers.append("command_returncode_nonzero")
    internal = _internal_peak_rss_kb(case.get("stdout_json"))
    external = case.get("time_max_rss_kb")
    observed = [value for value in (internal, external) if isinstance(value, int)]
    effective = max(observed) if observed else None
    if effective is None:
        blockers.append("rss_measurement_missing")
    elif effective > ceiling_kb:
        blockers.append("rss_over_ceiling")

    evaluated = dict(case)
    evaluated["internal_peak_rss_kb"] = internal
    evaluated["external_peak_rss_kb"] = external
    evaluated["effective_peak_rss_kb"] = effective
    evaluated["ceiling_kb"] = ceiling_kb
    evaluated["passed"] = not blockers
    evaluated["blockers"] = blockers
    # Keep receipts compact while preserving enough stderr/stdout to debug failures.
    if isinstance(evaluated.get("stdout"), str) and len(evaluated["stdout"]) > 4000:
        evaluated["stdout"] = evaluated["stdout"][:4000] + "...<truncated>"
    if isinstance(evaluated.get("stderr"), str) and len(evaluated["stderr"]) > 4000:
        evaluated["stderr"] = evaluated["stderr"][:4000] + "...<truncated>"
    return evaluated


def run_gate(
    *,
    doggystyle_root: Path = DEFAULT_DOGGYSTYLE_ROOT,
    python_executable: str = sys.executable,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
    ceiling_kb: int = MEMORY_CEILING_KB,
    runner: Callable[[GateCase], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    receipt_dir = Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    runner = runner or run_case
    cases = [evaluate_case(runner(case), ceiling_kb=ceiling_kb) for case in build_cases(Path(doggystyle_root), python_executable)]
    receipt_path = receipt_dir / f"diogenes_memory_gate_{stamp()}.json"
    receipt = {
        "schema": "lucidota.diogenes.memory_gate.v1",
        "generated_at": utc_now(),
        "ceiling_kb": ceiling_kb,
        "ceiling_mb": ceiling_kb / 1024,
        "doggystyle_root": str(Path(doggystyle_root)),
        "source_refs": [
            "https://gitlab.com/mfspx/doggystyle",
            "01_REPOS/doggystyle/README.md",
            "00_PROJECT_BRAIN/RFCS/RFC-050-DIOGENES-KERNEL.md",
            "00_PROJECT_BRAIN/RFCS/RFC-090-PERCYPHONAI.md",
            "ALGOS/percyphon.py",
        ],
        "scope": "CKDOG1 kernel hot paths and PercyphonAI procedural scaffold only; external engines require separate admission gates.",
        "external_engines_not_in_scope": ["River", "Bytewax", "DeepSeek/Ollama", "browser extraction workers"],
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
        "receipt_path": str(receipt_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DIOGENES/CKDOG1 <=50MB memory gate and write receipt.")
    parser.add_argument("--doggystyle-root", type=Path, default=DEFAULT_DOGGYSTYLE_ROOT)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    parser.add_argument("--ceiling-kb", type=int, default=MEMORY_CEILING_KB)
    args = parser.parse_args(list(argv) if argv is not None else None)
    receipt = run_gate(
        doggystyle_root=args.doggystyle_root,
        python_executable=args.python_executable,
        receipt_dir=args.receipt_dir,
        ceiling_kb=args.ceiling_kb,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
