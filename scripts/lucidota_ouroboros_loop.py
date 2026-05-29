#!/usr/bin/env python3
"""Bounded Ouroboros loop runner.

Inspects real repo targets, classifies them, runs the smallest safe validation
available for each target, and writes per-cycle JSONL plus a summary receipt.
It never deletes or mutates target artifacts; runtime writes stay under 05_OUTPUTS
unless a caller provides another receipt root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "ouroboros_loop"
ACTIVE_PATHS = {
    "scripts/lucidota_ouroboros_loop.py",
    "scripts/lucidota_synthesis_pass.py",
    "scripts/lucidota_synthesis_pass.sh",
    "scripts/lucidota_pipeline_synthesis.py",
    "scripts/lucidota_acceptance.py",
    "scripts/lucidota_ci_gate.py",
    "scripts/tickletrunk_scan.py",
    "scripts/knowledge_library_check.py",
    "scripts/script_bucket_manifest.py",
    "tests/test_lucidota_ouroboros_loop.py",
    "tests/test_lucidota_synthesis_pass.py",
}
SUPPORTING_PREFIXES = ("scripts/", "tests/", "06_SCHEMA/", "00_PROJECT_BRAIN/")
GENERATED_PREFIXES = ("05_OUTPUTS/", ".pytest_cache/", "__pycache__/")
RECEIPT_SUFFIXES = (".json", ".jsonl")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_path(path: str | Path) -> str:
    name = rel(path)
    lower = name.lower()
    if any(part in lower for part in ("/__pycache__/", "/.pytest_cache/")) or lower.endswith(".pyc"):
        return "GENERATED"
    if name.startswith(GENERATED_PREFIXES):
        return "RECEIPT" if lower.endswith(RECEIPT_SUFFIXES) else "GENERATED"
    if "/legacy/" in lower or lower.startswith("scripts/legacy/"):
        return "LEGACY_USEFUL"
    if any(token in lower for token in ("archive", "archived")):
        return "ARCHIVE"
    if name in ACTIVE_PATHS:
        return "ACTIVE"
    if any(name.startswith(prefix) for prefix in SUPPORTING_PREFIXES):
        return "SUPPORTING"
    return "UNKNOWN"


def discover_targets(limit: int | None = None) -> list[Path]:
    roots = [ROOT / "scripts", ROOT / "tests", ROOT / "06_SCHEMA"]
    suffixes = {".py", ".sh", ".sql", ".json"}
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                continue
            found.append(path)
    active = [resolve_repo_path(p) for p in sorted(ACTIVE_PATHS) if resolve_repo_path(p).exists()]
    ordered = list(dict.fromkeys(active + found))
    return ordered[:limit] if limit else ordered


def validation_command(path: Path) -> list[str] | None:
    relative = rel(path)
    if path.suffix == ".py":
        return [sys.executable, "-m", "py_compile", relative]
    if path.suffix == ".sh":
        return ["bash", "-n", relative]
    if path.suffix == ".json":
        return [sys.executable, "-m", "json.tool", relative]
    return None


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
        rc = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
        stderr_raw = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
        stderr = (stderr_raw + f"\nTIMEOUT after {timeout_sec} seconds").strip()
    return {
        "cmd": shlex.join(cmd),
        "rc": rc,
        "passed": rc == 0,
        "timed_out": timed_out,
        "stdout_tail": stdout[-1200:],
        "stderr_tail": stderr[-1200:],
    }


def inspect_target(path: Path) -> dict[str, Any]:
    exists = path.exists()
    text_head = ""
    if path.is_file():
        try:
            text_head = path.read_text(encoding="utf-8", errors="replace")[:600]
        except Exception as exc:  # defensive receipt detail only
            text_head = f"<unreadable:{exc}>"
    stat = path.stat() if exists else None
    return {
        "path": rel(path),
        "exists": exists,
        "size_bytes": stat.st_size if stat else None,
        "sha256": sha256_file(path),
        "classification": classify_path(path),
        "suffix": path.suffix,
        "has_shebang": text_head.startswith("#!"),
        "has_module_docstring": path.suffix == ".py" and ('"""' in text_head[:120] or "'''" in text_head[:120]),
        "has_strict_shell": path.suffix == ".sh" and "set -euo pipefail" in text_head[:240],
    }


def weaknesses_for(info: dict[str, Any], validation: dict[str, Any] | None) -> list[dict[str, str]]:
    weaknesses: list[dict[str, str]] = []
    path = info["path"]
    if not info["exists"]:
        weaknesses.append({"rank": "critical", "kind": "missing_target", "evidence": path})
        return weaknesses
    if info["classification"] == "UNKNOWN":
        weaknesses.append({"rank": "medium", "kind": "unknown_classification", "evidence": path})
    if info["suffix"] == ".sh" and not info["has_strict_shell"]:
        weaknesses.append({"rank": "medium", "kind": "shell_without_strict_error_handling", "evidence": path})
    if info["suffix"] == ".py" and not info["has_module_docstring"]:
        weaknesses.append({"rank": "low", "kind": "python_missing_top_docstring", "evidence": path})
    if validation and not validation["passed"]:
        weaknesses.append({"rank": "high", "kind": "validation_failed", "evidence": validation["cmd"]})
    if validation and validation.get("timed_out"):
        weaknesses.append({"rank": "high", "kind": "validation_timeout", "evidence": validation["cmd"]})
    return weaknesses


def cycle_record(index: int, total: int, path: Path, timeout_sec: int, next_path: Path | None) -> dict[str, Any]:
    info = inspect_target(path)
    cmd = validation_command(path) if info["exists"] else None
    validation = run_command(cmd, timeout_sec) if cmd else None
    weaknesses = weaknesses_for(info, validation)
    blockers = [w for w in weaknesses if w["rank"] in {"critical", "high"}]
    return {
        "schema": "lucidota.ouroboros_loop.cycle.v1",
        "loop": index,
        "loops_requested": total,
        "current_mode": "validation" if validation else "audit",
        "cycle_verdict": "PASS" if not blockers else "PARTIAL_PASS",
        "evidence_read": [info["path"], validation["cmd"] if validation else "metadata_only"],
        "target": info,
        "weaknesses_found": weaknesses,
        "changes_made": [],
        "validation": validation,
        "failures_blockers": blockers,
        "next_loop": rel(next_path) if next_path else "complete_batch",
        "generated_at": now(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_loops(*, loops: int, targets: list[Path], receipt_root: Path, timeout_sec: int) -> dict[str, Any]:
    if loops < 1:
        raise ValueError("loops_must_be_positive")
    if not targets:
        raise ValueError("no_targets_available")
    receipt_root.mkdir(parents=True, exist_ok=True)
    run_id = stamp()
    ledger_path = receipt_root / f"ouroboros_cycles_{run_id}.jsonl"
    weakness_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    validation_failures = 0
    cycles_written = 0
    sample_cycles: list[dict[str, Any]] = []
    with ledger_path.open("w", encoding="utf-8") as ledger:
        for i in range(1, loops + 1):
            path = targets[(i - 1) % len(targets)]
            next_path = targets[i % len(targets)] if i < loops else None
            record = cycle_record(i, loops, path, timeout_sec, next_path)
            ledger.write(json.dumps(record, sort_keys=True) + "\n")
            cycles_written += 1
            class_counter[record["target"]["classification"]] += 1
            for weakness in record["weaknesses_found"]:
                weakness_counter[weakness["kind"]] += 1
            if record.get("validation") and not record["validation"]["passed"]:
                validation_failures += 1
            if len(sample_cycles) < 5 or i > loops - 5:
                sample_cycles.append(record)
    status = "PASS" if validation_failures == 0 else "PARTIAL_PASS"
    summary = {
        "schema": "lucidota.ouroboros_loop.summary.v1",
        "action": "bounded_ouroboros_loop_batch",
        "generated_at": now(),
        "loops_requested": loops,
        "loops_executed": cycles_written,
        "targets_available": len(targets),
        "ledger_path": rel(ledger_path),
        "classification_counts": dict(sorted(class_counter.items())),
        "weakness_counts": dict(sorted(weakness_counter.items())),
        "validation_failures": validation_failures,
        "runtime_writes_limited_to_receipts": True,
        "destructive_actions_performed": False,
        "canonical_graph_writes_requested": False,
        "db_writes_requested": False,
        "sample_cycles": sample_cycles,
        "status": status,
    }
    report_path = receipt_root / f"ouroboros_summary_{run_id}.json"
    summary["report_path"] = rel(report_path)
    write_json(report_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded receipt-backed Ouroboros inspection/validation loops.")
    parser.add_argument("--loops", type=int, default=100)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--receipt-root", default=str(OUT))
    parser.add_argument("--target", action="append", default=[], help="Explicit target path; repeatable. Defaults to discovered repo targets.")
    parser.add_argument("--target-limit", type=int, default=0, help="Limit discovered targets before loop rotation.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    target_paths = [resolve_repo_path(t) for t in args.target] if args.target else discover_targets(args.target_limit or None)
    summary = run_loops(
        loops=args.loops,
        targets=target_paths,
        receipt_root=resolve_repo_path(args.receipt_root),
        timeout_sec=args.timeout_sec,
    )
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    print("REPORT_PATH=" + summary["report_path"])
    print("LEDGER_PATH=" + summary["ledger_path"])
    print("OUROBOROS_LOOP=" + summary["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
