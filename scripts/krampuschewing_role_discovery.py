#!/usr/bin/env python3
"""KRAMPUSCHEWING executable-spine / role discovery.

Read-only: does not start, stop, restart, move files, delete files, train models,
or materialize canonical graph rows.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "krampuschewing"
SERVICE = Path("/home/mfspx/.config/systemd/user/krampuschewing.service")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def run_read(cmd: list[str], *, timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {
            "command": " ".join(cmd),
            "rc": proc.returncode,
            "stdout_tail": proc.stdout[-12000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except Exception as exc:
        return {"command": " ".join(cmd), "rc": 124, "stdout_tail": "", "stderr_tail": str(exc)}


def read_text(path: Path, max_chars: int = 200_000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max_chars)
    except Exception:
        return ""


def parse_exec_start(service_text: str) -> str | None:
    for line in service_text.splitlines():
        if line.startswith("ExecStart="):
            return line.split("=", 1)[1].strip()
    return None


def grep_file(path: Path, patterns: list[str], limit: int = 50) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    text = read_text(path, max_chars=1_000_000)
    for idx, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        for pat in patterns:
            if pat.lower() in low:
                matches.append({"path": rel(path), "line": idx, "pattern": pat, "text": line[:500]})
                break
        if len(matches) >= limit:
            break
    return matches


def find_scripts() -> list[str]:
    names = [
        "scripts/krampuschewing_watcher.sh",
        "scripts/korpus_krampii.py",
        "scripts/krampus_rechronologize.py",
        "scripts/krampus_time_machine.py",
        "scripts/chronological_migrator.sh",
        "scripts/krampus_chrono_ledger.py",
        "scripts/krampus_bounded_inventory.py",
        "scripts/krampus_inventory.py",
        "scripts/spine_krampus_worker.py",
        "scripts/lucidota_brain_ingest.py",
        "scripts/lucidota_decision_delta.py",
    ]
    for p in sorted((ROOT / "scripts").glob("*krampus*")):
        r = rel(p)
        if r not in names:
            names.append(r)
    return [n for n in names if (ROOT / n).exists()]


def find_reports() -> list[str]:
    reports: list[str] = []
    for base in [ROOT / "05_OUTPUTS", ROOT / "04_RUNTIME", ROOT / "09_STORAGE"]:
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            low = str(p).lower()
            if p.is_file() and ("krampus" in low or "chewing" in low):
                reports.append(rel(p))
                if len(reports) >= 300:
                    return reports
    return reports


def rg_mentions(paths: list[Path], regex: str, limit: int = 120) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pat = re.compile(regex, re.I)
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        text = read_text(path, max_chars=500_000)
        for idx, line in enumerate(text.splitlines(), 1):
            if pat.search(line):
                out.append({"path": rel(path), "line": idx, "text": line[:500]})
                if len(out) >= limit:
                    return out
    return out


def build_receipt(root_arg: str) -> dict[str, Any]:
    root = (ROOT / root_arg).resolve() if not Path(root_arg).is_absolute() else Path(root_arg).resolve()
    service_text = read_text(SERVICE)
    exec_start = parse_exec_start(service_text)
    status = run_read(["systemctl", "--user", "status", "krampuschewing.service", "--no-pager"], timeout=10)
    journal = run_read(["journalctl", "--user", "-u", "krampuschewing.service", "--no-pager", "-n", "250"], timeout=20)
    scripts = find_scripts()
    script_paths = [ROOT / s for s in scripts]
    log_paths = [ROOT / "04_RUNTIME" / "krampuschewing_watcher.log", ROOT / "04_RUNTIME" / "krampus_time_machine.log"]
    search_paths = script_paths + [p for p in log_paths if p.exists()]
    graph_barrier_hits = []
    for p in search_paths:
        graph_barrier_hits.extend(grep_file(p, ["enforce_graph_promotion_path", "direct canonical graph write blocked", "graph_promotion_path"], limit=30))
    river_mentions = rg_mentions(search_paths, r"\bRiver\b|DBSTREAM|river_model|vibe_telemetry|brain_map", limit=80)
    diff_mentions = rg_mentions(search_paths, r"\bdiff\b|patch|changed files|dev cycle|delta|decision_delta", limit=80)
    case_mentions = rg_mentions(search_paths, r"\bcase\b|RTB|BCFSA|tenant|landlord|evidence|forensic|chrono|timeline", limit=80)
    rechrono_emitters = rg_mentions(script_paths, r"KRAMPUS FORENSIC RECHRONO|RECHRONO", limit=40)
    decision_ingest_reports = [rel(p) for p in sorted((ROOT / "05_OUTPUTS").rglob("*krampuschewing*decision*ingest*report*.json"))]
    blockers: list[str] = []
    if not SERVICE.exists():
        blockers.append("krampuschewing_service_unit_missing")
    if not root.exists():
        blockers.append("krampuschewing_root_missing")
    if not exec_start:
        blockers.append("service_exec_start_missing")
    return {
        "schema": "lucidota.krampuschewing.spine_discovery.v1",
        "generated_at_utc": now(),
        "repo_root": str(ROOT),
        "selected_root": str(root),
        "service_unit_path": str(SERVICE),
        "service_unit_text": service_text,
        "exec_start": exec_start,
        "service_status": status,
        "recent_journal_tail": journal,
        "likely_entrypoints": [exec_start] if exec_start else [],
        "daemon_entrypoint": "scripts/krampuschewing_watcher.sh" if (ROOT / "scripts/krampuschewing_watcher.sh").exists() else None,
        "scripts_found": scripts,
        "reports_found": find_reports(),
        "decision_ingest_reports_found": decision_ingest_reports,
        "scripts_that_emit_rechrono": rechrono_emitters,
        "scripts_that_emit_decision_ingest_report": [],
        "graph_barrier_hits": graph_barrier_hits,
        "river_mentions": river_mentions,
        "diff_mentions": diff_mentions,
        "case_mentions": case_mentions,
        "input_dirs": [rel(root)],
        "output_dirs": ["05_OUTPUTS/korpus_krampii", "05_OUTPUTS/krampus_inventory", "05_OUTPUTS/absurd", "03_VAULT/korpus_krampii/DIGESTED", "09_STORAGE/graph_staging_candidates"],
        "databases_touched": ["lucidota_korpus", "lucidota_chatdump", "lucidota_commdump", "lucidota_control", "lucidota_go attempted by child path and blocked"],
        "safe_to_run_now": [
            "python3 scripts/krampus_bounded_inventory.py --target KRAMPUSCHEWING --dry-run --max-files 500 --max-bytes 1048576",
            "python3 scripts/spine_krampus_worker.py --action audit --dry-run",
            "python3 scripts/krampuschewing_master_index.py --root KRAMPUSCHEWING",
        ],
        "unsafe_to_run_now": [
            "systemctl --user restart krampuschewing.service",
            "bash scripts/krampuschewing_watcher.sh",
            "python3 scripts/korpus_krampii.py --json ingest KRAMPUSCHEWING without storage-only/graph gate",
            "python3 scripts/krampus_time_machine.py --json run KRAMPUSCHEWING",
            "graph materialization helpers",
        ],
        "blockers": blockers,
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "files_moved": [],
        "files_deleted": [],
        "verdict": "PASS" if not blockers else "PARTIAL_FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="KRAMPUSCHEWING spine/role discovery (read-only)")
    ap.add_argument("--root", default="KRAMPUSCHEWING")
    ap.add_argument("--output")
    args = ap.parse_args()
    payload = build_receipt(args.root)
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else OUT / f"krampuschewing_spine_discovery_{stamp()}.json"
    if not out.is_absolute():
        out = ROOT / out
    payload["report_path"] = rel(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(out))
    print("KRAMPUSCHEWING_SPINE_DISCOVERY=" + payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
