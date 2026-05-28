#!/usr/bin/env python3
"""Check that ABSURD is current and DBOS is legacy-compatibility only."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "00_PROJECT_BRAIN" / "DURABLE_WORKFLOW_DECISION.json"
OUT = ROOT / "05_OUTPUTS" / "durable_workflow"

BAD_DBOS_PATTERNS = [
    re.compile(r"\bDBOS\b[^\n]{0,100}\b(is|as|=|:)\s*(the\s+)?(canonical|current|future|selected)\b", re.I),
    re.compile(r"\b(canonical|current|future|selected)\b[^\n]{0,100}\bDBOS\b", re.I),
    re.compile(r"\bdurable[_ -]?workflow[_ -]?substrate\b\s*[:=]\s*[\"']?DBOS[\"']?", re.I),
    re.compile(r"\bnew[_ -]?work[_ -]?should[_ -]?target\b\s*[:=]\s*[\"']?DBOS\b", re.I),
    re.compile(r"\bnew\s+DBOS\s+architecture\s+allowed\b\s*[:=]\s*true", re.I),
]

SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "target", "__pycache__", ".pytest_cache",
    "01_REPOS", "03_VAULT", "04_RUNTIME", "05_OUTPUTS", "09_STORAGE", "KRAMPUSCHEWING",
    "BOOKS", "reference_only", ".lucidota_agents",
}
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".py", ".sh", ".toml", ".yaml", ".yml", ".sql"}
MAX_SCAN_BYTES = 2_000_000
LEGACY_DBOS_PATH_MARKERS = (
    "dbos", "DBOS", "legacy", "migration", "test_", "fixtures",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_decision(blockers: list[str], files_read: list[str]) -> dict[str, Any]:
    files_read.append(rel(DECISION))
    try:
        data = json.loads(DECISION.read_text(encoding="utf-8"))
    except Exception as exc:
        blockers.append(f"decision_file_unreadable:{exc}")
        return {}
    if not isinstance(data, dict):
        blockers.append("decision_file_not_object")
        return {}
    cur = data.get("current_decision") if isinstance(data.get("current_decision"), dict) else {}
    if data.get("schema") != "lucidota.durable_workflow_decision.v1":
        blockers.append("invalid_decision_schema")
    if cur.get("durable_workflow_substrate") != "ABSURD":
        blockers.append("absurd_not_canonical")
    if cur.get("dbos_status") != "legacy_compatibility_only":
        blockers.append("dbos_not_legacy_only")
    if cur.get("new_dbos_architecture_allowed") is not False:
        blockers.append("new_dbos_architecture_not_disabled")
    if cur.get("new_work_should_target") != "ABSURD/Postgres":
        blockers.append("new_work_target_not_absurd_postgres")
    return data


def iter_active_text_files() -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        base = Path(dirpath)
        rel_parts = set(base.relative_to(ROOT).parts) if base != ROOT else set()
        if rel_parts & SKIP_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = base / name
            if p.suffix not in TEXT_SUFFIXES and p.name not in {"claw", "check_diogenes.sh"}:
                continue
            try:
                if p.stat().st_size > MAX_SCAN_BYTES:
                    continue
            except OSError:
                continue
            out.append(p)
    return out


def allowed_legacy_context(path: Path, line: str) -> bool:
    r = rel(path)
    if r == "00_PROJECT_BRAIN/DURABLE_WORKFLOW_DECISION.json":
        return True
    if r == "scripts/durable_workflow_decision_check.py":
        return True
    lower_path = r.lower()
    lower_line = line.lower()
    if any(marker.lower() in lower_path for marker in LEGACY_DBOS_PATH_MARKERS):
        return True
    if "legacy" in lower_line or "old script" in lower_line or "existing receipt" in lower_line or "legacy-only" in lower_line:
        return True
    return False


def scan_dbos_canonical_claims() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    blocked: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    files_read: list[str] = []
    for path in iter_active_text_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        relp = rel(path)
        for lineno, line in enumerate(text.splitlines(), 1):
            if "DBOS" not in line and "dbos" not in line:
                continue
            for pat in BAD_DBOS_PATTERNS:
                if pat.search(line):
                    item = {"path": relp, "line": lineno, "text": line.strip()[:500], "pattern": pat.pattern}
                    if allowed_legacy_context(path, line):
                        allowed.append(item)
                    else:
                        blocked.append(item)
                    files_read.append(relp)
                    break
    return blocked, allowed, sorted(set(files_read))


def build_receipt() -> dict[str, Any]:
    blockers: list[str] = []
    files_read: list[str] = []
    decision = load_decision(blockers, files_read)
    blocked, allowed, scan_files = scan_dbos_canonical_claims()
    files_read.extend(scan_files)
    for item in blocked:
        blockers.append(f"active_dbos_canonical_claim:{item['path']}:{item['line']}")
    verdict = "PASS" if not blockers else "FAIL"
    return {
        "schema": "lucidota.durable_workflow_decision_check.v1",
        "generated_at": now(),
        "verdict": verdict,
        "durable_workflow_substrate": (decision.get("current_decision") or {}).get("durable_workflow_substrate"),
        "dbos_status": (decision.get("current_decision") or {}).get("dbos_status"),
        "new_dbos_architecture_allowed": (decision.get("current_decision") or {}).get("new_dbos_architecture_allowed"),
        "new_work_should_target": (decision.get("current_decision") or {}).get("new_work_should_target"),
        "blocked_matches": blocked,
        "allowed_legacy_matches": allowed,
        "blockers": blockers,
        "files_read": sorted(set(files_read)),
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "canonical_graph_writes_performed": False,
        "db_writes_performed": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate ABSURD current / DBOS legacy durable workflow decision")
    ap.add_argument("--output")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else OUT / f"durable_workflow_decision_check_{stamp()}.json"
    payload = build_receipt()
    payload["report_path"] = rel(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(out))
    print("DURABLE_WORKFLOW_DECISION_CHECK=" + payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
