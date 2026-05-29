#!/usr/bin/env python3
"""Print LUCIDOTA build progress.

Legacy BUILD_PLAN_AUDIT.md may be archived. Active fallback is the status ledger;
missing legacy files are not a crash condition.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "00_PROJECT_BRAIN" / "BUILD_PLAN_AUDIT.md"
LEDGER_JSON = ROOT / "05_OUTPUTS" / "status_ledger.json"


def bar(progress: int) -> str:
    progress = max(0, min(100, int(progress)))
    filled = progress // 10
    return "[" + "█" * filled + "░" * (10 - filled) + f"] {progress}%"


def print_legacy_plan(text: str) -> None:
    overall = re.search(r"## Overall Build Bar: (.+)", text)
    print("Source: BUILD_PLAN_AUDIT.md")
    print(f"Overall: {overall.group(1) if overall else 'unknown'}")
    print("Phases:")
    in_bars = False
    for line in text.splitlines():
        if line == "## Phase Bars":
            in_bars = True
            continue
        if in_bars and line.startswith("## "):
            break
        if in_bars and line.startswith("- **"):
            print(line.replace("- **", "  ").replace("** `", ": ").replace("`", ""))


def load_ledger() -> dict[str, Any]:
    if not LEDGER_JSON.exists():
        return {}
    return json.loads(LEDGER_JSON.read_text(encoding="utf-8"))


def rows_from_ledger(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("software", "hardware_runtime_targets", "phases", "plans_workstreams"):
        for row in data.get(key, []):
            if isinstance(row, dict):
                rows.append({**row, "section": key})
    return rows


def print_ledger_progress(data: dict[str, Any]) -> None:
    rows = rows_from_ledger(data)
    progress_values = [int(row.get("progress", 0) or 0) for row in rows]
    overall = round(sum(progress_values) / len(progress_values)) if progress_values else 0
    blockers = data.get("open_blockers", []) if isinstance(data.get("open_blockers", []), list) else []
    print("Source: status ledger")
    print(f"Overall: {bar(overall)} from {len(progress_values)} ledger rows")
    print("Top active rows:")
    active = [row for row in rows if str(row.get("status")) not in {"verified", "archived", "deprecated"}]
    for row in sorted(active, key=lambda r: int(r.get("progress", 0) or 0))[:12]:
        print(f"  {row.get('name', 'UNKNOWN')}: {row.get('loading_bar') or bar(row.get('progress', 0))} {row.get('status', '')}")
    print(f"Open blockers: {len(blockers)}")
    for blocker in blockers[:8]:
        print(f"  {blocker.get('severity', 'unknown')}: {blocker.get('blocker_key', 'UNKNOWN')} -> {blocker.get('next_action', '')}")


def main() -> int:
    if PLAN.exists():
        print_legacy_plan(PLAN.read_text(encoding="utf-8"))
        return 0
    data = load_ledger()
    if not data:
        print("Source: none")
        print("Overall: unknown")
        print("Open blockers: unknown")
        return 2
    print_ledger_progress(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
