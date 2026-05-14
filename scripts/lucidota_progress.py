#!/usr/bin/env python3
"""Print LUCIDOTA build progress bars from BUILD_PLAN_AUDIT.md."""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "00_PROJECT_BRAIN" / "BUILD_PLAN_AUDIT.md"

def main() -> int:
    text = PLAN.read_text(encoding="utf-8")
    overall = re.search(r"## Overall Build Bar: (.+)", text)
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
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
