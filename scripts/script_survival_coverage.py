#!/usr/bin/env python3
"""Report script survival-audit coverage against TICKLETRUNK script entries."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TICKLETRUNK = ROOT / "00_PROJECT_BRAIN" / "TICKLETRUNK.json"
AUDIT = ROOT / "scripts" / "SCRIPT_AUDIT_MANIFEST.jsonl"
OUT = ROOT / "05_OUTPUTS" / "script_audit"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def tickletrunk_scripts() -> set[str]:
    data = json.loads(TICKLETRUNK.read_text(encoding="utf-8"))
    rows = data.get("toolboxes", {}).get("SCRIPTS", [])
    out: set[str] = set()
    for row in rows:
        p = row.get("relative_path") or row.get("path")
        if p:
            out.add(rel(p))
    return out


def audited_scripts() -> set[str]:
    out: set[str] = set()
    if not AUDIT.exists():
        return out
    with AUDIT.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = row.get("path")
            if p:
                out.add(rel(p))
    return out


def build(strict: bool) -> dict[str, Any]:
    scripts = tickletrunk_scripts()
    audited = audited_scripts()
    missing = sorted(scripts - audited)
    extra = sorted(audited - scripts)
    coverage = (len(audited & scripts) / len(scripts)) if scripts else 1.0
    blockers = []
    if strict and missing:
        blockers.append("script_survival_coverage_incomplete")
    return {
        "schema": "lucidota.script_survival_coverage.v1",
        "generated_at": now(),
        "tickletrunk_path": rel(TICKLETRUNK),
        "audit_manifest": rel(AUDIT),
        "tickletrunk_scripts": len(scripts),
        "audited_tickletrunk_scripts": len(audited & scripts),
        "audited_not_in_tickletrunk": len(extra),
        "unaudited_tickletrunk_scripts": len(missing),
        "coverage_ratio": round(coverage, 6),
        "strict": strict,
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "unaudited_sample": missing[:100],
        "extra_audited_sample": extra[:100],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure script survival audit coverage against TICKLETRUNK SCRIPTS.")
    ap.add_argument("--strict", action="store_true", help="Fail if any TICKLETRUNK script remains unaudited.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    payload = build(args.strict)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"script_survival_coverage_{stamp()}.json"
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    print("REPORT_PATH=" + rel(path))
    print("SCRIPT_SURVIVAL_COVERAGE=" + payload["status"])
    print("TICKLETRUNK_SCRIPTS=" + str(payload["tickletrunk_scripts"]))
    print("UNAUDITED_TICKLETRUNK_SCRIPTS=" + str(payload["unaudited_tickletrunk_scripts"]))
    return 0 if payload["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
