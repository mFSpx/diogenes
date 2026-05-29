#!/usr/bin/env python3
"""Compile launch gaps into executable workflow cards; no DB/graph writes."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "05_OUTPUTS/status_ledger.json"
OUT = ROOT / "05_OUTPUTS/gaps"
MD = ROOT / "00_PROJECT_BRAIN/LUCIDOTA_GAPS_ACTION_WORKFLOWS.md"
ACTIVE_ROOT_LIMIT = 100


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def sha_key(x: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def loc(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
    except Exception:
        return 0


def workflow(source: str, title: str, gap: str, action: str, gate: str, severity: str = "high", commands: list[str] | None = None) -> dict[str, Any]:
    payload = {"source": source, "title": title, "gap": gap, "action": action, "gate": gate}
    return {
        "workflow_id": "gap:" + sha_key(payload),
        "source": source,
        "severity": severity,
        "title": title,
        "gap": gap,
        "action": action,
        "closure_gate": gate,
        "commands": commands or [],
        "status": "queued",
    }


def ledger_gaps(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for b in data.get("open_blockers", []):
        rows.append(workflow("status_ledger.open_blockers", b.get("blocker_key", "blocker"), b.get("meaning", ""), b.get("next_action", ""), "blocker removed or downgraded with evidence", b.get("severity", "high")))
    for section in ("software", "hardware_runtime_targets", "phases", "plans_workstreams"):
        for e in data.get(section, []):
            if e.get("status") in {"blocked", "not_started", "scaffolded"}:
                rows.append(workflow(f"status_ledger.{section}", e["name"], e.get("blockers") or "not launch-complete", e.get("next_action", ""), "ledger entry reaches executed/verified or blocker is named harder", "medium"))
    return rows


def code_gaps() -> list[dict[str, Any]]:
    rows = []
    for p in sorted((ROOT / "scripts").iterdir()):
        if not p.is_file() or p.suffix not in {".py", ".sh"}:
            continue
        n = loc(p)
        if n > ACTIVE_ROOT_LIMIT:
            rows.append(workflow("active_root_loc_audit", rel(p), f"{n} nonblank LOC in active root; >100 LOC must justify itself", "split to data/params/schema, port stable core to Rust, or move preserved body to KRAMPUS/legacy behind a tiny wrapper", "active root file <=100 LOC or has written exception with proof", "high" if n > 300 else "medium", ["python3 scripts/slop_audit_law.py --paths " + rel(p)]))
    if any((ROOT / "scripts/legacy/ahoy").glob("ahoy_*.py")):
        rows.append(workflow("paused_lab", "AHOY paused-preserved", "AHOY must be kept but cannot affect smooth launch", "keep under scripts/legacy/ahoy + KRAMPUSCHEWING/Paused_AHOY; block launch when LUCIDOTA_AHOY_PAUSED=1", "root ahoy_*.py count is zero and fabric press reports ahoy_paused", "medium"))
    return rows


def render_md(report: dict[str, Any]) -> str:
    lines = ["# LUCIDOTA GAPS → ACTIONABLE WORKFLOWS", "", f"Generated: `{report['generated_at']}`", "", f"Total queued gaps: **{len(report['workflows'])}**", ""]
    for w in report["workflows"]:
        lines += [f"## {w['title']}", "", f"- Severity: `{w['severity']}`", f"- Source: `{w['source']}`", f"- Gap: {w['gap']}", f"- Action: {w['action']}", f"- Closure gate: {w['closure_gate']}", ""]
        if w["commands"]:
            lines += ["```bash", *w["commands"], "```", ""]
    return "\n".join(lines)


def main() -> int:
    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    workflows = ledger_gaps(data) + code_gaps()
    report = {"schema": "lucidota.gap_workflows.v1", "generated_at": now(), "db_writes_performed": False, "graph_writes_performed": False, "canonical_mutation_allowed": False, "workflows": workflows}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"gap_workflows_{stamp()}.json"
    report["report_path"] = rel(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    MD.write_text(render_md(report), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    print("GAP_WORKFLOWS=" + str(len(workflows)))
    return 0 if workflows else 2


if __name__ == "__main__":
    raise SystemExit(main())
