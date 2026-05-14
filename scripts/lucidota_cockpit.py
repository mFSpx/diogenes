#!/usr/bin/env python3
"""One-screen LUCIDOTA cockpit: compact bars, counters, Indy queues, governor."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)

BAR_RE = re.compile(r"(?P<label>.*?)\s*(?P<bar>[█░]+)\s*(?P<pct>\d+%)\s*(?P<tail>.*)")
COUNTER_ORDER = [
    "workflow_events",
    "registered_workflows",
    "scheduled_workflows",
    "signoff_pending",
    "signoff_approved",
    "wake_pending",
    "wake_delivered",
    "cas_artifacts",
    "body_capture_captures",
    "body_capture_bundles",
    "river_scores",
    "bytewax_hints",
    "treelite_runs",
]


def j(script: str, *args: str) -> dict[str, Any]:
    cmd = [str(PY), str(ROOT / "scripts" / script), *args, "--json"]
    r = subprocess.run(cmd, text=True, capture_output=True, check=False)
    try:
        data = json.loads(r.stdout)
    except Exception:
        data = {"ok": False, "error": r.stderr[-300:] or r.stdout[-300:] or f"exit {r.returncode}"}
    data["_returncode"] = r.returncode
    return data


def parse_phase(line: str) -> dict[str, str]:
    m = BAR_RE.match(line)
    if not m:
        return {"label": line[:64], "bar": "", "pct": "?", "tail": ""}
    return {k: (m.group(k) or "").strip(" —") for k in ("label", "bar", "pct", "tail")}


def phase_key(phase: dict[str, str]) -> tuple[int, str]:
    pct = int(phase.get("pct", "0%").rstrip("%") or 0) if phase.get("pct", "").rstrip("%").isdigit() else 0
    return (pct, phase.get("label", ""))


def compact_counters(counters: dict[str, Any]) -> str:
    parts = []
    for key in COUNTER_ORDER:
        if key in counters:
            short = {
                "workflow_events": "wf",
                "registered_workflows": "flows",
                "scheduled_workflows": "sched",
                "signoff_pending": "sign_pending",
                "signoff_approved": "sign_ok",
                "wake_pending": "wake_pending",
                "wake_delivered": "wake_done",
                "cas_artifacts": "cas",
                "body_capture_captures": "captures",
                "body_capture_bundles": "bundles",
                "river_scores": "river",
                "bytewax_hints": "bytewax",
                "treelite_runs": "treelite",
            }[key]
            parts.append(f"{short}={counters[key]}")
    leftovers = [f"{k}={v}" for k, v in counters.items() if k not in COUNTER_ORDER]
    return "  ".join(parts + leftovers)


def scraper_line(big: dict[str, Any]) -> str:
    sf = big.get("scraper_fleet", {})
    scripts = len(sf.get("local_scripts", []))
    adapters = len(sf.get("authorized_adapters", []))
    targets = len(sf.get("manifest_scraper_targets", []))
    return (
        f"{sf.get('status', 'unknown')}  "
        f"scripts={scripts} adapters={adapters} "
        f"browser_required={sf.get('browser_required_adapters', '?')} manifest_targets={targets}"
    )


def governor_line(mg: dict[str, Any]) -> str:
    if not mg.get("ok"):
        return f"unavailable ({mg.get('error', 'no report')})"
    decision = mg.get("decision", "unknown")
    rationale = mg.get("rationale", "")
    gpu = mg.get("gpu", {})
    gpu_bits = ""
    if gpu.get("status") == "ok":
        gpu_bits = f" gpu={gpu.get('used_mb')}MB/{gpu.get('total_mb')}MB free={gpu.get('free_mb')}MB"
    elif gpu:
        gpu_bits = f" gpu={gpu.get('status')}"
    return f"{decision}  {rationale}{gpu_bits}".strip()


def priority_queue(queue: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rank = {"high": 0, "normal": 1, "low": 2}
    return sorted(queue, key=lambda q: (rank.get(q.get("urgency"), 9), q.get("created_at", "")), reverse=False)[:limit]


def build_report(limit: int, include_governor: bool) -> dict[str, Any]:
    big = j("lucidota_big_board.py")
    indy = j("lucidota_indy_brief.py")
    mg = j("lucidota_model_governor.py") if include_governor else {"ok": True, "decision": "skipped"}
    phases = [parse_phase(p) for p in big.get("bars", {}).get("phases", [])]
    lowest = sorted(phases, key=phase_key)[:limit]
    indy_queue = priority_queue(indy.get("queue", []), limit)
    report = {
        "ok": all(x.get("ok") for x in (big, indy)),
        "governor_ok": bool(mg.get("ok")),
        "big_board": big,
        "indy": indy,
        "model_governor": mg,
        "summary": {
            "overall": big.get("bars", {}).get("overall", "unknown"),
            "lowest_phases": lowest,
            "counters": big.get("counters", {}),
            "scraper_fleet": scraper_line(big),
            "indy_counters": indy.get("counters", {}),
            "indy_queue": indy_queue,
            "governor": governor_line(mg),
        },
    }
    return report


def render(report: dict[str, Any]) -> str:
    s = report["summary"]
    indy = report["indy"]
    mg = report["model_governor"]
    lines = [
        "LUCIDOTA COCKPIT",
        "================",
        f"Overall      {s['overall']}",
        f"Governor     {s['governor']}",
        f"Counters     {compact_counters(s['counters'])}",
        f"Scrapers     {s['scraper_fleet']}",
    ]
    ic = s.get("indy_counters", {})
    lines.append(
        "Indy         "
        f"{indy.get('indy_phase', {}).get('bar', 'unknown')}  "
        f"active_memory={ic.get('active_memory', '?')} quiet_queue={ic.get('quiet_queue', '?')} auth={ic.get('auth_records', '?')}"
    )
    lines += ["", "Lowest build bars:"]
    for phase in s["lowest_phases"]:
        label = phase["label"].replace(" / ", "/")
        lines.append(f"  {phase['pct']:>4} {phase['bar']}  {label}")
    lines += ["", "Indy queue:"]
    if s["indy_queue"]:
        for q in s["indy_queue"]:
            due = f" due={q['due_at']}" if q.get("due_at") else ""
            lines.append(f"  - {q.get('urgency','?')}/{q.get('item_type','?')}: {q.get('title','')}{due}")
    else:
        lines.append("  - none queued")
    if not mg.get("ok"):
        lines += ["", "Action: seed/check resident loadout before relying on governor decisions."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-cockpit")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5, help="rows to show for low bars and Indy queue")
    ap.add_argument("--no-governor", action="store_true", help="skip DB-writing model governor check")
    args = ap.parse_args()
    report = build_report(max(1, args.limit), not args.no_governor)
    print(json.dumps(report, sort_keys=True) if args.json else render(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
