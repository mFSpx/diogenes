#!/usr/bin/env python3
"""Emit compact GOALS swarm packets from the current swarm brief."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "goals"
BRIEF = ROOT / "GOALS" / "SWARM_CURRENT_BRIEF.md"

if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.goal_agent_packet import build_packet  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_brief() -> str:
    return BRIEF.read_text(encoding="utf-8", errors="replace")


def parse_brief_sections(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        if line.startswith("### "):
            if current:
                sections.append(current)
            current = {"title": line[4:].strip(), "bullets": []}
        elif current and line.strip().startswith("- "):
            current["bullets"].append(line.strip()[2:].strip())
    if current:
        sections.append(current)
    return sections


def packet_for_section(section: dict[str, Any]) -> dict[str, Any]:
    title = str(section.get("title") or "")
    bullets = [str(x) for x in section.get("bullets") or []]
    task = f"{title}. " + " ".join(bullets[:2])
    files = []
    if "goal_scenario_batch" in title or "goal_scenario_compare" in title:
        files = ["scripts/goal_scenario_batch.py", "scripts/goal_scenario_compare.py", "GOALS/SWARM_CURRENT_BRIEF.md"]
    elif "language_router" in title:
        files = ["scripts/language_router.py", "GOALS/SWARM_CURRENT_BRIEF.md"]
    elif "goal_agent_packet" in title:
        files = ["scripts/goal_agent_packet.py", "GOALS/SWARM_CURRENT_BRIEF.md"]
    elif "model_output_contract_audit" in title:
        files = ["scripts/model_output_contract_audit.py", "GOALS/SWARM_CURRENT_BRIEF.md"]
    elif "absurd_queue_spine" in title or "absurd_consume_one" in title:
        files = ["scripts/absurd_queue_spine.py", "scripts/absurd_consume_one.py", "GOALS/SWARM_CURRENT_BRIEF.md"]
    elif "korpus_embedding_worker" in title:
        files = ["scripts/korpus_embedding_worker.sh", "scripts/legacy/lucidota_indy_library_ingest.py", "GOALS/SWARM_CURRENT_BRIEF.md"]
    complexity = "repeat" if any(k in title for k in ("compare", "audit")) else "standard"
    target = "local" if "korpus" in title or "queue" in title else "groq|local"
    packet = build_packet(target=target, task=task, files=files, complexity=complexity, checks=["run focused tests", "name receipt paths"])
    packet["output_contract"] = packet.get("model_policy", {}).get("output_contract") or packet.get("coding_prompt", {}).get("output_contract")
    packet["swarm_section"] = title
    packet["swarm_bullets"] = bullets
    packet["target"] = target
    packet["task"] = task
    packet["complexity"] = complexity
    return packet


def build_swarm_brief() -> dict[str, Any]:
    text = read_brief()
    sections = parse_brief_sections(text)
    packets = [packet_for_section(section) for section in sections]
    return {
        "schema": "lucidota.goals.swarm_brief.v1",
        "generated_at": now(),
        "objective": "Turn the current swarm brief into bounded worker packets for new sessions.",
        "intent": "swarm_brief_emit",
        "ontology_mode": "GO25_STRICT",
        "ontology_terms": ["OBJECT", "EVENT", "EDGE"],
        "evidence_refs": [rel(BRIEF)],
        "status": "PASS",
        "next_action": "Use the emitted packets to launch bounded follow-on sessions or queue them as needed.",
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
        "brief_path": rel(BRIEF),
        "packet_count": len(packets),
        "packets": packets,
    }


def write_report(report: dict[str, Any], *, prefix: str = "goal_swarm_brief") -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{prefix}_{stamp()}.json"
    report["receipt_path"] = rel(path)
    report["report_path"] = report["receipt_path"]
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def launch_swarm_brief(report: dict[str, Any], *, queue: str = "goal_swarm", workflow: str = "goal_swarm", limit: int | None = None) -> dict[str, Any]:
    packets = list(report.get("packets") or [])
    if limit is not None:
        packets = packets[: max(0, limit)]
    launches: list[dict[str, Any]] = []
    for packet in packets:
        cmd = [
            sys.executable,
            "scripts/goal_swarm_dispatch.py",
            "--target",
            str(packet.get("target") or "generic"),
            "--task",
            str(packet.get("task") or ""),
            "--complexity",
            str(packet.get("complexity") or "standard"),
            "--queue",
            queue,
            "--workflow",
            workflow,
            "--jobs",
            "1",
            "--json",
        ]
        for path in packet.get("files") or []:
            cmd.extend(["--file", str(path)])
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        stdout_lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        dispatch_report_path = next((line.split("=", 1)[1].strip() for line in stdout_lines if line.startswith("REPORT_PATH=")), "")
        dispatch_report = {}
        if dispatch_report_path:
            try:
                dispatch_report = json.loads((ROOT / dispatch_report_path).read_text(encoding="utf-8"))
            except Exception:
                dispatch_report = {}
        launches.append(
            {
                "target": packet.get("target"),
                "task": packet.get("task"),
                "complexity": packet.get("complexity"),
                "dispatch_report_path": dispatch_report_path,
                "dispatch_job_uuids": [str(job.get("job_uuid")) for job in dispatch_report.get("jobs", []) if job.get("job_uuid")],
                "returncode": proc.returncode,
            }
        )
    launch_report = {
        "schema": "lucidota.goals.swarm_brief_launch.v1",
        "generated_at": now(),
        "objective": "Launch bounded swarm packets into queued follow-on sessions.",
        "intent": "swarm_brief_launch",
        "ontology_mode": "GO25_STRICT",
        "ontology_terms": ["OBJECT", "EVENT", "EDGE"],
        "evidence_refs": [report.get("receipt_path") or report.get("report_path") or rel(BRIEF)],
        "status": "PASS" if all(item["returncode"] == 0 for item in launches) else "FAIL",
        "next_action": "Consume the queued follow-on sessions or tighten the failing packet.",
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
        "brief_receipt_path": report.get("receipt_path") or report.get("report_path"),
        "queue": queue,
        "workflow": workflow,
        "packet_count": len(packets),
        "launch_count": len(launches),
        "launches": launches,
    }
    return write_report(launch_report, prefix="goal_swarm_brief_launch")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--queue", default="goal_swarm")
    ap.add_argument("--workflow", default="goal_swarm")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = write_report(build_swarm_brief())
    if args.launch:
        report = launch_swarm_brief(report, queue=args.queue, workflow=args.workflow, limit=args.limit)
    print("REPORT_PATH=" + report["report_path"])
    print("GOAL_SWARM_BRIEF=PASS" if report["status"] == "PASS" else "GOAL_SWARM_BRIEF=FAIL")
    print(json.dumps(report, sort_keys=True) if args.json else report["report_path"])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
