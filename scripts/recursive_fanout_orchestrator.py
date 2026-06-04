#!/usr/bin/env python3
"""Emit an explicit recursive fanout plan using existing GOALS packet/dispatch surfaces."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "goals"
PYTHON = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.goal_agent_packet import build_packet  # noqa: E402

LANE_TOPICS = [
    "topology",
    "vibe_packet_shape",
    "groq_packet_shape",
    "dispatch_bridge",
    "receipt_surface",
    "verification_surface",
]
WORKER_SPECS = [
    ("vibe", "code_patch", "simple"),
    ("vibe", "test_patch", "simple"),
    ("groq", "code_review", "standard"),
    ("groq", "blocker_scan", "standard"),
]
OWNED_PATHS = ["scripts/recursive_fanout_orchestrator.py", "tests/test_recursive_fanout_orchestrator.py"]
CHECKS = [
    ".venv/bin/python -m pytest -q tests/test_recursive_fanout_orchestrator.py",
    "receipt JSON shows 6 mini-orchestrators and 24 total workers",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def worker_task(*, lane_id: str, topic: str, family: str, role: str) -> str:
    return (
        f"Recursive fanout lane {lane_id} / {topic}: {family} worker owns {role}. "
        "Return code, tests, or an exact blocker only. Reuse existing GOALS packet/dispatch surfaces."
    )


def dispatch_cmd(*, family: str, target: str, task: str, complexity: str, workflow: str) -> list[str]:
    cmd = [
        PYTHON,
        "scripts/goal_swarm_dispatch.py",
        "--target",
        target,
        "--task",
        task,
        "--complexity",
        complexity,
        "--workflow",
        workflow,
        "--queue",
        "goal_swarm",
        "--jobs",
        "1",
        "--json",
    ]
    for path in OWNED_PATHS:
        cmd.extend(["--file", path])
    if family == "groq":
        cmd.extend(
            [
                "--command",
                PYTHON,
                "scripts/groq_goal_delegate.py",
                "--task",
                task,
                "--kind",
                "code-slice",
                "--model",
                "llama-3.1-8b-instant",
                "--max-tokens",
                "512",
                "--json",
            ]
        )
    return cmd


def build_worker(*, lane_index: int, topic: str, family: str, role: str, complexity: str) -> dict[str, Any]:
    lane_id = f"lane_{lane_index:02d}"
    worker_id = f"{family}_{role}"
    task = worker_task(lane_id=lane_id, topic=topic, family=family, role=role)
    target = family
    return {
        "worker_id": worker_id,
        "family": family,
        "role": role,
        "owned_paths": OWNED_PATHS,
        "task": task,
        "packet": build_packet(target=target, task=task, files=OWNED_PATHS, complexity=complexity, checks=CHECKS),
        "dispatch_cmd": dispatch_cmd(
            family=family,
            target=target,
            task=task,
            complexity=complexity,
            workflow=f"recursive_fanout:{lane_id}:{worker_id}",
        ),
    }


def build_lane(*, lane_index: int, topic: str) -> dict[str, Any]:
    workers = [build_worker(lane_index=lane_index, topic=topic, family=family, role=role, complexity=complexity) for family, role, complexity in WORKER_SPECS]
    return {
        "mini_orchestrator_id": f"mini_orchestrator_{lane_index:02d}",
        "lane_id": f"lane_{lane_index:02d}",
        "topic": topic,
        "spawn_contract": {
            "worker_count": 4,
            "family_counts": {"vibe": 2, "groq": 2},
            "selection_rule": "choose_best_minimal_bundle",
            "packet_surface": "scripts/goal_agent_packet.py",
            "dispatch_surface": "scripts/goal_swarm_dispatch.py",
        },
        "workers": workers,
    }


def build_fanout_plan() -> dict[str, Any]:
    lanes = [build_lane(lane_index=i, topic=topic) for i, topic in enumerate(LANE_TOPICS, start=1)]
    return {
        "schema": "lucidota.recursive_fanout_orchestrator.v1",
        "generated_at": now(),
        "objective": "Make recursive fanout explicit for the DB-backed test receipt gate lane without inventing a daemon.",
        "owner": "MINI-ORCHESTRATOR A",
        "dispatch_surface": "scripts/goal_swarm_dispatch.py",
        "packet_surface": "scripts/goal_agent_packet.py",
        "mini_orchestrator_count": len(lanes),
        "worker_count": sum(len(lane["workers"]) for lane in lanes),
        "per_lane_worker_counts": [len(lane["workers"]) for lane in lanes],
        "mini_orchestrators": lanes,
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
    }


def write_receipt(plan: dict[str, Any], receipt: str | None = None) -> Path:
    path = Path(receipt) if receipt else OUT / f"recursive_fanout_orchestrator_{stamp()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    plan["report_path"] = rel(path)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    plan = build_fanout_plan()
    path = write_receipt(plan, receipt=args.receipt)
    print("REPORT_PATH=" + rel(path))
    print("RECURSIVE_FANOUT_ORCHESTRATOR=PASS")
    if args.json:
        print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
