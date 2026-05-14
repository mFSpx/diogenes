#!/usr/bin/env python3
"""DBOS wrapper for the Survey Protocol.

This makes Survey a durable workflow-shaped action without changing Survey's CLI.
DBOS is command brain; Survey is the tool; workflow_event remains the common event surface.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from dbos import DBOS, DBOSConfig

ROOT = Path(__file__).resolve().parents[1]


@DBOS.step()
def run_survey_step(args: list[str]) -> dict:
    cmd = [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "lucidota_survey.py"), *args]
    if not Path(cmd[0]).exists():
        cmd[0] = "python3"
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"survey exited {proc.returncode}")
    return json.loads(proc.stdout)


@DBOS.workflow()
def survey_workflow(args: list[str]) -> dict:
    result = run_survey_step(args)
    return {
        "target": result.get("target"),
        "decision": result.get("decision"),
        "sha256": result.get("sha256"),
        "pivots": len(result.get("pivot_candidates") or []),
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-dbos-survey")
    ap.add_argument("survey_args", nargs=argparse.REMAINDER)
    ns = ap.parse_args()
    config: DBOSConfig = {
        "name": "lucidota-dbos-survey",
        "system_database_url": os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state"),
    }
    DBOS(config=config)
    DBOS.launch()
    result = survey_workflow(ns.survey_args)
    print(json.dumps({"ok": True, "dbos_survey": result}, sort_keys=True))
    DBOS.destroy(destroy_registry=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
