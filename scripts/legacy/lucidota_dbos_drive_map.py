#!/usr/bin/env python3
"""DBOS Drive-map workflow, local-records only.

No Google Drive connector is touched here. This makes the current safe Drive
import/map scaffold a DBOS-owned, replayable workflow.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import psycopg
from dbos import DBOS, DBOSConfig

ROOT = Path(__file__).resolve().parents[1]
STATE_DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)


def emit(run_id: str, phase: str, status: str, detail: dict[str, Any]) -> str:
    with psycopg.connect(STATE_DB) as conn:
        event_id = conn.execute(
            """
            INSERT INTO lucidota_control.workflow_event
              (workflow_id, run_id, phase, status, source, detail)
            VALUES ('drive-map-workflow', %s, %s, %s, 'lucidota_dbos_drive_map', %s::jsonb)
            RETURNING event_id
            """,
            (run_id, phase, status, json.dumps(detail, sort_keys=True)),
        ).fetchone()[0]
        conn.commit()
    return str(event_id)


@DBOS.step()
def drive_manifest_step() -> dict[str, Any]:
    check = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "lucidota_drive_import_manifest.py"), "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(check.stderr or check.stdout)
    manifest = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "lucidota_drive_manifest.py"), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if manifest.returncode != 0:
        raise RuntimeError(manifest.stderr or manifest.stdout)
    data = json.loads(manifest.stdout)
    return {
        "mode": "local_records_only_no_drive_api",
        "target_count": data.get("count", len(data.get("targets", []))),
        "top_targets": data.get("targets", [])[:5],
        "check": check.stdout.strip(),
    }


@DBOS.workflow()
def drive_map_workflow(run_id: str) -> dict[str, Any]:
    emit(run_id, "drive_map", "running", {"mode": "local_records_only_no_drive_api"})
    result = drive_manifest_step()
    event_id = emit(run_id, "drive_map", "succeeded", result)
    return {"run_id": run_id, "event_id": event_id, **result}


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-dbos-drive-map")
    ap.add_argument("--run-id", default=f"drive-map-{uuid.uuid4()}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    DBOS(config={"name": "lucidota-dbos-drive-map", "system_database_url": STATE_DB})
    DBOS.launch()
    result = drive_map_workflow(args.run_id)
    DBOS.destroy(destroy_registry=True)
    report = {"ok": True, "drive_map": result}
    print(json.dumps(report, sort_keys=True, default=str) if args.json else report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
