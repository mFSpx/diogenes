#!/usr/bin/env python3
from __future__ import annotations
import json, os
from pathlib import Path
import psycopg
ROOT=Path(__file__).resolve().parents[1]
DB=os.environ.get('DBOS_SYSTEM_DATABASE_URL','postgresql://mfspx@/lucidota_state')
with psycopg.connect(DB) as conn:
    conn.execute((ROOT/'06_SCHEMA/006_workflow_registry.sql').read_text())
    rows=conn.execute("select workflow_name,phase,status,command from lucidota_control.workflow_registry order by phase, workflow_name").fetchall()
    conn.commit()
print(json.dumps({"ok": True, "workflows": [dict(zip(['name','phase','status','command'], r)) for r in rows]}, sort_keys=True))
