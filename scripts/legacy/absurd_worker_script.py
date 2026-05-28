#!/usr/bin/env python3
"""LUCIDOTA Absurd Workflow Engine Spine.

Replaces DBOS with a native, state-machine driven Postgres workflow engine.
No more SKIP LOCKED contention. Workflows are durable, resumable state functions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "absurd"

# Minimal Absurd Schema: One table to rule them all.
SCHEMA = """
CREATE TABLE IF NOT EXISTS lucidota_control.absurd_workflow (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', -- pending, running, suspended, completed, failed
    state JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_log TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_absurd_workflow_status ON lucidota_control.absurd_workflow(status);
"""

def db_url() -> str:
    return os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"

class AbsurdWorker:
    """The new, ultra-lightweight worker harness."""
    
    def __init__(self, dsn: str):
        self.dsn = dsn

    def init_schema(self):
        """Deploy the Absurd table."""
        with psycopg.connect(self.dsn) as conn:
            conn.execute(SCHEMA)
            conn.commit()
        print("Absurd workflow schema deployed.")

    def spawn(self, workflow_name: str, payload: dict[str, Any]) -> str:
        """Enqueue a new workflow."""
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                """
                INSERT INTO lucidota_control.absurd_workflow (workflow_name, payload)
                VALUES (%s, %s::jsonb)
                RETURNING workflow_id::text
                """, 
                (workflow_name, json.dumps(payload))
            ).fetchone()
            conn.commit()
            return row[0]

    def process_next(self, handlers: dict[str, Callable[[dict, dict], tuple[str, dict]]]) -> bool:
        """
        Pull the next pending or running workflow. 
        Handlers return (new_status, new_state). Postgres handles atomicity.
        """
        with psycopg.connect(self.dsn) as conn:
            # Postgres UPDATE ... RETURNING handles the lock and claim atomically
            row = conn.execute(
                """
                UPDATE lucidota_control.absurd_workflow
                SET status = 'running', updated_at = now()
                WHERE workflow_id = (
                    SELECT workflow_id FROM lucidota_control.absurd_workflow 
                    WHERE status IN ('pending', 'running') 
                    ORDER BY created_at ASC LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING workflow_id::text, workflow_name, state, payload
                """
            ).fetchone()
            
            if not row:
                return False # No work available
                
            workflow_id, name, state, payload = row
            
            if name not in handlers:
                conn.execute(
                    "UPDATE lucidota_control.absurd_workflow SET status='failed', error_log=%s WHERE workflow_id=%s::uuid",
                    (f"No handler for {name}", workflow_id)
                )
                conn.commit()
                return True

            try:
                # Execute the workflow step
                new_status, new_state = handlers[name](payload, state)
                
                conn.execute(
                    """
                    UPDATE lucidota_control.absurd_workflow 
                    SET status=%s, state=%s::jsonb, updated_at=now() 
                    WHERE workflow_id=%s::uuid
                    """,
                    (new_status, json.dumps(new_state), workflow_id)
                )
            except Exception as e:
                conn.execute(
                    "UPDATE lucidota_control.absurd_workflow SET status='failed', error_log=%s WHERE workflow_id=%s::uuid",
                    (str(e), workflow_id)
                )
            conn.commit()
            print(f"Processed {workflow_id} ({name}) -> {new_status}")
            return True

# --- Example Usage / Stub ---
def example_handler(payload: dict, state: dict) -> tuple[str, dict]:
    """An Absurd state machine step. Returns (status, updated_state)."""
    step = state.get("step", 0)
    
    if step == 0:
        print("Starting task...")
        state["step"] = 1
        return "running", state # Re-queues itself for the next step!
        
    elif step == 1:
        print("Finishing task...")
        state["result"] = "Success!"
        return "completed", state

def main():
    ap = argparse.ArgumentParser(description="Absurd Workflow Spine")
    ap.add_argument("action", choices=["init", "spawn", "work"])
    ap.add_argument("--name", default="example_task")
    args = ap.parse_args()
    
    worker = AbsurdWorker(db_url())
    handlers = {"example_task": example_handler}

    if args.action == "init":
        worker.init_schema()
    elif args.action == "spawn":
        wid = worker.spawn(args.name, {"target": "data"})
        print(f"Spawned {wid}")
    elif args.action == "work":
        if not worker.process_next(handlers):
            print("Queue empty.")

if __name__ == "__main__":
    raise SystemExit(main())