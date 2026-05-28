#!/usr/bin/env python3
"""LUCIDOTA Absurd Workflow Engine Spine.

Small durable state-machine queue on Postgres.

Critical rule: the control-table row is locked only while it is claimed or
finished. Handler code runs outside the control transaction so long CAS/file/IO
work cannot hold a row lock indefinitely.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable

import psycopg

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = """
CREATE SCHEMA IF NOT EXISTS lucidota_control;
CREATE TABLE IF NOT EXISTS lucidota_control.absurd_workflow (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    state JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_log TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_absurd_workflow_status ON lucidota_control.absurd_workflow(status);
CREATE INDEX IF NOT EXISTS idx_absurd_workflow_pending_created
    ON lucidota_control.absurd_workflow(created_at)
    WHERE status = 'pending';
"""

def db_url() -> str:
    return os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"

class AbsurdWorker:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def init_schema(self):
        with psycopg.connect(self.dsn) as conn:
            conn.execute(SCHEMA)
            conn.commit()
        print("Absurd workflow schema deployed.")

    def spawn(self, workflow_name: str, payload: dict[str, Any]) -> str:
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
        """Claim, execute, and persist exactly one workflow step.

        Handler contract:
        - return ("completed", state) for a finished workflow
        - return ("failed", state) for an intentional terminal failure
        - return ("running", state) to continue later; this is persisted as
          status='pending' after committing the updated state, so the next phase
          reads a fully durable state row and no row sits in a hot running loop.
        """
        row = self._claim_next()
        if not row:
            return False

        workflow_id, name, state, payload = row
        state = dict(state or {})
        payload = dict(payload or {})

        if name not in handlers:
            err_msg = f"No handler for {name}"
            self._finish(workflow_id, "failed", state, err_msg)
            print(f"Processed {workflow_id} ({name}) -> failed")
            return True

        requested_status = "failed"
        try:
            requested_status, new_state = handlers[name](payload, state)
            persisted_status = self._finish(workflow_id, requested_status, new_state)
        except psycopg.Error as exc:
            err_msg = _format_exception("database exception", exc)
            print(f"[Absurd Core ERROR] Step crashed: {err_msg}")
            self._finish(workflow_id, "failed", _state_with_error(state, err_msg), err_msg)
            persisted_status = "failed"
        except Exception as exc:
            err_msg = _format_exception("handler exception", exc)
            print(f"[Absurd Core ERROR] Step crashed: {err_msg}")
            self._finish(workflow_id, "failed", _state_with_error(state, err_msg), err_msg)
            persisted_status = "failed"

        suffix = "" if persisted_status == requested_status else f" (handler requested {requested_status})"
        print(f"Processed {workflow_id} ({name}) -> {persisted_status}{suffix}")
        return True

    def _claim_next(self):
        """Atomically mark the oldest pending workflow step running and commit."""
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                """
                UPDATE lucidota_control.absurd_workflow
                SET status = 'running', error_log = NULL, updated_at = now()
                WHERE workflow_id = (
                    SELECT workflow_id FROM lucidota_control.absurd_workflow 
                    WHERE status = 'pending'
                    ORDER BY created_at ASC LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING workflow_id::text, workflow_name, state, payload
                """
            ).fetchone()
            conn.commit()
            return row

    def _finish(
        self,
        workflow_id: str,
        requested_status: str,
        state: dict[str, Any] | None,
        error_log: str | None = None,
    ) -> str:
        """Persist terminal state, failure state, or durable continuation state."""
        if requested_status not in {"pending", "running", "completed", "failed"}:
            raise ValueError(f"Invalid workflow status returned by handler: {requested_status!r}")

        # "running" means "the workflow has more phases". Store it as pending
        # after committing the new state so another poll can claim the next phase
        # from durable JSONB instead of spinning on a live running row.
        persisted_status = "pending" if requested_status == "running" else requested_status
        new_state = dict(state or {})
        if requested_status == "running":
            new_state["_absurd_last_requested_status"] = "running"

        if persisted_status == "failed" and not error_log:
            error_log = _state_error(new_state) or "handler returned failed"

        with psycopg.connect(self.dsn) as conn:
            conn.execute(
                """
                UPDATE lucidota_control.absurd_workflow
                SET status=%s,
                    state=%s::jsonb,
                    error_log=%s,
                    updated_at=now()
                WHERE workflow_id=%s::uuid
                """,
                (persisted_status, json.dumps(new_state), error_log, workflow_id),
            )
            conn.commit()
        return persisted_status


def _state_error(state: dict[str, Any]) -> str | None:
    value = state.get("error") or state.get("_last_error")
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _state_with_error(state: dict[str, Any], error_log: str) -> dict[str, Any]:
    new_state = dict(state or {})
    new_state["_last_error"] = error_log
    return new_state


def _format_exception(kind: str, exc: BaseException) -> str:
    return f"{kind}: {exc}\n{traceback.format_exc()}"

def main():
    ap = argparse.ArgumentParser(description="Absurd Workflow Spine")
    ap.add_argument("action", choices=["init", "spawn", "work"])
    ap.add_argument("--name", default="example_task")
    args = ap.parse_args()
    worker = AbsurdWorker(db_url())
    if args.action == "init":
        worker.init_schema()
    elif args.action == "spawn":
        print(f"Spawned {worker.spawn(args.name, {'target': 'data'})}")
    elif args.action == "work":
        print("Worker polling stub.")

if __name__ == "__main__":
    raise SystemExit(main())
