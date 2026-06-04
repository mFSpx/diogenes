#!/usr/bin/env python3
"""Queue bounded canonical manual work orders into ABSURD/Postgres."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "manual_canon"
DB = os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
QUEUE = "manual_canon"
PY = "python3"


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_stamp = stamp()
    jobs = [
        ("manual.root", 1, "root"),
        ("manual.api", 2, "api"),
        ("manual.runtime", 3, "runtime"),
        ("manual.contradiction", 4, "contradiction"),
        ("manual.final", 5, "final"),
        ("manual.html", 6, "html"),
    ]
    manifest = {"schema": "lucidota.manual_canon.queue_manifest.v1", "generated_at": now_z(), "queue": QUEUE, "jobs": []}
    queue_rows: list[dict[str, Any]] = []
    with psycopg.connect(DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, max_attempts, notes)
                VALUES (%s, 'Manual canon worker', 3, 'Chunked manual volumes; no giant prompt; receipt-backed docs only')
                ON CONFLICT (queue_name) DO UPDATE SET updated_at = now(), notes = EXCLUDED.notes
                """,
                (QUEUE,),
            )
        conn.commit()
    with psycopg.connect(DB) as conn:
        for workflow, priority, volume in jobs:
            payload = {"command": [PY, "scripts/manual_canon_worker.py", "--volume", volume], "volume": volume, "timeout_seconds": 120}
            idempotency = sha({"queue": QUEUE, "workflow": workflow, "payload": payload})
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_job
                      (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                    VALUES (%s, %s, 'external_command', %s, %s::jsonb, %s, 3, %s::jsonb)
                    ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at = now()
                    RETURNING job_uuid::text, (xmax = 0) AS inserted_new
                    """,
                    (QUEUE, workflow, idempotency, json.dumps(payload), priority, json.dumps({"source": "manual_canon_queue"})),
                )
                job_uuid, inserted_new = cur.fetchone()
                if inserted_new:
                    cur.execute(
                        "INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, detail) VALUES (%s,%s,'enqueued',%s::jsonb)",
                        (job_uuid, QUEUE, json.dumps({"workflow": workflow, "volume": volume})),
                    )
            conn.commit()
            queue_rows.append({"workflow": workflow, "volume": volume, "job_uuid": job_uuid, "inserted_new": bool(inserted_new), "idempotency_key": idempotency})
            manifest["jobs"].append({"workflow": workflow, "volume": volume, "priority": priority, "payload": payload, "job_uuid": job_uuid})
    manifest_path = OUT / f"manual_canon_queue_{run_stamp}.json"
    jsonl_path = OUT / f"manual_canon_queue_{run_stamp}.jsonl"
    receipt_path = OUT / f"manual_canon_queue_receipt_{run_stamp}.json"
    receipt = {
        "schema": "lucidota.manual_canon.queue_receipt.v1",
        "generated_at": manifest["generated_at"],
        "status": "PASS",
        "queue": QUEUE,
        "job_count": len(queue_rows),
        "queue_rows": queue_rows,
        "manifest_path": rel(manifest_path),
        "jsonl_path": rel(jsonl_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in queue_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    receipt["manifest_path"] = rel(manifest_path)
    receipt["jsonl_path"] = rel(jsonl_path)
    receipt["receipt_path"] = rel(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RECEIPT_PATH=" + rel(receipt_path))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
