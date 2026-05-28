#!/usr/bin/env python3
"""KORPUS derived-compute queue worker.

Native Postgres queue with single-row leases (FOR UPDATE SKIP LOCKED).
Designed for post-raw-ingest enrichment so River, MinHash, and graph promotion
never hold the evidence storage pump hostage again.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import korpus_krampii as kk  # noqa: E402

STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
QUEUE_SCHEMA = ROOT / "06_SCHEMA" / "020_korpus_derived_compute_queue.sql"
WORKFLOW_ID = "korpus-derived-compute-worker"


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def ensure_schema(conn: psycopg.Connection) -> None:
    # Deliberately only apply the queue migration here. The heavy KORPUS/GO
    # schemas already exist on this host, and workers should not replay broad
    # DDL while raw ingest is under load.
    conn.execute(QUEUE_SCHEMA.read_text(encoding="utf-8"))
    conn.commit()


def worker_id(prefix: str = "korpus-derived") -> str:
    return f"{prefix}:{socket.gethostname()}:{os.getpid()}"


def queue_summary(conn: psycopg.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT task_type, status, count(*)::bigint AS job_count,
               min(created_at) AS oldest_job, max(updated_at) AS newest_update
        FROM lucidota_korpus.derived_compute_queue
        GROUP BY task_type, status
        ORDER BY task_type, status
        """
    ).fetchall()
    return [dict(r) for r in rows]


def claim_job(conn: psycopg.Connection, wid: str, lease_seconds: int, task_types: list[str] | None) -> dict[str, Any] | None:
    params: list[Any] = []
    task_filter = ""
    if task_types:
        task_filter = "AND task_type = ANY(%s)"
        params.append(task_types)
    claim_sql = f"""
        WITH job AS (
            SELECT job_uuid
            FROM lucidota_korpus.derived_compute_queue
            WHERE (
                (status = 'queued' AND run_after <= now())
                OR (status = 'running' AND locked_until < now() AND attempts < max_attempts)
            )
            {task_filter}
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE lucidota_korpus.derived_compute_queue q
        SET status = 'running',
            locked_by = %s,
            locked_until = now() + (%s * interval '1 second'),
            attempts = attempts + 1,
            last_error = '',
            updated_at = now()
        FROM job
        WHERE q.job_uuid = job.job_uuid
        RETURNING q.*
    """
    params.extend([wid, lease_seconds])
    row = conn.execute(claim_sql, params).fetchone()
    return dict(row) if row else None


def mark_succeeded(conn: psycopg.Connection, job_uuid: str, detail: dict[str, Any] | None = None) -> None:
    payload_merge = jdump({"last_result": detail or {}})
    conn.execute(
        """
        UPDATE lucidota_korpus.derived_compute_queue
        SET status='succeeded', locked_by='', locked_until=NULL, last_error='',
            payload = payload || %s::jsonb,
            finished_at=now(), updated_at=now()
        WHERE job_uuid=%s::uuid
        """,
        (payload_merge, job_uuid),
    )


def mark_failed(conn: psycopg.Connection, job: dict[str, Any], exc: BaseException) -> str:
    attempts = int(job.get("attempts") or 0)
    max_attempts = int(job.get("max_attempts") or 3)
    dead = attempts >= max_attempts
    status = "dead" if dead else "queued"
    backoff_seconds = min(3600, 60 * (2 ** max(0, attempts - 1)))
    err = f"{type(exc).__name__}: {exc}"[:4000]
    detail = {"last_traceback": traceback.format_exc(limit=8)[-8000:]}
    conn.execute(
        """
        UPDATE lucidota_korpus.derived_compute_queue
        SET status=%s,
            locked_by='',
            locked_until=NULL,
            last_error=%s,
            payload = payload || %s::jsonb,
            run_after = CASE WHEN %s THEN run_after ELSE now() + (%s * interval '1 second') END,
            finished_at = CASE WHEN %s THEN now() ELSE finished_at END,
            updated_at=now()
        WHERE job_uuid=%s::uuid
        """,
        (status, err, jdump(detail), dead, backoff_seconds, dead, str(job["job_uuid"])),
    )
    return status


def ensure_replay_batch(conn: psycopg.Connection, label: str) -> str:
    row = conn.execute(
        """
        SELECT batch_uuid::text AS batch_uuid
        FROM lucidota_korpus.ingest_batch
        WHERE batch_label=%s AND status='running'
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (label,),
    ).fetchone()
    if row:
        return str(row["batch_uuid"])
    return str(
        conn.execute(
            """
            INSERT INTO lucidota_korpus.ingest_batch(batch_label, status, root_paths, options, detail)
            VALUES (%s, 'running', '[]'::jsonb, %s::jsonb, %s::jsonb)
            RETURNING batch_uuid::text AS batch_uuid
            """,
            (label, jdump({"derived_compute_replay": True}), jdump({"purpose": label})),
        ).fetchone()["batch_uuid"]
    )


def fetch_component(conn: psycopg.Connection, component_uuid: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT c.component_uuid::text, c.file_uuid::text, c.component_index, c.component_kind,
               c.language, c.title, c.symbol, c.start_line, c.end_line, c.sha256,
               c.token_count, c.content, c.go_terms, c.minhash_signature, c.entropy,
               c.graph_item_uuid::text AS component_graph_uuid,
               f.sha256 AS file_sha256, f.first_seen_path, f.file_kind, f.mime,
               f.graph_item_uuid::text AS file_graph_uuid
        FROM lucidota_korpus.component c
        JOIN lucidota_korpus.file_object f ON f.file_uuid=c.file_uuid
        WHERE c.component_uuid=%s::uuid
        """,
        (component_uuid,),
    ).fetchone()
    if not row:
        raise ValueError(f"component not found: {component_uuid}")
    comp = dict(row)
    comp.setdefault("go_terms", [])
    comp.setdefault("minhash_signature", [])
    return comp


def promote_file(conn: psycopg.Connection, file_uuid: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT file_uuid::text, sha256, first_seen_path, file_kind, mime, graph_item_uuid::text
        FROM lucidota_korpus.file_object
        WHERE file_uuid=%s::uuid
        """,
        (file_uuid,),
    ).fetchone()
    if not row:
        raise ValueError(f"file_object not found: {file_uuid}")
    file_row = dict(row)
    if file_row.get("graph_item_uuid"):
        return {"graph_item_uuid": file_row["graph_item_uuid"], "already_present": True}
    path = file_row.get("first_seen_path") or file_row["sha256"]
    graph_uuid = kk.insert_graph_item(
        conn,
        "EVIDENCE",
        f"Korpus file {Path(path).name} [{file_row['sha256'][:12]}]",
        f"korpus://sha256/{file_row['sha256']}",
        {
            "kind": "korpus_file",
            "sha256": file_row["sha256"],
            "path": path,
            "file_kind": file_row.get("file_kind", ""),
            "mime": file_row.get("mime", ""),
            "evidence_note": "File graph node promoted out-of-band from derived_compute_queue.",
        },
        layer="digital_world",
        role="korpus_file",
    )
    conn.execute(
        "UPDATE lucidota_korpus.file_object SET graph_item_uuid=%s::uuid WHERE file_uuid=%s::uuid",
        (graph_uuid, file_uuid),
    )
    return {"graph_item_uuid": graph_uuid, "already_present": False}


def handle_river_replay(conn: psycopg.Connection, job: dict[str, Any]) -> dict[str, Any]:
    comp = fetch_component(conn, str(job["target_uuid"]))
    os.environ[kk.RIVER_BULK_LIGHT_ENV] = "full"
    batch_uuid = (job.get("payload") or {}).get("batch_uuid") or ensure_replay_batch(conn, "derived-river-replay")
    decision = kk.river_decide_component(conn, batch_uuid, comp["file_uuid"], comp["component_uuid"], comp)
    return {"decision_route": decision.get("decision_route"), "score": decision.get("score"), "batch_uuid": batch_uuid}


def handle_near_duplicate(conn: psycopg.Connection, job: dict[str, Any]) -> dict[str, Any]:
    comp = fetch_component(conn, str(job["target_uuid"]))
    os.environ[kk.NEAR_DUP_MODE_ENV] = "full"
    os.environ[kk.RIVER_BULK_LIGHT_ENV] = "full"
    inserted = kk.insert_near_duplicates(conn, comp["component_uuid"], comp)
    return {"near_duplicates_inserted": inserted}


def handle_graph_file(conn: psycopg.Connection, job: dict[str, Any]) -> dict[str, Any]:
    return promote_file(conn, str(job["target_uuid"]))


def handle_graph_component(conn: psycopg.Connection, job: dict[str, Any]) -> dict[str, Any]:
    comp = fetch_component(conn, str(job["target_uuid"]))
    file_graph = comp.get("file_graph_uuid")
    if not file_graph:
        file_graph = promote_file(conn, comp["file_uuid"])["graph_item_uuid"]
    if comp.get("component_graph_uuid"):
        comp_graph = comp["component_graph_uuid"]
        already = True
    else:
        comp_graph = kk.insert_graph_item(
            conn,
            (comp.get("go_terms") or ["COMMENT"])[0],
            f"{Path(comp.get('first_seen_path') or comp['file_sha256']).name}:{comp['component_kind']}:{comp.get('title') or comp.get('symbol') or comp['component_index']}",
            f"korpus-component://{comp['component_uuid']}",
            {
                "kind": "korpus_component",
                "component_uuid": comp["component_uuid"],
                "file_sha256": comp["file_sha256"],
                "component_sha256": comp["sha256"],
                "go_terms": comp.get("go_terms", []),
                "evidence_note": "Component graph node promoted out-of-band from derived_compute_queue.",
            },
            layer="map",
            role="korpus_component",
        )
        conn.execute(
            "UPDATE lucidota_korpus.component SET graph_item_uuid=%s::uuid WHERE component_uuid=%s::uuid",
            (comp_graph, comp["component_uuid"]),
        )
        already = False
    if file_graph:
        kk.insert_graph_edge(
            conn,
            file_graph,
            comp_graph,
            "FILE_HAS_COMPONENT",
            {"component_kind": comp["component_kind"], "index": comp["component_index"], "derived_compute_queue": True},
            evidence_uuid=file_graph,
        )
    return {"component_graph_item_uuid": comp_graph, "file_graph_item_uuid": file_graph, "already_present": already}


def dispatch_task(conn: psycopg.Connection, job: dict[str, Any]) -> dict[str, Any]:
    task_type = str(job["task_type"])
    if task_type == "river_replay_component":
        return handle_river_replay(conn, job)
    if task_type == "near_duplicate_scan":
        return handle_near_duplicate(conn, job)
    if task_type == "graph_promote_file":
        return handle_graph_file(conn, job)
    if task_type == "graph_promote_component":
        return handle_graph_component(conn, job)
    raise NotImplementedError(f"task handler not implemented yet: {task_type}")


def process_one(conn: psycopg.Connection, wid: str, lease_seconds: int, task_types: list[str] | None) -> dict[str, Any]:
    job = claim_job(conn, wid, lease_seconds, task_types)
    if not job:
        conn.commit()
        return {"claimed": False}
    job_uuid = str(job["job_uuid"])
    try:
        result = dispatch_task(conn, job)
        mark_succeeded(conn, job_uuid, result)
        conn.commit()
        return {"claimed": True, "ok": True, "job_uuid": job_uuid, "task_type": job["task_type"], "result": result}
    except Exception as exc:
        status = mark_failed(conn, job, exc)
        conn.commit()
        return {"claimed": True, "ok": False, "job_uuid": job_uuid, "task_type": job["task_type"], "status": status, "error": str(exc)[:500]}


def backfill(conn: psycopg.Connection, task_type: str, limit: int, priority: int, dry_run: bool) -> dict[str, Any]:
    specs = {
        "river_replay_component": ("component", """
            SELECT c.component_uuid AS target_uuid
            FROM lucidota_korpus.component c
            WHERE c.content <> ''
            ORDER BY c.component_uuid
            LIMIT %s
        """),
        "near_duplicate_scan": ("component", """
            SELECT c.component_uuid AS target_uuid
            FROM lucidota_korpus.component c
            WHERE c.minhash_signature <> '[]'::jsonb
            ORDER BY c.component_uuid
            LIMIT %s
        """),
        "graph_promote_file": ("file_object", """
            SELECT f.file_uuid AS target_uuid
            FROM lucidota_korpus.file_object f
            WHERE f.graph_item_uuid IS NULL
            ORDER BY f.file_uuid
            LIMIT %s
        """),
        "graph_promote_component": ("component", """
            SELECT c.component_uuid AS target_uuid
            FROM lucidota_korpus.component c
            WHERE c.graph_item_uuid IS NULL
            ORDER BY c.component_uuid
            LIMIT %s
        """),
    }
    if task_type not in specs:
        raise ValueError(f"no backfill spec for task_type={task_type}")
    target_table, candidate_sql = specs[task_type]
    if dry_run:
        count = conn.execute(f"SELECT count(*) AS n FROM ({candidate_sql}) s", (limit,)).fetchone()["n"]
        return {"task_type": task_type, "target_table": target_table, "dry_run": True, "candidate_count": int(count)}
    rows = conn.execute(
        f"""
        WITH candidates AS ({candidate_sql})
        INSERT INTO lucidota_korpus.derived_compute_queue(task_type, target_table, target_uuid, priority, payload)
        SELECT %s, %s, target_uuid, %s, %s::jsonb
        FROM candidates
        ON CONFLICT(task_type, target_table, target_uuid) DO NOTHING
        RETURNING job_uuid::text
        """,
        (limit, task_type, target_table, priority, jdump({"backfilled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})),
    ).fetchall()
    return {"task_type": task_type, "target_table": target_table, "dry_run": False, "inserted": len(rows)}


def parse_task_types(raw: str) -> list[str] | None:
    if not raw:
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(prog="korpus-derived-compute-worker")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--db-url", default=STORAGE_DSN)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status")

    p = sub.add_parser("backfill")
    p.add_argument("--task-type", required=True, choices=["river_replay_component", "near_duplicate_scan", "graph_promote_file", "graph_promote_component"])
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--priority", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("work")
    p.add_argument("--limit", type=int, default=1, help="Max jobs to process before exiting; 0 means loop forever.")
    p.add_argument("--idle-sleep", type=float, default=5.0)
    p.add_argument("--lease-seconds", type=int, default=300)
    p.add_argument("--worker-id", default="")
    p.add_argument("--task-types", default="", help="Comma-separated allowlist, e.g. near_duplicate_scan,graph_promote_file")

    args = ap.parse_args()
    results: dict[str, Any]
    with psycopg.connect(args.db_url, row_factory=dict_row) as conn:
        ensure_schema(conn)
        if args.cmd == "status":
            results = {"ok": True, "summary": queue_summary(conn)}
        elif args.cmd == "backfill":
            results = {"ok": True, **backfill(conn, args.task_type, args.limit, args.priority, args.dry_run)}
            conn.commit()
        elif args.cmd == "work":
            wid = args.worker_id or worker_id()
            task_types = parse_task_types(args.task_types)
            processed = succeeded = failed = 0
            samples: list[dict[str, Any]] = []
            while args.limit == 0 or processed < args.limit:
                item = process_one(conn, wid, args.lease_seconds, task_types)
                if not item.get("claimed"):
                    if args.limit == 0:
                        time.sleep(args.idle_sleep)
                        continue
                    break
                processed += 1
                if item.get("ok"):
                    succeeded += 1
                else:
                    failed += 1
                if len(samples) < 20:
                    samples.append(item)
            results = {"ok": failed == 0, "worker_id": wid, "processed": processed, "succeeded": succeeded, "failed": failed, "samples": samples}
            kk.emit_event(WORKFLOW_ID, wid, "work", "succeeded" if failed == 0 else "failed", results)
        else:
            raise ValueError(args.cmd)
    print(jdump(results) if args.json else json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if results.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
