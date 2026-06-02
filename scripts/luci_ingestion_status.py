#!/usr/bin/env python3
"""Repo-wide LUCI ingestion completion contract.

This is a status/receipt rail, not an ingestion worker. It normalizes the local
meaning of "ingestion done" across the active lanes so the front door can say
DONE only when the DB/receipt evidence supports that exact claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "05_OUTPUTS" / "ingestion_status"
STATE_DSN = os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL") or os.environ.get("LUCIDOTA_GO_STATE_DSN") or "postgresql:///lucidota_state"
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
ELIGIBLE_KRAMPUS_EXT = {".pdf", ".docx", ".odt", ".txt", ".md"}
KRAMPUS_ARCHIVE_EXT = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"}
KRAMPUS_SPECIAL_ARCHIVE_LANES = {"C_ARCHIVE.zip": "c_archive_email"}
MARKDOWN_KEEP_NAMES = {"README.md"}
MARKDOWN_SKIP_PARTS = {".git", ".venv", "01_REPOS", "03_VAULT"}
SCHEMA = "lucidota.ingestion_completion_contract.v1"
FALSE_VICTORY_GUARD = "per-lane evidence only; no full-ingestion claim unless all required lanes are done"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def storage_scalar(conn: psycopg.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return int(row[0]) if row else 0


def table_count(conn: psycopg.Connection, qualified: str) -> int:
    return storage_scalar(conn, f"SELECT count(*) FROM {qualified}")


def existing_source_paths(conn: psycopg.Connection, names: list[str]) -> set[str]:
    if not names:
        return set()
    possible: list[str] = []
    for name in names:
        absolute = str(ROOT / "KRAMPUSCHEWING" / name)
        possible.extend([absolute, f"KRAMPUSCHEWING/{name}", f"/home/mfspx/LUCIDOTA/KRAMPUSCHEWING/{name}"])
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT source_path FROM lucidota_korpus.corpus_chunk WHERE source_path = ANY(%s)",
            (possible,),
        )
        return {str(row[0]) for row in cur.fetchall()}


def krampus_top_level_contract(conn: psycopg.Connection) -> dict[str, Any]:
    root = ROOT / "KRAMPUSCHEWING"
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ELIGIBLE_KRAMPUS_EXT) if root.exists() else []
    names = [p.name for p in files]
    existing = existing_source_paths(conn, names)
    pending: list[str] = []
    processed = 0
    for p in files:
        variants = {str(p), rel(p), f"KRAMPUSCHEWING/{p.name}"}
        if variants & existing:
            processed += 1
        else:
            pending.append(rel(p))
    return {
        "done": len(pending) == 0,
        "eligible_files": len(files),
        "processed_files": processed,
        "pending_files": len(pending),
        "pending_sample": pending[:20],
        "required_evidence": [
            "lucidota_korpus.corpus_chunk rows keyed by each KRAMPUSCHEWING top-level document source_path",
            "05_OUTPUTS/receipts/krampus_pdf_*.json or successor receipt for parser run",
        ],
        "source": "KRAMPUSCHEWING loose top-level .pdf/.docx/.odt/.txt/.md files. Archive containers are in scope under krampus_archive_members.",
    }


def archive_member_prefix_counts(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT split_part(source_path, '!', 1) AS archive_prefix, count(*)
            FROM lucidota_korpus.corpus_chunk
            WHERE position('!' in source_path) > 0
            GROUP BY archive_prefix
            """
        )
        return {str(row[0]): int(row[1]) for row in cur.fetchall()}


def archive_receipt_summary() -> tuple[list[Path], dict[str, Any]]:
    receipts = sorted(
        (ROOT / "05_OUTPUTS" / "receipts").glob("krampus_archive_*.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )
    latest: dict[str, Any] = {}
    if receipts:
        try:
            latest = json.loads(receipts[-1].read_text(encoding="utf-8"))
        except Exception as exc:
            latest = {"read_error": f"{type(exc).__name__}: {exc}"}
    return receipts, latest


def krampus_archive_contract(conn: psycopg.Connection) -> dict[str, Any]:
    root = ROOT / "KRAMPUSCHEWING"
    all_archives = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in KRAMPUS_ARCHIVE_EXT) if root.exists() else []
    excluded = [p for p in all_archives if p.name in KRAMPUS_SPECIAL_ARCHIVE_LANES]
    archives = [p for p in all_archives if p.name not in KRAMPUS_SPECIAL_ARCHIVE_LANES]
    prefix_counts = archive_member_prefix_counts(conn)
    opened: list[dict[str, Any]] = []
    pending: list[str] = []
    member_chunks = 0
    for archive in archives:
        chunks = prefix_counts.get(f"KRAMPUSCHEWING/{archive.name}", 0) + prefix_counts.get(archive.name, 0)
        member_chunks += chunks
        if chunks:
            opened.append({"archive": rel(archive), "member_chunks": chunks})
        else:
            pending.append(rel(archive))
    receipts, latest = archive_receipt_summary()
    return {
        "done": len(pending) == 0,
        "archives_present": len(archives),
        "archives_opened_with_chunks": len(opened),
        "pending_unopened_archives": len(pending),
        "pending_sample": pending[:20],
        "archive_member_chunks": member_chunks,
        "excluded_special_archive_count": len(excluded),
        "excluded_special_archives": [rel(p) for p in excluded],
        "excluded_special_archive_lanes": {rel(root / name): lane for name, lane in KRAMPUS_SPECIAL_ARCHIVE_LANES.items() if (root / name).exists()},
        "receipt_count": len(receipts),
        "latest_receipt": rel(receipts[-1]) if receipts else "",
        "latest_receipt_status": latest.get("status", ""),
        "latest_receipt_dry_run": bool(latest.get("dry_run")) if latest else False,
        "latest_receipt_archive_filter": latest.get("archive_filter", ""),
        "latest_receipt_chunks_inserted": latest.get("chunks_inserted", 0),
        "latest_receipt_members_seen": latest.get("members_seen", 0),
        "latest_receipt_nested_archives_opened": latest.get("nested_archives_opened", 0),
        "latest_receipt_source_files_deleted": bool(latest.get("source_files_deleted")) if latest else False,
        "required_evidence": [
            "KRAMPUS archive containers opened recursively, including nested archive members",
            "lucidota_korpus.corpus_chunk.source_path values shaped like KRAMPUSCHEWING/<archive>!<member>[!<nested-member>]",
            "05_OUTPUTS/receipts/krampus_archive_*.json with source_files_deleted=true after confirmed ingest",
        ],
        "source": "KRAMPUSCHEWING archive containers are ingestion inputs; once confirmed by receipt, successful originals can be retired to save disk.",
    }


def markdown_active_candidate_count() -> int:
    count = 0
    for path in ROOT.rglob("*.md"):
        try:
            parts = path.relative_to(ROOT).parts
        except ValueError:
            continue
        if any(part in MARKDOWN_SKIP_PARTS for part in parts):
            continue
        if path.name in MARKDOWN_KEEP_NAMES:
            continue
        count += 1
    return count


def markdown_contract(conn: psycopg.Connection) -> dict[str, Any]:
    candidates = markdown_active_candidate_count()
    rows = table_count(conn, "lucidota_indy.markdown_artifact")
    archived = storage_scalar(conn, "SELECT count(*) FROM lucidota_indy.markdown_artifact WHERE status='archived'")
    ingested = storage_scalar(conn, "SELECT count(*) FROM lucidota_indy.markdown_artifact WHERE status='ingested'")
    return {
        "done": candidates == 0 and rows > 0,
        "active_candidates": candidates,
        "markdown_artifact_rows": rows,
        "archived_rows": archived,
        "ingested_rows": ingested,
        "required_evidence": [
            "lucidota_indy.markdown_artifact rows for discovered markdown",
            "03_VAULT/ingested_markdown/<run_id>/manifest.json when --execute archives active markdown",
        ],
    }


def c_archive_contract(conn: psycopg.Connection) -> dict[str, Any]:
    chunks = storage_scalar(
        conn,
        "SELECT count(*) FROM lucidota_korpus.corpus_chunk WHERE extractor IN ('c_archive_email_mime_reingest_v1','c_archive_email_stream_v1')",
    )
    receipts = sorted((ROOT / "05_OUTPUTS" / "ingestion_audit").glob("c_archive_email_mime_reingest_*.json"))
    archive_exists = (ROOT / "KRAMPUSCHEWING" / "C_ARCHIVE.zip").exists()
    return {
        "done": chunks > 0 and bool(receipts),
        "chunks": chunks,
        "receipt_count": len(receipts),
        "latest_receipt": rel(receipts[-1]) if receipts else "",
        "archive_exists": archive_exists,
        "required_evidence": [
            "decoded email chunks in lucidota_korpus.corpus_chunk",
            "05_OUTPUTS/ingestion_audit/c_archive_email_mime_reingest_*.json",
        ],
    }


def embedding_contract(conn: psycopg.Connection) -> dict[str, Any]:
    total = table_count(conn, "lucidota_korpus.corpus_chunk")
    null_all = storage_scalar(conn, "SELECT count(*) FROM lucidota_korpus.corpus_chunk WHERE embedding IS NULL")
    enqueue_receipts = sorted((ROOT / "05_OUTPUTS" / "embedding_enqueue").glob("embed_fill_enqueuer_*.json"))
    try:
        from scripts.lucidota_ingestion_quality_audit import embedding_quality_sql_where  # type: ignore

        quality_where = embedding_quality_sql_where()
        quality_null = storage_scalar(conn, f"SELECT count(*) FROM lucidota_korpus.corpus_chunk WHERE {quality_where}")
    except Exception as exc:
        quality_where = "embedding IS NULL"
        quality_null = null_all
        quality_error = f"{type(exc).__name__}: {exc}"
    else:
        quality_error = ""
    return {
        "done": quality_null == 0,
        "corpus_chunks": total,
        "embedding_null_all": null_all,
        "embedding_null_quality_gate": quality_null,
        "quality_gate_sql": quality_where,
        "quality_gate_error": quality_error,
        "enqueue_receipt_count": len(enqueue_receipts),
        "latest_enqueue_receipt": rel(enqueue_receipts[-1]) if enqueue_receipts else "",
        "required_evidence": [
            "lucidota_korpus.corpus_chunk.embedding filled for readable-text quality gate",
            "embedding drain/enqueue receipt proving backlog was admitted, skipped by governor, or drained",
        ],
    }


def db_counts(conn: psycopg.Connection) -> dict[str, int]:
    return {
        "corpus_chunk": table_count(conn, "lucidota_korpus.corpus_chunk"),
        "markdown_artifact": table_count(conn, "lucidota_indy.markdown_artifact"),
        "graph_item": table_count(conn, "lucidota_go.graph_item"),
    }


def contract_status(contracts: dict[str, dict[str, Any]], errors: list[str]) -> str:
    if errors:
        return "BLOCKED"
    return "DONE" if all(bool(c.get("done")) for c in contracts.values()) else "IN_PROGRESS"


def next_actions(contracts: dict[str, dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    k = contracts.get("krampus_top_level_documents", {})
    if k.get("pending_files", 0):
        actions.append("Run a bounded non-destructive KRAMPUS document ingest smoke, then execute in small batches until pending_files=0.")
    a = contracts.get("krampus_archive_members", {})
    if a.get("pending_unopened_archives", 0):
        actions.append("Run luci ingest archive --max-members N in bounded batches; archive containers and nested archives must be opened recursively.")
    e = contracts.get("embedding_backlog", {})
    if e.get("embedding_null_quality_gate", 0):
        actions.append("Run embedding enqueue/drain under governor caps until embedding_null_quality_gate=0 or a SKIPPED receipt explains pressure.")
    m = contracts.get("markdown_archive", {})
    if m.get("active_candidates", 0):
        actions.append("Run luci ingest markdown --execute when ready to archive active markdown breadcrumbs into 03_VAULT.")
    c = contracts.get("c_archive_email", {})
    if not c.get("done"):
        actions.append("Run C_ARCHIVE email reingest with --execute --limit-emails first, then full stream when receipt quality is acceptable.")
    return actions[:8]


def write_state_ledger(payload: dict[str, Any], receipt_path: str, state_dsn: str) -> dict[str, str]:
    identity = {
        "schema": SCHEMA,
        "run_id": payload["run_id"],
        "status": payload["status"],
    }
    event_id = sha256_text(stable_json(identity))
    raw_ref = f"inline://luci-ingestion-status/{payload['run_id']}"
    text = stable_json({"status": payload["status"], "contracts": payload["contracts"], "db_counts": payload["db_counts"]})
    receipt_sha = sha256_text(stable_json(payload))
    with psycopg.connect(state_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            raw_row = cur.execute(
                """
                INSERT INTO lucidota_control.raw_artifact(raw_ref, raw_sha256, hash_algo, source, actor, byte_count, char_count, mime_type, storage_hint, detail)
                VALUES (%s,%s,'sha256','luci_ingestion_status','operator',%s,%s,'application/json','receipt_or_status',%s::jsonb)
                ON CONFLICT (raw_ref) DO UPDATE SET
                  raw_sha256=EXCLUDED.raw_sha256,
                  byte_count=EXCLUDED.byte_count,
                  char_count=EXCLUDED.char_count,
                  detail=EXCLUDED.detail
                RETURNING raw_artifact_uuid::text
                """,
                (raw_ref, sha256_text(text), len(text.encode()), len(text), json.dumps({"receipt_path": receipt_path, "status": payload["status"]})),
            ).fetchone()
            raw_artifact_uuid = raw_row["raw_artifact_uuid"]
            cur.execute(
                """
                INSERT INTO lucidota_control.event_envelope(event_id, ts, source, actor, raw_ref, raw_artifact_uuid, verbatim_hash, hash_algo, text, entities, claims, actions_requested, artifacts_referenced, risk_flags, route_candidates, board_features, embedding_ref, detail)
                VALUES (%s, now(), 'luci_ingestion_status', 'operator', %s, %s::uuid, %s, 'sha256', %s, '[]'::jsonb, '[]'::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, NULL, %s::jsonb)
                ON CONFLICT (event_id) DO UPDATE SET
                  ts=EXCLUDED.ts,
                  raw_ref=EXCLUDED.raw_ref,
                  raw_artifact_uuid=EXCLUDED.raw_artifact_uuid,
                  verbatim_hash=EXCLUDED.verbatim_hash,
                  text=EXCLUDED.text,
                  actions_requested=EXCLUDED.actions_requested,
                  artifacts_referenced=EXCLUDED.artifacts_referenced,
                  risk_flags=EXCLUDED.risk_flags,
                  route_candidates=EXCLUDED.route_candidates,
                  board_features=EXCLUDED.board_features,
                  detail=EXCLUDED.detail
                """,
                (
                    event_id,
                    raw_ref,
                    raw_artifact_uuid,
                    sha256_text(text),
                    text,
                    json.dumps(payload["next_actions"]),
                    json.dumps([receipt_path]),
                    json.dumps([payload["status"], "false_victory_guard"]),
                    json.dumps(list(payload["contracts"].keys())),
                    json.dumps({"db_counts": payload["db_counts"], "done": payload["done"]}),
                    json.dumps({"schema": SCHEMA, "receipt_path": receipt_path}),
                ),
            )
            work_order_row = cur.execute(
                """
                INSERT INTO lucidota_control.work_order(event_id, lane, work_kind, status, payload, idempotency_key)
                VALUES (%s, 'audit', 'luci_ingestion_status', 'succeeded', %s::jsonb, %s)
                ON CONFLICT (idempotency_key) DO UPDATE SET
                  event_id=EXCLUDED.event_id,
                  status=EXCLUDED.status,
                  payload=EXCLUDED.payload,
                  updated_at=now()
                RETURNING work_order_uuid::text
                """,
                (event_id, json.dumps(payload, default=str), f"luci-ingestion-status:{payload['run_id']}"),
            ).fetchone()
            work_order_uuid = work_order_row["work_order_uuid"]
            receipt_row = cur.execute(
                """
                SELECT work_receipt_uuid::text FROM lucidota_control.work_receipt
                WHERE work_order_uuid=%s::uuid AND receipt_path=%s
                ORDER BY created_at DESC LIMIT 1
                """,
                (work_order_uuid, receipt_path),
            ).fetchone()
            if receipt_row:
                work_receipt_uuid = receipt_row["work_receipt_uuid"]
                cur.execute(
                    """
                    UPDATE lucidota_control.work_receipt
                    SET event_id=%s, receipt_sha256=%s, verdict=%s, cost=%s::jsonb, gain=%s::jsonb, artifact_refs=%s::jsonb, detail=%s::jsonb
                    WHERE work_receipt_uuid=%s::uuid
                    """,
                    (
                        event_id,
                        receipt_sha,
                        "promote" if payload["done"] else "retry",
                        json.dumps({"status_probe": "cheap", "canonical_graph_writes": False}),
                        json.dumps({"ingestion_contract": payload["status"], "done": payload["done"]}),
                        json.dumps([raw_ref, receipt_path]),
                        json.dumps({"contracts": payload["contracts"], "db_counts": payload["db_counts"]}),
                        work_receipt_uuid,
                    ),
                )
            else:
                receipt_row = cur.execute(
                    """
                    INSERT INTO lucidota_control.work_receipt(event_id, work_order_uuid, receipt_path, receipt_sha256, verdict, cost, gain, artifact_refs, canonical_graph_writes_performed, graph_write_mode, detail)
                    VALUES (%s, %s::uuid, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, false, 'staged_only', %s::jsonb)
                    RETURNING work_receipt_uuid::text
                    """,
                    (
                        event_id,
                        work_order_uuid,
                        receipt_path,
                        receipt_sha,
                        "promote" if payload["done"] else "retry",
                        json.dumps({"status_probe": "cheap", "canonical_graph_writes": False}),
                        json.dumps({"ingestion_contract": payload["status"], "done": payload["done"]}),
                        json.dumps([raw_ref, receipt_path]),
                        json.dumps({"contracts": payload["contracts"], "db_counts": payload["db_counts"]}),
                    ),
                ).fetchone()
                work_receipt_uuid = receipt_row["work_receipt_uuid"]
        conn.commit()
    return {
        "event_id": event_id,
        "raw_artifact_uuid": raw_artifact_uuid,
        "work_order_uuid": work_order_uuid,
        "work_receipt_uuid": work_receipt_uuid,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or "luci-ingestion-status"
    errors: list[str] = []
    contracts: dict[str, dict[str, Any]] = {}
    counts = {"corpus_chunk": 0, "markdown_artifact": 0, "graph_item": 0}
    with psycopg.connect(args.storage_dsn) as storage_conn:
        counts = db_counts(storage_conn)
        contracts["markdown_archive"] = markdown_contract(storage_conn)
        contracts["krampus_top_level_documents"] = krampus_top_level_contract(storage_conn)
        contracts["krampus_archive_members"] = krampus_archive_contract(storage_conn)
        contracts["c_archive_email"] = c_archive_contract(storage_conn)
        contracts["embedding_backlog"] = embedding_contract(storage_conn)
    status = contract_status(contracts, errors)
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / f"luci_ingestion_status_{sha256_text(run_id)[:16]}.json"
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": now(),
        "run_id": run_id,
        "status": status,
        "done": status == "DONE",
        "false_victory_guard": FALSE_VICTORY_GUARD,
        "db_counts": counts,
        "contracts": contracts,
        "errors": errors,
        "next_actions": next_actions(contracts),
        "receipt_path": rel(receipt_path),
        "canonical_graph_writes_performed": False,
    }
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    try:
        db_write = write_state_ledger(payload, payload["receipt_path"], args.state_dsn)
    except Exception as exc:
        errors.append(f"state_ledger_failed:{type(exc).__name__}:{str(exc)[:240]}")
        payload["status"] = "BLOCKED"
        payload["done"] = False
        payload["errors"] = errors
        db_write = {}
    payload["db_write"] = db_write
    payload["visible_response"] = {
        "summary": f"Ingestion contract status: {payload['status']} ({'all required lanes done' if payload['done'] else 'remaining lane work exists'}).",
        "work_order_id": db_write.get("work_order_uuid", ""),
        "work_receipt_id": db_write.get("work_receipt_uuid", ""),
        "attempt_id": db_write.get("work_order_uuid", ""),
        "raw_artifact_id": db_write.get("raw_artifact_uuid", ""),
        "receipt_path": payload["receipt_path"],
        "next": payload["next_actions"][:3],
    }
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LUCI ingestion completion status contract.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--state-dsn", default=STATE_DSN)
    ap.add_argument("--storage-dsn", default=STORAGE_DSN)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    if args.json:
        print(json.dumps(payload, sort_keys=True, default=str))
    else:
        print(f"INGESTION_STATUS={payload['status']}")
        print(f"DONE={str(payload['done']).lower()}")
        print(f"WORK_ORDER_ID={payload['visible_response']['work_order_id']}")
        print(f"WORK_RECEIPT_ID={payload['visible_response']['work_receipt_id']}")
        print(f"RECEIPT_PATH={payload['receipt_path']}")
        for action in payload.get("next_actions", [])[:5]:
            print(f"NEXT={action}")
    return 0 if payload["status"] in {"DONE", "IN_PROGRESS", "BLOCKED"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
