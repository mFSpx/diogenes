#!/usr/bin/env python3
"""Stateless Matrix/Conduit adapter for Indy_READs.

Matrix is treated as an open chat protocol adapter only.  This script does not
run a bot loop and does not own process lifetime; Conduit/Synapse/systemd can
hand inbound events to this driver, and ABSURD/Postgres keep the durable state.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "05_OUTPUTS" / "indy_conduit"
DIALOGUE_TABLE = "ironclaw.waking_dialogue_stream"
QUEUE_NAME = "matrix_intake"


def now_z() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")).hexdigest()


URL_RE = re.compile(r"https?://[^\s<>()]+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
COMMAND_RE = re.compile(r"(^|\s)/(\w[\w-]*)")
HASH_TAG_RE = re.compile(r"(?<!\w)#([A-Za-z][A-Za-z0-9_-]*)")


def clean_text(raw_text: str) -> str:
    return re.sub(r"\s+", " ", raw_text).strip()


def extract_entities(raw_text: str) -> dict[str, Any]:
    return {
        "urls": sorted(set(URL_RE.findall(raw_text))),
        "emails": sorted(set(EMAIL_RE.findall(raw_text))),
        "slash_commands": sorted(set(match.group(2) for match in COMMAND_RE.finditer(raw_text))),
        "hashtags": sorted(set(HASH_TAG_RE.findall(raw_text))),
    }


def event_content(event: dict[str, Any]) -> dict[str, Any]:
    content = event.get("content")
    return content if isinstance(content, dict) else {}


def attachment_metadata(content: dict[str, Any]) -> dict[str, Any] | None:
    if content.get("msgtype") != "m.file":
        return None
    info = content.get("info") if isinstance(content.get("info"), dict) else {}
    encrypted = content.get("file") if isinstance(content.get("file"), dict) else {}
    hashes = encrypted.get("hashes") if isinstance(encrypted.get("hashes"), dict) else {}
    return {
        "body": str(content.get("body") or ""),
        "mxc_url": str(content.get("url") or encrypted.get("url") or ""),
        "size": info.get("size"),
        "mimetype": info.get("mimetype"),
        "sha256": hashes.get("sha256") or hashes.get("sha-256") or content.get("sha256"),
    }


def matrix_ref(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": str(event.get("event_id") or ""),
        "room_id": str(event.get("room_id") or ""),
        "sender": str(event.get("sender") or ""),
    }


def build_dialogue_row(event: dict[str, Any]) -> dict[str, Any]:
    content = event_content(event)
    raw_text = str(content.get("body") or "")
    normalized_text = clean_text(raw_text)
    msgtype = str(content.get("msgtype") or "m.text")
    attachment = attachment_metadata(content)
    matrix = {
        **matrix_ref(event),
        "msgtype": msgtype,
        "origin_server_ts": event.get("origin_server_ts"),
        "content_keys": sorted(str(k) for k in content.keys()),
        "body_sha256": sha256_text(raw_text),
        "attachment": attachment,
    }
    entities = extract_entities(raw_text)
    if attachment:
        entities["attachments"] = [attachment]
    return {
        "comms_channel": "matrix",
        "raw_text": raw_text,
        "clean_text": normalized_text,
        "extracted_entities": entities,
        "processed_status": "queued",
        "sender_id": matrix["sender"],
        "room_id": matrix["room_id"],
        "event_id": matrix["event_id"],
        "receipt_id": "",
        "source_payload": {
            "schema": "lucidota.indy.matrix_event_ref.v1",
            "matrix": matrix,
            "raw_event_sha256": sha256_obj(event),
            "body_attached_as_reference_only": True,
        },
    }


def flow_widget_action() -> dict[str, Any]:
    return {
        "kind": "chat_platform_widget_request",
        "widget_key": "lucidota.promptflow_canvas",
        "home": "active_operator_chat_surface",
        "manual_panel": "postgrest_html_manual",
        "requires_operator_stage_validate_run": True,
    }


def build_absurd_job(event: dict[str, Any], *, kind: str) -> dict[str, Any]:
    ref = matrix_ref(event)
    if kind == "file":
        content = event_content(event)
        payload = {
            "schema": "lucidota.matrix.file_atomize_payload.v1",
            "matrix_event_ref": ref,
            "attachment": attachment_metadata(content),
            "custody": {"bytes_embedded": False, "download_requires_matrix_media_client": True},
        }
        return {
            "queue_name": QUEUE_NAME,
            "workflow_name": "matrix.file.atomize",
            "job_kind": "matrix_file_atomize",
            "idempotency_key": "matrix-file:" + sha256_obj(payload),
            "payload": payload,
            "priority": 50,
            "max_attempts": 3,
        }
    if kind == "flow":
        payload = {
            "schema": "lucidota.matrix.widget_request_payload.v1",
            "matrix_event_ref": ref,
            "ui_action": flow_widget_action(),
            "visible_not_hidden_automation": True,
        }
        return {
            "queue_name": QUEUE_NAME,
            "workflow_name": "matrix.widget.open",
            "job_kind": "matrix_widget_open_request",
            "idempotency_key": "matrix-widget:" + sha256_obj(payload),
            "payload": payload,
            "priority": 40,
            "max_attempts": 1,
        }
    raise ValueError(f"unknown_absurd_job_kind:{kind}")


def build_plan(matrix_event: dict[str, Any]) -> dict[str, Any]:
    row = build_dialogue_row(matrix_event)
    content = event_content(matrix_event)
    body = row["clean_text"].strip()
    jobs: list[dict[str, Any]] = []
    ui_action: dict[str, Any] | None = None
    if content.get("msgtype") == "m.file":
        jobs.append(build_absurd_job(matrix_event, kind="file"))
    if body == "/flow":
        ui_action = flow_widget_action()
        jobs.append(build_absurd_job(matrix_event, kind="flow"))
    return {
        "schema": "lucidota.indy.matrix_conduit_plan.v1",
        "generated_at": now_z(),
        "dialogue_row": row,
        "absurd_jobs": jobs,
        "ui_action": ui_action,
        "execution_law": {
            "protocol_adapter_only": True,
            "homeserver_lifetime_owned_by_systemd": True,
            "postgres_owns_state": True,
            "absurd_owns_work": True,
            "chat_platform_owns_window": True,
            "adapter_does_not_choose_product_root": True,
            "no_hidden_execution_on_drag_or_message": True,
        },
    }


def read_queued_dialogue_rows(conn: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    """Read-only pull path for Indy_READs/LLM context assembly.

    This does not claim, update, or mutate rows. Indy_READs can use it to see the
    original `raw_text` plus deterministic `clean_text` and entity hooks once the
    configured database role has ironclaw read permission.
    """
    bounded_limit = max(1, min(int(limit), 100))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, received_at, sender_id, room_id, event_id, raw_text, clean_text, extracted_entities, receipt_id, created_at
            FROM ironclaw.waking_dialogue_stream
            WHERE comms_channel = 'matrix'
              AND processed_status = 'queued'
            ORDER BY received_at ASC, created_at ASC
            LIMIT %s;
            """,
            (bounded_limit,),
        )
        rows = cur.fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append({
            "id": row[0],
            "received_at": row[1],
            "sender_id": row[2],
            "room_id": row[3],
            "event_id": row[4],
            "raw_text": row[5],
            "clean_text": row[6],
            "extracted_entities": row[7] or {},
            "receipt_id": row[8],
            "created_at": row[9],
            "read_only": True,
        })
    return result


def execute_plan(conn: Any, plan: dict[str, Any]) -> dict[str, Any]:
    row = plan["dialogue_row"]
    inserted: dict[str, Any] = {"dialogue_id": None, "jobs": []}
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ironclaw.waking_dialogue_stream
              (comms_channel, sender_id, room_id, event_id, raw_text, clean_text, extracted_entities, processed_status, receipt_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            ON CONFLICT (comms_channel, event_id) DO UPDATE SET
              raw_text = EXCLUDED.raw_text,
              clean_text = EXCLUDED.clean_text,
              extracted_entities = EXCLUDED.extracted_entities,
              processed_status = EXCLUDED.processed_status,
              receipt_id = EXCLUDED.receipt_id,
              updated_at = now()
            RETURNING id::text;
            """,
            (
                row["comms_channel"],
                row["sender_id"],
                row["room_id"],
                row["event_id"],
                row["raw_text"],
                row["clean_text"],
                json.dumps(row["extracted_entities"], sort_keys=True),
                row["processed_status"],
                row["receipt_id"],
            ),
        )
        fetched = cur.fetchone()
        inserted["dialogue_id"] = fetched[0] if fetched else None
        for job in plan["absurd_jobs"]:
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_job
                  (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
                ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET
                  updated_at = lucidota_control.absurd_queue_job.updated_at
                RETURNING job_uuid::text, (xmax = 0) AS inserted;
                """,
                (
                    job["queue_name"],
                    job["workflow_name"],
                    job["job_kind"],
                    job["idempotency_key"],
                    json.dumps(job["payload"], sort_keys=True),
                    job["priority"],
                    job["max_attempts"],
                    json.dumps({"source": "indy_conduit_driver", "matrix_event_ref": job["payload"].get("matrix_event_ref")}, sort_keys=True),
                ),
            )
            job_row = cur.fetchone()
            inserted["jobs"].append({"job_uuid": job_row[0] if job_row else None, "inserted": bool(job_row[1]) if job_row else False, "job_kind": job["job_kind"]})
    return inserted


def receipt_id_for_plan(plan: dict[str, Any]) -> str:
    row = plan["dialogue_row"]
    return "matrix_conduit:" + sha256_obj(
        {
            "schema": plan["schema"],
            "event_id": row["event_id"],
            "room_id": row["room_id"],
            "sender_id": row["sender_id"],
            "raw_text_sha256": sha256_text(row["raw_text"]),
        }
    )[:16]


def process_event_payload(
    payload: dict[str, Any],
    *,
    dry_run: bool,
    output_dir: Path,
    database_url: str | None = None,
) -> dict[str, Any]:
    """Importable one-shot conduit path for chat surfaces.

    Dry-run writes only a receipt. Execute mode writes through the approved
    `ironclaw.waking_dialogue_stream` row contract and optional ABSURD jobs.
    """
    plan = build_plan(payload)
    plan["dialogue_row"]["receipt_id"] = receipt_id_for_plan(plan)
    executed = False
    db_result: dict[str, Any] | None = None
    error = ""
    if not dry_run:
        try:
            import psycopg  # type: ignore

            dsn = database_url or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or os.environ.get("LUCIDOTA_DATABASE_URL") or "postgresql:///lucidota_state"
            with psycopg.connect(dsn) as conn:
                db_result = execute_plan(conn, plan)
            executed = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    receipt = write_receipt(output_dir, plan, executed=executed, db_result=db_result, error=error)
    return {
        "ok": not error,
        "executed": executed,
        "receipt_path": str(receipt),
        "receipt_id": plan["dialogue_row"]["receipt_id"],
        "event_id": plan["dialogue_row"]["event_id"],
        "absurd_jobs": len(plan["absurd_jobs"]),
        "ui_action": plan.get("ui_action"),
        "db_result": db_result,
        "error": error,
    }


def write_receipt(out_dir: Path, plan: dict[str, Any], *, executed: bool, db_result: dict[str, Any] | None, error: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_obj(plan)[:16]
    path = out_dir / f"matrix_conduit_{stamp()}_{digest}.json"
    receipt = {
        "schema": "lucidota.indy.matrix_conduit_receipt.v1",
        "generated_at": now_z(),
        "executed": executed,
        "db_result": db_result,
        "error": error,
        "plan": plan,
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_event(args: argparse.Namespace) -> dict[str, Any]:
    if args.event_json:
        return json.loads(args.event_json)
    if args.event_file:
        return json.loads(Path(args.event_file).read_text(encoding="utf-8"))
    return json.load(sys.stdin)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stateless Matrix event adapter for Indy_READs/Postgres/ABSURD.")
    parser.add_argument("--event-json", help="Matrix event JSON object")
    parser.add_argument("--event-file", help="Path to Matrix event JSON object")
    parser.add_argument("--database-url", default=os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or os.environ.get("LUCIDOTA_DATABASE_URL") or "postgresql:///lucidota_state")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--dry-run", action="store_true", help="Plan and receipt only; no database writes")
    parser.add_argument("--execute", action="store_true", help="Write to Postgres/ABSURD using approved tables")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.execute:
        parser.error("choose exactly one of --dry-run/--execute")
    event = load_event(args)
    payload = process_event_payload(
        event,
        dry_run=not args.execute,
        output_dir=Path(args.output_dir),
        database_url=args.database_url,
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"RECEIPT_PATH={rel(payload['receipt_path'])}")
        if payload.get("error"):
            print(f"ERROR={payload['error']}", file=sys.stderr)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
