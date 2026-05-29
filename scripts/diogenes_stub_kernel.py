#!/usr/bin/env python3
"""Marrow Loop v0 command receipt kernel.

Default is dry-run. A receipt is always written. DB writes are disabled unless
--execute and --execute-db are both explicitly passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "marrow_loop"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def maybe_db_write(receipt: dict[str, Any], database_url: str | None) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not database_url:
        blockers.append("DATABASE_URL_not_set")
        return False, blockers
    try:
        import psycopg  # type: ignore
    except Exception as exc:
        blockers.append(f"psycopg_unavailable:{exc}")
        return False, blockers
    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('lucidota_control.event_outbox')")
                if cur.fetchone()[0] is None:
                    blockers.append("lucidota_control.event_outbox_missing")
                    return False, blockers
                cur.execute(
                    """
                    INSERT INTO lucidota_control.event_outbox(event_type, payload)
                    VALUES ('marrow_loop.command_receipt', %s::jsonb)
                    """,
                    (json.dumps(receipt),),
                )
            conn.commit()
        return True, blockers
    except Exception as exc:
        blockers.append(f"db_write_failed:{exc}")
        return False, blockers


def main() -> int:
    parser = argparse.ArgumentParser(description="DIOGENES/Marrow Loop v0 command receipt stub")
    parser.add_argument("--raw-command", required=True)
    parser.add_argument("--normalized-intent", required=True)
    parser.add_argument("--authority-class", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--execute", action="store_true", help="execute local non-DB side effects where implemented; default dry-run")
    parser.add_argument("--execute-db", dest="execute_db", action="store_true", help="allow explicit DB receipt write when combined with --execute")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command_uuid = str(uuid.uuid4())
    payload = {
        "raw_command": args.raw_command,
        "normalized_intent": args.normalized_intent,
        "authority_class": args.authority_class,
        "source": args.source,
    }
    receipt = {
        "schema": "lucidota.marrow_loop.command_receipt.v0",
        "generated_at": utc_now(),
        "command_uuid": command_uuid,
        "raw_command": args.raw_command,
        "normalized_intent": args.normalized_intent,
        "authority_class": args.authority_class,
        "source": args.source,
        "payload_sha256": sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"))),
        "dry_run": not args.execute,
        "execute_requested": bool(args.execute),
        "execute_db_requested": bool(args.execute_db),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "blockers": [],
    }

    if args.execute_db and not args.execute:
        receipt["blockers"].append("execute_db_requires_execute")
    elif args.execute and args.execute_db:
        performed, blockers = maybe_db_write(receipt, args.database_url)
        receipt["db_writes_performed"] = performed
        receipt["blockers"].extend(blockers)

    receipt_path = OUT_DIR / f"command_receipt_{stamp()}.json"
    receipt["receipt_path"] = str(receipt_path.relative_to(ROOT))
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(f"RECEIPT_PATH={receipt_path.relative_to(ROOT)}")
    return 0 if not receipt["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
