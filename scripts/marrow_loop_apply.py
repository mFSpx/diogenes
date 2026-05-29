#!/usr/bin/env python3
"""Apply a Marrow Loop command receipt to the append-only local marrow state."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "marrow_loop"
STATE_PATH = OUT_DIR / "marrow_state.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_receipt(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Marrow Loop receipt to append-only state")
    parser.add_argument("--receipt", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default; do not mutate marrow_state.md")
    mode.add_argument("--execute", action="store_true", help="append to marrow_state.md")
    args = parser.parse_args()

    dry_run = not args.execute
    receipt_path = Path(args.receipt)
    if not receipt_path.is_absolute():
        receipt_path = ROOT / receipt_path
    blockers: list[str] = []
    receipt: dict[str, Any] = {}
    if not receipt_path.exists():
        blockers.append("receipt_missing")
    else:
        receipt = load_receipt(receipt_path)

    before = sha256_file(STATE_PATH)
    appended_text = ""
    if receipt:
        appended_text = (
            f"\n## Marrow Command {receipt.get('command_uuid')}\n\n"
            f"- generated_at: {utc_now()}\n"
            f"- receipt_path: `{receipt_path.relative_to(ROOT)}`\n"
            f"- raw_command: {receipt.get('raw_command', '')}\n"
            f"- normalized_intent: {receipt.get('normalized_intent', '')}\n"
            f"- authority_class: {receipt.get('authority_class', '')}\n"
            f"- source: {receipt.get('source', '')}\n"
            f"- db_writes_performed: {receipt.get('db_writes_performed', False)}\n"
            f"- graph_writes_performed: {receipt.get('graph_writes_performed', False)}\n"
        )

    execute_performed = False
    if args.execute and not blockers:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with STATE_PATH.open("a", encoding="utf-8") as fh:
            if before is None:
                fh.write("# LUCIDOTA Marrow Loop State\n")
            fh.write(appended_text)
        execute_performed = True
    after = sha256_file(STATE_PATH) if execute_performed else None

    report = {
        "schema": "lucidota.marrow_loop.apply_report.v0",
        "generated_at": utc_now(),
        "receipt_path": str(receipt_path.relative_to(ROOT)) if receipt_path.exists() else str(args.receipt),
        "command_uuid": receipt.get("command_uuid"),
        "raw_command": receipt.get("raw_command"),
        "normalized_intent": receipt.get("normalized_intent"),
        "execute_performed": execute_performed,
        "dry_run": dry_run,
        "state_path": str(STATE_PATH.relative_to(ROOT)),
        "state_sha256_before": before,
        "state_sha256_after": after,
        "would_append": appended_text if dry_run else "",
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "blockers": blockers,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"marrow_apply_report_{stamp()}.json"
    report["report_path"] = str(report_path.relative_to(ROOT))
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"APPLY_REPORT_PATH={report_path.relative_to(ROOT)}")
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
