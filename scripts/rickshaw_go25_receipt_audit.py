#!/usr/bin/env python3
"""Audit Rickshaw GO-25 Groq stdout receipts without mutating graph state."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "rickshaw_reingest"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def extract_go25_payload(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return the first raw GO-25 batch object from a mixed stdout log."""
    for match in re.finditer(r"```json\s*(\{.*?\})\s*```", text, flags=re.S):
        try:
            obj = json.loads(match.group(1))
        except Exception as exc:
            return None, f"json_parse_failed:{type(exc).__name__}:{str(exc)[:220]}"
        if isinstance(obj, dict) and obj.get("schema") == "lucidota.go25.staging_batch.v1":
            return obj, None
    for line in text.splitlines():
        raw = line.strip()
        if not raw.startswith('{"schema":"lucidota.go25.staging_batch.v1"') and not raw.startswith(
            '{"schema": "lucidota.go25.staging_batch.v1"'
        ):
            continue
        try:
            obj = json.loads(raw)
        except Exception as exc:  # json.JSONDecodeError on normal path
            return None, f"json_parse_failed:{type(exc).__name__}:{str(exc)[:220]}"
        if not isinstance(obj, dict):
            return None, "go25_payload_not_object"
        return obj, None
    return None, "missing_raw_go25_json_line"


def batch_number(path: Path) -> int | None:
    match = re.search(r"batch_(\d+)_", path.name)
    return int(match.group(1)) if match else None


def validate_payload(payload: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        temp_path = handle.name
    proc = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "scripts/lucidota_go_ingest.py", "validate-packet", temp_path],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    try:
        parsed = json.loads(proc.stdout)
    except Exception:
        parsed = {"ok": False, "errors": ["validator_non_json_stdout"], "stdout": proc.stdout[:2000]}
    parsed["validator_returncode"] = proc.returncode
    return proc.returncode == 0 and bool(parsed.get("ok")), parsed, proc.stderr


def component_uuid(payload: dict[str, Any]) -> str | None:
    packets = payload.get("packets")
    if not isinstance(packets, list) or not packets:
        return None
    packet = packets[0] if isinstance(packets[0], dict) else {}
    item = packet.get("proposed_item") if isinstance(packet, dict) else {}
    payload_obj = item.get("payload") if isinstance(item, dict) else {}
    value = payload_obj.get("source_component_uuid") if isinstance(payload_obj, dict) else None
    if value:
        return str(value)
    source_id = packet.get("source_id") if isinstance(packet, dict) else None
    if isinstance(source_id, str) and source_id.startswith("component_uuid:"):
        return source_id.split(":", 1)[1]
    return None


def audit_stdout_files(files: list[Path], *, validate: bool = True) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_components: Counter[str] = Counter()
    duplicate_rows: list[dict[str, Any]] = []
    for path in sorted(files):
        batch = batch_number(path)
        payload, error = extract_go25_payload(path.read_text(encoding="utf-8", errors="replace"))
        row: dict[str, Any] = {
            "batch": batch,
            "stdout_path": rel(path),
            "json_ok": payload is not None,
            "validation_ok": False,
            "errors": [] if error is None else [error],
            "component_uuid": None,
        }
        if payload is not None:
            cu = component_uuid(payload)
            row["component_uuid"] = cu
            if cu:
                seen_components[cu] += 1
                if seen_components[cu] > 1:
                    duplicate_rows.append({"batch": batch, "component_uuid": cu, "stdout_path": rel(path)})
            if validate:
                ok, validation, stderr = validate_payload(payload)
                row["validation_ok"] = ok
                row["validation"] = validation
                if stderr:
                    row["validator_stderr"] = stderr[:2000]
            else:
                row["validation_ok"] = True
        rows.append(row)
    counts = {
        "stdout_files": len(rows),
        "valid_json": sum(1 for row in rows if row["json_ok"]),
        "validation_ok": sum(1 for row in rows if row["validation_ok"]),
        "json_failed": sum(1 for row in rows if not row["json_ok"]),
        "duplicate_component_batches": len(duplicate_rows),
    }
    return {
        "schema": "lucidota.rickshaw.go25_receipt_audit.v1",
        "generated_at": now(),
        "counts": counts,
        "duplicate_component_batches": duplicate_rows,
        "rows": rows,
        "canonical_graph_writes": False,
        "canonical_graph_materialization": False,
        "db_writes_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=999)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    files = [
        path
        for path in OUT.glob("groq_go25_batch_[0-9][0-9][0-9]_*.stdout.jsonl")
        if (batch_number(path) or -1) >= args.start and (batch_number(path) or -1) <= args.end
    ]
    report = audit_stdout_files(files, validate=not args.no_validate)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"rickshaw_go25_receipt_audit_{args.start:03d}_{args.end:03d}_{stamp()}.json"
    report["report_path"] = rel(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    print("RICKSHAW_GO25_RECEIPT_AUDIT=" + ("PASS" if report["counts"]["validation_ok"] == report["counts"]["stdout_files"] else "FAIL"))
    return 0 if report["counts"]["validation_ok"] == report["counts"]["stdout_files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
