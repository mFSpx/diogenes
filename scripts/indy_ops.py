#!/usr/bin/env python3
"""Pure stateless Indy operations.

These helpers intentionally do not touch the database. They atomize input files
into structured records that ABSURD can persist inside a transaction.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ATOMIZE_JSON_SCHEMA = "lucidota.indy_ops.atomize_json.result.v1"
ATOMIZE_CSV_SCHEMA = "lucidota.indy_ops.atomize_csv.result.v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _stable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _json_records(payload: Any, *, source_path: str, source_sha256: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for index, (key, value) in enumerate(payload.items()):
            records.append({
                "record_index": index,
                "record_key": str(key),
                "record_value": value,
                "record_text": _stable_text(value),
                "source_kind": "json",
                "source_path": source_path,
                "source_sha256": source_sha256,
            })
        if not records:
            records.append({
                "record_index": 0,
                "record_key": "{}",
                "record_value": {},
                "record_text": "{}",
                "source_kind": "json",
                "source_path": source_path,
                "source_sha256": source_sha256,
            })
        return records
    if isinstance(payload, list):
        for index, value in enumerate(payload):
            records.append({
                "record_index": index,
                "record_key": f"[{index}]",
                "record_value": value,
                "record_text": _stable_text(value),
                "source_kind": "json",
                "source_path": source_path,
                "source_sha256": source_sha256,
            })
        return records
    return [{
        "record_index": 0,
        "record_key": "value",
        "record_value": payload,
        "record_text": _stable_text(payload),
        "source_kind": "json",
        "source_path": source_path,
        "source_sha256": source_sha256,
    }]


def handle_atomize_json_file(path: str | Path) -> dict[str, Any]:
    source = _path(path)
    raw = source.read_text(encoding="utf-8")
    payload = json.loads(raw)
    source_sha256 = sha256_text(raw)
    records = _json_records(payload, source_path=str(source), source_sha256=source_sha256)
    return {
        "schema": ATOMIZE_JSON_SCHEMA,
        "outcome": "succeeded",
        "source_kind": "json",
        "source_path": str(source),
        "source_sha256": source_sha256,
        "record_count": len(records),
        "records": records,
        "source_preview": _stable_text(payload)[:1000],
    }


def handle_atomize_csv_file(path: str | Path) -> dict[str, Any]:
    source = _path(path)
    raw = source.read_text(encoding="utf-8", errors="ignore")
    source_sha256 = sha256_text(raw)
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(raw.splitlines())
    headers = list(reader.fieldnames or [])
    for index, row in enumerate(reader):
        rows.append({
            "record_index": index,
            "record_key": f"row_{index}",
            "record_value": row,
            "record_text": _stable_text(row),
            "source_kind": "csv",
            "source_path": str(source),
            "source_sha256": source_sha256,
            "csv_headers": headers,
        })
    return {
        "schema": ATOMIZE_CSV_SCHEMA,
        "outcome": "succeeded",
        "source_kind": "csv",
        "source_path": str(source),
        "source_sha256": source_sha256,
        "header_fields": headers,
        "record_count": len(rows),
        "records": rows,
        "source_preview": raw[:1000],
    }
