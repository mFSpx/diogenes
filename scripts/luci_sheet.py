#!/usr/bin/env python3
"""LUCIDOTA Sheet Layer CLI.

Dry by default: lists/plans/explains database-native spreadsheet sheets. Use this
as the stable interface for `luci sheet ...`; actual DB execution can be wired
behind the same manifest without turning every spreadsheet task into Python.
"""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_sheet(manifest: dict[str, Any], sheet_id: str) -> dict[str, Any] | None:
    for sheet in manifest.get("sheets", []):
        if sheet.get("id") == sheet_id:
            return sheet
    return None


def emit(payload: Any, json_out: bool) -> int:
    if json_out:
        print(json.dumps(payload, indent=None, sort_keys=True))
    else:
        if isinstance(payload, list):
            for item in payload:
                print(item)
        elif isinstance(payload, dict):
            for k, v in payload.items():
                print(f"{k}: {v}")
        else:
            print(payload)
    return 0


def unknown(sheet_id: str, json_out: bool) -> int:
    print(json.dumps({"error": "unknown_sheet", "sheet": sheet_id}, sort_keys=True) if json_out else f"unknown_sheet: {sheet_id}")
    return 2


def validate_query(sql: str) -> list[str]:
    errors: list[str] = []
    upper = sql.upper()
    if "SELECT *" in upper:
        errors.append("select_star_forbidden")
    if "LIMIT" not in upper and "REFRESH MATERIALIZED VIEW" not in upper and not upper.strip().startswith("COPY"):
        errors.append("live_query_without_limit")
    return errors


def export_sql(sheet: dict[str, Any], fmt: str) -> str:
    query = str(sheet.get("query") or "")
    return f"COPY ({query}) TO STDOUT WITH (FORMAT {fmt}, HEADER true)"


def main() -> int:
    ap = argparse.ArgumentParser(prog="luci sheet")
    ap.add_argument("--manifest", default="04_RUNTIME/lucidota_sheet_manifest.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["list", "validate", "current"]:
        p = sub.add_parser(name)
        p.add_argument("--json", action="store_true")
    for name in ["show", "explain", "refresh", "diff", "promote"]:
        p = sub.add_parser(name)
        p.add_argument("sheet")
        p.add_argument("--json", action="store_true")
    p = sub.add_parser("export")
    p.add_argument("sheet")
    p.add_argument("--format", choices=["csv", "json", "parquet"], default="csv")
    p.add_argument("--json", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    json_out = bool(getattr(args, "json", False))

    if args.cmd == "list":
        sheets = [
            {"id": s["id"], "kind": s["kind"], "class": s["class"], "object": s["database_object"]}
            for s in manifest.get("sheets", [])
        ]
        return emit({"schema": manifest.get("schema"), "sheets": sheets}, json_out)

    if args.cmd == "current":
        payload = {
            "schema": manifest.get("schema"),
            "sheet_ids": [s["id"] for s in manifest.get("sheets", [])],
            "current_route": "/sheet_current",
            "current_object": "lucidota_sheet.sheet_current",
            "operator_use": "inspect live spreadsheet-style status, active work, and next batch through PostgREST",
            "routing_order": manifest.get("routing_order"),
        }
        return emit(payload, json_out)

    if args.cmd == "validate":
        errors = []
        for sheet in manifest.get("sheets", []):
            errors.extend(f"{sheet.get('id')}:{e}" for e in validate_query(str(sheet.get("query", ""))))
            if not sheet.get("receipt_required"):
                errors.append(f"{sheet.get('id')}:receipt_not_required")
        return emit({"ok": not errors, "errors": errors}, json_out)

    sheet = find_sheet(manifest, args.sheet)
    if sheet is None:
        return unknown(args.sheet, json_out)

    if args.cmd == "show":
        return emit(sheet, json_out)

    if args.cmd == "explain":
        query = str(sheet.get("query") or "")
        payload = {
            "sheet": sheet["id"],
            "execution": "dry_run",
            "kind": sheet["kind"],
            "query": query,
            "query_hash": sha256(query.encode()).hexdigest(),
            "max_rows": sheet["max_rows"],
            "budget_ms": sheet["budget_ms"],
            "routing_order": manifest.get("routing_order"),
        }
        return emit(payload, json_out)

    if args.cmd == "refresh":
        sql = str(sheet.get("refresh_sql") or f"REFRESH MATERIALIZED VIEW {sheet.get('database_object')}")
        payload = {
            "operation": "refresh_projection",
            "sheet": sheet["id"],
            "execution": "dry_run",
            "receipt_required": bool(sheet.get("receipt_required")),
            "source_tables": sheet.get("source_tables", []),
            "sql": sql,
            "query_hash": sha256(sql.encode()).hexdigest(),
        }
        return emit(payload, json_out)

    if args.cmd == "export":
        sql = export_sql(sheet, args.format)
        payload = {
            "operation": "export_sheet",
            "sheet": sheet["id"],
            "format": args.format,
            "execution": "dry_run",
            "receipt_required": bool(sheet.get("receipt_required")),
            "sql": sql,
            "query_hash": sha256(sql.encode()).hexdigest(),
        }
        return emit(payload, json_out)

    if args.cmd in {"diff", "promote"}:
        operation = "diff_sheet" if args.cmd == "diff" else "promote_sheet"
        payload = {
            "operation": operation,
            "sheet": sheet["id"],
            "execution": "dry_run",
            "receipt_required": bool(sheet.get("receipt_required")),
            "note": "planned surface; requires live DB executor/approval before mutation",
        }
        return emit(payload, json_out)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
