#!/usr/bin/env python3
"""Seed general LUCIDOTA workflow registry rows into the sheet-task layer.

Dry-run by default. `--apply` writes only lucidota_sheet.sheet_task rows for the
explicit workflow targets in the registry; no graph/canon mutation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "04_RUNTIME/lucidota_workflow_registry.json"
DEFAULT_RECEIPT = ROOT / "05_OUTPUTS/runtime/lucidota_workflow_seed_latest.json"
ALLOWED = {
    "FILTER_SHEET",
    "STATUS_SHEET",
    "PIVOT_SHEET",
    "SCORE_SHEET",
    "DIFF_SHEET",
    "REFRESH_SHEET",
    "EXPORT_SHEET",
    "IMPORT_SHEET",
    "PROMOTION_SHEET",
    "DEADLETTER_SHEET",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_workflow(w: dict[str, Any], max_rows_limit: int) -> list[str]:
    errors: list[str] = []
    if w.get("task_type") != "SHEET_TASK":
        errors.append(f"{w.get('target')}:task_type_not_sheet_task")
    if w.get("task_class") not in ALLOWED:
        errors.append(f"{w.get('target')}:invalid_task_class")
    if not w.get("receipt_required"):
        errors.append(f"{w.get('target')}:receipt_not_required")
    if w.get("body_policy") != "refs_not_bodies":
        errors.append(f"{w.get('target')}:body_policy_not_refs")
    query = str(w.get("query_sql") or "")
    upper = query.upper()
    if "SELECT *" in upper:
        errors.append(f"{w.get('target')}:select_star_forbidden")
    if "LIMIT" not in upper and not upper.strip().startswith("REFRESH MATERIALIZED VIEW"):
        errors.append(f"{w.get('target')}:live_query_without_limit")
    try:
        max_rows = int(w.get("max_rows", 0))
    except Exception:
        max_rows = 0
    if max_rows < 1 or max_rows > max_rows_limit:
        errors.append(f"{w.get('target')}:max_rows_out_of_bounds")
    return errors


def normalize_task(w: dict[str, Any]) -> dict[str, Any]:
    query = str(w.get("query_sql") or "")
    return {
        "task_type": "SHEET_TASK",
        "task_class": w["task_class"],
        "target": w["target"],
        "title": w.get("title", w["target"]),
        "status": w.get("status", "OPEN"),
        "friction_score": int(w.get("friction_score", 50)),
        "receipt_count": 0,
        "source_tables": list(w.get("source_tables", [])),
        "query_sql": query,
        "query_hash": sha256(query.encode()).hexdigest(),
        "max_rows": int(w.get("max_rows", 1000)),
        "budget_ms": int(w.get("budget_ms", 5000)),
        "domain": w.get("domain", "general"),
        "body_policy": w.get("body_policy", "refs_not_bodies"),
        "receipt_required": bool(w.get("receipt_required", True)),
        "script_candidates": list(w.get("script_candidates", [])),
    }


def apply_tasks(tasks: list[dict[str, Any]], dsn: str) -> dict[str, int]:
    try:
        import psycopg
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"psycopg unavailable: {exc}") from exc

    targets = [t["target"] for t in tasks]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM lucidota_sheet.sheet_task WHERE target = ANY(%s)", (targets,))
            deleted = cur.rowcount
            inserted = 0
            for t in tasks:
                cur.execute(
                    """
                    INSERT INTO lucidota_sheet.sheet_task(
                      task_type, task_class, target, title, status, friction_score,
                      receipt_count, source_tables, query_sql, max_rows, budget_ms
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        t["task_type"],
                        t["task_class"],
                        t["target"],
                        t["title"],
                        t["status"],
                        t["friction_score"],
                        t["receipt_count"],
                        t["source_tables"],
                        t["query_sql"],
                        t["max_rows"],
                        t["budget_ms"],
                    ),
                )
                inserted += 1
        conn.commit()
    return {"deleted": int(deleted), "inserted": inserted}


def emit(payload: dict[str, Any], json_out: bool) -> None:
    if json_out:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"STATUS={payload['status']} EXECUTION={payload['execution']} TASKS={payload['tasks_seen']} APPLIED={payload.get('applied', False)}")
        print(f"RECEIPT={payload['receipt_path']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    ap.add_argument("--apply", action="store_true", help="write sheet_task rows for registry workflow targets")
    ap.add_argument("--dsn", default=os.environ.get("LUCIDOTA_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    registry_path = Path(args.registry)
    registry = load_registry(registry_path)
    max_rows_limit = int(registry.get("limits", {}).get("max_db_rows_per_batch", 1000))
    errors: list[str] = []
    tasks: list[dict[str, Any]] = []
    for workflow in registry.get("workflows", []):
        errors.extend(validate_workflow(workflow, max_rows_limit))
        tasks.append(normalize_task(workflow))

    apply_result = {"deleted": 0, "inserted": 0}
    status = "PASS" if not errors else "FAIL"
    if args.apply and not errors:
        try:
            apply_result = apply_tasks(tasks, args.dsn)
        except Exception as exc:
            status = "FAIL"
            errors.append(f"apply_failed:{exc}")

    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "lucidota.workflow_task_seed.receipt.v1",
        "generated_at": now(),
        "status": status,
        "execution": "apply" if args.apply else "dry_run",
        "registry_path": rel(registry_path),
        "registry_hash": sha256(registry_path.read_bytes()).hexdigest(),
        "routing_law": registry.get("routing_law", []),
        "tasks_seen": len(tasks),
        "task_targets": [t["target"] for t in tasks],
        "tasks": tasks,
        "errors": errors,
        "would_apply": bool(args.apply),
        "applied": bool(args.apply and status == "PASS"),
        "apply_result": apply_result,
        "db_writes_performed": bool(args.apply and status == "PASS"),
        "graph_writes_performed": False,
        "receipt_path": rel(receipt_path),
    }
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    emit(payload, args.json)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
