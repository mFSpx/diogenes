#!/usr/bin/env python3
"""Safely stage RunPod embedding JSONL into Postgres.

Behavior:
- Default input is ``05_OUTPUTS/runpod/ingest_artifacts/chunk_embeddings_1128_keyed.jsonl``.
- Validates each JSONL row for required fields and embedding integrity.
- Verifies optional expected row count and file SHA-256.
- Builds an offline SQL/COPY + upsert plan.
- In dry-run mode, only writes a receipt.
- In execute mode, performs bounded batch UPSERTs and requires
  explicit ``--execute`` with ``--database-url`` / ``DATABASE_URL``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import psycopg
except Exception:  # pragma: no cover - environment-dependent optional dependency
    psycopg = None  # type: ignore

SCHEMA = "lucidota.runpod.embedding_stage_import.v1"
DEFAULT_INPUT = Path("05_OUTPUTS/runpod/ingest_artifacts/chunk_embeddings_1128_keyed.jsonl")
DEFAULT_RECEIPT = Path("05_OUTPUTS/runpod/ingest_artifacts/chunk_embeddings_1128_keyed_stage_import_receipt.json")
DEFAULT_IMPORT_TABLE = "lucidota_scratch.runpod_chunk_embedding_stage"
DEFAULT_BATCH_SIZE = 250
REQUIRED_FIELDS = ("chunk_id", "text_sha256", "model", "dimensions", "embedding")
DEFAULT_COLUMNS = [
    "chunk_id",
    "text_sha256",
    "status",
    "provider",
    "model",
    "dimensions",
    "embedding_json",
    "error",
    "source_path",
    "chunk_text_preview",
]


def _now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _coerce_embedding(values: Any) -> list[float]:
    if not isinstance(values, list):
        raise ValueError("embedding must be a JSON array")
    output: list[float] = []
    for idx, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"embedding[{idx}] must be numeric")
        as_float = float(item)
        if not math.isfinite(as_float):
            raise ValueError(f"embedding[{idx}] must be finite")
        output.append(as_float)
    return output


def _validate_required_fields(row: dict[str, Any], line_no: int) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in row or row[field] is None:
            errors.append(f"missing required field: {field}")
            continue

    if errors:
        return errors, None

    if not isinstance(row.get("chunk_id"), str) or not str(row["chunk_id"]).strip():
        errors.append("chunk_id must be a non-empty string")
    if not isinstance(row.get("text_sha256"), str) or not row["text_sha256"]:
        errors.append("text_sha256 must be a non-empty string")
    if not isinstance(row.get("model"), str):
        errors.append("model must be a string")

    try:
        dimensions = int(row.get("dimensions", 0))
    except Exception:
        dimensions = None
        errors.append("dimensions must be an integer")

    try:
        embedding = _coerce_embedding(row.get("embedding"))
    except Exception as exc:
        errors.append(str(exc))
        embedding = []

    if isinstance(dimensions, int) and dimensions > 0:
        if dimensions != len(embedding):
            errors.append("dimensions must equal len(embedding)")
    elif isinstance(dimensions, int) and dimensions == 0:
        errors.append("dimensions must be > 0")

    if line_no and not (row.get("status") is None or isinstance(row.get("status"), str)):
        errors.append("status must be a string")

    if errors:
        return errors, None

    normalized: dict[str, Any] = {
        "chunk_id": str(row["chunk_id"]).strip(),
        "text_sha256": str(row["text_sha256"]).strip(),
        "status": str(row.get("status", "EMBEDDED")) if row.get("status") is not None else "EMBEDDED",
        "provider": str(row.get("provider", "")),
        "model": str(row["model"]),
        "dimensions": dimensions,
        "embedding": embedding,
        "error": row.get("error") if row.get("error") is None else str(row.get("error")),
        "source_path": str(row.get("source_path", "")),
        "chunk_text_preview": str(row.get("chunk_text_preview", "")),
    }

    return [], normalized


def read_and_validate_jsonl(
    path: Path,
    *,
    expected_rows: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    if not path.exists():
        raise FileNotFoundError(path)

    observed_sha = file_sha256(path)
    observed_rows = 0
    valid_rows: list[dict[str, Any]] = []
    row_errors: list[dict[str, Any]] = []

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not raw_line.strip():
            continue

        observed_rows += 1
        try:
            payload = json.loads(raw_line)
        except Exception as exc:
            row_errors.append({"line": line_no, "error": f"invalid json: {exc}"})
            continue

        if not isinstance(payload, dict):
            row_errors.append({"line": line_no, "error": "row is not a JSON object"})
            continue

        row_errors_list, normalized = _validate_required_fields(payload, line_no)
        if row_errors_list:
            row_errors.append({"line": line_no, "error": "; ".join(row_errors_list)})
            continue
        valid_rows.append(normalized)

    row_count_ok = expected_rows is None or observed_rows == expected_rows
    expected_sha = (expected_sha256 or "").lower() if expected_sha256 else None
    sha_ok = expected_sha is None or observed_sha == expected_sha

    report = {
        "input_path": str(path),
        "observed_lines": observed_rows,
        "valid_rows": len(valid_rows),
        "invalid_rows": len(row_errors),
        "row_errors": row_errors,
        "expected_rows": expected_rows,
        "row_count_ok": row_count_ok,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha,
        "sha256_ok": sha_ok,
    }

    return valid_rows, report, bool((not row_errors) and row_count_ok and sha_ok)


def vector_as_pg_literal(values: list[float]) -> str:
    if not values:
        return "[]"
    return "[" + ",".join(f"{v:.8g}" for v in values) + "]"


def build_import_plan(
    rows: list[dict[str, Any]],
    *,
    import_table: str,
    columns: list[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    cols = columns or DEFAULT_COLUMNS
    copy_csv_path = "<RENDERED_IMPORT_PATH>"
    update_cols = [c for c in cols if c != "chunk_id"]
    columns_sql = ", ".join(cols)

    copy_stmt = """
-- SQL/COPY Plan for staging import from a prepared CSV stream.
-- 1) Render a CSV with columns: {columns_sql}.
-- 2) Run in psql:
\\copy {import_table}({columns_sql}) FROM '{copy_csv_path}' WITH (FORMAT csv, QUOTE '"', DELIMITER ',');
-- NOTE: this importer validates and executes upserts directly without writing to staging files.
""".format(
        columns_sql=columns_sql,
        import_table=import_table,
        copy_csv_path=copy_csv_path,
    ).strip()

    placeholders = ", ".join(["%s::jsonb" if c == "embedding_json" else "%s" for c in cols])
    conflict_clause = "chunk_id"
    update_clause = ", ".join(
        [f"{col} = EXCLUDED.{col}" for col in update_cols],
    )
    upsert_stmt = (
        f"INSERT INTO {import_table} ({columns_sql}) VALUES ({placeholders})\n"
        f"ON CONFLICT ({conflict_clause}) DO UPDATE SET\n    {update_clause};"
    )

    return {
        "ok": True,
        "status": "PLAN_READY",
        "table": import_table,
        "columns": cols,
        "batch_size": batch_size,
        "copy_stmt": copy_stmt.strip(),
        "upsert_stmt": upsert_stmt,
        "estimated_rows": len(rows),
        "notes": "Dry-run mode only. Execute mode performs parameterized UPSERT batches.",
    }


def _row_to_db_values(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    values: list[Any] = []
    for col in columns:
        if col == "embedding_json":
            values.append(json.dumps(row["embedding"], separators=(",", ":")))
        else:
            values.append(row.get(col))
    return tuple(values)


def execute_import(
    *,
    database_url: str,
    rows: list[dict[str, Any]],
    import_table: str,
    columns: list[str],
    batch_size: int,
    connect_fn: Callable[..., Any] = None,
) -> dict[str, Any]:
    if connect_fn is None:
        if psycopg is None:  # pragma: no cover - import-path guard
            raise RuntimeError("psycopg unavailable")
        connect_fn = psycopg.connect

    safe_batch_size = max(1, min(1000, batch_size))
    update_cols = [c for c in columns if c != "chunk_id"]
    columns_sql = ", ".join(columns)
    placeholders = ", ".join(["%s::jsonb" if c == "embedding_json" else "%s" for c in columns])
    update_clause = ", ".join([f"{col}=EXCLUDED.{col}" for col in update_cols])
    sql = (
        f"INSERT INTO {import_table} ({columns_sql}) VALUES ({placeholders}) "
        f"ON CONFLICT (chunk_id) DO UPDATE SET {update_clause}"
    )

    executed = 0
    with connect_fn(database_url) as conn:
        with conn.cursor() as cur:
            for start in range(0, len(rows), safe_batch_size):
                batch = rows[start : start + safe_batch_size]
                payload = [_row_to_db_values(r, columns) for r in batch]
                cur.executemany(sql, payload)
                executed += len(payload)
        conn.commit()

    return {
        "status": "DONE",
        "executed_rows": executed,
        "executed_batches": (len(rows) + safe_batch_size - 1) // safe_batch_size,
        "upsert_sql": sql,
        "batch_size": safe_batch_size,
        "table": import_table,
    }


def load_env_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.database_url:
        args.database_url = os.environ.get("DATABASE_URL") or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="runpod-embedding-stage-import",
        description="Validate and stage RunPod embedding JSONL into Postgres safely.",
    )
    ap.add_argument("--input", default=DEFAULT_INPUT, type=Path, help="Path to source embedding JSONL")
    ap.add_argument("--import-table", default=DEFAULT_IMPORT_TABLE, help="Target staging table")
    ap.add_argument("--expected-rows", type=int, help="Expected source row count")
    ap.add_argument("--expected-sha256", help="Expected source sha256 hex")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Max rows per DB batch")
    ap.add_argument("--execute", action="store_true", help="Actually write to DB (disabled by default)")
    ap.add_argument("--database-url", help="Postgres DSN (required with --execute)")
    ap.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT, help="Receipt output path")
    ap.add_argument("--json", action="store_true", help="Print receipt as JSON")
    return ap.parse_args(argv)


def build_receipt(
    *,
    args: argparse.Namespace,
    validation: dict[str, Any],
    import_plan: dict[str, Any],
    validation_ok: bool,
    execution: dict[str, Any] | None = None,
    execute_error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "PASS" if validation_ok and not execute_error else "FAIL",
        "generated_at": _now_z(),
        "input": str(args.input),
        "import_table": args.import_table,
        "batch_size": args.batch_size,
        "dry_run": not args.execute,
        "execute": bool(args.execute),
        "expected_rows": args.expected_rows,
        "expected_sha256": args.expected_sha256,
        "validation": validation,
        "import_plan": import_plan,
        "execution_status": None if execution is None and not execute_error else (execution.get("status") if execution else None),
        "execution": execution,
        "error": execute_error,
    }


def main(argv: list[str] | None = None) -> int:
    args = load_env_args(parse_args(argv))
    try:
        rows, validation, validation_ok = read_and_validate_jsonl(
            args.input,
            expected_rows=args.expected_rows,
            expected_sha256=args.expected_sha256,
        )
    except FileNotFoundError:
        payload = {
            "schema": SCHEMA,
            "status": "FAIL",
            "generated_at": _now_z(),
            "input": str(args.input),
            "validation": {"status": "FILE_NOT_FOUND"},
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True) if args.json else json.dumps(payload, indent=2, sort_keys=True))
        return 1

    import_plan = build_import_plan(rows, import_table=args.import_table, batch_size=args.batch_size)

    execution: dict[str, Any] | None = None
    execute_error: str | None = None
    if args.execute:
        if not args.database_url:
            execute_error = "Missing DATABASE_URL for --execute"
        elif not validation_ok:
            execute_error = "Validation failed; execution skipped"
        else:
            try:
                execution = execute_import(
                    database_url=args.database_url,
                    rows=rows,
                    import_table=args.import_table,
                    columns=import_plan["columns"],
                    batch_size=args.batch_size,
                )
            except Exception as exc:
                execute_error = f"Execution failed: {exc}"

    status_payload = build_receipt(
        args=args,
        validation=validation,
        import_plan=import_plan,
        validation_ok=validation_ok,
        execution=execution,
        execute_error=execute_error,
    )

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(status_payload, indent=2, sort_keys=True) if args.json else json.dumps(status_payload, sort_keys=True))

    if args.execute:
        return 0 if execute_error is None else 1
    return 0 if validation_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
