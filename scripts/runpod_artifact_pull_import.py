#!/usr/bin/env python3
"""RunPod/Jupyter artifact pull + local Postgres import plan skeleton.

This CLI pulls a remote file via Jupyter Contents API, verifies optional size/
sha256, writes to a local output path, and can emit a deterministic import
plan for COPY + UPSERT without requiring real DB credentials when in dry-run mode.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests

SCHEMA = "lucidota.runpod.artifact_pull_import.v1"
DEFAULT_TIMEOUT = 30


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size


def _quoted_path(remote_path: str) -> str:
    safe = "/"
    return quote(remote_path.lstrip("/"), safe=safe)


def build_contents_url(base_url: str, remote_path: str) -> str:
    return f"{base_url.rstrip('/')}/api/contents/{_quoted_path(remote_path)}?download=1"


def pull_artifact(
    *,
    jupyter_url: str,
    token: str | None,
    remote_path: str,
    output: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    request_get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"token {token}"

    response = request_get(
        build_contents_url(jupyter_url, remote_path),
        headers=headers,
        timeout=timeout,
        stream=True,
    )
    if response.status_code != 200:  # avoid hard dependency on raise_for_status() in tests
        return {
            "ok": False,
            "status": f"HTTP_{response.status_code}",
            "status_code": response.status_code,
            "error": getattr(response, "text", "")[:400],
            "url": build_contents_url(jupyter_url, remote_path),
        }

    content_type = (getattr(response, "headers", {}) or {}).get("content-type", "")
    payload: bytes | None = None
    if "application/json" in content_type.lower():
        try:
            envelope = response.json()
        except Exception:
            envelope = None
        if isinstance(envelope, dict) and "content" in envelope:
            content = envelope.get("content") or ""
            fmt = (envelope.get("format") or "text").lower()
            if fmt == "base64":
                payload = base64.b64decode(content)
            elif fmt == "text":
                payload = str(content).encode("utf-8")

    hasher = hashlib.sha256()
    bytes_read = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as out:
        if payload is not None:
            out.write(payload)
            bytes_read = len(payload)
            hasher.update(payload)
        else:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                out.write(chunk)
                bytes_read += len(chunk)
                hasher.update(chunk)

    observed_sha = hasher.hexdigest()
    observed_size = bytes_read if bytes_read > 0 else file_size_bytes(output)

    size_ok = expected_size is None or observed_size == expected_size
    sha_ok = expected_sha256 is None or observed_sha.lower() == expected_sha256.lower()

    if expected_size is not None and not size_ok:
        return {
            "ok": False,
            "status": "SIZE_MISMATCH",
            "remote_path": remote_path,
            "observed_size": observed_size,
            "expected_size": expected_size,
            "output": str(output),
            "sha256": observed_sha,
            "sha256_ok": sha_ok,
            "size_ok": size_ok,
        }

    if expected_sha256 is not None and not sha_ok:
        return {
            "ok": False,
            "status": "SHA256_MISMATCH",
            "remote_path": remote_path,
            "observed_sha256": observed_sha,
            "expected_sha256": expected_sha256,
            "output": str(output),
            "sha256_ok": sha_ok,
            "size_ok": size_ok,
        }

    return {
        "ok": True,
        "status": "PULLED",
        "remote_path": remote_path,
        "output": str(output),
        "observed_size": observed_size,
        "expected_size": expected_size,
        "observed_sha256": observed_sha,
        "expected_sha256": expected_sha256,
        "sha256_ok": sha_ok,
        "size_ok": size_ok,
    }


def _infer_columns_from_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            row = next(reader, [])
            return [c.strip() for c in row if c.strip()]

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    return []
                if isinstance(obj, dict):
                    return list(obj.keys())
                return []

    return []


def _clean_sql_id(value: str) -> str:
    return value.strip().strip("\"")


def build_import_plan(
    *,
    output: Path,
    table: str,
    import_columns: list[str] | None = None,
    conflict_columns: list[str] | None = None,
    update_columns: list[str] | None = None,
) -> dict[str, Any]:
    columns = list(import_columns or [])
    if not columns:
        columns = _infer_columns_from_file(output)

    if not columns:
        return {
            "ok": False,
            "status": "MISSING_IMPORT_COLUMNS",
            "table": table,
            "copy_stmt": None,
            "upsert_stmt": None,
            "message": "Provide --import-columns for non-CSV/TSV/JSONL or when header is unavailable.",
            "copy_path": str(output),
        }

    columns_sql = ", ".join(_clean_sql_id(c) for c in columns)
    conflict = conflict_columns or ["id"]
    conflict_sql = ", ".join(_clean_sql_id(c) for c in conflict)

    is_csv = output.suffix.lower() in {".csv", ".tsv", ".txt"}
    delimiter = "\t" if output.suffix.lower() == ".tsv" else ","

    copy_stmt = (
        f"\\copy {table} ({columns_sql}) FROM '{output.resolve()}' "
        f"WITH (FORMAT csv, HEADER true, DELIMITER '{delimiter}');"
        if is_csv
        else f"-- COPY format cannot be inferred for {output.name}; provide --import-columns + --delimiter override manually"
    )

    if not update_columns:
        set_columns = [c for c in columns if c not in conflict]
    else:
        set_columns = update_columns

    update_sql = ", ".join(f"{_clean_sql_id(c)} = EXCLUDED.{_clean_sql_id(c)}" for c in set_columns)
    upsert_stmt = (
        f"INSERT INTO {table} ({columns_sql}) VALUES ({{row_values}})"
        f"\nON CONFLICT ({conflict_sql}) DO UPDATE SET {update_sql};"
        if update_sql
        else f"INSERT INTO {table} ({columns_sql}) VALUES ({{row_values}})\nON CONFLICT ({conflict_sql}) DO NOTHING;"
    )

    return {
        "ok": True,
        "status": "PLAN_READY",
        "table": table,
        "copy_stmt": copy_stmt,
        "upsert_stmt": upsert_stmt,
        "columns": columns,
        "conflict_columns": conflict,
        "update_columns": set_columns,
        "copy_path": str(output),
        "notes": "Dry-run/import-plan mode emits SQL only; execute-mode is intentionally guarded behind credentials and --execute.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="runpod-artifact-pull-import",
        description="Pull a Jupyter Contents artifact and optionally build Postgres import SQL plan.",
    )
    ap.add_argument("--jupyter-url", help="RunPod/Jupyter base URL (or env JUPYTER_URL)")
    ap.add_argument("--jupyter-token", help="RunPod/Jupyter API token (or env JUPYTER_TOKEN)")
    ap.add_argument("--remote-path", required=True, help="Remote artifact path in Jupyter contents API")
    ap.add_argument("--output", required=True, type=Path, help="Local output path")
    ap.add_argument("--expected-size", type=int, help="Optional expected size in bytes")
    ap.add_argument("--expected-sha256", help="Optional expected sha256 hex")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--import-table", help="Target Postgres table for plan generation")
    ap.add_argument("--import-columns", nargs="+", help="Columns for import (space/comma separated)")
    ap.add_argument("--conflict-columns", nargs="+", default=["id"], help="Conflict columns for UPSERT")
    ap.add_argument("--update-columns", nargs="+", help="Explicit UPSERT update columns")
    ap.add_argument("--dry-run", action="store_true", help="Do not execute DB writes; emit import plan only")
    ap.add_argument("--execute", action="store_true", help="Execute DB import (requires --import-table and DB credentials)")
    ap.add_argument("--database-url", help="DB URL, default env DATABASE_URL")
    ap.add_argument("--receipt", type=Path, default=Path("05_OUTPUTS/runpod_artifact_pull_import_receipt.json"))
    ap.add_argument("--json", action="store_true", help="Print receipt JSON")
    return ap.parse_args(argv)


def load_env_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.jupyter_url:
        args.jupyter_url = os.environ.get("JUPYTER_URL") or os.environ.get("RUNPOD_JUPYTER_URL")
    if not args.jupyter_token:
        args.jupyter_token = os.environ.get("JUPYTER_TOKEN") or os.environ.get("RUNPOD_JUPYTER_TOKEN")
    if not args.database_url:
        args.database_url = os.environ.get("DATABASE_URL") or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")

    if args.import_columns:
        parsed: list[str] = []
        for token in args.import_columns:
            parsed.extend(part.strip() for part in token.split(",") if part.strip())
        args.import_columns = parsed

    return args


def build_receipt(
    *,
    args: argparse.Namespace,
    pull_result: dict[str, Any],
    import_plan: dict[str, Any] | None = None,
    execution_status: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "PASS" if pull_result.get("ok") else "FAIL",
        "jupyter_url": args.jupyter_url,
        "remote_path": args.remote_path,
        "output": str(args.output),
        "expected_size": args.expected_size,
        "expected_sha256": args.expected_sha256,
        "import_requested": bool(args.import_table),
        "dry_run": args.dry_run,
        "execute": args.execute,
        "import_table": args.import_table,
        "import_plan": import_plan,
        "execution_status": execution_status,
        "pull": pull_result,
    }


def main(argv: list[str] | None = None) -> int:
    args = load_env_args(parse_args(argv))
    if not args.jupyter_url:
        return _fail("missing jupyter URL; set --jupyter-url or JUPYTER_URL")

    pull_result = pull_artifact(
        jupyter_url=args.jupyter_url,
        token=args.jupyter_token,
        remote_path=args.remote_path,
        output=args.output,
        expected_size=args.expected_size,
        expected_sha256=args.expected_sha256,
        timeout=args.timeout,
    )
    execution_status = None
    plan = None

    if args.import_table:
        if args.dry_run or not args.execute:
            plan = build_import_plan(
                output=args.output,
                table=args.import_table,
                import_columns=args.import_columns,
                conflict_columns=args.conflict_columns,
                update_columns=args.update_columns,
            )
        else:
            if not args.database_url:
                execution_status = "FAILED_MISSING_DATABASE_URL"
            else:
                # Intentional skeleton behavior: execution is deferred to avoid implicit
                # external side effects in minimal mode. This keeps tests safe and focuses
                # this CLI on artifact pull + import plan correctness.
                execution_status = "DEFERRED_IN_SKELETON_MODE"

    receipt = build_receipt(args=args, pull_result=pull_result, import_plan=plan, execution_status=execution_status)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))

    if not pull_result.get("ok"):
        return 1
    if execution_status in {"FAILED_MISSING_DATABASE_URL"}:
        return 1
    return 0


def _fail(message: str) -> int:
    payload = {"schema": SCHEMA, "status": "FAIL", "error": message}
    print(json.dumps(payload, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
