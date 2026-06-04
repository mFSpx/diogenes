#!/usr/bin/env python3
"""DB-backed test receipt gate.

Tests are receipts, not vibes. Postgres is the source of truth; file-plane
artifacts are only inputs to the dependency signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from spine_common import rel, sha256_bytes, sha256_json


DEFAULT_DSN = (
    os.environ.get("LUCIDOTA_AUDIT_DATABASE_URL")
    or os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
    or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
    or os.environ.get("LUCIDOTA_GO_STATE_DSN")
    or "postgresql:///lucidota_state"
)
DEFAULT_OPENAPI_BASE_URL = os.environ.get("POSTGREST_BASE_URL", "http://127.0.0.1:3000").rstrip("/")


class DBBlocked(RuntimeError):
    """Raised when the gate cannot read/write DB truth."""


@dataclass(frozen=True)
class GitProbe:
    branch: str
    commit: str
    diff_hash: str
    status_hash: str
    dirty: bool


@dataclass(frozen=True)
class GateResult:
    exit_code: int
    status: str
    receipt_uuid: str | None
    decision: str
    dependency_signature: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def normalize_command(command: list[str]) -> str:
    return shlex.join(command)


def _classify_watch(path: Path) -> str:
    s = str(path).replace("\\", "/")
    suffix = path.suffix.lower()
    if suffix == ".sql" or "/06_SCHEMA/" in s or s.endswith(".migration.sql"):
        return "sql"
    if suffix in {".service", ".timer", ".socket"} or "/services/" in s:
        return "systemd"
    return "file"


def _surface_tokens(command_text: str, watched_records: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    try:
        tokens.add(Path(shlex.split(command_text)[0]).name.lower())
    except Exception:
        tokens.add(command_text.lower().split()[0] if command_text.split() else "")
    for row in watched_records:
        tokens.update(part.lower() for part in Path(str(row["path"])).parts)
    return tokens


def _file_hash(path: Path) -> str:
    if not path.exists():
        return sha256_text(f"MISSING:{rel(path)}")
    return sha256_bytes(path.read_bytes())


def _hash_json(value: Any) -> str:
    return sha256_text(stable_json(value))


def probe_git(watched_paths: Iterable[str] | None = None, cwd: Path | None = None) -> GitProbe:
    base = cwd or ROOT
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=base,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        branch = ""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = ""

    status_text = ""
    diff_text = ""
    try:
        status_text = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            cwd=base,
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8", "replace")
    except Exception:
        status_text = ""

    diff_cmd = ["git", "diff", "--no-ext-diff", "--binary"]
    paths = list(dict.fromkeys(str(Path(p)) for p in (watched_paths or [])))
    if paths:
        diff_cmd += ["--", *paths]
    try:
        diff_text = subprocess.run(
            diff_cmd,
            cwd=base,
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8", "replace")
    except Exception:
        diff_text = ""

    dirty = bool(status_text.strip())
    return GitProbe(
        branch=branch,
        commit=commit,
        diff_hash=sha256_text(diff_text),
        status_hash=sha256_text(status_text),
        dirty=dirty,
    )


def openapi_probe(base_url: str = DEFAULT_OPENAPI_BASE_URL) -> dict[str, Any]:
    import urllib.request
    import urllib.error

    try:
        with urllib.request.urlopen(base_url + "/", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "{}")
            paths = payload.get("paths", {}) if isinstance(payload, dict) else {}
            return {"included": True, "base_url": base_url, "hash": sha256_json(paths)}
    except Exception:
        return {"included": False, "base_url": base_url, "hash": None}


def env_probe(env_keys: Iterable[str]) -> dict[str, Any]:
    entries = []
    for key in sorted(set(env_keys)):
        entries.append(
            {
                "name": key,
                "value_sha256": sha256_text(os.environ.get(key, "")),
                "source": "env",
            }
        )
    return {
        "tracked_scope": "worker" if entries else "none",
        "entries": entries,
        "env_config_hash": _hash_json(entries),
    }


def build_dependency_payload(
    *,
    scope: str,
    command_text: str,
    watched_paths: list[str],
    cwd: Path,
    include_openapi: bool | None,
    env_keys: Iterable[str],
    git_probe: GitProbe | None = None,
    git_probe_fn: Any | None = None,
    openapi_fetcher: Any | None = None,
) -> dict[str, Any]:
    watched_records = []
    sql_hashes = []
    systemd_hashes = []
    for raw in watched_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = (cwd / path).resolve()
        kind = _classify_watch(path)
        row = {
            "path": rel(path),
            "kind": kind,
            "exists": path.exists(),
            "sha256": _file_hash(path),
        }
        watched_records.append(row)
        if kind == "sql":
            sql_hashes.append(row["sha256"])
        elif kind == "systemd":
            systemd_hashes.append(row["sha256"])

    git = git_probe or (git_probe_fn(watched_paths) if callable(git_probe_fn) else git_probe_fn)
    if git is None:
        git = probe_git(watched_paths, cwd=cwd)

    surface_tokens = _surface_tokens(command_text, watched_records)
    api_included = include_openapi if include_openapi is not None else (
        any(item["kind"] in {"sql", "systemd"} for item in watched_records)
        or bool(surface_tokens & {"luci", "indy", "postgrest", "api"})
    )
    api = openapi_fetcher() if api_included and callable(openapi_fetcher) else (openapi_probe() if api_included else {"included": False, "base_url": DEFAULT_OPENAPI_BASE_URL, "hash": None})

    payload = {
        "schema_version": "v1",
        "scope": scope,
        "cwd": rel(cwd),
        "command_text": command_text,
        "command_sha256": sha256_text(command_text),
        "watched_files": watched_records,
        "sql_migration_hashes": sql_hashes,
        "systemd_unit_hashes": systemd_hashes,
        "postgrest_openapi": api,
        "env_config": env_probe(env_keys),
        "git": {
            "branch": git.branch,
            "commit": git.commit,
            "git_diff_hash": git.diff_hash,
            "git_status_hash": git.status_hash,
            "dirty": git.dirty,
        },
        "invalidation_rules": [
            {"rule_id": "COMMAND_CHANGED", "field": "command_sha256", "severity": "hard"},
            {"rule_id": "FILE_HASH_MISMATCH", "field": "watched_files", "severity": "hard"},
            {"rule_id": "MIGRATION_HASH_MISMATCH", "field": "sql_migration_hashes", "severity": "hard"},
            {"rule_id": "SYSTEMD_UNIT_HASH_MISMATCH", "field": "systemd_unit_hashes", "severity": "hard"},
            {"rule_id": "OPENAPI_HASH_MISMATCH", "field": "postgrest_openapi.hash", "severity": "hard"},
            {"rule_id": "ENV_CONFIG_HASH_MISMATCH", "field": "env_config.env_config_hash", "severity": "hard"},
            {"rule_id": "GIT_DIFF_HASH_MISMATCH", "field": "git.git_diff_hash", "severity": "hard"},
            {"rule_id": "GIT_STATUS_HASH_MISMATCH", "field": "git.git_status_hash", "severity": "hard"},
        ],
    }
    payload["selected_tier"] = select_tier(payload)
    payload["dependency_signature"] = dependency_signature(payload)
    return payload


def dependency_signature(payload: dict[str, Any]) -> str:
    material = dict(payload)
    material.pop("dependency_signature", None)
    return sha256_text(stable_json(material))


def select_tier(payload: dict[str, Any]) -> str:
    watched_records = list(payload.get("watched_files", []))
    tokens = _surface_tokens(str(payload.get("command_text", "")), watched_records)
    if tokens & {"luci", "indy", "postgrest"}:
        return "T2"
    if any(item.get("kind") == "systemd" for item in watched_records):
        return "T2"
    if any(item.get("kind") == "sql" for item in watched_records):
        return "T1"
    return "T0"


class InMemoryReceiptStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.locks: set[tuple[str, str]] = set()

    @contextmanager
    def lock(self, scope: str, dependency_signature: str) -> Iterator[None]:
        key = (scope, dependency_signature)
        if key in self.locks:
            raise DBBlocked("duplicate locked request")
        self.locks.add(key)
        try:
            yield
        finally:
            self.locks.discard(key)

    def latest_pass(self, scope: str, dependency_signature: str) -> dict[str, Any] | None:
        rows = [
            row
            for row in self.rows
            if row["scope"] == scope
            and row["dependency_signature"] == dependency_signature
            and row["status"] == "passed"
            and row.get("invalidated_by") is None
        ]
        rows.sort(key=lambda row: row["completed_at"] or row["started_at"] or "")
        return rows[-1] if rows else None

    def next_attempt(self, scope: str, dependency_signature: str) -> int:
        attempts = [row["attempt"] for row in self.rows if row["scope"] == scope and row["dependency_signature"] == dependency_signature]
        return (max(attempts) if attempts else 0) + 1

    def insert_receipt(self, row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        row.setdefault("receipt_uuid", f"fake-{len(self.rows)+1:04d}")
        self.rows.append(row)
        return row

    def update_receipt(self, receipt_uuid: str, updates: dict[str, Any]) -> dict[str, Any]:
        for row in self.rows:
            if row["receipt_uuid"] == receipt_uuid:
                row.update(updates)
                return row
        raise KeyError(receipt_uuid)

    def insert_pass(
        self,
        *,
        scope: str,
        command_text: str,
        dependency_signature: str,
        dependency_payload_json: dict[str, Any],
        runner_id: str,
    ) -> dict[str, Any]:
        row = {
            "receipt_uuid": f"fake-{len(self.rows)+1:04d}",
            "command_text": command_text,
            "scope": scope,
            "dependency_signature": dependency_signature,
            "dependency_payload_json": dependency_payload_json,
            "status": "passed",
            "exit_code": 0,
            "stdout_sha256": "",
            "stderr_sha256": "",
            "started_at": now_iso(),
            "completed_at": now_iso(),
            "runner_id": runner_id,
            "invalidated_by": None,
            "metadata_json": {},
            "attempt": self.next_attempt(scope, dependency_signature),
        }
        return self.insert_receipt(row)

    def count(self, *, status: str | None = None) -> int:
        if status is None:
            return len(self.rows)
        return sum(1 for row in self.rows if row["status"] == status)


class DBReceiptStore:
    def __init__(self, dsn: str = DEFAULT_DSN) -> None:
        self.dsn = dsn

    def connect(self):
        try:
            return psycopg.connect(self.dsn, row_factory=dict_row, autocommit=False)
        except Exception as exc:  # pragma: no cover - exercised in DB unavailable test
            raise DBBlocked(str(exc)) from exc

    @contextmanager
    def lock(self, scope: str, dependency_signature: str) -> Iterator[None]:
        key = _advisory_lock_key(scope, dependency_signature)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_lock(%s)", (key,))
                conn.commit()
            try:
                yield
            finally:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (key,))
                    conn.commit()

    def latest_pass(self, scope: str, dependency_signature: str) -> dict[str, Any] | None:
        sql = """
        SELECT *
        FROM lucidota_audit.test_execution_receipts
        WHERE scope = %s
          AND dependency_signature = %s
          AND status = 'passed'
          AND invalidated_by IS NULL
        ORDER BY completed_at DESC NULLS LAST, started_at DESC NULLS LAST, created_at DESC
        LIMIT 1
        """
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (scope, dependency_signature))
            row = cur.fetchone()
            conn.commit()
            return dict(row) if row else None

    def next_attempt(self, scope: str, dependency_signature: str) -> int:
        return 1

    def insert_receipt(self, row: dict[str, Any]) -> dict[str, Any]:
        sql = """
        INSERT INTO lucidota_audit.test_execution_receipts (
            command_text, scope, dependency_signature, dependency_payload_json,
            status, exit_code, stdout_sha256, stderr_sha256, started_at, completed_at,
            runner_id, invalidated_by, metadata_json
        ) VALUES (
            %(command_text)s, %(scope)s, %(dependency_signature)s, %(dependency_payload_json)s::jsonb,
            %(status)s, %(exit_code)s, %(stdout_sha256)s, %(stderr_sha256)s, %(started_at)s, %(completed_at)s,
            %(runner_id)s, %(invalidated_by)s, %(metadata_json)s::jsonb
        )
        RETURNING receipt_uuid::text
        """
        payload = dict(row)
        payload["dependency_payload_json"] = stable_json(payload["dependency_payload_json"])
        payload["metadata_json"] = stable_json(payload["metadata_json"])
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql, payload)
            receipt_uuid = cur.fetchone()["receipt_uuid"]
            conn.commit()
        payload["receipt_uuid"] = receipt_uuid
        return payload

    def update_receipt(self, receipt_uuid: str, updates: dict[str, Any]) -> dict[str, Any]:
        fields = ", ".join(f"{k} = %({k})s" for k in updates)
        params = dict(updates)
        if "metadata_json" in params:
            params["metadata_json"] = stable_json(params["metadata_json"])
        params["receipt_uuid"] = receipt_uuid
        sql = f"""
        UPDATE lucidota_audit.test_execution_receipts
        SET {fields},
            updated_at = now()
        WHERE receipt_uuid = %(receipt_uuid)s::uuid
        RETURNING receipt_uuid::text
        """
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            conn.commit()
            return {"receipt_uuid": row["receipt_uuid"] if row else receipt_uuid, **updates}


def _advisory_lock_key(scope: str, dependency_signature: str) -> int:
    digest = hashlib.sha256(f"{scope}:{dependency_signature}".encode("utf-8")).hexdigest()[:16]
    value = int(digest, 16)
    return value if value < 2**63 else value - 2**64


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DB-backed test receipt gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run or skip a command using DB receipts")
    run.add_argument("--scope", required=True)
    run.add_argument("--watch", action="append", default=[], help="File to hash into the signature; repeatable.")
    run.add_argument("--env-key", action="append", default=[], help="Environment key to hash into the signature; repeatable.")
    run.add_argument("--include-openapi", action="store_true", help="Force OpenAPI hash into the signature.")
    run.add_argument("--dsn", default=DEFAULT_DSN)
    run.add_argument("--cwd", default=str(ROOT))
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _parse_command(command: list[str]) -> list[str]:
    if not command:
        return []
    if command[0] == "--":
        command = command[1:]
    return command


def _run_child(command: list[str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, cwd=cwd, capture_output=True, check=False)


def run_gate(
    argv: list[str],
    *,
    store: Any | None = None,
    command_runner: Any = _run_child,
    openapi_fetcher: Any | None = None,
    git_probe: Any | None = None,
) -> GateResult:
    args = _build_parser().parse_args(argv)
    if args.cmd != "run":
        raise SystemExit(2)
    command = _parse_command(args.command)
    if not command:
        raise SystemExit("command after -- is required")
    cwd = Path(args.cwd).resolve()
    command_text = normalize_command(command)

    payload = build_dependency_payload(
        scope=args.scope,
        command_text=command_text,
        watched_paths=args.watch,
        cwd=cwd,
        include_openapi=True if args.include_openapi else None,
        env_keys=args.env_key,
        git_probe_fn=git_probe,
        openapi_fetcher=openapi_fetcher,
    )
    signature = payload["dependency_signature"]
    try:
        store_obj = store if store is not None else create_store(args.dsn)
        tier = payload["selected_tier"]
        with store_obj.lock(args.scope, signature):
            latest = store_obj.latest_pass(args.scope, signature)
            if latest:
                skipped = store_obj.insert_receipt(
                    {
                        "command_text": command_text,
                        "scope": args.scope,
                        "dependency_signature": signature,
                        "dependency_payload_json": payload,
                        "status": "skipped",
                        "exit_code": 0,
                        "stdout_sha256": sha256_text("SKIPPED_ALREADY_VERIFIED\n"),
                        "stderr_sha256": "",
                        "started_at": now_iso(),
                        "completed_at": now_iso(),
                        "runner_id": _runner_id(),
                        "invalidated_by": None,
                        "metadata_json": {
                            "decision": "skip",
                            "reason": "existing_passing_receipt",
                            "tier": tier,
                            "latest_pass_receipt_uuid": latest.get("receipt_uuid"),
                        },
                    }
                )
                print(f"SKIPPED_ALREADY_VERIFIED scope={args.scope} receipt_uuid={skipped['receipt_uuid']} dependency_signature={signature}")
                return GateResult(0, "skipped", skipped["receipt_uuid"], "skip", signature)

            running = store_obj.insert_receipt(
                {
                    "command_text": command_text,
                    "scope": args.scope,
                    "dependency_signature": signature,
                    "dependency_payload_json": payload,
                    "status": "running",
                    "exit_code": None,
                    "stdout_sha256": "",
                    "stderr_sha256": "",
                    "started_at": now_iso(),
                    "completed_at": None,
                    "runner_id": _runner_id(),
                    "invalidated_by": None,
                    "metadata_json": {"decision": "run", "tier": tier},
                }
            )
            print(f"RUNNING scope={args.scope} tier={tier} receipt_uuid={running['receipt_uuid']} dependency_signature={signature}")
            completed = command_runner(command, cwd=cwd)
            stdout = _coerce_bytes(getattr(completed, "stdout", b""))
            stderr = _coerce_bytes(getattr(completed, "stderr", b""))
            status = "passed" if getattr(completed, "returncode", 1) == 0 else "failed"
            updates = {
                "status": status,
                "exit_code": int(getattr(completed, "returncode", 1)),
                "stdout_sha256": sha256_bytes(stdout),
                "stderr_sha256": sha256_bytes(stderr),
                "completed_at": now_iso(),
                "metadata_json": {
                    "decision": "run",
                    "tier": tier,
                    "stdout_bytes": len(stdout),
                    "stderr_bytes": len(stderr),
                },
            }
            store_obj.update_receipt(running["receipt_uuid"], updates)
            if stdout:
                sys.stdout.write(stdout.decode("utf-8", "replace"))
            if stderr:
                sys.stderr.write(stderr.decode("utf-8", "replace"))
            print(f"RECEIPT_UUID={running['receipt_uuid']} STATUS={status.upper()} EXIT_CODE={updates['exit_code']} DEPENDENCY_SIGNATURE={signature}")
            return GateResult(updates["exit_code"], status, running["receipt_uuid"], "run", signature)
    except DBBlocked as exc:
        print(f"DB_BLOCKED {exc}")
        return GateResult(3, "blocked", None, "blocked", signature)


def _coerce_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return str(value).encode("utf-8")


def _runner_id() -> str:
    return f"{os.environ.get('USER') or 'unknown'}@{socket.gethostname()}:{os.getpid()}"


def create_store(dsn: str = DEFAULT_DSN) -> DBReceiptStore:
    return DBReceiptStore(dsn)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    result = run_gate(args)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
