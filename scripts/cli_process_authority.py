#!/usr/bin/env python3
"""Monitor a CLI subprocess, inject auth tokens on prompts, and write DB receipts.

This wrapper is intentionally generic: it can manage Codex/Claude-style CLI
programs or any other local assistant binary that reads from stdin and emits
auth/login prompts on stdout/stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "cli_process_authority"
DB_DSN = os.environ.get("LUCIDOTA_STATE_DSN") or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
DEFAULT_AUTH_ENV_VARS = [
    "LUCIDOTA_CLI_AUTH_TOKEN",
    "CODEX_OAUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
]
DEFAULT_AUTH_PATTERNS = [
    r"(?i)\bauth(?:entication)? required\b",
    r"(?i)\blogin\b",
    r"(?i)\bauthori[sz]e\b",
    r"(?i)\boauth\b",
    r"(?i)\btoken\b",
    r"(?i)\bsign in\b",
    r"(?i)\bsession (?:expired|timeout)\b",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_output_dir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def read_stream(name: str, stream, sink: "queue.Queue[tuple[str, str | None]]") -> None:
    try:
        for line in iter(stream.readline, ""):
            sink.put((name, line))
    finally:
        sink.put((name, None))


def pick_auth_token(env_names: Iterable[str]) -> tuple[str | None, str | None]:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value, env_name
    return None, None


@dataclass
class AttemptResult:
    status: str
    exit_code: int | None
    pid: int | None
    auth_prompt_seen: bool
    auth_injected: bool
    restart_reason: str | None
    stdout_tail: str
    stderr_tail: str
    command_line: str


def run_attempt(
    command: list[str],
    *,
    timeout_seconds: float,
    auth_patterns: list[re.Pattern[str]],
    auth_env_vars: list[str],
) -> AttemptResult:
    token_value, token_env = pick_auth_token(auth_env_vars)
    env = os.environ.copy()
    for env_name in auth_env_vars:
        if env.get(env_name):
            continue
        # Preserve explicit auth env vars from the operator environment.
        if env_name in os.environ:
            env[env_name] = os.environ[env_name]

    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    sink: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
    stdout_thread = threading.Thread(target=read_stream, args=("stdout", proc.stdout, sink), daemon=True)
    stderr_thread = threading.Thread(target=read_stream, args=("stderr", proc.stderr, sink), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    stdout_tail: deque[str] = deque(maxlen=200)
    stderr_tail: deque[str] = deque(maxlen=200)
    auth_prompt_seen = False
    auth_injected = False
    stdout_eof = False
    stderr_eof = False
    started = time.monotonic()
    restart_reason: str | None = None

    def print_line(name: str, line: str) -> None:
        target = sys.stdout if name == "stdout" else sys.stderr
        target.write(line)
        target.flush()

    while True:
        try:
            name, line = sink.get(timeout=0.1)
        except queue.Empty:
            line = None
            name = ""

        if line is not None:
            print_line(name, line)
            (stdout_tail if name == "stdout" else stderr_tail).append(line.rstrip("\n"))
            combined = line
            if any(pattern.search(combined) for pattern in auth_patterns):
                auth_prompt_seen = True
                if token_value and proc.stdin and not auth_injected:
                    proc.stdin.write(token_value + "\n")
                    proc.stdin.flush()
                    auth_injected = True
                    stdout_tail.append("[wrapper] injected auth token from " + (token_env or "env"))
            continue

        if name == "stdout":
            stdout_eof = True
        elif name == "stderr":
            stderr_eof = True

        exit_code = proc.poll()
        if exit_code is not None and stdout_eof and stderr_eof:
            return AttemptResult(
                status="succeeded" if exit_code == 0 else "failed",
                exit_code=exit_code,
                pid=proc.pid,
                auth_prompt_seen=auth_prompt_seen,
                auth_injected=auth_injected,
                restart_reason=None,
                stdout_tail="\n".join(stdout_tail),
                stderr_tail="\n".join(stderr_tail),
                command_line=" ".join(command),
            )

        if time.monotonic() - started > timeout_seconds:
            restart_reason = "timeout"
            break

        if auth_prompt_seen and not token_value:
            restart_reason = "auth_prompt_without_token"
            break

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=2)
    except Exception:
        pass
    return AttemptResult(
        status="timeout" if restart_reason == "timeout" else "auth_failed",
        exit_code=proc.returncode,
        pid=proc.pid,
        auth_prompt_seen=auth_prompt_seen,
        auth_injected=auth_injected,
        restart_reason=restart_reason,
        stdout_tail="\n".join(stdout_tail),
        stderr_tail="\n".join(stderr_tail),
        command_line=" ".join(command),
    )


def write_receipt(result: dict[str, object], receipt_path: Path) -> None:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_db_schema() -> None:
    ddl = """
    CREATE SCHEMA IF NOT EXISTS lucidota_control;
    CREATE OR REPLACE VIEW lucidota_canon.cli_process_receipts AS
    SELECT
        receipt_uuid,
        received_at,
        command_line,
        command_sha256,
        process_pid,
        timeout_seconds,
        restart_count,
        auth_env_var,
        auth_prompt_seen,
        auth_injected,
        status,
        exit_code,
        stdout_tail,
        stderr_tail,
        receipt_path,
        detail,
        created_at,
        updated_at,
        stdout_tail_sha256,
        stderr_tail_sha256,
        stdout_archive_ref,
        stderr_archive_ref,
        stdout_archived_at,
        stderr_archived_at
    FROM lucidota_control.cli_process_receipt;
    """
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def write_db_receipt(payload: dict[str, object]) -> None:
    ensure_db_schema()
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.cli_process_receipt (
                    receipt_uuid, received_at, command_line, command_sha256, process_pid,
                    timeout_seconds, restart_count, auth_env_var, auth_prompt_seen,
                    auth_injected, status, exit_code, stdout_tail, stderr_tail,
                    stdout_tail_sha256, stderr_tail_sha256, stdout_archive_ref,
                    stderr_archive_ref, stdout_archived_at, stderr_archived_at,
                    receipt_path, detail
                ) VALUES (
                    %(receipt_uuid)s, %(received_at)s, %(command_line)s, %(command_sha256)s,
                    %(process_pid)s, %(timeout_seconds)s, %(restart_count)s, %(auth_env_var)s,
                    %(auth_prompt_seen)s, %(auth_injected)s, %(status)s, %(exit_code)s,
                    %(stdout_tail)s, %(stderr_tail)s, %(stdout_tail_sha256)s, %(stderr_tail_sha256)s,
                    %(stdout_archive_ref)s, %(stderr_archive_ref)s, %(stdout_archived_at)s, %(stderr_archived_at)s,
                    %(receipt_path)s, %(detail)s::jsonb
                )
                ON CONFLICT (receipt_uuid) DO UPDATE SET
                    received_at = EXCLUDED.received_at,
                    command_line = EXCLUDED.command_line,
                    command_sha256 = EXCLUDED.command_sha256,
                    process_pid = EXCLUDED.process_pid,
                    timeout_seconds = EXCLUDED.timeout_seconds,
                    restart_count = EXCLUDED.restart_count,
                    auth_env_var = EXCLUDED.auth_env_var,
                    auth_prompt_seen = EXCLUDED.auth_prompt_seen,
                    auth_injected = EXCLUDED.auth_injected,
                    status = EXCLUDED.status,
                    exit_code = EXCLUDED.exit_code,
                    stdout_tail = EXCLUDED.stdout_tail,
                    stderr_tail = EXCLUDED.stderr_tail,
                    stdout_tail_sha256 = EXCLUDED.stdout_tail_sha256,
                    stderr_tail_sha256 = EXCLUDED.stderr_tail_sha256,
                    stdout_archive_ref = EXCLUDED.stdout_archive_ref,
                    stderr_archive_ref = EXCLUDED.stderr_archive_ref,
                    stdout_archived_at = EXCLUDED.stdout_archived_at,
                    stderr_archived_at = EXCLUDED.stderr_archived_at,
                    receipt_path = EXCLUDED.receipt_path,
                    detail = EXCLUDED.detail,
                    updated_at = now()
                """,
                payload,
            )
        conn.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a CLI subprocess with auth-prompt handling and DB receipts.")
    ap.add_argument("--timeout-seconds", type=float, default=18000.0)
    ap.add_argument("--max-restarts", type=int, default=1)
    ap.add_argument("--receipt-path", default=str(OUT / f"cli_process_authority_{stamp()}.json"))
    ap.add_argument("--auth-env-var", action="append", default=[])
    ap.add_argument("--auth-prompt-regex", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        print("missing command", file=sys.stderr)
        return 2

    auth_env_vars = args.auth_env_var or DEFAULT_AUTH_ENV_VARS
    auth_patterns = [re.compile(p) for p in (args.auth_prompt_regex or DEFAULT_AUTH_PATTERNS)]
    receipt_path = Path(args.receipt_path)
    ensure_output_dir()

    restart_count = 0
    final: AttemptResult | None = None
    while True:
        attempt = run_attempt(
            command,
            timeout_seconds=args.timeout_seconds,
            auth_patterns=auth_patterns,
            auth_env_vars=auth_env_vars,
        )
        final = attempt
        if attempt.status == "succeeded":
            break
        if attempt.restart_reason == "timeout" and restart_count < max(0, args.max_restarts):
            restart_count += 1
            print(f"[wrapper] timeout; restarting attempt {restart_count}", file=sys.stderr)
            continue
        break

    payload = {
        "schema": "lucidota.cli_process_authority.v1",
        "receipt_uuid": str(uuid.uuid4()),
        "received_at": utc_now(),
        "command_line": final.command_line if final else " ".join(command),
        "command_sha256": sha256_text(" ".join(command)),
        "process_pid": final.pid if final else None,
        "timeout_seconds": args.timeout_seconds,
        "restart_count": restart_count,
        "auth_env_var": ",".join(auth_env_vars),
        "auth_prompt_seen": bool(final.auth_prompt_seen if final else False),
        "auth_injected": bool(final.auth_injected if final else False),
        "status": final.status if final else "failed",
        "exit_code": final.exit_code if final else None,
        "stdout_tail": final.stdout_tail if final else "",
        "stderr_tail": final.stderr_tail if final else "",
        "stdout_tail_sha256": sha256_text(final.stdout_tail if final else ""),
        "stderr_tail_sha256": sha256_text(final.stderr_tail if final else ""),
        "stdout_archive_ref": "",
        "stderr_archive_ref": "",
        "stdout_archived_at": None,
        "stderr_archived_at": None,
        "receipt_path": rel(receipt_path),
        "detail": json.dumps(
            {
                "restart_reason": final.restart_reason if final else None,
                "command": command,
                "auth_patterns": [p.pattern for p in auth_patterns],
            },
            sort_keys=True,
        ),
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }

    write_receipt(payload, receipt_path)
    write_db_receipt(payload)

    report = {
        "schema": payload["schema"],
        "status": payload["status"],
        "restart_count": restart_count,
        "auth_prompt_seen": payload["auth_prompt_seen"],
        "auth_injected": payload["auth_injected"],
        "exit_code": payload["exit_code"],
        "receipt_path": rel(receipt_path),
    }
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if payload["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
