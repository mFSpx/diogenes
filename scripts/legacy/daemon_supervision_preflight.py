#!/usr/bin/env python3
"""Daemon launch/supervision preflight with machine-readable failure records.

MVP control-plane rule: inspect and optionally start missing local loops; never
kill healthy daemons and never reset queues from this script.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS/supervision"
LOG_DIR = ROOT / "04_RUNTIME/lucidota_daemons"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def db(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"daemon_supervision_preflight_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(p)}")
    return p


@dataclass(frozen=True)
class LoopSpec:
    name: str
    needle: str
    command: list[str]
    log_name: str
    required: bool = True
    env: dict[str, str] | None = None


def py() -> str:
    venv = ROOT / ".venv/bin/python"
    return str(venv) if venv.exists() else "python3"


def timestamped_krampus_command() -> tuple[list[str], str, str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ledger = f"05_OUTPUTS/korpus_krampii/krampus_chrono_ledger_{ts}.csv"
    brain_state = f"03_VAULT/krampus_dbstream_brain.{ts}.pkl"
    brain_map = f"05_OUTPUTS/korpus_krampii/krampus_brain_map.{ts}.jsonl"
    return (
        [
            py(),
            "scripts/krampus_time_machine.py",
            "--json",
            "run",
            "03_VAULT/korpus_krampii/DIGESTED",
            "--mode",
            "both",
            "--case",
            "KRAMPUSCHEWING",
            "--label",
            "mvp_supervised_backfill",
            "--ledger",
            ledger,
            "--log",
            f"05_OUTPUTS/korpus_krampii/krampus_time_machine_{ts}.jsonl",
            "--brain-state",
            brain_state,
            "--brain-map",
            brain_map,
            "--save-every",
            "100",
            "--commit-every",
            "100",
            "--max-file-mb",
            "128",
            "--max-text-mb",
            "32",
            "--no-default-excludes",
            "--keep-going",
        ],
        ledger,
        brain_state,
        brain_map,
    )


def loop_specs() -> list[LoopSpec]:
    krampus_cmd, _, _, _ = timestamped_krampus_command()
    base_env = {"PYTHONPATH": ".", "LUCIDOTA_DRAFT_ONLY": "1"}
    return [
        LoopSpec(
            "absurd-worker",
            "scripts/unified_absurd_ingest_worker.py work",
            ["python3", "scripts/unified_absurd_ingest_worker.py", "work"],
            "absurd_worker_supervised.log",
            env=base_env,
        ),
        LoopSpec(
            "bytewax-abductive-blender",
            "scripts/bytewax_abductive_blender.py loop",
            ["python3", "scripts/bytewax_abductive_blender.py", "loop", "--limit", "400", "--idle-sleep", "2", "--activity-window-seconds", "2"],
            "bytewax_abductive_blender_supervised.log",
            env={**base_env, "LUCIDOTA_DUCKDB_MEMORY_LIMIT": "1536MB"},
        ),
        LoopSpec(
            "abcd-sequence-runner",
            "scripts/updated_abcd_sequence_runner.py --execute --continue-on-failure --continuous",
            ["python3", "scripts/updated_abcd_sequence_runner.py", "--execute", "--continue-on-failure", "--continuous", "--idle-sleep", "20"],
            "abcd_sequence_runner_supervised.log",
            env=base_env,
        ),
        LoopSpec(
            "krampus-time-machine",
            "scripts/krampus_time_machine.py --json run 03_VAULT/korpus_krampii/DIGESTED",
            krampus_cmd,
            "krampus_time_machine_supervised.log",
            env=base_env,
        ),
        LoopSpec(
            "krampuschewing-watcher",
            "scripts/krampuschewing_watcher.sh",
            ["bash", "scripts/krampuschewing_watcher.sh"],
            "krampuschewing_watcher_supervised.log",
            env={"LUCIDOTA_HOME": str(ROOT)},
        ),
        LoopSpec(
            "river-stream-worker",
            "scripts/lucidota_stream_river_worker.sh",
            ["bash", "scripts/lucidota_stream_river_worker.sh"],
            "lucidota_stream_river_worker_supervised.log",
            required=False,
            env={"LUCIDOTA_HOME": str(ROOT)},
        ),
    ]


def run(cmd: list[str], timeout: int = 40) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {"command": " ".join(cmd), "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
    except Exception as exc:
        return {"command": " ".join(cmd), "returncode": None, "stdout_tail": "", "stderr_tail": f"{type(exc).__name__}:{exc}"}


def ps_rows() -> list[dict[str, Any]]:
    cp = subprocess.run(["ps", "-eo", "pid,ppid,user,stat,etime,rss,cmd"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    rows: list[dict[str, Any]] = []
    for line in cp.stdout.splitlines()[1:]:
        parts = line.split(None, 6)
        if len(parts) != 7:
            continue
        pid, ppid, user, stat, etime, rss, cmd = parts
        rows.append({"pid": int(pid), "ppid": int(ppid), "user": user, "stat": stat, "etime": etime, "rss_kb": int(rss), "cmd": cmd})
    return rows


def find_processes(needle: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        cmd = str(row.get("cmd") or "")
        if needle in cmd and not any(skip in cmd for skip in ("bash -c", "grep ", "awk ", "sed ")):
            out.append(row)
    return out


def start_loop(spec: LoopSpec) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / spec.log_name
    env = os.environ.copy()
    env.update(spec.env or {})
    with log_path.open("ab", buffering=0) as log:
        proc = subprocess.Popen(
            spec.command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {"started": True, "pid": proc.pid, "log_path": rel(log_path), "command": spec.command}


def check_loop(spec: LoopSpec, rows: list[dict[str, Any]], ensure: bool) -> dict[str, Any]:
    procs = find_processes(spec.needle, rows)
    result: dict[str, Any] = {
        "daemon_name": spec.name,
        "kind": "mvp_loop",
        "needle": spec.needle,
        "required": spec.required,
        "processes": procs,
        "process_count": len(procs),
        "status": "ready" if procs else ("missing" if spec.required else "optional_missing"),
        "blockers": [] if procs or not spec.required else ["process_missing"],
        "ensure_action": None,
    }
    if ensure and not procs:
        try:
            result["ensure_action"] = start_loop(spec)
            result["status"] = "started"
            result["blockers"] = []
        except Exception as exc:
            result["ensure_action"] = {"started": False, "error": f"{type(exc).__name__}: {exc}"}
            result["status"] = "failed_to_start"
            if spec.required:
                result["blockers"] = ["process_missing", "start_failed"]
    return result


def check_daemon(name: str, check_cmd: str | None, systemd_unit: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"daemon_name": name, "kind": "service_probe", "check_command": check_cmd, "systemd_unit": systemd_unit, "script_exists_executable": None, "check": None, "systemd": None, "status": "unknown", "blockers": []}
    if check_cmd:
        script = ROOT / check_cmd.split()[0]
        result["script_exists_executable"] = script.exists() and os.access(script, os.X_OK)
        if result["script_exists_executable"]:
            result["check"] = run(check_cmd.split())
        else:
            result["blockers"].append("check_script_missing_or_not_executable")
    if systemd_unit and shutil.which("systemctl"):
        result["systemd"] = run(["systemctl", "--user", "is-active", systemd_unit], timeout=10)
    healthy = (result.get("check") and result["check"]["returncode"] == 0) or (result.get("systemd") and str(result["systemd"].get("stdout_tail", "")).strip() == "active")
    result["status"] = "ready" if healthy else "failed"
    if not healthy and not result["blockers"]:
        result["blockers"].append("daemon_health_check_failed")
    return result


def ensure_table(conn: psycopg.Connection[Any]) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS lucidota_control")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lucidota_control.daemon_preflight(
          preflight_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          daemon_name text NOT NULL,
          command_path text NOT NULL DEFAULT '',
          exists_executable boolean NOT NULL DEFAULT false,
          status text NOT NULL,
          detail jsonb NOT NULL DEFAULT '{}'::jsonb,
          created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def record(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    with psycopg.connect(db(args)) as conn:
        ensure_table(conn)
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.daemon_preflight(daemon_name,command_path,exists_executable,status,detail)
                    VALUES (%s,%s,%s,%s,%s::jsonb)
                    """,
                    (
                        r["daemon_name"],
                        r.get("needle") or r.get("check_command") or "",
                        bool(r.get("process_count") or r.get("script_exists_executable")),
                        r["status"],
                        json.dumps(r, default=str),
                    ),
                )
        conn.commit()


def main() -> int:
    p = argparse.ArgumentParser(description="Check/ensure MVP daemon supervision and record machine-readable failures")
    p.add_argument("--database-url")
    p.add_argument("--execute", action="store_true", help="write daemon_preflight rows")
    p.add_argument("--ensure", action="store_true", help="start missing MVP loops without killing existing processes")
    p.add_argument("--skip-service-probes", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = ps_rows()
    loop_results = [check_loop(spec, rows, ensure=args.ensure) for spec in loop_specs()]
    service_results: list[dict[str, Any]] = []
    if not args.skip_service_probes:
        service_results = [
            check_daemon("chrono-ledger", "scripts/check_chrono_ledger_service.sh", "lucidota-chrono-ledger.service"),
            check_daemon("rust-intake", "scripts/check_lucidota_intake_service.sh", "lucidota-intake.service"),
        ]

    all_results = loop_results + service_results
    blockers = [f"{r['daemon_name']}:{b}" for r in all_results for b in r.get("blockers", [])]
    report = {
        "schema": "lucidota.daemon_supervision_preflight.v2",
        "action": "daemon_supervision_preflight",
        "execute_performed": bool(args.execute),
        "ensure_performed": bool(args.ensure),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "kill_performed": False,
        "loops": loop_results,
        "service_probes": service_results,
        "blockers": blockers,
        "status": "PASS" if not blockers else "BLOCKED",
    }
    if args.execute:
        record(args, all_results)
        report["db_writes_performed"] = True
    path = write_report("execute" if args.execute else "dry_run", report)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    print("DAEMON_SUPERVISION=" + report["status"])
    print("SUPERVISION_REPORT=" + rel(path))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
