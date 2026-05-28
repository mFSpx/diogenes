#!/usr/bin/env python3
"""Resource-governed process collar for LUCIDOTA workers.

Blueprint: measure first, decide safe concurrency, then either refuse spawn,
adopt an existing PID, or spawn a bounded child with a receipt.  No daemon here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Optional in tests; available in the live LUCIDOTA venv.
    import psycopg  # type: ignore
except Exception:  # pragma: no cover - exercised only without psycopg installed.
    psycopg = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path("05_OUTPUTS/runtime")
SCHEMA_FILE = ROOT / "06_SCHEMA/122_resource_governor.sql"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except Exception:
        return str(path)


@dataclass(frozen=True)
class ResourcePolicy:
    default_workers: int = 1
    max_workers: int = 4
    min_mem_available_mb: int = 1024
    max_swap_used_pct: float = 25.0
    max_disk_used_pct: float = 90.0
    max_load_per_cpu: float = 1.2
    max_vram_used_pct: float = 90.0


@dataclass(frozen=True)
class PidRegistration:
    pid: int
    owner: str
    purpose: str
    cwd: str
    command: list[str]
    max_memory_mb: int
    max_cpu_percent: float
    kill_policy: str
    status: str = "running"
    started_at: str = ""
    ended_at: str | None = None
    exit_code: int | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["started_at"] = self.started_at or now()
        return record


def write_receipt(root: Path, prefix: str, payload: dict[str, Any]) -> Path:
    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{prefix}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["receipt_path"] = rel(path, root)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def write_pid_registry(root: Path, entry: PidRegistration, telemetry: dict[str, Any] | None = None, db_result: dict[str, Any] | None = None) -> Path:
    payload = {
        "schema": "lucidota.resource_governor.pid_registry.v1",
        "registration": entry.to_record(),
        "telemetry": telemetry or {},
        "db_result": db_result or {"attempted": False},
    }
    return write_receipt(root, f"pid_registry_{entry.pid}_{entry.status}", payload)


def meminfo() -> dict[str, Any]:
    raw: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, rest = line.split(":", 1)
            raw[k] = int(rest.strip().split()[0])
    except Exception:
        return {"available_mb": None, "swap_used_pct": None, "raw_kb": raw}
    swap_total = raw.get("SwapTotal", 0)
    swap_free = raw.get("SwapFree", 0)
    swap_used_pct = 0.0 if swap_total <= 0 else round((swap_total - swap_free) / swap_total * 100, 2)
    return {
        "total_mb": round(raw.get("MemTotal", 0) / 1024, 2),
        "available_mb": round(raw.get("MemAvailable", 0) / 1024, 2),
        "swap_total_mb": round(swap_total / 1024, 2),
        "swap_free_mb": round(swap_free / 1024, 2),
        "swap_used_pct": swap_used_pct,
        "raw_kb": raw,
    }


def disk_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_pct": round(usage.used / usage.total * 100, 2) if usage.total else None,
    }


def vram_snapshot() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "gpus": [], "error": "nvidia-smi_not_found"}
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
    if proc.returncode != 0:
        return {"available": False, "gpus": [], "error": proc.stderr.strip()[:500]}
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            total = float(parts[2])
            used = float(parts[3])
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": total,
                    "memory_used_mb": used,
                    "memory_used_pct": round(used / total * 100, 2) if total else None,
                    "utilization_gpu_pct": float(parts[4]),
                    "temperature_c": float(parts[5]),
                }
            )
    return {"available": True, "gpus": gpus}


def pg_activity(database_url: str | None) -> dict[str, Any]:
    if not database_url or psycopg is None:
        return {"available": False, "connection_count": None, "rows": [], "error": "database_url_or_psycopg_missing"}
    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            rows = conn.execute(
                """
                SELECT pid, usename, datname, state, wait_event_type, wait_event,
                       EXTRACT(EPOCH FROM now()-xact_start)::int AS xact_age_seconds,
                       EXTRACT(EPOCH FROM now()-query_start)::int AS query_age_seconds,
                       left(query, 240) AS query
                FROM pg_stat_activity
                WHERE datname = current_database()
                ORDER BY xact_start NULLS LAST, query_start NULLS LAST
                """
            ).fetchall()
        keys = ["pid", "usename", "datname", "state", "wait_event_type", "wait_event", "xact_age_seconds", "query_age_seconds", "query"]
        return {"available": True, "connection_count": len(rows), "rows": [dict(zip(keys, row)) for row in rows]}
    except Exception as exc:
        return {"available": False, "connection_count": None, "rows": [], "error": f"{type(exc).__name__}:{exc}"}


def collect_telemetry(root: Path = ROOT, database_url: str | None = None) -> dict[str, Any]:
    load = os.getloadavg()
    return {
        "schema": "lucidota.resource_governor.telemetry.v1",
        "captured_at": now(),
        "cpu": {"count": os.cpu_count() or 1, "loadavg_1m": load[0], "loadavg_5m": load[1], "loadavg_15m": load[2]},
        "memory": meminfo(),
        "disk": disk_snapshot(root),
        "vram": vram_snapshot(),
        "postgres": pg_activity(database_url),
    }


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def governance_decision(snapshot: dict[str, Any], requested_workers: int | None = None, policy: ResourcePolicy = ResourcePolicy()) -> dict[str, Any]:
    requested = int(requested_workers or policy.default_workers or 1)
    safe = max(1, min(requested, policy.max_workers))
    reasons: list[str] = []
    cpu = snapshot.get("cpu", {})
    cpu_count = max(1, int(cpu.get("count") or 1))
    if _num(cpu.get("loadavg_1m")) / cpu_count > policy.max_load_per_cpu:
        reasons.append("load_pressure")
        safe = max(1, min(safe, requested // 2 or 1))
    mem = snapshot.get("memory", {})
    if _num(mem.get("available_mb"), 10**9) < policy.min_mem_available_mb:
        reasons.append("mem_available_below_floor")
        safe = 1
    if _num(mem.get("swap_used_pct")) > policy.max_swap_used_pct:
        reasons.append("swap_pressure")
        safe = 1
    if _num(snapshot.get("disk", {}).get("used_pct")) > policy.max_disk_used_pct:
        reasons.append("disk_pressure")
        safe = 1
    for gpu in snapshot.get("vram", {}).get("gpus", []) or []:
        pct = gpu.get("memory_used_pct")
        if pct is None:
            total = _num(gpu.get("memory_total_mb"))
            pct = (_num(gpu.get("memory_used_mb")) / total * 100) if total else 0
        if _num(pct) > policy.max_vram_used_pct:
            reasons.append("vram_pressure")
            safe = 1
            break
    throttle = bool(reasons)
    deltas = []
    if throttle:
        deltas.append({"heuristic": "reduce_worker_count", "from": requested, "to": safe, "because": reasons})
        deltas.append({"heuristic": "pause_before_spawn", "because": reasons})
    else:
        deltas.append({"heuristic": "maintain_worker_count", "workers": safe})
    return {"requested_workers": requested, "safe_workers": safe, "throttle": throttle, "reasons": reasons, "learning_deltas": deltas}


def pg_supervision_plan(rows: list[dict[str, Any]], max_idle_xact_seconds: int = 300, protected_pids: set[int] | None = None) -> dict[str, Any]:
    protected = set(protected_pids or set())
    candidates = []
    for row in rows:
        pid = int(row.get("pid") or 0)
        age = int(row.get("xact_age_seconds") or 0)
        if pid in protected:
            continue
        if str(row.get("state")) == "idle in transaction" and age >= max_idle_xact_seconds:
            item = dict(row)
            item["reason"] = "stale_idle_in_transaction"
            candidates.append(item)
    return {
        "schema": "lucidota.resource_governor.pg_supervision_plan.v1",
        "max_idle_xact_seconds": max_idle_xact_seconds,
        "protected_pids": sorted(protected),
        "terminate_candidates": candidates,
    }


def ensure_db_tables(conn: Any) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    conn.execute("CREATE SCHEMA IF NOT EXISTS lucidota_control")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lucidota_control.pid_registry(
          registry_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          pid integer NOT NULL,
          owner text NOT NULL,
          purpose text NOT NULL,
          cwd text NOT NULL DEFAULT '',
          command jsonb NOT NULL DEFAULT '[]'::jsonb,
          max_memory_mb integer NOT NULL DEFAULT 0,
          max_cpu_percent numeric NOT NULL DEFAULT 0,
          kill_policy text NOT NULL DEFAULT '',
          status text NOT NULL,
          receipt_path text NOT NULL DEFAULT '',
          telemetry jsonb NOT NULL DEFAULT '{}'::jsonb,
          detail jsonb NOT NULL DEFAULT '{}'::jsonb,
          observed_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lucidota_control.resource_throttle_receipt(
          throttle_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          reason text NOT NULL,
          safe_workers integer NOT NULL DEFAULT 1,
          requested_workers integer NOT NULL DEFAULT 1,
          receipt_path text NOT NULL DEFAULT '',
          detail jsonb NOT NULL DEFAULT '{}'::jsonb,
          created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lucidota_control.pg_supervision_receipt(
          supervision_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          action text NOT NULL,
          candidate_count integer NOT NULL DEFAULT 0,
          terminated_count integer NOT NULL DEFAULT 0,
          receipt_path text NOT NULL DEFAULT '',
          detail jsonb NOT NULL DEFAULT '{}'::jsonb,
          created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def record_pid_db(database_url: str | None, entry: PidRegistration, receipt_path: Path, telemetry: dict[str, Any]) -> dict[str, Any]:
    if not database_url or psycopg is None:
        return {"attempted": False, "ok": False, "error": "database_url_or_psycopg_missing"}
    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            ensure_db_tables(conn)
            conn.execute(
                """
                INSERT INTO lucidota_control.pid_registry
                (pid, owner, purpose, cwd, command, max_memory_mb, max_cpu_percent, kill_policy, status, receipt_path, telemetry, detail)
                VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                """,
                (
                    entry.pid,
                    entry.owner,
                    entry.purpose,
                    entry.cwd,
                    json.dumps(entry.command),
                    entry.max_memory_mb,
                    entry.max_cpu_percent,
                    entry.kill_policy,
                    entry.status,
                    rel(receipt_path),
                    json.dumps(telemetry, default=str),
                    json.dumps({"ended_at": entry.ended_at, "exit_code": entry.exit_code}, default=str),
                ),
            )
            conn.commit()
        return {"attempted": True, "ok": True}
    except Exception as exc:
        return {"attempted": True, "ok": False, "error": f"{type(exc).__name__}:{exc}"}


def read_proc_command(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
    except Exception:
        return []


def read_proc_cwd(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        return ""


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmd_preflight(args: argparse.Namespace) -> tuple[int, Path]:
    root = Path(args.root)
    snap = collect_telemetry(root, args.database_url)
    decision = governance_decision(snap, args.requested_workers, ResourcePolicy(max_workers=args.max_workers))
    payload = {"schema": "lucidota.resource_governor.preflight.v1", "telemetry": snap, "decision": decision}
    path = write_receipt(root, "resource_preflight", payload)
    if decision["throttle"]:
        throttle = {"schema": "lucidota.resource_governor.resource_throttle.v1", "decision": decision, "telemetry": snap, "preflight_receipt": rel(path, root)}
        write_receipt(root, "RESOURCE_THROTTLE", throttle)
    return 0, path


def cmd_adopt(args: argparse.Namespace) -> tuple[int, Path]:
    root = Path(args.root)
    snap = collect_telemetry(root, args.database_url)
    cmd = args.command or read_proc_command(args.pid)
    cwd = args.cwd or read_proc_cwd(args.pid)
    entry = PidRegistration(
        pid=args.pid,
        owner=args.owner,
        purpose=args.purpose,
        cwd=cwd,
        command=cmd,
        max_memory_mb=args.max_memory_mb,
        max_cpu_percent=args.max_cpu_percent,
        kill_policy=args.kill_policy,
        status="running" if pid_alive(args.pid) else "missing",
    )
    path = write_pid_registry(root, entry, telemetry=snap, db_result={"attempted": False})
    db_result = record_pid_db(args.database_url, entry, path, snap) if args.execute else {"attempted": False}
    if args.execute:
        path = write_pid_registry(root, entry, telemetry=snap, db_result=db_result)
    return (0 if entry.status == "running" else 2), path


def cmd_spawn(args: argparse.Namespace) -> tuple[int, Path]:
    root = Path(args.root)
    snap = collect_telemetry(root, args.database_url)
    decision = governance_decision(snap, args.requested_workers, ResourcePolicy(max_workers=args.max_workers))
    if decision["throttle"] and not args.force:
        path = write_receipt(root, "RESOURCE_THROTTLE", {"schema": "lucidota.resource_governor.resource_throttle.v1", "decision": decision, "telemetry": snap, "spawn_blocked": True, "command": args.worker_cmd})
        return 3, path
    planned = PidRegistration(0, args.owner, args.purpose, str(root), args.worker_cmd, args.max_memory_mb, args.max_cpu_percent, args.kill_policy, "planned")
    if not args.execute:
        path = write_pid_registry(root, planned, telemetry={"decision": decision, "dry_run": True}, db_result={"attempted": False})
        return 0, path
    proc = subprocess.Popen(args.worker_cmd, cwd=root, stdin=subprocess.DEVNULL, start_new_session=True)
    running = PidRegistration(proc.pid, args.owner, args.purpose, str(root), args.worker_cmd, args.max_memory_mb, args.max_cpu_percent, args.kill_policy, "running")
    path = write_pid_registry(root, running, telemetry={"decision": decision, "snapshot": snap}, db_result={"attempted": False})
    db_result = record_pid_db(args.database_url, running, path, {"decision": decision, "snapshot": snap}) if args.execute else {"attempted": False}
    path = write_pid_registry(root, running, telemetry={"decision": decision, "snapshot": snap}, db_result=db_result)
    if args.wait:
        code = proc.wait(timeout=args.timeout if args.timeout else None)
        finished = PidRegistration(proc.pid, args.owner, args.purpose, str(root), args.worker_cmd, args.max_memory_mb, args.max_cpu_percent, args.kill_policy, "succeeded" if code == 0 else "failed", ended_at=now(), exit_code=code)
        path = write_pid_registry(root, finished, telemetry={"decision": decision}, db_result=record_pid_db(args.database_url, finished, path, {"decision": decision}) if args.execute else {"attempted": False})
        return code, path
    return 0, path


def cmd_pg_supervise(args: argparse.Namespace) -> tuple[int, Path]:
    root = Path(args.root)
    activity = pg_activity(args.database_url)
    protected = {os.getpid(), os.getppid(), *(int(p) for p in args.protect_pid)}
    plan = pg_supervision_plan(activity.get("rows", []), args.max_idle_xact_seconds, protected)
    terminated: list[dict[str, Any]] = []
    if args.execute and plan["terminate_candidates"] and psycopg is not None and args.database_url:
        with psycopg.connect(args.database_url, connect_timeout=5) as conn:
            for item in plan["terminate_candidates"]:
                ok = conn.execute("SELECT pg_terminate_backend(%s)", (item["pid"],)).fetchone()[0]
                terminated.append({"pid": item["pid"], "terminated": bool(ok)})
            ensure_db_tables(conn)
            conn.execute(
                """
                INSERT INTO lucidota_control.pg_supervision_receipt(action,candidate_count,terminated_count,detail)
                VALUES (%s,%s,%s,%s::jsonb)
                """,
                ("terminate_stale_idle_xact", len(plan["terminate_candidates"]), sum(1 for t in terminated if t["terminated"]), json.dumps({"plan": plan, "terminated": terminated}, default=str)),
            )
            conn.commit()
    payload = {"schema": "lucidota.resource_governor.pg_supervision_receipt.v1", "activity": activity, "plan": plan, "execute_performed": bool(args.execute), "terminated": terminated}
    path = write_receipt(root, "pg_supervision", payload)
    return 0, path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LUCIDOTA resource governor: telemetry, PID registry, and PG supervision")
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--database-url", default=os.environ.get("LUCIDOTA_STATE_DSN") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--json", action="store_true")
    sub = p.add_subparsers(dest="action", required=True)
    pre = sub.add_parser("preflight")
    add_late_global_flags(pre)
    pre.add_argument("--requested-workers", type=int, default=1)
    pre.add_argument("--max-workers", type=int, default=4)
    adopt = sub.add_parser("adopt")
    add_late_global_flags(adopt)
    adopt.add_argument("--pid", type=int, required=True)
    adopt.add_argument("--owner", required=True)
    adopt.add_argument("--purpose", required=True)
    adopt.add_argument("--cwd", default="")
    adopt.add_argument("--command", nargs="*")
    adopt.add_argument("--max-memory-mb", type=int, default=1024)
    adopt.add_argument("--max-cpu-percent", type=float, default=100.0)
    adopt.add_argument("--kill-policy", default="terminate_then_kill_after_5s")
    sp = sub.add_parser("spawn")
    add_late_global_flags(sp)
    sp.add_argument("--owner", required=True)
    sp.add_argument("--purpose", required=True)
    sp.add_argument("--requested-workers", type=int, default=1)
    sp.add_argument("--max-workers", type=int, default=4)
    sp.add_argument("--max-memory-mb", type=int, default=1024)
    sp.add_argument("--max-cpu-percent", type=float, default=100.0)
    sp.add_argument("--kill-policy", default="terminate_then_kill_after_5s")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--wait", action="store_true")
    sp.add_argument("--timeout", type=int, default=0)
    sp.add_argument("worker_cmd", nargs=argparse.REMAINDER)
    pg = sub.add_parser("pg-supervise")
    add_late_global_flags(pg)
    pg.add_argument("--max-idle-xact-seconds", type=int, default=300)
    pg.add_argument("--protect-pid", action="append", default=[])
    return p


def add_late_global_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execute", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "preflight":
        code, path = cmd_preflight(args)
    elif args.action == "adopt":
        code, path = cmd_adopt(args)
    elif args.action == "spawn":
        if args.worker_cmd and args.worker_cmd[0] == "--":
            args.worker_cmd = args.worker_cmd[1:]
        if not args.worker_cmd:
            raise SystemExit("spawn requires command after --")
        code, path = cmd_spawn(args)
    else:
        code, path = cmd_pg_supervise(args)
    print(f"REPORT_PATH={rel(path, Path(args.root))}")
    if args.json:
        print(path.read_text())
    status = "PASS" if code == 0 else "THROTTLED" if code == 3 else "BLOCKED"
    print(f"RESOURCE_GOVERNOR={status}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
