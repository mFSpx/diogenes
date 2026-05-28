#!/usr/bin/env python3
"""Restartable DBOS wrapper for one Krampus brain-sidecar child.

This does not replace the live shell watcher yet.  It gives each per-file brain
sidecar run a deterministic DBOS workflow ID, timeout envelope, retry/resume
surface, and LUCIDOTA workflow_event audit trail.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psycopg
from dbos import DBOS, DBOSConfig, SetWorkflowID

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)
STATE_DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")
CONTROL_SCHEMA = ROOT / "06_SCHEMA" / "001_lucidota_control.sql"
WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
BRAIN_SCRIPT = ROOT / "scripts" / "lucidota_brain_ingest.py"
DEFAULT_STATE = ROOT / "03_VAULT" / "krampus_dbstream_brain.pkl"
DEFAULT_MAP = ROOT / "05_OUTPUTS" / "korpus_krampii" / "krampus_brain_map.jsonl"
DEFAULT_AUDIT = ROOT / "05_OUTPUTS" / "korpus_krampii" / "brain_sidecar_audit.jsonl"
WORKFLOW_NAME = "krampus-brain-sidecar-child"

BINARY_OR_MEDIA_SUFFIXES = {
    ".3gp",
    ".7z",
    ".aac",
    ".avi",
    ".bmp",
    ".bz2",
    ".db",
    ".dmg",
    ".doc",
    ".docx",
    ".exe",
    ".flac",
    ".gif",
    ".gz",
    ".heic",
    ".ico",
    ".iso",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".odp",
    ".ods",
    ".odt",
    ".ogg",
    ".parquet",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tif",
    ".tiff",
    ".wav",
    ".webm",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}


def timeout_policy(path: str, requested_seconds: int) -> dict[str, Any]:
    """Return the actual timeout envelope for one brain-sidecar child.

    requested_seconds > 0 is an explicit operator override. Otherwise:
    - media/binary: 10s max because it should become a skip receipt;
    - <=1 MiB text-ish work: 10s;
    - <=16 MiB normal brain text window: 30s;
    - >16 MiB: 20s max because brain_ingest should skip as oversize sidecar;
    - >1 GiB: 60s absolute emergency cap, still expected to skip.
    """
    p = Path(path).expanduser().resolve(strict=False)
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    suffix = p.suffix.lower()
    if requested_seconds and requested_seconds > 0:
        return {"timeout_seconds": requested_seconds, "policy": "operator_override", "size_bytes": size, "suffix": suffix}
    mib = 1024 * 1024
    gib = 1024 * mib
    if suffix in BINARY_OR_MEDIA_SUFFIXES:
        return {"timeout_seconds": 10, "policy": "binary_media_fast_skip", "size_bytes": size, "suffix": suffix}
    if size <= mib:
        return {"timeout_seconds": 10, "policy": "small_text", "size_bytes": size, "suffix": suffix}
    if size <= 16 * mib:
        return {"timeout_seconds": 30, "policy": "normal_text_window", "size_bytes": size, "suffix": suffix}
    if size <= gib:
        return {"timeout_seconds": 20, "policy": "oversize_sidecar_expected_skip", "size_bytes": size, "suffix": suffix}
    return {"timeout_seconds": 60, "policy": "huge_sidecar_expected_skip", "size_bytes": size, "suffix": suffix}


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def audit_path() -> Path:
    return Path(os.environ.get("LUCIDOTA_BRAIN_AUDIT_JSONL", str(DEFAULT_AUDIT))).expanduser()


def append_audit(event: dict[str, Any]) -> None:
    path = audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(jdump({"at": now(), "tool": "lucidota_dbos_brain_child", **event}) + "\n")


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(ROOT))
    except Exception:
        return str(path)


def stable_workflow_id(path: str, state_path: str, map_jsonl: str, dry_run: bool = False) -> str:
    p = Path(path).expanduser().resolve(strict=False)
    try:
        st = p.stat()
        sig = f"{p}|{st.st_size}|{st.st_mtime_ns}|{Path(state_path).resolve(strict=False)}|{Path(map_jsonl).resolve(strict=False)}|dry={dry_run}"
    except OSError:
        sig = f"{p}|missing|{Path(state_path).resolve(strict=False)}|{Path(map_jsonl).resolve(strict=False)}|dry={dry_run}"
    return "krampus-brain-child-" + hashlib.sha256(sig.encode()).hexdigest()[:32]


def ensure_schemas() -> None:
    """Verify the control schema exists without running broad DDL.

    The live ingestion box has dashboard/DBOS readers constantly touching
    lucidota_control. Re-applying the full schema here is unnecessary and can
    contend on constraint DDL. This child workflow only needs workflow_event.
    """
    with psycopg.connect(STATE_DB, connect_timeout=3) as conn:
        ok = conn.execute("SELECT to_regclass('lucidota_control.workflow_event') IS NOT NULL").fetchone()[0]
        if not ok:
            raise RuntimeError("missing lucidota_control.workflow_event; apply 06_SCHEMA/001_lucidota_control.sql outside live ingest")
        conn.commit()


def emit_event(run_id: str, phase: str, status: str, detail: dict[str, Any]) -> str:
    with psycopg.connect(STATE_DB, connect_timeout=3) as conn:
        event_id = conn.execute(
            """
            INSERT INTO lucidota_control.workflow_event
              (workflow_id, run_id, phase, status, source, detail)
            VALUES (%s, %s, %s, %s, 'lucidota_dbos_brain_child', %s::jsonb)
            RETURNING event_id::text
            """,
            (WORKFLOW_NAME, run_id, phase, status, jdump(detail)),
        ).fetchone()[0]
        conn.commit()
    return str(event_id)


def command_for(path: str, state_path: str, map_jsonl: str, dry_run: bool) -> list[str]:
    cmd = [
        str(PY),
        str(BRAIN_SCRIPT),
        "--json",
        "--state-path",
        str(Path(state_path).expanduser()),
        "--map-jsonl",
        str(Path(map_jsonl).expanduser()),
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.append(str(Path(path).expanduser()))
    return cmd


def parse_last_json(stdout: str) -> Any:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"stdout_tail": stdout[-2000:]}


@DBOS.step()
def brain_child_step(path: str, state_path: str, map_jsonl: str, timeout_seconds: int, dry_run: bool) -> dict[str, Any]:
    run_id = stable_workflow_id(path, state_path, map_jsonl, dry_run)
    p = Path(path).expanduser().resolve(strict=False)
    if not p.exists():
        event_id = emit_event(run_id, "brain_child", "failed", {"path": str(p), "reason": "path_missing"})
        raise FileNotFoundError(f"brain child target missing: {p}; event={event_id}")

    policy = timeout_policy(str(p), timeout_seconds)
    actual_timeout = int(policy["timeout_seconds"])
    cmd = command_for(str(p), state_path, map_jsonl, dry_run)
    emit_event(
        run_id,
        "brain_child",
        "running",
        {"path": str(p), "size_bytes": p.stat().st_size, "dry_run": dry_run, "timeout": policy, "command": cmd},
    )
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=max(1, actual_timeout), check=False)
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.time() - started, 3)
        timeout_detail = {
            "kind": "dbos_brain_child_timeout",
            "workflow_id": run_id,
            "path": str(p),
            "status": "failed",
            "reason": "timeout",
            "timeout": policy,
            "elapsed_s": elapsed,
            "dry_run": dry_run,
        }
        if not dry_run:
            append_audit(timeout_detail)
        event_id = emit_event(
            run_id,
            "brain_child",
            "failed",
            {
                "path": str(p),
                "timed_out": True,
                "timeout": policy,
                "elapsed_s": elapsed,
                "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            },
        )
        raise TimeoutError(f"brain child timed out after {actual_timeout}s; event={event_id}; path={p}; policy={policy}") from exc

    elapsed = round(time.time() - started, 3)
    parsed = parse_last_json(proc.stdout)
    detail = {
        "path": str(p),
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "dry_run": dry_run,
        "timeout": policy,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "result": parsed,
    }
    if proc.returncode == 0:
        event_id = emit_event(run_id, "brain_child", "succeeded", detail)
        return {"ok": True, "event_id": event_id, "workflow_run_id": run_id, **detail}
    if not dry_run:
        append_audit(
            {
                "kind": "dbos_brain_child_failed",
                "workflow_id": run_id,
                "path": str(p),
                "status": "failed",
                "reason": "nonzero_returncode",
                "returncode": proc.returncode,
                "timeout": policy,
                "elapsed_s": elapsed,
                "stderr_tail": proc.stderr[-2000:],
                "stdout_tail": proc.stdout[-2000:],
            }
        )
    event_id = emit_event(run_id, "brain_child", "failed", detail)
    raise RuntimeError(f"brain child failed rc={proc.returncode}; event={event_id}; stderr={proc.stderr[-500:]}")


@DBOS.workflow(name=WORKFLOW_NAME)
def brain_child_workflow(path: str, state_path: str, map_jsonl: str, timeout_seconds: int, dry_run: bool) -> dict[str, Any]:
    run_id = stable_workflow_id(path, state_path, map_jsonl, dry_run)
    emit_event(run_id, "brain_child", "queued", {"path": path, "state_path": state_path, "map_jsonl": map_jsonl, "dry_run": dry_run})
    return brain_child_step(path, state_path, map_jsonl, timeout_seconds, dry_run)


def active_children() -> list[dict[str, Any]]:
    out = subprocess.run(
        ["pgrep", "-af", "lucidota_brain_ingest.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.splitlines()
    rows: list[dict[str, Any]] = []
    for line in out:
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        pid = parts[0]
        cmd = parts[1] if len(parts) > 1 else ""
        if "lucidota_dbos_brain_child.py" in cmd:
            continue
        try:
            ps = subprocess.check_output(["ps", "-p", pid, "-o", "pid=,ppid=,etime=,%cpu=,%mem=,rss=,stat=,args="], text=True).strip()
        except Exception:
            ps = line
        target = ""
        try:
            args = [x.decode("utf-8", "replace") for x in (Path("/proc") / pid / "cmdline").read_bytes().split(b"\0") if x]
            if args:
                target = args[-1]
        except Exception:
            pass
        rows.append({"pid": pid, "target": target, "ps": ps})
    return rows


def map_status(path: str, map_jsonl: str, limit: int = 5) -> list[dict[str, Any]]:
    p = str(Path(path).expanduser().resolve(strict=False))
    m = Path(map_jsonl).expanduser()
    if not m.exists():
        return []
    found: list[dict[str, Any]] = []
    # Current map is modest; full scan keeps it simple and exact.
    for line in m.read_text(encoding="utf-8", errors="replace").splitlines():
        if p not in line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            item = {"raw": line[-1000:]}
        found.append(item)
    return found[-limit:]


def event_status(run_id: str, limit: int = 10) -> list[dict[str, Any]]:
    with psycopg.connect(STATE_DB, connect_timeout=3) as conn:
        rows = conn.execute(
            """
            SELECT event_id::text, phase, status, source, detail, created_at::text
            FROM lucidota_control.workflow_event
            WHERE workflow_id=%s AND run_id=%s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (WORKFLOW_NAME, run_id, limit),
        ).fetchall()
    return [
        {"event_id": r[0], "phase": r[1], "status": r[2], "source": r[3], "detail": r[4], "created_at": r[5]}
        for r in rows
    ]


def audit_summary(limit: int = 20) -> dict[str, Any]:
    path = audit_path()
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    tail: list[dict[str, Any]] = []
    total = 0
    bad_lines = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            status = str(item.get("status") or "unknown")
            reason = str(item.get("reason") or "")
            kind = str(item.get("kind") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            tail.append(item)
            if len(tail) > limit:
                tail = tail[-limit:]
    return {
        "path": repo_rel(path),
        "exists": path.exists(),
        "total_lines": total,
        "bad_lines": bad_lines,
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "kind_counts": kind_counts,
        "tail": tail,
    }


def workflow_status_to_dict(status: Any) -> Any:
    if status is None:
        return None
    if hasattr(status, "__dict__"):
        return dict(status.__dict__)
    return str(status)


def init_dbos() -> None:
    config: DBOSConfig = {"name": "lucidota-dbos-brain-child", "system_database_url": STATE_DB}
    DBOS(config=config)
    DBOS.launch()


def cmd_current(args: argparse.Namespace) -> dict[str, Any]:
    return {"ok": True, "active_children": active_children()}


def cmd_audit(args: argparse.Namespace) -> dict[str, Any]:
    return {"ok": True, "audit": audit_summary(args.limit)}


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    ensure_schemas()
    run_id = args.workflow_id or stable_workflow_id(args.path, args.state_path, args.map_jsonl, args.dry_run)
    init_dbos()
    status = workflow_status_to_dict(DBOS.get_workflow_status(run_id))
    DBOS.destroy(destroy_registry=True)
    return {
        "ok": True,
        "workflow_name": WORKFLOW_NAME,
        "workflow_id": run_id,
        "dbos_status": status,
        "events": event_status(run_id, args.limit),
        "map_entries": map_status(args.path, args.map_jsonl, args.limit) if args.path else [],
        "active_children": active_children(),
    }


def cmd_run_file(args: argparse.Namespace) -> dict[str, Any]:
    ensure_schemas()
    run_id = args.workflow_id or stable_workflow_id(args.path, args.state_path, args.map_jsonl, args.dry_run)
    init_dbos()
    with SetWorkflowID(run_id):
        result = brain_child_workflow(args.path, args.state_path, args.map_jsonl, args.timeout_seconds, args.dry_run)
    status = workflow_status_to_dict(DBOS.get_workflow_status(run_id))
    DBOS.destroy(destroy_registry=True)
    return {"ok": True, "workflow_name": WORKFLOW_NAME, "workflow_id": run_id, "dbos_status": status, "result": result}


def cmd_resume(args: argparse.Namespace) -> dict[str, Any]:
    ensure_schemas()
    init_dbos()
    handle = DBOS.resume_workflow(args.workflow_id)
    try:
        result = handle.get_result(timeout_seconds=args.timeout_seconds)
    except TypeError:
        result = handle.get_result()
    status = workflow_status_to_dict(DBOS.get_workflow_status(args.workflow_id))
    DBOS.destroy(destroy_registry=True)
    return {"ok": True, "workflow_name": WORKFLOW_NAME, "workflow_id": args.workflow_id, "dbos_status": status, "result": result}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lucidota-dbos-brain-child")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("current")
    p.set_defaults(func=cmd_current)

    p = sub.add_parser("audit")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("run-file")
    p.add_argument("path")
    p.add_argument("--state-path", default=str(DEFAULT_STATE))
    p.add_argument("--map-jsonl", default=str(DEFAULT_MAP))
    p.add_argument("--timeout-seconds", type=int, default=0, help="0 = dynamic policy; >0 = explicit override")
    p.add_argument("--workflow-id", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run_file)

    p = sub.add_parser("status")
    p.add_argument("--path", default="")
    p.add_argument("--state-path", default=str(DEFAULT_STATE))
    p.add_argument("--map-jsonl", default=str(DEFAULT_MAP))
    p.add_argument("--workflow-id", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("resume")
    p.add_argument("workflow_id")
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.set_defaults(func=cmd_resume)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
