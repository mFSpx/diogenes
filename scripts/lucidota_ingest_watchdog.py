#!/usr/bin/env python3
"""LUCIDOTA ingestion watchdog / observation dashboard.

Cron-safe, conservative maintenance loop:
- observe live KRAMPUS/KORPUS/brain sidecar state;
- write JSON/Markdown/HTML observation dashboard;
- apply only narrow safe remediation: terminate stale binary/media/PDF brain-sidecar
  children that are known to be safe to skip in the sidecar, and restart the
  dashboard writer if it disappears;
- never moves evidence files, never kills the watcher/migrator parent, never
  runs broad schema DDL.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.runtime_dsns import resolve_state_dsn, resolve_storage_dsn

STATE_DB = resolve_state_dsn("postgresql://mfspx@/lucidota_state")
STORAGE_DB = resolve_storage_dsn("postgresql:///lucidota_storage")
GRAPH_DB = os.environ.get("LUCIDOTA_GRAPH_DATABASE_URL", "postgresql://mfspx@/lucidota_graph")
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)
WATCH_DIR = ROOT / "KRAMPUSCHEWING"
OUT_DIR = ROOT / "05_OUTPUTS"
RUN_DIR = ROOT / "04_RUNTIME"
STATUS_JSON = OUT_DIR / "ingestion_watchdog_status.json"
LATEST_MD = OUT_DIR / "ingestion_watchdog_latest.md"
DASHBOARD_HTML = OUT_DIR / "ingestion_observation_dashboard.html"
LEGACY_LIVE_DASHBOARD_HTML = OUT_DIR / "ingestion_live_dashboard.html"
RAW_BIG_BOARD_HTML = OUT_DIR / "ingestion_raw_big_board.html"
LIVE_DASHBOARD_REFRESH_SECONDS = int(os.environ.get("LUCIDOTA_LIVE_DASHBOARD_REFRESH_SECONDS", "15"))
REQUIRE_RAW_BIG_BOARD = os.environ.get("LUCIDOTA_REQUIRE_RAW_BIG_BOARD", "0") == "1"
LOG = RUN_DIR / "ingestion_watchdog.log"
DONE_MARKER = OUT_DIR / "INGEST_DONE_RELAUNCH_ME.md"
DONE_COUNT = RUN_DIR / "ingestion_watchdog_done_count.txt"
STREAM_STATE = RUN_DIR / "korpus_result_stream_state.json"
AUDIT_JSONL = OUT_DIR / "korpus_krampii" / "brain_sidecar_audit.jsonl"
BRAIN_MAP = OUT_DIR / "korpus_krampii" / "krampus_brain_map.jsonl"
LEDGER = OUT_DIR / "korpus_krampii" / "latest_chrono_ledger.csv"
CHRONO_LOG = RUN_DIR / "chronological_migrator.log"
CRON_MARKER = "# LUCIDOTA-INGEST-WATCHDOG"
DASHBOARD_THROTTLED_INTERVAL = int(os.environ.get("LUCIDOTA_DASHBOARD_THROTTLED_INTERVAL", "5"))
IO_FULL_WARN_PCT = float(os.environ.get("LUCIDOTA_IO_FULL_WARN_PCT", "15"))
IO_FULL_CRIT_PCT = float(os.environ.get("LUCIDOTA_IO_FULL_CRIT_PCT", "35"))
MEM_AVAIL_WARN_PCT = float(os.environ.get("LUCIDOTA_MEM_AVAIL_WARN_PCT", "15"))
SWAP_USED_WARN_PCT = float(os.environ.get("LUCIDOTA_SWAP_USED_WARN_PCT", "50"))
GPU_MEM_WARN_PCT = float(os.environ.get("LUCIDOTA_GPU_MEM_WARN_PCT", "85"))

BINARY_OR_MEDIA_SUFFIXES = {
    ".3gp", ".7z", ".aac", ".avi", ".bmp", ".bz2", ".db", ".dmg",
    ".doc", ".docx", ".exe", ".flac", ".gif", ".gz", ".heic", ".ico",
    ".iso", ".jpeg", ".jpg", ".m4a", ".m4v", ".mkv", ".mov", ".mp3",
    ".mp4", ".odp", ".ods", ".odt", ".ogg", ".parquet", ".pdf", ".png",
    ".ppt", ".pptx", ".rar", ".sqlite", ".sqlite3", ".tar", ".tif",
    ".tiff", ".wav", ".webm", ".webp", ".xls", ".xlsx", ".zip",
}
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md", ".py", ".rst", ".svg", ".tsv", ".txt", ".yaml", ".yml"}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def localnow() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def previous_status() -> dict[str, Any]:
    return read_json_file(STATUS_JSON)


def log_line(message: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as out:
        out.write(f"[{localnow()}] {message}\n")


def scalar(db_url: str, sql: str, params: tuple = ()) -> Any:
    try:
        with psycopg.connect(db_url, connect_timeout=3) as conn:
            return conn.execute(sql, params).fetchone()[0]
    except Exception as exc:
        return f"error:{type(exc).__name__}:{str(exc)[:100]}"


def rows(db_url: str, sql: str, params: tuple = ()) -> list[tuple]:
    try:
        with psycopg.connect(db_url, connect_timeout=3) as conn:
            return list(conn.execute(sql, params).fetchall())
    except Exception:
        return []


def emit_event(phase: str, status: str, detail: dict[str, Any]) -> str:
    try:
        with psycopg.connect(STATE_DB, connect_timeout=3) as conn:
            ok = conn.execute("SELECT to_regclass('lucidota_control.workflow_event') IS NOT NULL").fetchone()[0]
            if not ok:
                return "workflow_event_missing"
            event_id = conn.execute(
                """
                INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
                VALUES ('ingestion-watchdog-maintenance', %s, %s, %s, 'lucidota_ingest_watchdog', %s::jsonb)
                RETURNING event_id::text
                """,
                (f"watchdog-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}", phase, status, jdump(detail)),
            ).fetchone()[0]
            conn.commit()
            return str(event_id)
    except Exception as exc:
        return f"event_failed:{type(exc).__name__}:{str(exc)[:120]}"


def run(cmd: list[str], timeout: int = 5) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def pgrep(pattern: str) -> list[str]:
    proc = subprocess.run(["pgrep", "-f", pattern], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return [x.strip() for x in proc.stdout.splitlines() if x.strip()]


def cmdline(pid: str) -> list[str]:
    try:
        return [x.decode("utf-8", "replace") for x in (Path("/proc") / pid / "cmdline").read_bytes().split(b"\0") if x]
    except Exception:
        return []


def etimes(pid: str) -> int:
    try:
        return int(subprocess.check_output(["ps", "-p", pid, "-o", "etimes="], text=True).strip() or "0")
    except Exception:
        return 0


def psline(pid: str) -> str:
    try:
        return subprocess.check_output(["ps", "-p", pid, "-o", "pid=,ppid=,etime=,%cpu=,%mem=,rss=,stat=,args="], text=True).strip()
    except Exception:
        return ""


def parse_meminfo() -> dict[str, Any]:
    data: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            parts = line.replace(":", "").split()
            if len(parts) >= 2:
                data[parts[0]] = int(parts[1]) * 1024
    except OSError:
        return {"status": "missing"}
    total = data.get("MemTotal", 0)
    avail = data.get("MemAvailable", 0)
    swap_total = data.get("SwapTotal", 0)
    swap_free = data.get("SwapFree", 0)
    swap_used = max(0, swap_total - swap_free)
    return {
        "status": "ok",
        "mem_total_bytes": total,
        "mem_available_bytes": avail,
        "mem_available_pct": bar_pct(avail, total),
        "swap_total_bytes": swap_total,
        "swap_used_bytes": swap_used,
        "swap_used_pct": bar_pct(swap_used, swap_total) if swap_total else 0.0,
    }


def parse_psi_file(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            bucket = parts[0]
            vals = {}
            for item in parts[1:]:
                if "=" in item:
                    k, v = item.split("=", 1)
                    try:
                        vals[k] = float(v)
                    except ValueError:
                        vals[k] = v
            out[bucket] = vals
    except OSError:
        out["status"] = "missing"
    return out


def pressure_report() -> dict[str, Any]:
    return {
        "cpu": parse_psi_file(Path("/proc/pressure/cpu")),
        "memory": parse_psi_file(Path("/proc/pressure/memory")),
        "io": parse_psi_file(Path("/proc/pressure/io")),
    }


def disk_report() -> dict[str, Any]:
    rows_out = []
    for target in {"/", str(ROOT), str(WATCH_DIR), str(OUT_DIR)}:
        try:
            st = os.statvfs(target)
            total = st.f_blocks * st.f_frsize
            free = st.f_bavail * st.f_frsize
            files_total = st.f_files
            files_free = st.f_favail
            rows_out.append(
                {
                    "target": target,
                    "bytes_total": total,
                    "bytes_free": free,
                    "bytes_used_pct": 100.0 - bar_pct(free, total),
                    "inodes_total": files_total,
                    "inodes_free": files_free,
                    "inodes_used_pct": 100.0 - bar_pct(files_free, files_total),
                }
            )
        except OSError as exc:
            rows_out.append({"target": target, "error": str(exc)})
    return {"mounts": rows_out}


def gpu_report() -> dict[str, Any]:
    if not shutil_which("nvidia-smi"):
        return {"status": "missing"}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0:
            return {"status": "error", "stderr": proc.stderr[-300:]}
        gpus = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 8:
                used = float(parts[4] or 0)
                total = float(parts[5] or 0)
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "temp_c": float(parts[2] or 0),
                        "util_pct": float(parts[3] or 0),
                        "memory_used_mib": used,
                        "memory_total_mib": total,
                        "memory_used_pct": bar_pct(used, total),
                        "power_draw_w": parts[6],
                        "power_limit_w": parts[7],
                    }
                )
        return {"status": "ok", "gpus": gpus}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}:{str(exc)[:160]}"}


def shutil_which(name: str) -> str | None:
    # Tiny local replacement to avoid another import in this long-running cron script.
    for part in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(part) / name
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


def fd_report(processes: dict[str, Any]) -> dict[str, Any]:
    rows_out = []
    for group, items in processes.items():
        for item in items:
            pid = str(item.get("pid"))
            try:
                fd_count = len(list((Path("/proc") / pid / "fd").iterdir()))
            except OSError:
                fd_count = 0
            soft = hard = 0
            try:
                for line in (Path("/proc") / pid / "limits").read_text().splitlines():
                    if line.startswith("Max open files"):
                        parts = line.split()
                        soft = int(parts[3]) if parts[3].isdigit() else 0
                        hard = int(parts[4]) if parts[4].isdigit() else 0
                        break
            except OSError:
                pass
            rows_out.append(
                {
                    "group": group,
                    "pid": pid,
                    "fd_count": fd_count,
                    "fd_soft_limit": soft,
                    "fd_hard_limit": hard,
                    "fd_used_pct": bar_pct(fd_count, soft) if soft else 0.0,
                }
            )
    return {"processes": rows_out}


def kernel_hints() -> dict[str, Any]:
    cmd = ["dmesg", "-T"]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3, check=False)
        if proc.returncode != 0:
            return {"status": "unavailable", "stderr": proc.stderr[-200:]}
        rx = re.compile(r"(oom|out of memory|killed process|gpu|nvrm|xid|i/o error|ext4|btrfs|blocked for more than)", re.I)
        hits = [line for line in proc.stdout.splitlines() if rx.search(line)]
        return {"status": "ok", "tail": hits[-20:], "count": len(hits)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}:{str(exc)[:160]}"}


def hardware_report(processes: dict[str, Any]) -> dict[str, Any]:
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = 0.0
    return {
        "load": {"load1": load1, "load5": load5, "load15": load15, "cpu_count": os.cpu_count() or 1},
        "memory": parse_meminfo(),
        "pressure": pressure_report(),
        "disk": disk_report(),
        "gpu": gpu_report(),
        "fds": fd_report(processes),
        "kernel_hints": kernel_hints(),
    }


def active_processes() -> dict[str, Any]:
    patterns = {
        "watcher": "krampuschewing_watcher.sh",
        "chrono": "chronological_migrator.sh",
        "brain_child": "lucidota_brain_ingest.py",
        "absurd_brain_child": "lucidota_absurd_brain_child.py run-file",
        "korpus_ingest": "korpus_krampii.py.*ingest",
        "dashboard_writer": "lucidota_big_board.py --ingest-html --watch",
    }
    out: dict[str, Any] = {}
    self_pid = str(os.getpid())
    for name, pat in patterns.items():
        items = []
        for pid in pgrep(pat):
            if pid == self_pid:
                continue
            args = cmdline(pid)
            joined = " ".join(args)
            if "lucidota_ingest_watchdog.py" in joined:
                continue
            items.append({"pid": pid, "etimes": etimes(pid), "cmdline": args, "ps": psline(pid)})
        out[name] = items
    return out


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def fmt_bytes(n: Any) -> str:
    try:
        v = float(n)
    except Exception:
        return str(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(v) < 1024 or unit == "TB":
            return f"{v:.1f} {unit}" if unit != "B" else f"{int(v)} B"
        v /= 1024
    return str(n)


def bar_pct(done: int | float, total: int | float) -> float:
    try:
        if float(total) <= 0:
            return 0.0
        return max(0.0, min(100.0, float(done) / float(total) * 100.0))
    except Exception:
        return 0.0


def timeout_policy_for(path: str) -> dict[str, Any]:
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    suffix = p.suffix.lower()
    mib = 1024 * 1024
    gib = 1024 * mib
    if suffix in BINARY_OR_MEDIA_SUFFIXES:
        timeout = 10
        policy = "binary_media_fast_skip"
    elif size <= mib:
        timeout = 10
        policy = "small_text"
    elif size <= 16 * mib:
        timeout = 30
        policy = "normal_text_window"
    elif size <= gib:
        timeout = 20
        policy = "oversize_sidecar_expected_skip"
    else:
        timeout = 60
        policy = "huge_sidecar_expected_skip"
    # Direct legacy child gets grace because it is not ABSURD-timeboxed itself.
    grace = max(30, timeout * 2)
    return {"size_bytes": size, "suffix": suffix, "timeout_seconds": timeout, "legacy_grace_seconds": grace, "policy": policy}


def active_brain_children(processes: dict[str, Any]) -> list[dict[str, Any]]:
    children = []
    for item in processes.get("brain_child", []):
        args = item.get("cmdline") or []
        target = ""
        if args:
            target = args[-1]
        if not target.startswith("/") and target:
            target = str((ROOT / target).resolve(strict=False))
        item = dict(item)
        item["target"] = target
        item["timeout_policy"] = timeout_policy_for(target) if target else {}
        children.append(item)
    return children


def audit_summary(limit: int = 8) -> dict[str, Any]:
    total = bad = 0
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    tail: list[dict[str, Any]] = []
    if AUDIT_JSONL.exists():
        for line in AUDIT_JSONL.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue
            status = str(obj.get("status") or "unknown")
            reason = str(obj.get("reason") or "")
            status_counts[status] = status_counts.get(status, 0) + 1
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            tail.append(obj)
            if len(tail) > limit:
                tail = tail[-limit:]
    return {"path": str(AUDIT_JSONL.relative_to(ROOT)), "exists": AUDIT_JSONL.exists(), "total": total, "bad_lines": bad, "status_counts": status_counts, "reason_counts": reason_counts, "tail": tail}


def last_brain_marker() -> dict[str, Any]:
    if not CHRONO_LOG.exists():
        return {"index": 0, "line": ""}
    data = CHRONO_LOG.read_bytes()[-4_000_000:].decode("utf-8", "replace")
    matches = list(re.finditer(r"\[(.*?)\] brain\[(\d+)\]: (.*)", data))
    if not matches:
        return {"index": 0, "line": ""}
    m = matches[-1]
    return {"at": m.group(1), "index": int(m.group(2)), "path": m.group(3), "line": m.group(0)}


def tail_line(path: Path, max_bytes: int = 1_000_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(-max_bytes, os.SEEK_END)
            data = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    lines = [line for line in data.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def latest_korpus_result_stream() -> dict[str, Any]:
    try:
        files = sorted((OUT_DIR / "korpus_krampii").glob("*.korpus.results.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        files = []
    if not files:
        return {"exists": False}
    path = files[0]
    try:
        st = path.stat()
    except OSError:
        return {"exists": False}
    state = read_json_file(STREAM_STATE)
    path_key = str(path)
    lines = 0
    if state.get("path") == path_key and int(state.get("bytes") or 0) <= st.st_size:
        lines = int(state.get("lines") or 0)
        offset = int(state.get("bytes") or 0)
        try:
            with path.open("rb") as fh:
                fh.seek(offset)
                lines += fh.read().count(b"\n")
        except OSError:
            pass
    else:
        lines = count_lines(path)
    STREAM_STATE.parent.mkdir(parents=True, exist_ok=True)
    STREAM_STATE.write_text(jdump({"path": path_key, "bytes": st.st_size, "lines": lines, "mtime": st.st_mtime}) + "\n", encoding="utf-8")
    last = tail_line(path)
    parsed: dict[str, Any] = {}
    if last:
        try:
            obj = json.loads(last)
            parsed = {
                "path": obj.get("path"),
                "file_kind": obj.get("file_kind"),
                "size_bytes": obj.get("size_bytes"),
                "ok": obj.get("ok"),
                "components": len(obj.get("components") or []),
                "error": obj.get("error"),
                "deferred_reason": (obj.get("detail") or {}).get("deferred_reason"),
            }
        except Exception as exc:
            parsed = {"parse_error": f"{type(exc).__name__}:{str(exc)[:120]}", "raw_tail": last[-500:]}
    return {
        "exists": True,
        "path": str(path.relative_to(ROOT)),
        "lines": lines,
        "bytes": st.st_size,
        "mtime": dt.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
        "age_seconds": max(0, int(time.time() - st.st_mtime)),
        "last": parsed,
    }


def ingest_report(processes: dict[str, Any], light: bool = False) -> dict[str, Any]:
    cached = (previous_status().get("report") or {}) if light else {}
    if light and cached:
        report = dict(cached)
        report["korpus_result_stream"] = latest_korpus_result_stream()
        report["active_brain_children"] = active_brain_children(processes)
        report["light_mode"] = True
        return report
    raw = rows(
        STORAGE_DB,
        """
        SELECT count(*), count(DISTINCT sha256_hash), count(*)-count(DISTINCT sha256_hash),
               coalesce(sum(file_size_bytes),0), max(ingested_at)
        FROM lucidota_archaeology.raw_file_inventory
        WHERE absolute_path LIKE %s
        """,
        (str(WATCH_DIR) + "/%",),
    )
    raw_tuple = raw[0] if raw else (0, 0, 0, 0, None)
    k = {
        "files": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_korpus.file_object"),
        "occurrences": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_korpus.file_occurrence"),
        "components": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_korpus.component"),
        "entities": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_korpus.entity"),
        "concepts": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_korpus.concept"),
        "deferred": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_korpus.file_object WHERE status='deferred'"),
        "river_decisions": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_korpus.river_decision"),
        "vibe_telemetry": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_korpus.vibe_telemetry"),
    }
    watch_files = 0
    watch_bytes = 0
    if WATCH_DIR.exists():
        for root, _dirs, files in os.walk(WATCH_DIR):
            watch_files += len(files)
            for name in files:
                try:
                    watch_bytes += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    ledger_items = max(0, count_lines(LEDGER) - 1)
    brain_map_lines = count_lines(BRAIN_MAP)
    marker = last_brain_marker()
    return {
        "watch_files": watch_files,
        "watch_bytes": watch_bytes,
        "raw_inventory": {
            "rows": int(raw_tuple[0] or 0),
            "unique_hashes": int(raw_tuple[1] or 0),
            "duplicate_paths": int(raw_tuple[2] or 0),
            "bytes": int(raw_tuple[3] or 0),
            "latest": str(raw_tuple[4]),
        },
        "ledger_items": ledger_items,
        "brain_marker": marker,
        "brain_map_lines": brain_map_lines,
        "korpus_result_stream": latest_korpus_result_stream(),
        "korpus": k,
        "active_brain_children": active_brain_children(processes),
        "light_mode": False,
    }


def locks_and_queues(light: bool = False) -> dict[str, Any]:
    locks = {}
    for name, db in [("state", STATE_DB), ("storage", STORAGE_DB), ("graph", GRAPH_DB)]:
        waiting = scalar(db, "SELECT count(*) FROM pg_locks WHERE pid IN (SELECT pid FROM pg_stat_activity WHERE datname=current_database()) AND NOT granted")
        lock_waiters = scalar(db, "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock'")
        active = rows(db, "SELECT pid::text, state, wait_event_type, wait_event, left(query,160) FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid() AND state<>'idle' ORDER BY pid LIMIT 20")
        locks[name] = {"waiting_locks": waiting, "lock_waiters": lock_waiters, "active_nonidle": active}
    if light:
        queues = (previous_status().get("locks_queues") or {}).get("queues", {})
    else:
        queues = {
            "absurd_status": rows(STATE_DB, "SELECT status, count(*) FROM absurd.workflow_status GROUP BY status ORDER BY status"),
            "event_outbox": rows(STATE_DB, "SELECT status, count(*) FROM lucidota_control.event_outbox GROUP BY status ORDER BY status"),
            "indy_side_queue": rows(STATE_DB, "SELECT status, count(*) FROM lucidota_indy.side_queue GROUP BY status ORDER BY status"),
            "korpus_batches": rows(STORAGE_DB, "SELECT status, count(*) FROM lucidota_korpus.ingest_batch GROUP BY status ORDER BY status"),
            "korpus_files": rows(STORAGE_DB, "SELECT status, count(*) FROM lucidota_korpus.file_object GROUP BY status ORDER BY status"),
            "korpus_embedding_queue": rows(STORAGE_DB, "SELECT status, count(*) FROM lucidota_korpus.embedding_queue GROUP BY status ORDER BY status"),
            "korpus_derived_compute_queue": rows(STORAGE_DB, "SELECT task_type, status, count(*), min(created_at)::text FROM lucidota_korpus.derived_compute_queue GROUP BY task_type, status ORDER BY task_type, status"),
        }
    return {"locks": locks, "queues": queues}


def is_done(processes: dict[str, Any], report: dict[str, Any]) -> bool:
    # The watcher and dashboard may be long-running daemons. The finite big drop is
    # done when chronological_migrator and per-file children are gone, and the log
    # contains the terminal marker.
    no_finite_workers = not processes.get("chrono") and not processes.get("brain_child") and not processes.get("absurd_brain_child") and not processes.get("korpus_ingest")
    log_done = False
    if CHRONO_LOG.exists():
        tail = CHRONO_LOG.read_bytes()[-200_000:].decode("utf-8", "replace")
        log_done = "chronological migrator done" in tail
    return bool(no_finite_workers and log_done)


def update_done_count(done: bool) -> int:
    if done:
        n = 0
        if DONE_COUNT.exists():
            try:
                n = int(DONE_COUNT.read_text().strip() or "0")
            except Exception:
                n = 0
        n += 1
        DONE_COUNT.write_text(str(n), encoding="utf-8")
        return n
    DONE_COUNT.write_text("0", encoding="utf-8")
    return 0


def dashboard_interval(processes: dict[str, Any]) -> float:
    for item in processes.get("dashboard_writer", []):
        args = item.get("cmdline") or []
        for i, arg in enumerate(args):
            if arg == "--interval" and i + 1 < len(args):
                try:
                    return float(args[i + 1])
                except ValueError:
                    return 0.0
    return 0.0


def dashboard_output_paths(processes: dict[str, Any]) -> list[str]:
    outputs = []
    for item in processes.get("dashboard_writer", []):
        args = item.get("cmdline") or []
        for i, arg in enumerate(args):
            if arg == "--output" and i + 1 < len(args):
                outputs.append(args[i + 1])
    return outputs


def start_dashboard_writer(interval: int = DASHBOARD_THROTTLED_INTERVAL) -> int:
    log_path = RUN_DIR / "ingestion_dashboard_writer.log"
    pid_path = RUN_DIR / "ingestion_dashboard_writer.pid"
    with log_path.open("ab") as logfh:
        proc = subprocess.Popen(
            [
                str(PY),
                "-u",
                str(ROOT / "scripts" / "lucidota_big_board.py"),
                "--ingest-html",
                "--watch",
                "--interval",
                str(interval),
                "--output",
                str(RAW_BIG_BOARD_HTML),
            ],
            cwd=ROOT,
            stdout=logfh,
            stderr=logfh,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    return int(proc.pid)


def cancel_storage_autovacuum() -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    try:
        with psycopg.connect(STORAGE_DB, connect_timeout=3, autocommit=True) as conn:
            rows_ = list(
                conn.execute(
                    """
                    SELECT pid, query
                    FROM pg_stat_activity
                    WHERE datname=current_database()
                      AND state='active'
                      AND query LIKE 'autovacuum:%'
                    """
                )
            )
            for pid, query in rows_:
                ok = conn.execute("SELECT pg_cancel_backend(%s)", (pid,)).fetchone()[0]
                actions.append({"pid": str(pid), "ok": bool(ok), "query": str(query)[:220]})
    except Exception as exc:
        actions.append({"error": f"{type(exc).__name__}:{str(exc)[:160]}"})
    return actions


def safe_maintenance(processes: dict[str, Any], hardware: dict[str, Any], dry_run: bool = False) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    dashboard_touched = False
    # Fix only legacy direct brain children that are on known sidecar-skip file
    # classes and are older than the dynamic policy's grace window.
    for child in active_brain_children(processes):
        target = child.get("target") or ""
        policy = child.get("timeout_policy") or {}
        suffix = policy.get("suffix")
        age = int(child.get("etimes") or 0)
        grace = int(policy.get("legacy_grace_seconds") or 999999)
        if suffix in BINARY_OR_MEDIA_SUFFIXES and age > grace:
            action = {
                "kind": "terminate_stale_brain_sidecar_child",
                "pid": child.get("pid"),
                "target": target,
                "age_seconds": age,
                "policy": policy,
                "dry_run": dry_run,
            }
            if not dry_run:
                try:
                    os.kill(int(child["pid"]), signal.SIGTERM)
                    action["result"] = "SIGTERM_sent"
                    event_id = emit_event("remediate_stale_child", "succeeded", action)
                    action["workflow_event"] = event_id
                except Exception as exc:
                    action["result"] = f"failed:{type(exc).__name__}:{exc}"
                    emit_event("remediate_stale_child", "failed", action)
            actions.append(action)

    # Raw Big Board is optional now that the canonical observation dashboard is live.
    # Do not resurrect it during heavy IO unless explicitly requested.
    if REQUIRE_RAW_BIG_BOARD and not processes.get("dashboard_writer"):
        action = {"kind": "restart_dashboard_writer", "dry_run": dry_run}
        if not dry_run:
            try:
                pid = start_dashboard_writer(DASHBOARD_THROTTLED_INTERVAL)
                action["result"] = f"started:{pid}"
                action["workflow_event"] = emit_event("restart_dashboard_writer", "succeeded", action)
            except Exception as exc:
                action["result"] = f"failed:{type(exc).__name__}:{exc}"
                emit_event("restart_dashboard_writer", "failed", action)
        actions.append(action)
        dashboard_touched = True

    # Collapse the old naming mess: the raw Big Board feed must not overwrite
    # ingestion_live_dashboard.html now that that path is a redirect to the
    # canonical observation dashboard.
    legacy_outputs = {str(LEGACY_LIVE_DASHBOARD_HTML), str(LEGACY_LIVE_DASHBOARD_HTML.relative_to(ROOT))}
    active_outputs = set(dashboard_output_paths(processes))
    if processes.get("dashboard_writer") and active_outputs.intersection(legacy_outputs):
        action = {
            "kind": "migrate_dashboard_writer_output",
            "old_outputs": sorted(active_outputs.intersection(legacy_outputs)),
            "new_output": str(RAW_BIG_BOARD_HTML),
            "dry_run": dry_run,
        }
        if not dry_run:
            stopped = []
            for item in processes.get("dashboard_writer", []):
                try:
                    os.kill(int(item["pid"]), signal.SIGTERM)
                    stopped.append(item["pid"])
                except Exception:
                    pass
            time.sleep(1)
            try:
                pid = start_dashboard_writer(DASHBOARD_THROTTLED_INTERVAL)
                action["result"] = f"stopped={stopped};started={pid}"
                action["workflow_event"] = emit_event("migrate_dashboard_writer_output", "succeeded", action)
            except Exception as exc:
                action["result"] = f"failed:{type(exc).__name__}:{exc}"
                emit_event("migrate_dashboard_writer_output", "failed", action)
        actions.append(action)
        dashboard_touched = True

    # If the metal is screaming on IO, slow the read-only dashboard writer first.
    io_full = float((((hardware.get("pressure") or {}).get("io") or {}).get("full") or {}).get("avg10") or 0.0)
    if io_full >= IO_FULL_CRIT_PCT:
        action = {"kind": "cancel_autovacuum_for_ingest_io_pressure", "io_full_avg10": io_full, "dry_run": dry_run}
        if not dry_run:
            result = cancel_storage_autovacuum()
            action["result"] = result
            if result:
                action["workflow_event"] = emit_event("cancel_autovacuum_io_pressure", "succeeded", action)
        actions.append(action)

    current_interval = dashboard_interval(processes)
    if not dashboard_touched and processes.get("dashboard_writer") and io_full >= IO_FULL_WARN_PCT and 0 < current_interval < DASHBOARD_THROTTLED_INTERVAL:
        action = {
            "kind": "throttle_dashboard_writer_for_io_pressure",
            "io_full_avg10": io_full,
            "old_interval": current_interval,
            "new_interval": DASHBOARD_THROTTLED_INTERVAL,
            "dry_run": dry_run,
        }
        if not dry_run:
            stopped = []
            for item in processes.get("dashboard_writer", []):
                try:
                    os.kill(int(item["pid"]), signal.SIGTERM)
                    stopped.append(item["pid"])
                except Exception:
                    pass
            time.sleep(1)
            try:
                pid = start_dashboard_writer(DASHBOARD_THROTTLED_INTERVAL)
                action["result"] = f"stopped={stopped};started={pid}"
                action["workflow_event"] = emit_event("throttle_dashboard_writer", "succeeded", action)
            except Exception as exc:
                action["result"] = f"failed:{type(exc).__name__}:{exc}"
                emit_event("throttle_dashboard_writer", "failed", action)
        actions.append(action)
    return actions


def pct_bars(report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = report["raw_inventory"]
    k = report["korpus"]
    watch_files = int(report.get("watch_files") or 0)
    watch_bytes = int(report.get("watch_bytes") or 0)
    ledger = int(report.get("ledger_items") or 0)
    brain_idx = int((report.get("brain_marker") or {}).get("index") or 0)
    bars = [
        {"name": "Chrono ledger vs disk", "pct": bar_pct(ledger, watch_files), "done": ledger, "total": watch_files},
        {"name": "Brain pass vs ledger", "pct": bar_pct(brain_idx, ledger), "done": brain_idx, "total": ledger},
        {"name": "KORPUS current result stream", "pct": bar_pct((report.get("korpus_result_stream") or {}).get("lines") or 0, watch_files), "done": (report.get("korpus_result_stream") or {}).get("lines") or 0, "total": watch_files},
        {"name": "Raw inventory paths", "pct": bar_pct(raw["rows"], watch_files), "done": raw["rows"], "total": watch_files},
        {"name": "Raw inventory bytes", "pct": bar_pct(raw["bytes"], watch_bytes), "done": raw["bytes"], "total": watch_bytes, "bytes": True},
        {"name": "KORPUS file objects vs disk", "pct": bar_pct(k["files"], watch_files), "done": k["files"], "total": watch_files},
        {"name": "KORPUS components / River", "pct": bar_pct(k["river_decisions"], k["components"]), "done": k["river_decisions"], "total": k["components"]},
        {"name": "Vibe telemetry / components", "pct": bar_pct(k["vibe_telemetry"], k["components"]), "done": k["vibe_telemetry"], "total": k["components"]},
    ]
    return bars


def health(report: dict[str, Any], locks_queues: dict[str, Any], actions: list[dict[str, Any]], done: bool) -> dict[str, Any]:
    issues = []
    critical = []
    if done:
        return {"level": "done", "summary": "FINITE INGEST COMPLETE", "issues": []}
    processes = report.get("processes", {})
    hardware = report.get("hardware", {})
    if not processes.get("watcher"):
        critical.append("watcher process not detected")
    if not processes.get("chrono") and not processes.get("korpus_ingest"):
        issues.append("chronological migrator not detected and done marker not confirmed")
    waiting = sum(int(v.get("waiting_locks") or 0) for v in locks_queues.get("locks", {}).values() if isinstance(v.get("waiting_locks"), int))
    if waiting:
        issues.append(f"DB waiting locks: {waiting}")
    mem = hardware.get("memory") or {}
    if mem.get("status") == "ok":
        if float(mem.get("mem_available_pct") or 0.0) < MEM_AVAIL_WARN_PCT:
            issues.append(f"low available RAM: {float(mem.get('mem_available_pct') or 0):.1f}%")
        if float(mem.get("swap_used_pct") or 0.0) > SWAP_USED_WARN_PCT:
            issues.append(f"high swap usage: {float(mem.get('swap_used_pct') or 0):.1f}%")
    io_full = float((((hardware.get("pressure") or {}).get("io") or {}).get("full") or {}).get("avg10") or 0.0)
    if io_full >= IO_FULL_CRIT_PCT:
        critical.append(f"critical IO stall pressure: full.avg10={io_full:.1f}%")
    elif io_full >= IO_FULL_WARN_PCT:
        issues.append(f"high IO stall pressure: full.avg10={io_full:.1f}%")
    load = hardware.get("load") or {}
    cpu_count = int(load.get("cpu_count") or 1)
    if float(load.get("load1") or 0.0) > cpu_count * 2:
        issues.append(f"high CPU load: {float(load.get('load1') or 0):.2f}/{cpu_count} cores")
    for gpu in (hardware.get("gpu") or {}).get("gpus", []):
        if float(gpu.get("memory_used_pct") or 0.0) > GPU_MEM_WARN_PCT:
            issues.append(f"GPU {gpu.get('index')} VRAM high: {float(gpu.get('memory_used_pct') or 0):.1f}%")
    fd_hot = [x for x in ((hardware.get("fds") or {}).get("processes") or []) if float(x.get("fd_used_pct") or 0.0) > 80]
    if fd_hot:
        issues.append(f"file descriptor pressure: {len(fd_hot)} process(es) over 80% soft limit")
    kh = hardware.get("kernel_hints") or {}
    if kh.get("status") == "ok" and kh.get("tail"):
        critical.append(f"kernel warning hints present: {len(kh.get('tail') or [])} recent matching lines")
    for child in report.get("active_brain_children", []):
        pol = child.get("timeout_policy") or {}
        if pol.get("suffix") in BINARY_OR_MEDIA_SUFFIXES and int(child.get("etimes") or 0) > int(pol.get("legacy_grace_seconds") or 999999):
            issues.append(f"stale legacy sidecar child {child.get('pid')} on {pol.get('suffix')}")
    if actions:
        issues.append(f"maintenance actions this tick: {len(actions)}")
    if critical:
        return {"level": "critical", "summary": "hardware/pipeline saturation needs attention", "issues": critical + issues}
    if issues:
        return {"level": "warn", "summary": "watching / safe maintenance active", "issues": issues}
    return {"level": "ok", "summary": "watching / no unsafe condition detected", "issues": []}


def render_md(status: dict[str, Any]) -> str:
    bars = status["bars"]
    lines = [
        "# LUCIDOTA Ingestion Watchdog",
        "",
        f"Generated: `{status['generated_at_local']}`",
        f"Health: **{status['health']['level'].upper()}** — {status['health']['summary']}",
        f"Done count: `{status['done_count']}`",
        f"Telemetry mode: `{'critical-light' if status.get('light_mode') else 'full'}`",
        "",
        "## Progress",
        "",
    ]
    for b in bars:
        done = fmt_bytes(b["done"]) if b.get("bytes") else f"{int(b['done']):,}"
        total = fmt_bytes(b["total"]) if b.get("bytes") else f"{int(b['total']):,}"
        lines.append(f"- **{b['name']}** `{b['pct']:.1f}%` — {done}/{total}")
    hw = status.get("hardware") or {}
    mem = hw.get("memory") or {}
    io_full = ((((hw.get("pressure") or {}).get("io") or {}).get("full") or {}).get("avg10") or 0.0)
    load = hw.get("load") or {}
    gpu = hw.get("gpu") or {}
    gpu_txt = "missing"
    if gpu.get("status") == "ok":
        gpu_txt = "; ".join(
            f"GPU{x.get('index')} {x.get('memory_used_mib')}/{x.get('memory_total_mib')} MiB VRAM {x.get('util_pct')}% util"
            for x in gpu.get("gpus", [])
        ) or "ok/no GPUs"
    lines += [
        "",
        "## Metal",
        "",
        f"- Load: `{load.get('load1', 0):.2f}` / cores `{load.get('cpu_count', '?')}`",
        f"- RAM available: `{fmt_bytes(mem.get('mem_available_bytes', 0))}` / `{fmt_bytes(mem.get('mem_total_bytes', 0))}` (`{float(mem.get('mem_available_pct') or 0):.1f}%`)",
        f"- Swap used: `{fmt_bytes(mem.get('swap_used_bytes', 0))}` (`{float(mem.get('swap_used_pct') or 0):.1f}%`)",
        f"- IO pressure full.avg10: `{float(io_full):.1f}%`",
        f"- GPU: `{gpu_txt}`",
    ]
    stream = status["report"].get("korpus_result_stream") or {}
    last = stream.get("last") or {}
    if stream.get("exists"):
        lines += [
            "",
            "## Live KORPUS stream",
            "",
            f"- Current result lines: `{stream.get('lines')}` in `{stream.get('path')}`",
            f"- Last result age: `{stream.get('age_seconds')}s`",
            f"- Last file: `{last.get('path')}`",
        ]
    lines += ["", "## Current brain child", ""]
    for c in status["report"].get("active_brain_children", [])[:5]:
        lines.append(f"- PID `{c.get('pid')}` age `{c.get('etimes')}s` target `{c.get('target')}` policy `{c.get('timeout_policy',{}).get('policy')}`")
    if not status["report"].get("active_brain_children"):
        lines.append("- none")
    lines += ["", "## Actions this tick", ""]
    if status["actions"]:
        for a in status["actions"]:
            lines.append(f"- `{a.get('kind')}` pid={a.get('pid','')} result={a.get('result','pending')} target={a.get('target','')}")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Outputs",
        "",
        f"- Canonical dashboard: `{DASHBOARD_HTML.relative_to(ROOT)}`",
        f"- Legacy live URL redirects here: `{LEGACY_LIVE_DASHBOARD_HTML.relative_to(ROOT)}`",
        f"- Raw Big Board feed: `{RAW_BIG_BOARD_HTML.relative_to(ROOT)}`",
        f"- JSON: `{STATUS_JSON.relative_to(ROOT)}`",
        f"- Audit: `{AUDIT_JSONL.relative_to(ROOT)}`",
    ]
    return "\n".join(lines) + "\n"


def render_html(status: dict[str, Any]) -> str:
    def esc(x: Any) -> str:
        return html.escape(str(x if x is not None else ""))
    def num(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default
    def progress(b: dict[str, Any]) -> str:
        pct = float(b["pct"])
        done = fmt_bytes(b["done"]) if b.get("bytes") else f"{int(b['done']):,}"
        total = fmt_bytes(b["total"]) if b.get("bytes") else f"{int(b['total']):,}"
        return f"<div class='barrow'><div class='barhead'><b>{esc(b['name'])}</b><span>{pct:.1f}% · {esc(done)} / {esc(total)}</span></div><div class='bar'><i style='width:{pct:.3f}%'></i></div></div>"
    h = status["health"]
    level = h["level"]
    bars = "\n".join(progress(b) for b in status["bars"])
    proc = status["processes"]
    child_rows = "".join(
        f"<tr><td>{esc(c.get('pid'))}</td><td>{esc(c.get('etimes'))}s</td><td>{esc((c.get('timeout_policy') or {}).get('policy'))}</td><td>{esc(c.get('target'))}</td></tr>"
        for c in status["report"].get("active_brain_children", [])
    ) or "<tr><td colspan='4'>none</td></tr>"
    action_rows = "".join(
        f"<tr><td>{esc(a.get('kind'))}</td><td>{esc(a.get('result','pending'))}</td><td>{esc(a.get('pid',''))}</td><td>{esc(a.get('target',''))}</td></tr>"
        for a in status["actions"]
    ) or "<tr><td colspan='4'>none</td></tr>"
    issue_list = "".join(f"<li>{esc(x)}</li>" for x in h.get("issues", [])) or "<li>none</li>"
    audit = status["audit"]
    audit_rows = "".join(
        f"<tr><td>{esc(x.get('status'))}</td><td>{esc(x.get('reason'))}</td><td>{esc(x.get('path') or (x.get('metadata') or {}).get('path'))}</td></tr>"
        for x in audit.get("tail", [])[-8:]
    ) or "<tr><td colspan='3'>no audit entries yet</td></tr>"
    lock_rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{esc(v.get('waiting_locks'))}</td><td>{esc(v.get('lock_waiters'))}</td><td>{esc(len(v.get('active_nonidle') or []))}</td></tr>"
        for name, v in status["locks_queues"].get("locks", {}).items()
    )
    queues = status["locks_queues"].get("queues", {})
    queue_text = esc(json.dumps(queues, default=str))
    stream = status["report"].get("korpus_result_stream") or {}
    stream_last = stream.get("last") or {}
    hw = status.get("hardware") or {}
    load = hw.get("load") or {}
    mem = hw.get("memory") or {}
    pressure = hw.get("pressure") or {}
    io_pressure = pressure.get("io") or {}
    mem_pressure = pressure.get("memory") or {}
    cpu_pressure = pressure.get("cpu") or {}
    io_full_avg10 = num((io_pressure.get("full") or {}).get("avg10"))
    io_some_avg10 = num((io_pressure.get("some") or {}).get("avg10"))
    mem_full_avg10 = num((mem_pressure.get("full") or {}).get("avg10"))
    cpu_some_avg10 = num((cpu_pressure.get("some") or {}).get("avg10"))
    disk_mounts = (hw.get("disk") or {}).get("mounts") or []
    primary_disk = next((m for m in disk_mounts if m.get("target") == "/"), disk_mounts[0] if disk_mounts else {})
    disk_rows = "".join(
        f"<tr><td>{esc(m.get('target'))}</td><td>{esc(fmt_bytes(m.get('bytes_free', 0)))}</td><td>{num(m.get('bytes_used_pct')):.1f}%</td><td>{num(m.get('inodes_used_pct')):.1f}%</td></tr>"
        for m in disk_mounts
        if not m.get("error")
    ) or "<tr><td colspan='4'>disk telemetry unavailable</td></tr>"
    fd_procs = (hw.get("fds") or {}).get("processes") or []
    max_fd_pct = max((num(x.get("fd_used_pct")) for x in fd_procs), default=0.0)
    fd_rows = "".join(
        f"<tr><td>{esc(x.get('group'))}</td><td>{esc(x.get('pid'))}</td><td>{esc(x.get('fd_count'))}/{esc(x.get('fd_soft_limit'))}</td><td>{num(x.get('fd_used_pct')):.1f}%</td></tr>"
        for x in sorted(fd_procs, key=lambda item: num(item.get("fd_used_pct")), reverse=True)[:12]
    ) or "<tr><td colspan='4'>no watched process FDs</td></tr>"
    gpu = hw.get("gpu") or {}
    gpu_cards = gpu.get("gpus") or []
    if gpu.get("status") == "ok" and gpu_cards:
        max_gpu_vram = max((num(x.get("memory_used_pct")) for x in gpu_cards), default=0.0)
        gpu_summary = "; ".join(
            f"GPU{x.get('index')} {x.get('memory_used_mib')}/{x.get('memory_total_mib')} MiB {num(x.get('util_pct')):.0f}% util"
            for x in gpu_cards
        )
    elif gpu.get("status") == "ok":
        max_gpu_vram = 0.0
        gpu_summary = "ok/no GPU rows"
    else:
        max_gpu_vram = 0.0
        gpu_summary = gpu.get("status", "missing")
    kh = hw.get("kernel_hints") or {}
    kh_tail = kh.get("tail") or []
    kernel_rows = "".join(f"<tr><td>{esc(line)}</td></tr>" for line in kh_tail[-8:]) or "<tr><td>no matching kernel/OOM/GPU/I/O hints</td></tr>"
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta http-equiv='refresh' content='{LIVE_DASHBOARD_REFRESH_SECONDS}'>
<title>LUCIDOTA Observation Dashboard</title>
<style>
:root{{--bg:#060711;--panel:#101827;--panel2:#151f31;--text:#eff7ff;--muted:#90a7c0;--line:#26384f;--good:#53ffb1;--warn:#ffd166;--bad:#ff5c8a;--cyan:#67e8ff}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at top left,#172945,#060711 38%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,Segoe UI,Arial,sans-serif}}
header{{padding:22px 28px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:20px;align-items:flex-end;background:#070b13cc;position:sticky;top:0}}
h1{{margin:0;font-size:26px;letter-spacing:.08em}} .sub{{color:var(--muted);font-size:12px}} .clock{{font-variant-numeric:tabular-nums;color:var(--good)}}
main{{padding:20px;display:grid;grid-template-columns:1.15fr .85fr;gap:16px}} section{{background:linear-gradient(180deg,var(--panel),#0b111c);border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:0 18px 45px #0008}} .wide{{grid-column:1/-1}}
h2{{margin:0 0 12px;color:var(--cyan);font-size:14px;text-transform:uppercase;letter-spacing:.14em}} .health{{font-size:28px;font-weight:800}} .ok{{color:var(--good)}} .warn{{color:var(--warn)}} .done{{color:var(--cyan)}} .critical{{color:var(--bad)}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}} .metric{{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:12px;min-height:82px}} .metric b{{display:block;color:var(--good);font-size:22px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}} .metric span{{display:block;color:var(--text);font-size:11px;text-transform:uppercase;letter-spacing:.08em}} .metric small{{color:var(--muted)}}
.barrow{{margin:10px 0 14px}} .barhead{{display:flex;justify-content:space-between;gap:12px;font-size:13px}} .barhead span{{color:var(--muted);font-variant-numeric:tabular-nums}} .bar{{height:16px;border-radius:999px;overflow:hidden;background:#08101b;border:1px solid var(--line)}} .bar i{{display:block;height:100%;background:linear-gradient(90deg,#4ade80,#67e8ff)}}
table{{width:100%;border-collapse:collapse;font-size:12px}} th,td{{padding:7px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{color:var(--muted);text-transform:uppercase;letter-spacing:.08em}} code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}} pre{{max-height:220px;overflow:auto;background:#060912;border:1px solid var(--line);border-radius:12px;padding:10px;color:#cfe3ff}}
@media(max-width:1100px){{main{{grid-template-columns:1fr}} .grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head>
<body><header><div><h1>LUCIDOTA OBSERVATION DASHBOARD</h1><div class='sub'>CANONICAL dashboard · old ingestion_live_dashboard.html redirects here · raw feed optional: {esc(RAW_BIG_BOARD_HTML.relative_to(ROOT))}</div><div class='sub'>live loop writes ~30s, adaptive 60s under IO critical · cron safety net every 5m · safe maintenance only · telemetry: {esc('critical-light' if status.get('light_mode') else 'full')}</div></div><div><div class='clock'>{esc(status['generated_at_local'])}</div><div class='sub'>browser auto-refresh {LIVE_DASHBOARD_REFRESH_SECONDS}s</div></div></header>
<main>
<section class='wide'><h2>Health</h2><div class='health {esc(level)}'>{esc(level.upper())}: {esc(h['summary'])}</div><ul>{issue_list}</ul></section>
<section><h2>Progress Bars</h2>{bars}</section>
<section><h2>Core Metrics</h2><div class='grid'>
<div class='metric'><b>{esc(status['report']['watch_files'])}</b><span>files on disk</span><small>{esc(fmt_bytes(status['report']['watch_bytes']))}</small></div>
<div class='metric'><b>{esc(status['report']['raw_inventory']['rows'])}</b><span>raw inventory rows</span><small>{esc(status['report']['raw_inventory']['unique_hashes'])} unique</small></div>
<div class='metric'><b>{esc(status['report']['brain_marker'].get('index'))}</b><span>brain index</span><small>{esc(status['report']['brain_marker'].get('path','')[-80:])}</small></div>
<div class='metric'><b>{esc(stream.get('lines', 0))}</b><span>live KORPUS result lines</span><small>last {esc(stream.get('age_seconds', '?'))}s ago</small></div>
<div class='metric'><b>{esc(audit.get('total'))}</b><span>brain audit rows</span><small>{esc(audit.get('status_counts'))}</small></div>
<div class='metric'><b>{esc(len(proc.get('watcher', [])))}</b><span>watchers</span><small>expected >=1</small></div>
<div class='metric'><b>{esc(len(proc.get('chrono', [])))}</b><span>chrono migrators</span><small>finite worker</small></div>
<div class='metric'><b>{esc(len(proc.get('brain_child', [])))}</b><span>brain children</span><small>current child</small></div>
<div class='metric'><b>{esc(status['done_count'])}</b><span>done confirmations</span><small>self-disables at 2</small></div>
</div></section>
<section class='wide'><h2>Metal / Saturation</h2><div class='grid'>
<div class='metric'><b>{num(load.get('load1')):.2f}</b><span>load avg 1m</span><small>{num(load.get('load5')):.2f}/{num(load.get('load15')):.2f} · cores {esc(load.get('cpu_count'))}</small></div>
<div class='metric'><b>{num(mem.get('mem_available_pct')):.1f}%</b><span>RAM available</span><small>{esc(fmt_bytes(mem.get('mem_available_bytes', 0)))} / {esc(fmt_bytes(mem.get('mem_total_bytes', 0)))}</small></div>
<div class='metric'><b>{num(mem.get('swap_used_pct')):.1f}%</b><span>swap used</span><small>{esc(fmt_bytes(mem.get('swap_used_bytes', 0)))} / {esc(fmt_bytes(mem.get('swap_total_bytes', 0)))}</small></div>
<div class='metric'><b>{io_full_avg10:.1f}%</b><span>IO full avg10</span><small>some {io_some_avg10:.1f}% · warn {IO_FULL_WARN_PCT:g}% · crit {IO_FULL_CRIT_PCT:g}%</small></div>
<div class='metric'><b>{max_gpu_vram:.1f}%</b><span>GPU VRAM max</span><small>{esc(gpu_summary)}</small></div>
<div class='metric'><b>{max_fd_pct:.1f}%</b><span>FD pressure max</span><small>watched workers only</small></div>
<div class='metric'><b>{num(primary_disk.get('bytes_used_pct')):.1f}%</b><span>disk used</span><small>{esc(fmt_bytes(primary_disk.get('bytes_free', 0)))} free on {esc(primary_disk.get('target', '?'))}</small></div>
<div class='metric'><b>{esc(len(kh_tail))}</b><span>kernel hints</span><small>{esc(kh.get('status', 'unknown'))} · OOM/GPU/I/O regex</small></div>
</div></section>
<section><h2>Pressure Stall Info</h2><div class='grid'>
<div class='metric'><b>{cpu_some_avg10:.1f}%</b><span>CPU some avg10</span><small>scheduler contention</small></div>
<div class='metric'><b>{mem_full_avg10:.1f}%</b><span>memory full avg10</span><small>OOM risk proxy</small></div>
<div class='metric'><b>{io_some_avg10:.1f}%</b><span>IO some avg10</span><small>at least one task stalled</small></div>
<div class='metric'><b>{io_full_avg10:.1f}%</b><span>IO full avg10</span><small>all runnable work stalled</small></div>
</div></section>
<section><h2>File Descriptors</h2><table><thead><tr><th>Group</th><th>PID</th><th>FDs</th><th>Soft-limit %</th></tr></thead><tbody>{fd_rows}</tbody></table></section>
<section><h2>Disk / Inodes</h2><table><thead><tr><th>Target</th><th>Free</th><th>Used</th><th>Inodes used</th></tr></thead><tbody>{disk_rows}</tbody></table></section>
<section class='wide'><h2>Live KORPUS Result Stream</h2><div class='grid'>
<div class='metric'><b>{esc(stream.get('lines', 0))}</b><span>result rows this run</span><small>{esc(stream.get('path', 'missing'))}</small></div>
<div class='metric'><b>{esc(stream.get('age_seconds', '?'))}s</b><span>last result age</span><small>mtime {esc(stream.get('mtime', ''))}</small></div>
<div class='metric'><b>{esc(stream_last.get('components', 0))}</b><span>last file components</span><small>{esc(stream_last.get('file_kind', ''))} · {esc(fmt_bytes(stream_last.get('size_bytes', 0)))}</small></div>
<div class='metric'><b>{esc(stream_last.get('ok', ''))}</b><span>last result ok</span><small>{esc(stream_last.get('deferred_reason') or stream_last.get('error') or '')}</small></div>
</div><pre>{esc(stream_last.get('path', ''))}</pre></section>
<section class='wide'><h2>Current Brain Child</h2><table><thead><tr><th>PID</th><th>Age</th><th>Policy</th><th>Target</th></tr></thead><tbody>{child_rows}</tbody></table></section>
<section><h2>Maintenance Actions This Tick</h2><table><thead><tr><th>Kind</th><th>Result</th><th>PID</th><th>Target</th></tr></thead><tbody>{action_rows}</tbody></table></section>
<section><h2>DB Locks</h2><table><thead><tr><th>DB</th><th>Waiting locks</th><th>Lock waiters</th><th>Active non-idle</th></tr></thead><tbody>{lock_rows}</tbody></table></section>
<section class='wide'><h2>Brain Audit Tail</h2><table><thead><tr><th>Status</th><th>Reason</th><th>Path</th></tr></thead><tbody>{audit_rows}</tbody></table></section>
<section class='wide'><h2>Kernel/OOM/GPU/I/O Hints</h2><table><thead><tr><th>dmesg match tail</th></tr></thead><tbody>{kernel_rows}</tbody></table></section>
<section class='wide'><h2>Queues Raw</h2><pre>{queue_text}</pre></section>
</main></body></html>"""


def legacy_redirect_html() -> str:
    canonical = DASHBOARD_HTML.name
    raw = RAW_BIG_BOARD_HTML.name
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={html.escape(canonical)}">
<title>LUCIDOTA dashboard moved</title></head>
<body style="font-family:system-ui;background:#060711;color:#eff7ff;padding:2rem">
<h1>LUCIDOTA dashboard moved</h1>
<p>The canonical dashboard is now <a href="{html.escape(canonical)}">{html.escape(canonical)}</a>.</p>
<p>The raw legacy Big Board feed is <a href="{html.escape(raw)}">{html.escape(raw)}</a>.</p>
</body></html>
"""


def write_outputs(status: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(jdump(status) + "\n", encoding="utf-8")
    LATEST_MD.write_text(render_md(status), encoding="utf-8")
    DASHBOARD_HTML.write_text(render_html(status), encoding="utf-8")
    LEGACY_LIVE_DASHBOARD_HTML.write_text(legacy_redirect_html(), encoding="utf-8")
    if status["health"]["level"] == "done":
        DONE_MARKER.write_text(
            f"# Ingest complete\n\nConfirmed by watchdog at `{status['generated_at_local']}`.\n\nOpen `{DASHBOARD_HTML.relative_to(ROOT)}` for final status.\n",
            encoding="utf-8",
        )


def self_disable_cron() -> dict[str, Any]:
    proc = subprocess.run(["crontab", "-l"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    old = proc.stdout if proc.returncode == 0 else ""
    new_lines = [line for line in old.splitlines() if CRON_MARKER not in line]
    new = "\n".join(new_lines).rstrip() + ("\n" if new_lines else "")
    setp = subprocess.run(["crontab", "-"], input=new, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"removed": old != new, "returncode": setp.returncode, "stderr": setp.stderr[-500:]}


def tick(args: argparse.Namespace) -> dict[str, Any]:
    processes = active_processes()
    hardware = hardware_report(processes)
    actions = safe_maintenance(processes, hardware, dry_run=args.dry_run)
    # Refresh after maintenance so dashboard sees latest process state.
    if actions and not args.dry_run:
        time.sleep(1)
        processes = active_processes()
        hardware = hardware_report(processes)
    io_full = float((((hardware.get("pressure") or {}).get("io") or {}).get("full") or {}).get("avg10") or 0.0)
    light_mode = io_full >= IO_FULL_CRIT_PCT
    report = ingest_report(processes, light=light_mode)
    report["processes"] = processes
    lq = locks_and_queues(light=light_mode)
    audit = audit_summary()
    done = is_done(processes, report)
    done_count = update_done_count(done)
    status = {
        "ok": True,
        "generated_at_utc": utcnow(),
        "generated_at_local": localnow(),
        "dry_run": args.dry_run,
        "report": report,
        "processes": processes,
        "locks_queues": lq,
        "hardware": hardware,
        "audit": audit,
        "bars": pct_bars(report),
        "actions": actions,
        "done": done,
        "done_count": done_count,
        "light_mode": light_mode,
    }
    status["health"] = health(status, lq, actions, done)
    write_outputs(status)
    if done and args.self_disable and done_count >= args.done_confirmations:
        disabled = self_disable_cron()
        status["self_disabled_cron"] = disabled
        STATUS_JSON.write_text(jdump(status) + "\n", encoding="utf-8")
        log_line(f"DONE: self-disabled cron={disabled}")
        emit_event("done_self_disabled", "succeeded", {"done_count": done_count, "cron": disabled})
    log_line(f"tick health={status['health']['level']} done={done} actions={len(actions)} brain={report['brain_marker'].get('index')} audit={audit.get('total')}")
    return status


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-ingest-watchdog")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-disable", action="store_true")
    ap.add_argument("--done-confirmations", type=int, default=2)
    args = ap.parse_args()
    status = tick(args)
    print(jdump(status) if args.json else render_md(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
