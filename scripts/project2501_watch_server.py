#!/usr/bin/env python3
"""Project 2501 realtime watch surface: batteries, task stream, subway map."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ALGOS.workshare_allocator import allocate_workshare, summarize_savings  # noqa: E402

OUT = ROOT / "05_OUTPUTS" / "project2501_watch"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return default


def tail_lines(path: Path, limit: int = 12) -> list[str]:
    text = read_text(path)
    return [line for line in text.splitlines() if line.strip()][-limit:]


def latest_file(root: Path, pattern: str) -> Path | None:
    try:
        matches = [p for p in root.glob(pattern) if p.is_file()]
    except Exception:
        return None
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def latest_json_status(path: Path | None) -> str:
    if not path:
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("status") or data.get("verdict") or data.get("passed") or "present")
    except Exception:
        return "unreadable"


def read_json(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def git_counts(root: Path) -> dict[str, int]:
    try:
        proc = subprocess.run(["git", "status", "--short"], cwd=root, text=True, capture_output=True, timeout=1.5)
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        return {"changed": -1, "modified": -1, "deleted": -1, "untracked": -1}
    return {
        "changed": len(lines),
        "modified": sum(1 for line in lines if line[:2].strip() == "M" or " M" in line[:2]),
        "deleted": sum(1 for line in lines if "D" in line[:2]),
        "untracked": sum(1 for line in lines if line.startswith("??")),
    }


def proc_meminfo() -> dict[str, Any]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                values[parts[0].rstrip(":")] = int(parts[1])
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}:{exc}"}
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    used = max(0, total - available)
    pct = round((used / total) * 100, 1) if total else 0.0
    return {
        "status": "ok" if total else "unavailable",
        "total_mb": round(total / 1024),
        "available_mb": round(available / 1024),
        "used_mb": round(used / 1024),
        "used_pct": pct,
        "swap_total_mb": round(values.get("SwapTotal", 0) / 1024),
        "swap_free_mb": round(values.get("SwapFree", 0) / 1024),
    }


def cpu_snapshot() -> dict[str, Any]:
    count = os.cpu_count() or 1
    try:
        load1, load5, load15 = os.getloadavg()
        pressure = round((load1 / max(count, 1)) * 100, 1)
        return {
            "status": "ok",
            "logical_cpus": count,
            "load1": round(load1, 2),
            "load5": round(load5, 2),
            "load15": round(load15, 2),
            "load_pressure_pct": pressure,
        }
    except Exception as exc:
        return {"status": "unavailable", "logical_cpus": count, "error": f"{type(exc).__name__}:{exc}"}


def thermal_snapshot(limit: int = 24) -> dict[str, Any]:
    sensors: list[dict[str, Any]] = []
    seen: set[Path] = set()
    paths = list(Path("/sys/class/thermal").glob("thermal_zone*/temp")) + list(Path("/sys/class/hwmon").glob("hwmon*/temp*_input"))
    for path in paths:
        if path in seen or len(sensors) >= limit:
            continue
        seen.add(path)
        try:
            raw = path.read_text(encoding="utf-8", errors="replace").strip()
            milli = float(raw)
            celsius = milli / 1000.0 if milli > 200 else milli
            label = path.parent.name
            for name_file in (path.parent / "type", path.parent / "name"):
                if name_file.exists():
                    label = name_file.read_text(encoding="utf-8", errors="replace").strip() or label
                    break
            sensors.append({"label": label, "path": str(path), "celsius": round(celsius, 1)})
        except Exception:
            continue
    max_c = max((row["celsius"] for row in sensors), default=None)
    return {"status": "ok" if sensors else "not_detected", "max_celsius": max_c, "sensors": sensors}


def gpu_snapshot() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"status": "not_detected", "tool": "nvidia-smi"}
    cmd = [
        exe,
        "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=1.5)
    except Exception as exc:
        return {"status": "error", "tool": "nvidia-smi", "error": f"{type(exc).__name__}:{exc}"}
    if proc.returncode != 0:
        return {"status": "error", "tool": "nvidia-smi", "stderr": proc.stderr[-500:]}
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        name, temp, util, mem_used, mem_total = parts[:5]
        power = parts[5] if len(parts) > 5 else ""
        def num(value: str) -> float | None:
            try:
                return float(value)
            except Exception:
                return None
        used = num(mem_used)
        total = num(mem_total)
        gpus.append(
            {
                "name": name,
                "temperature_celsius": num(temp),
                "utilization_pct": num(util),
                "memory_used_mb": used,
                "memory_total_mb": total,
                "memory_used_pct": round((used / total) * 100, 1) if used is not None and total else None,
                "power_watts": num(power),
            }
        )
    return {"status": "ok" if gpus else "not_detected", "tool": "nvidia-smi", "gpus": gpus}


def hardware_telemetry() -> dict[str, Any]:
    return {
        "schema": "lucidota.project2501.hardware_telemetry.v1",
        "generated_at": now(),
        "cpu": cpu_snapshot(),
        "memory": proc_meminfo(),
        "thermal": thermal_snapshot(),
        "gpu": gpu_snapshot(),
    }


def postgres_dsn() -> str:
    for key in ("LUCIDOTA_CONTROL_DATABASE_URL", "ABSURD_SYSTEM_DATABASE_URL", "DATABASE_URL", "LUCIDOTA_GO_STORAGE_DSN", "PG_DSN"):
        if os.environ.get(key):
            return str(os.environ[key])
    return "postgresql:///lucidota_state"


def postgres_recent_model_generation_events(conn: Any, limit: int = 12) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "lucidota.project2501.pg_model_generation_events.v1",
        "generated_at": now(),
        "available": False,
        "recent": [],
    }
    exists = bool(conn.execute("SELECT to_regclass('lucidota_control.model_generation_event') IS NOT NULL").fetchone()[0])
    out["available"] = exists
    if not exists:
        return out
    rows = conn.execute(
        """
        SELECT
          target,
          model_name,
          payload_size_bytes,
          payload_size_chars,
          latency_ms::float,
          raw_output,
          raw_output_chars,
          raw_response_present,
          execute_performed,
          receipt_path,
          receipt_sha256,
          queue_event_uuid::text,
          staged_at
        FROM lucidota_control.model_generation_event
        ORDER BY staged_at DESC
        LIMIT %s
        """,
        (max(1, min(50, int(limit))),),
    ).fetchall()
    recent: list[dict[str, Any]] = []
    for row in rows:
        recent.append(
            {
                "target": row[0],
                "model_name": row[1],
                "payload_size_bytes": int(row[2] or 0),
                "payload_size_chars": int(row[3] or 0),
                "latency_ms": row[4],
                "raw_output": row[5] or "",
                "raw_output_chars": int(row[6] or 0),
                "raw_response_present": bool(row[7]),
                "execute_performed": bool(row[8]),
                "receipt_path": row[9],
                "receipt_sha256": row[10],
                "queue_event_uuid": row[11],
                "staged_at": row[12].isoformat() if hasattr(row[12], "isoformat") else str(row[12]),
            }
        )
    out["recent"] = recent
    return out


def postgres_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "lucidota.project2501.postgres_snapshot.v1",
        "generated_at": now(),
        "status": "unknown",
        "dsn_source": "env_or_local_socket",
        "canonical_graph_writes_performed": False,
        "tables": [],
        "model_generation_events": {
            "schema": "lucidota.project2501.pg_model_generation_events.v1",
            "generated_at": now(),
            "available": False,
            "recent": [],
        },
    }
    if os.environ.get("LUCIDOTA_DISABLE_PG_AUTO_PROBE"):
        out["status"] = "disabled"
        out["dsn_source"] = "LUCIDOTA_DISABLE_PG_AUTO_PROBE"
        return out
    try:
        import psycopg  # type: ignore
    except Exception as exc:
        out["status"] = "missing_psycopg"
        out["error"] = f"{type(exc).__name__}:{exc}"
        return out
    wanted = [
        "lucidota_control.absurd_queue_job",
        "lucidota_control.absurd_queue_event",
        "lucidota_control.absurd_queue_dead_letter",
        "lucidota_control.conversation_command",
        "lucidota_control.model_generation_event",
        "lucidota_control.system_telemetry_rollup",
        "lucidota_runtime.model_candidate",
        "lucidota_runtime.resident_loadout",
        "lucidota_go.graph_item",
        "lucidota_go.graph_edge",
        "lucidota_go.graph_promotion_packet",
    ]
    started = time.monotonic()
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=1) as conn:
            conn.execute("SET statement_timeout = '900ms'")
            conn.execute("SET default_transaction_read_only = on")
            current = conn.execute("SELECT current_database(), current_user, now()").fetchone()
            out["database"] = current[0]
            out["user"] = current[1]
            out["server_now"] = current[2].isoformat() if hasattr(current[2], "isoformat") else str(current[2])
            for name in wanted:
                exists = bool(conn.execute("SELECT to_regclass(%s) IS NOT NULL", (name,)).fetchone()[0])
                row: dict[str, Any] = {"name": name, "exists": exists}
                if exists:
                    # Fixed allowlist only; safe identifier is obtained from to_regclass text.
                    count = conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                    row["count"] = int(count)
                out["tables"].append(row)
            out["model_generation_events"] = postgres_recent_model_generation_events(conn, 12)
            out["status"] = "ok"
    except Exception as exc:
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}:{str(exc)[:500]}"
    out["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
    return out


_PG_CACHE: dict[str, Any] = {"snapshot": None, "expires_at": 0.0, "running": False}
_PG_LOCK = threading.Lock()


def _refresh_pg_cache(ttl: float) -> None:
    try:
        snapshot = postgres_snapshot()
    finally:
        with _PG_LOCK:
            if "snapshot" not in locals():
                snapshot = {
                    "schema": "lucidota.project2501.postgres_snapshot.v1",
                    "generated_at": now(),
                    "status": "error",
                    "error": "refresh_failed_without_snapshot",
                    "canonical_graph_writes_performed": False,
                    "tables": [],
                }
            _PG_CACHE["snapshot"] = snapshot
            _PG_CACHE["expires_at"] = time.monotonic() + ttl
            _PG_CACHE["running"] = False


def postgres_snapshot_async(ttl: float = 2.0) -> dict[str, Any]:
    if os.environ.get("LUCIDOTA_DISABLE_PG_AUTO_PROBE"):
        return postgres_snapshot()
    ttl = max(0.5, float(ttl))
    with _PG_LOCK:
        cached = _PG_CACHE.get("snapshot")
        stale = time.monotonic() >= float(_PG_CACHE.get("expires_at") or 0)
        if stale and not _PG_CACHE.get("running"):
            _PG_CACHE["running"] = True
            threading.Thread(target=_refresh_pg_cache, args=(ttl,), daemon=True).start()
        if isinstance(cached, dict):
            out = dict(cached)
            out["stale"] = stale
            out["async_refresh_running"] = bool(_PG_CACHE.get("running"))
            return out
    return {
        "schema": "lucidota.project2501.postgres_snapshot.v1",
        "generated_at": now(),
        "status": "refreshing",
        "stale": True,
        "async_refresh_running": True,
        "canonical_graph_writes_performed": False,
        "tables": [],
    }


def parse_handoff(root: Path) -> dict[str, Any]:
    text = read_text(root / "GOALS" / "CURRENT_HANDOFF.md")
    out: dict[str, Any] = {"path": "GOALS/CURRENT_HANDOFF.md", "raw_chars": len(text)}
    for line in text.splitlines():
        if line.startswith("- Goal:"):
            out["goal"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Current step:"):
            out["current_step"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Status:"):
            out["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Next action:"):
            out["next_action"] = line.split(":", 1)[1].strip()
    return out


def percent_from_step(step: str | None) -> int:
    if not step or "/" not in step:
        return 0
    try:
        a, b = step.split("/", 1)
        return max(0, min(100, round(float(a.strip()) / float(b.strip()) * 100)))
    except Exception:
        return 0


def latest_model_receipt(root: Path, provider: str) -> tuple[Path | None, dict[str, Any] | None]:
    patterns = {
        "groq": "groq_chat_*.json",
        "cohere": "cohere_chat_*.json",
        "local_models": "local_model_chat_*.json",
    }
    path = latest_file(root / "05_OUTPUTS" / "model_invocations", patterns.get(provider, ""))
    return path, read_json(path)


def request_knobs(data: dict[str, Any] | None) -> dict[str, Any]:
    request = (data or {}).get("request") if isinstance(data, dict) else {}
    if not isinstance(request, dict):
        request = {}
    messages = request.get("messages")
    prompt_chars = 0
    system_chars = 0
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            chars = int(msg.get("content_chars") or len(str(msg.get("content_text", ""))) or len(str(msg.get("content", ""))))
            if msg.get("role") == "system":
                system_chars += chars
            elif msg.get("role") == "user":
                prompt_chars += chars
    prompt_chars = int(request.get("prompt_chars") or prompt_chars)
    system_chars = int(request.get("system_chars") or system_chars)
    return {
        "temperature": request.get("temperature"),
        "max_tokens": request.get("max_tokens") or request.get("max_gen_len"),
        "prompt_chars": prompt_chars,
        "system_chars": system_chars,
        "message_count": len(messages) if isinstance(messages, list) else None,
    }


def model_observability_pct(data: dict[str, Any] | None) -> int:
    if not data:
        return 0
    if data.get("status") == "BLOCKED" or data.get("blockers"):
        return 10
    if data.get("execute_performed") or data.get("model_calls_performed") or data.get("real_inference_performed"):
        return 100
    if data.get("mode") == "dry_run":
        return 55
    return 65 if data.get("status") == "PASS" else 20


def model_trace_detail(path: Path | None, data: dict[str, Any] | None) -> str:
    if not path or not data:
        return "no invocation receipt in 05_OUTPUTS/model_invocations"
    knobs = request_knobs(data)
    text = str(data.get("text") or "")
    usage = data.get("usage") or data.get("token_accounting")
    bits = [
        f"{data.get('mode', '?')}/{data.get('status', '?')}",
        f"model={data.get('model', '?')}",
        f"temp={knobs.get('temperature')}",
        f"max={knobs.get('max_tokens')}",
        f"in={knobs.get('system_chars', 0)}s/{knobs.get('prompt_chars', 0)}u chars",
        f"out={len(text)} chars",
    ]
    if usage:
        bits.append("usage=present")
    bits.append(f"receipt={rel(path)}")
    return " | ".join(bits)


def model_traces(root: Path) -> dict[str, Any]:
    providers: dict[str, dict[str, Any]] = {}
    for provider in ("groq", "cohere", "local_models"):
        path, data = latest_model_receipt(root, provider)
        knobs = request_knobs(data)
        providers[provider] = {
            "provider": (data or {}).get("provider", provider),
            "status": (data or {}).get("status", "missing_receipt"),
            "mode": (data or {}).get("mode"),
            "model": (data or {}).get("model"),
            "backend": (data or {}).get("backend"),
            "lane": (data or {}).get("lane"),
            "endpoint": (data or {}).get("endpoint"),
            "request": knobs,
            "response": {
                "text_chars": len(str((data or {}).get("text") or "")),
                "finish_reason": (data or {}).get("finish_reason"),
                "usage": (data or {}).get("usage") or (data or {}).get("token_accounting"),
                "raw_response_present": bool((data or {}).get("raw_response")),
            },
            "execute_performed": bool((data or {}).get("execute_performed") or (data or {}).get("model_calls_performed") or (data or {}).get("real_inference_performed")),
            "observability_pct": model_observability_pct(data),
            "receipt_path": rel(path) if path else None,
            "detail": model_trace_detail(path, data),
        }
    providers["codex"] = {
        "provider": "codex",
        "status": "main_window_metadata_not_exposed_to_repo",
        "mode": "interactive_session",
        "model": os.environ.get("CODEX_MODEL") or os.environ.get("OPENAI_MODEL") or "unexposed",
        "backend": "codex",
        "lane": "orchestrator",
        "endpoint": "chatgpt/codex session",
        "request": {"temperature": None, "max_tokens": None, "prompt_chars": None, "system_chars": None, "message_count": None},
        "response": {"text_chars": None, "finish_reason": None, "usage": None, "raw_response_present": False},
        "execute_performed": True,
        "observability_pct": 15,
        "receipt_path": None,
        "detail": "Codex main-window model/temperature/token metadata is not exposed to this repo process; showing gap instead of fake 25%.",
    }
    return {"schema": "lucidota.project2501.model_trace.v1", "generated_at": now(), "providers": providers}


def recent_model_generations(root: Path, limit: int = 12) -> dict[str, Any]:
    inv_dir = root / "05_OUTPUTS" / "model_invocations"
    try:
        paths = sorted(inv_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    except Exception:
        paths = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = read_json(path)
        if not data:
            continue
        trace = data.get("generation_trace") if isinstance(data.get("generation_trace"), dict) else {}
        target = trace.get("target") or data.get("provider") or data.get("target")
        model_name = trace.get("model_name") or data.get("model")
        if not target or not model_name:
            continue
        rows.append(
            {
                "target": target,
                "model_name": model_name,
                "mode": data.get("mode"),
                "status": data.get("status"),
                "payload_size_bytes": int(trace.get("payload_size_bytes") or 0),
                "payload_size_chars": int(trace.get("payload_size_chars") or 0),
                "latency_ms": trace.get("latency_ms") if trace.get("latency_ms") is not None else data.get("latency_ms"),
                "raw_output": str(trace.get("raw_output") if trace.get("raw_output") is not None else data.get("text") or ""),
                "raw_output_chars": int(trace.get("raw_output_chars") or len(str(data.get("text") or ""))),
                "raw_response_present": bool(trace.get("raw_response_present") or data.get("raw_response")),
                "execute_performed": bool(trace.get("execute_performed") or data.get("execute_performed") or data.get("real_inference_performed")),
                "receipt_path": rel(path),
            }
        )
    return {"schema": "lucidota.project2501.model_generation_ledger.v1", "generated_at": now(), "recent": rows}


def llxprt_orchestrator(root: Path) -> dict[str, Any]:
    path = latest_file(root / "05_OUTPUTS" / "llxprt_project2501", "llxprt_project2501_*.json")
    data = read_json(path) or {}
    profile = data.get("profile") if isinstance(data.get("profile"), dict) else {}
    provider_alias = data.get("provider_alias") if isinstance(data.get("provider_alias"), dict) else {}
    groq = data.get("groq") if isinstance(data.get("groq"), dict) else {}
    llxprt = data.get("llxprt") if isinstance(data.get("llxprt"), dict) else {}
    version = llxprt.get("version") if isinstance(llxprt.get("version"), dict) else {}
    return {
        "schema": "lucidota.project2501.llxprt_orchestrator.v1",
        "generated_at": now(),
        "status": data.get("status", "missing_receipt"),
        "profile": profile.get("name") or data.get("profile"),
        "provider_alias": provider_alias.get("name") or data.get("provider_alias"),
        "model": groq.get("model"),
        "base_url": groq.get("base_url"),
        "api_key_env_present": bool(groq.get("api_key_env_present")),
        "binary": llxprt.get("binary"),
        "version": version.get("stdout") if isinstance(version, dict) else None,
        "receipt_path": rel(path) if path else None,
    }


def route_map() -> dict[str, Any]:
    nodes = [
        {"id": "operator", "label": "operator", "x": 8, "y": 52, "line": "red"},
        {"id": "project2501", "label": "2501 admin", "x": 24, "y": 28, "line": "red"},
        {"id": "deterministic", "label": "90% algos", "x": 44, "y": 28, "line": "yellow"},
        {"id": "absurd", "label": "ABSURD queue", "x": 62, "y": 28, "line": "yellow"},
        {"id": "graph_gate", "label": "graph gate", "x": 80, "y": 28, "line": "green"},
        {"id": "codex", "label": "codex", "x": 44, "y": 68, "line": "blue"},
        {"id": "groq", "label": "groq", "x": 56, "y": 78, "line": "blue"},
        {"id": "cohere", "label": "cohere", "x": 68, "y": 78, "line": "blue"},
        {"id": "local", "label": "local", "x": 80, "y": 68, "line": "blue"},
        {"id": "receipts", "label": "receipts", "x": 92, "y": 52, "line": "green"},
    ]
    edges = [
        {"from": "operator", "to": "project2501", "line": "red", "label": "directive"},
        {"from": "project2501", "to": "deterministic", "line": "yellow", "label": "compile"},
        {"from": "deterministic", "to": "absurd", "line": "yellow", "label": "route"},
        {"from": "absurd", "to": "graph_gate", "line": "green", "label": "stage"},
        {"from": "graph_gate", "to": "receipts", "line": "green", "label": "prove"},
        {"from": "deterministic", "to": "codex", "line": "blue", "label": "residual"},
        {"from": "codex", "to": "groq", "line": "blue", "label": "25%"},
        {"from": "groq", "to": "cohere", "line": "blue", "label": "25%"},
        {"from": "cohere", "to": "local", "line": "blue", "label": "25%"},
        {"from": "local", "to": "receipts", "line": "blue", "label": "25%"},
    ]
    return {"schema": "lucidota.project2501.route_map.v1", "nodes": nodes, "edges": edges}


def batteries(root: Path, handoff: dict[str, Any], traces: dict[str, Any], hardware: dict[str, Any], postgres: dict[str, Any], llxprt: dict[str, Any]) -> list[dict[str, Any]]:
    latest_admin = latest_file(root / "05_OUTPUTS" / "model_admin_prompt", "project2501_admin_prompt_*.json")
    latest_graph = latest_file(root / "05_OUTPUTS" / "graph", "system_graph_safety_audit_*.json")
    git = git_counts(root)
    rows = [
        {"id": "deterministic_algos", "label": "deterministic algos / ML", "pct": 90, "detail": "target token savings", "color": "yellow"},
        {"id": "current_goal", "label": "goal handoff", "pct": percent_from_step(handoff.get("current_step")), "detail": handoff.get("current_step", "0/0"), "color": "cyan"},
        {"id": "project2501_admin", "label": "2501 admin prompt", "pct": 100 if latest_admin else 40, "detail": rel(latest_admin) if latest_admin else "needs receipt", "color": "red"},
        {
            "id": "llxprt_groq",
            "label": "LLXPRT Groq orchestrator",
            "pct": 100 if llxprt.get("status") == "PASS" and llxprt.get("api_key_env_present") else (70 if llxprt.get("status") == "PASS" else 0),
            "detail": f"profile={llxprt.get('profile')} provider={llxprt.get('provider_alias')} model={llxprt.get('model')} key={llxprt.get('api_key_env_present')} receipt={llxprt.get('receipt_path')}",
            "color": "blue",
        },
        {"id": "graph_safety", "label": "graph safety", "pct": 100 if latest_json_status(latest_graph) == "PASS" else 58, "detail": latest_json_status(latest_graph), "color": "green"},
        {
            "id": "postgres",
            "label": "PostgreSQL targeted async",
            "pct": {"ok": 100, "refreshing": 40, "disabled": 10, "missing_psycopg": 20}.get(str(postgres.get("status")), 15),
            "detail": f"status={postgres.get('status')} db={postgres.get('database')} tables={sum(1 for t in (postgres.get('tables') or []) if t.get('exists'))}/{len(postgres.get('tables') or [])} latency={postgres.get('latency_ms')}ms async={postgres.get('async_refresh_running')}",
            "color": "green",
        },
        {"id": "worktree_heat", "label": "worktree heat", "pct": max(0, min(100, 100 - max(git.get("changed", 0), 0) // 3)), "detail": f"changed={git.get('changed')}", "color": "purple"},
    ]
    for provider in ("codex", "groq", "cohere", "local_models"):
        trace = (traces.get("providers") or {}).get(provider, {})
        rows.append(
            {
                "id": provider,
                "label": provider.replace("_", " "),
                "pct": int(trace.get("observability_pct") or 0),
                "detail": str(trace.get("detail") or "no trace"),
                "color": "blue",
            }
        )
    cpu = hardware.get("cpu", {})
    memory = hardware.get("memory", {})
    thermal = hardware.get("thermal", {})
    gpu = hardware.get("gpu", {})
    rows.extend(
        [
            {
                "id": "cpu_load",
                "label": "CPU load",
                "pct": max(0, min(100, int(float(cpu.get("load_pressure_pct") or 0)))),
                "detail": f"load={cpu.get('load1')} / cpus={cpu.get('logical_cpus')} pressure={cpu.get('load_pressure_pct')}%",
                "color": "cyan",
            },
            {
                "id": "ram",
                "label": "RAM",
                "pct": max(0, min(100, int(float(memory.get("used_pct") or 0)))),
                "detail": f"used={memory.get('used_mb')}MB total={memory.get('total_mb')}MB avail={memory.get('available_mb')}MB",
                "color": "purple",
            },
            {
                "id": "thermal",
                "label": "temperature",
                "pct": max(0, min(100, int(float(thermal.get("max_celsius") or 0)))),
                "detail": f"status={thermal.get('status')} max={thermal.get('max_celsius')}C sensors={len(thermal.get('sensors') or [])}",
                "color": "red",
            },
            {
                "id": "gpu",
                "label": "GPU",
                "pct": max(0, min(100, int(max([float(g.get("utilization_pct") or g.get("memory_used_pct") or 0) for g in (gpu.get("gpus") or [])] or [0])))),
                "detail": " | ".join(
                    f"{g.get('name')} util={g.get('utilization_pct')}% mem={g.get('memory_used_mb')}/{g.get('memory_total_mb')}MB temp={g.get('temperature_celsius')}C"
                    for g in (gpu.get("gpus") or [])
                )
                or f"status={gpu.get('status')} tool={gpu.get('tool')}",
                "color": "green",
            },
        ]
    )
    return rows


def log_stream(root: Path) -> list[dict[str, str]]:
    logs: list[dict[str, str]] = []
    for line in tail_lines(root / "GOALS" / "CURRENT_HANDOFF.md", 8):
        logs.append({"source": "handoff", "text": line[:260]})
    for line in tail_lines(root / "GOALS" / "GOAL_LOG.md", 10):
        if line.startswith("## ") or line.startswith("- Completed:") or line.startswith("- Next action:") or line.startswith("Technical Summary"):
            logs.append({"source": "goal_log", "text": line[:260]})
    latest = sorted((root / "05_OUTPUTS").glob("*/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    for path in latest:
        logs.append({"source": "receipt", "text": rel(path)})
    if not logs:
        logs.append({"source": "boot", "text": "No logs yet; state collector alive."})
    return logs[-24:]


def collect_state(root: Path = ROOT) -> dict[str, Any]:
    handoff = parse_handoff(root)
    savings = summarize_savings(total_units=100.0, deterministic_target_pct=90.0)
    traces = model_traces(root)
    generations = recent_model_generations(root)
    hardware = hardware_telemetry()
    postgres = postgres_snapshot_async(float(os.environ.get("LUCIDOTA_PG_SNAPSHOT_TTL_SEC", "2.0")))
    pg_generations = postgres.get("model_generation_events") if isinstance(postgres.get("model_generation_events"), dict) else {"schema": "lucidota.project2501.pg_model_generation_events.v1", "generated_at": now(), "available": False, "recent": []}
    llxprt = llxprt_orchestrator(root)
    return {
        "schema": "lucidota.project2501.watch_state.v1",
        "generated_at": now(),
        "handoff": handoff,
        "git": git_counts(root),
        "savings": savings,
        "model_traces": traces,
        "model_generations": generations,
        "pg_model_generation_events": pg_generations,
        "llxprt": llxprt,
        "hardware": hardware,
        "postgres": postgres,
        "batteries": batteries(root, handoff, traces, hardware, postgres, llxprt),
        "route_map": route_map(),
        "logs": log_stream(root),
        "canonical_graph_writes_performed": False,
    }


def sse_event(payload: dict[str, Any]) -> str:
    return "event: state\n" + "data: " + json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n\n"


def render_html() -> str:
    return r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Project 2501 Watch</title><style>
:root{--bg:#05070b;--panel:#0c111b;--ink:#e9f8ff;--muted:#87a1b2;--red:#ff3b5f;--yellow:#ffd447;--green:#34e89e;--blue:#45a3ff;--cyan:#36f4ff;--purple:#a876ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#162136,#05070b 55%);color:var(--ink);font:14px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace}.shell{padding:18px;max-width:1500px;margin:0 auto}h1{margin:0 0 10px;font-size:22px;letter-spacing:.08em}.grid{display:grid;grid-template-columns:1fr 1.25fr;gap:14px}.panel{background:rgba(12,17,27,.86);border:1px solid #253345;border-radius:16px;padding:14px;box-shadow:0 0 30px #0008}.battery-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:10px}.battery-card{padding:10px;border:1px solid #26384b;border-radius:14px;background:#08101a}.battery-head{display:flex;justify-content:space-between;gap:8px;color:var(--muted);font-size:12px;margin-bottom:7px}.battery{position:relative;height:26px;border:2px solid #6f8397;border-radius:7px;padding:3px;margin-right:10px}.battery:after{content:"";position:absolute;right:-9px;top:7px;width:6px;height:10px;background:#6f8397;border-radius:0 3px 3px 0}.fill{height:100%;width:0;border-radius:4px;background:linear-gradient(90deg,var(--green),var(--cyan));box-shadow:0 0 12px currentColor;transition:width .25s}.detail{font-size:11px;color:#9ab}.task-stream{height:430px;overflow:auto;background:#03060a;border-radius:12px;border:1px solid #1e2a39;padding:10px}.log{padding:4px 0;border-bottom:1px solid #101a27;color:#d8f5ff}.src{color:var(--yellow);margin-right:8px}.subway-wrap{height:430px}.subway-map{width:100%;height:100%;background:#06101a;border-radius:12px;border:1px solid #1e2a39}.route-line{stroke-width:5;fill:none;stroke-linecap:round}.node{stroke:#f6fbff;stroke-width:2}.node-label{font-size:3.3px;fill:#e9f8ff}.red{stroke:var(--red);background:var(--red)}.yellow{stroke:var(--yellow);background:var(--yellow)}.green{stroke:var(--green);background:var(--green)}.blue{stroke:var(--blue);background:var(--blue)}.cyan{background:var(--cyan)}.purple{background:var(--purple)}.status{display:flex;justify-content:space-between;color:#9ab;margin-bottom:12px}.mono,.telemetry{white-space:pre-wrap;color:#bcd;background:#03060a;border:1px solid #1e2a39;border-radius:12px;padding:10px;overflow:auto}.telemetry{max-height:260px}.model-trace{max-height:320px}</style></head><body><div class="shell"><h1>PROJECT 2501 // LUCIDOTA WATCH</h1><div class="status"><span id="tick">booting</span><span>math in charge // canonical graph writes: blocked</span></div><div class="grid"><section class="panel"><h2>Battery lanes</h2><div id="batteries" class="battery-grid"></div><h2>Agentic task stream</h2><div id="task-stream" class="task-stream"></div></section><section class="panel"><h2>Routing map</h2><div class="subway-wrap"><svg id="subway-map" class="subway-map" viewBox="0 0 100 100"></svg></div><h2>LLXPRT Groq orchestrator</h2><div id="llxprt" class="telemetry"></div><h2>Model trace</h2><div id="model-trace" class="telemetry model-trace"></div><h2>Generation ledger</h2><div id="model-generations" class="telemetry model-trace"></div><h2>PG generation events</h2><div id="pg-model-generations" class="telemetry model-trace"></div><h2>Postgres live</h2><div id="postgres" class="telemetry"></div><h2>Hardware telemetry</h2><div id="hardware" class="telemetry"></div><h2>Workshare math</h2><div id="math" class="mono"></div></section></div></div><script>
const colors={red:'var(--red)',yellow:'var(--yellow)',green:'var(--green)',blue:'var(--blue)',cyan:'var(--cyan)',purple:'var(--purple)'};
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function renderBatteries(rows){document.getElementById('batteries').innerHTML=rows.map(b=>`<div class="battery-card"><div class="battery-head"><b>${esc(b.label)}</b><span>${b.pct}%</span></div><div class="battery"><div class="fill ${b.color||'green'}" style="width:${Math.max(0,Math.min(100,b.pct))}%"></div></div><div class="detail">${esc(b.detail)}</div></div>`).join('');}
function renderLogs(rows){document.getElementById('task-stream').innerHTML=rows.map(l=>`<div class="log"><span class="src">${esc(l.source)}</span>${esc(l.text)}</div>`).join('');}
function renderMap(map){let nodes=Object.fromEntries(map.nodes.map(n=>[n.id,n]));let svg=document.getElementById('subway-map');let edgeSvg=map.edges.map(e=>{let a=nodes[e.from],b=nodes[e.to];return `<path class="route-line ${e.line}" d="M${a.x},${a.y} L${b.x},${b.y}"/>`;}).join('');let nodeSvg=map.nodes.map(n=>`<circle class="node" cx="${n.x}" cy="${n.y}" r="3.2" fill="${colors[n.line]||'#fff'}"/><text class="node-label" x="${n.x+3.8}" y="${n.y+1.2}">${esc(n.label)}</text>`).join('');svg.innerHTML=edgeSvg+nodeSvg;}
function renderMath(s){document.getElementById('math').textContent=JSON.stringify({token_savings_pct:s.savings.token_savings_pct,planned_llm_units:s.savings.planned_llm_units,per_group_llm_units:s.savings.per_group_llm_units,git:s.git,handoff:s.handoff},null,2);}
function renderTelemetry(id,obj){document.getElementById(id).textContent=JSON.stringify(obj,null,2);}
function apply(s){document.getElementById('tick').textContent=s.generated_at;renderBatteries(s.batteries);renderLogs(s.logs);renderMap(s.route_map);renderTelemetry('llxprt',s.llxprt);renderTelemetry('model-trace',s.model_traces);renderTelemetry('model-generations',s.model_generations);renderTelemetry('pg-model-generations',s.pg_model_generation_events);renderTelemetry('postgres',s.postgres);renderTelemetry('hardware',s.hardware);renderMath(s);}
fetch('/api/state').then(r=>r.json()).then(apply);
const es=new EventSource('/events');es.addEventListener('state',ev=>apply(JSON.parse(ev.data)));es.onerror=()=>{document.getElementById('tick').textContent='stream reconnecting';};
</script></body></html>'''


class WatchHandler(BaseHTTPRequestHandler):
    root = ROOT
    interval = 1.0

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def send_body(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            return self.send_body(200, render_html().encode("utf-8"), "text/html; charset=utf-8")
        if self.path == "/api/state":
            body = json.dumps(collect_state(self.root), sort_keys=True).encode("utf-8")
            return self.send_body(200, body, "application/json")
        if self.path == "/events":
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-store")
            self.send_header("connection", "keep-alive")
            self.end_headers()
            while True:
                try:
                    self.wfile.write(sse_event(collect_state(self.root)).encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(self.interval)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    return
        if self.path == "/favicon.ico":
            self.send_response(204); self.end_headers(); return
        return self.send_body(404, b"not found", "text/plain")


def write_receipt(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"project2501_watch_server_{stamp()}.json"
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest = OUT / "project2501_watch_server_latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def open_browser(url: str) -> dict[str, Any]:
    try:
        ok = bool(webbrowser.open(url, new=2))
        return {"attempted": True, "ok": ok, "method": "webbrowser.open"}
    except Exception as exc:
        return {"attempted": True, "ok": False, "method": "webbrowser.open", "error": f"{type(exc).__name__}:{exc}"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Launch Project 2501 realtime watch dashboard.")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--open-browser", action="store_true")
    ap.add_argument("--once", action="store_true", help="Write one state receipt and exit without serving.")
    ap.add_argument("--json", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    url = f"http://{args.host}:{args.port}/"
    browser = {"attempted": False, "ok": False}
    if args.once:
        state = collect_state(ROOT)
        receipt = {"schema": "lucidota.project2501.watch_server_receipt.v1", "generated_at": now(), "url": url, "pid": os.getpid(), "once": True, "browser": browser, "state": state}
        path = write_receipt(receipt)
        if args.json:
            print(json.dumps(receipt, sort_keys=True))
        print("REPORT_PATH=" + rel(path)); print("PROJECT2501_WATCH_SERVER=PASS"); print("URL=" + url)
        return 0
    handler = type("Project2501WatchHandler", (WatchHandler,), {"root": ROOT, "interval": max(0.25, float(args.interval))})
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    actual_port = int(httpd.server_address[1])
    url = f"http://{args.host}:{actual_port}/"
    if args.open_browser:
        threading.Timer(0.25, open_browser, args=(url,)).start()
        browser = {"attempted": True, "ok": None, "method": "webbrowser.open_async"}
    receipt = {"schema": "lucidota.project2501.watch_server_receipt.v1", "generated_at": now(), "url": url, "pid": os.getpid(), "once": False, "browser": browser, "state": collect_state(ROOT)}
    path = write_receipt(receipt)
    print("REPORT_PATH=" + rel(path), flush=True); print("PROJECT2501_WATCH_SERVER=PASS", flush=True); print("URL=" + url, flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
