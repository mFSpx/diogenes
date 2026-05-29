#!/usr/bin/env python3
"""LUCIDOTA Bytewax-compatible abductive hypothesis blender.

Continuous local eventflow over command envelopes, workflow events, ABSURD rows,
and optional ActivityWatch-style telemetry windows.

The implementation uses real Bytewax when the package is installed and otherwise
runs the same deterministic map/reduce transforms through a zero-dependency
fallback. It never sends, publishes, files, or mutates the canonical graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ALGOS.bandit_router import BanditUpdate, update_policy
from ALGOS.rete_bandit_gate import apply_rete_bandit, bandit_update_from_decision
from core.runtime_dsns import resolve_state_dsn, resolve_storage_dsn
from core.telemetry.diogenes import compress_activity, sample_hardware_telemetry, staple_activity

STATE_DSN = resolve_state_dsn()
STORAGE_DSN = resolve_storage_dsn()
OUT_DIR = ROOT / "05_OUTPUTS" / "bytewax_abductive_blender"
RUNTIME_DIR = ROOT / "04_RUNTIME" / "bytewax_abductive_blender"
WORKER_SOURCE = "scripts/bytewax_abductive_blender.py"
DEFAULT_ACTIVITY_WINDOW_SECONDS = 2
DEFAULT_DUCKDB_MEMORY_LIMIT = os.environ.get("LUCIDOTA_DUCKDB_MEMORY_LIMIT", "1536MB")
DEFAULT_REPLICATION_SLOT = os.environ.get("LUCIDOTA_BYTEWAX_REPLICATION_SLOT", "lucidota_bytewax_abductive_slot")
DEFAULT_REPLICATION_PLUGIN = os.environ.get("LUCIDOTA_BYTEWAX_REPLICATION_PLUGIN", "test_decoding")
DEFAULT_REPLICATION_TIMEOUT_SECONDS = float(os.environ.get("LUCIDOTA_BYTEWAX_REPLICATION_TIMEOUT_SECONDS", "2.0"))
DEFAULT_UNIX_SOCKET = os.environ.get("LUCIDOTA_DIIOGENES_SOCKET", str(ROOT / "04_RUNTIME" / "diogenes" / "diogenes.sock"))
DEFAULT_LORA_SWAP_TOKEN_THRESHOLD = 512
MTIME_SNAPSHOT_LORA_SWAP_TOKEN_THRESHOLD = 256
TREELITE_DATE_ROUTER_ARTIFACT = "03_VAULT/router/treelite_router_v0.tl"
NEEDLE_SWARM_THROTTLE_TOK_PER_SEC = int(os.environ.get("LUCIDOTA_NEEDLE_SWARM_THROTTLE_TOK_PER_SEC", "7200"))
BYTEWAX_WORKFLOW_ID = "bytewax_abductive_blender"
MAX_TICK_LIMIT = int(os.environ.get("LUCIDOTA_BYTEWAX_ABDUCTIVE_MAX_TICK_LIMIT", "200"))
MAX_LOOP_CYCLES = int(os.environ.get("LUCIDOTA_BYTEWAX_ABDUCTIVE_MAX_LOOP_CYCLES", "100"))
MAX_IDLE_SLEEP_SECONDS = float(os.environ.get("LUCIDOTA_BYTEWAX_ABDUCTIVE_MAX_IDLE_SLEEP_SECONDS", "60.0"))

GO25 = [
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "FRICTION", "LEVERAGE", "VISIBILITY",
    "ACTION", "EVENT", "TIME", "PATTERN", "HYPOTHESIS", "CLAIM", "EVIDENCE",
    "ATOMIC_ID", "SIGNAL", "GLOW", "TERM", "TOOL", "ALGORITHM", "NAUGHTY",
    "NICE", "GROUP", "OPERATOR", "MODE", "COMMENT",
]
EPISTEMIC_FLAGS = ["FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE"]

INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|messages)\b", re.I),
    re.compile(r"\b(system|developer)\s+(prompt|message|instructions?)\b", re.I),
    re.compile(r"\bexfiltrat(e|ion)|steal\s+(secrets?|keys?|tokens?)\b", re.I),
    re.compile(r"\b(?:curl|wget)\b[^\n]{0,120}\|\s*(?:sh|bash)\b", re.I),
    re.compile(r"\brm\s+-rf\s+/(?:\s|$)", re.I),
    re.compile(r"\bsudo\b[^\n]{0,80}\b(drop_caches|tee\s+/proc|pkill|killall)\b", re.I),
]

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_learning;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_stream_run (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    status text NOT NULL CHECK (status IN ('succeeded','failed')),
    events_in integer NOT NULL DEFAULT 0,
    hints_out integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_hint (
    hint_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    phase text NOT NULL,
    status text NOT NULL,
    hint text NOT NULL,
    score integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_abductive_event (
    event_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    source_ref text NOT NULL,
    event_time timestamptz NOT NULL DEFAULT now(),
    text_surface text NOT NULL DEFAULT '',
    ontology_terms jsonb NOT NULL DEFAULT '[]'::jsonb,
    epistemic_flag text NOT NULL CHECK (epistemic_flag IN ('FACT','PROBABLE','POSSIBLE','BULLSHIT','SURE_MAYBE')),
    injection_flag boolean NOT NULL DEFAULT false,
    compressed_activity jsonb NOT NULL DEFAULT '{}'::jsonb,
    certainty_trace jsonb NOT NULL DEFAULT '{}'::jsonb,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source, source_ref)
);
ALTER TABLE lucidota_learning.bytewax_abductive_event
  ADD COLUMN IF NOT EXISTS certainty_trace jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_abductive_hint (
    hint_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    event_uuid uuid REFERENCES lucidota_learning.bytewax_abductive_event(event_uuid) ON DELETE SET NULL,
    source text NOT NULL,
    source_ref text NOT NULL,
    epistemic_flag text NOT NULL CHECK (epistemic_flag IN ('FACT','PROBABLE','POSSIBLE','BULLSHIT','SURE_MAYBE')),
    hypothesis text NOT NULL,
    support_score double precision NOT NULL DEFAULT 0.0,
    contradiction_score double precision NOT NULL DEFAULT 0.0,
    centrality_score double precision NOT NULL DEFAULT 0.0,
    injection_flag boolean NOT NULL DEFAULT false,
    ontology_terms jsonb NOT NULL DEFAULT '[]'::jsonb,
    certainty_trace jsonb NOT NULL DEFAULT '{}'::jsonb,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source, source_ref, hypothesis)
);
ALTER TABLE lucidota_learning.bytewax_abductive_hint
  ADD COLUMN IF NOT EXISTS certainty_trace jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_replication_receipt (
    receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slot_name text NOT NULL,
    plugin text NOT NULL,
    status text NOT NULL,
    events_seen integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_rete_bandit_decision (
    decision_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid REFERENCES lucidota_learning.bytewax_stream_run(run_id) ON DELETE SET NULL,
    event_uuid uuid REFERENCES lucidota_learning.bytewax_abductive_event(event_uuid) ON DELETE SET NULL,
    source text NOT NULL,
    source_ref text NOT NULL,
    context_id text NOT NULL,
    algorithm_pool jsonb NOT NULL DEFAULT '[]'::jsonb,
    selected_algorithm text NOT NULL,
    selected_engine text NOT NULL DEFAULT 'cpu_fairyfuse_ternary',
    parallel_engine_plan jsonb NOT NULL DEFAULT '{}'::jsonb,
    bandit_strategy text NOT NULL,
    rule_hits jsonb NOT NULL DEFAULT '[]'::jsonb,
    execution_status text NOT NULL,
    runtime_ms double precision NOT NULL DEFAULT 0,
    facts_yielded integer NOT NULL DEFAULT 0,
    reward double precision NOT NULL DEFAULT 0,
    regret_weights jsonb NOT NULL DEFAULT '{}'::jsonb,
    penalty_reason text,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source, source_ref, context_id, selected_algorithm)
);
ALTER TABLE lucidota_learning.bytewax_rete_bandit_decision
  ADD COLUMN IF NOT EXISTS selected_engine text NOT NULL DEFAULT 'cpu_fairyfuse_ternary',
  ADD COLUMN IF NOT EXISTS parallel_engine_plan jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_bandit_policy (
    algorithm_id text PRIMARY KEY,
    pulls bigint NOT NULL DEFAULT 0,
    total_reward double precision NOT NULL DEFAULT 0,
    last_reward double precision NOT NULL DEFAULT 0,
    last_context_id text NOT NULL DEFAULT '',
    updated_at timestamptz NOT NULL DEFAULT now(),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_learning.bytewax_abductive_cursor (
    cursor_name text PRIMARY KEY,
    last_seen_at timestamptz NOT NULL DEFAULT 'epoch'::timestamptz,
    last_seen_ref text NOT NULL DEFAULT '',
    updated_at timestamptz NOT NULL DEFAULT now(),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_evidence_resolution (
    resolution_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_ref text NOT NULL,
    ref_kind text NOT NULL,
    resolved boolean NOT NULL,
    resolver text NOT NULL DEFAULT 'scripts/graph_promotion_packet_reviewer.py',
    source_table text,
    source_uuid uuid,
    source_path text,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
"""

@dataclass(frozen=True)
class BlenderEvent:
    source: str
    source_ref: str
    event_time: str
    text_surface: str
    payload: dict[str, Any]
    compressed_activity: dict[str, Any]


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def emit(kind: str, detail: dict[str, Any]) -> None:
    print(f"===== {kind} :: {now_z()} =====", flush=True)
    print(jdump(detail), flush=True)
    print(f"===== /{kind} =====", flush=True)


def hardware_stress_score(state: dict[str, Any]) -> float:
    cpu = state.get("cpu") or {}
    mem = state.get("memory") or {}
    gpu_blob = state.get("gpu") or {}
    primary = gpu_blob.get("primary") or {}
    thermal = state.get("thermal") or {}
    power = state.get("power") or {}
    score = 0.0

    cpu_load = cpu.get("overall_percent")
    load_1m = cpu.get("load_avg_1m")
    load_5m = cpu.get("load_avg_5m")
    load_15m = cpu.get("load_avg_15m")
    cores = float(cpu.get("cores") or 0.0)
    if cpu_load is not None:
        score += min(1.0, float(cpu_load) / 100.0) * 0.28
    if cores > 0:
        load_pressure = 0.0
        for load in (load_1m, load_5m, load_15m):
            if load is None:
                continue
            load_pressure = max(load_pressure, float(load) / cores)
        score += min(1.0, load_pressure) * 0.24
        if load_pressure >= 1.5:
            score += 0.18
        elif load_pressure >= 1.0:
            score += 0.10
        elif load_pressure >= 0.75:
            score += 0.05
    if cpu_load is not None and float(cpu_load) <= 1.0:
        score += 0.12
    mem_percent = mem.get("percent")
    if mem_percent is not None:
        score += min(1.0, float(mem_percent) / 100.0) * 0.28
    gpu_util = primary.get("utilization_gpu_percent")
    if gpu_util is not None:
        score += min(1.0, float(gpu_util) / 100.0) * 0.12
    gpu_mem_total = primary.get("memory_total_mb") or 0
    gpu_mem_used = primary.get("memory_used_mb") or 0
    if gpu_mem_total:
        vram_frac = float(gpu_mem_used) / float(gpu_mem_total)
        score += min(1.0, vram_frac) * 0.18
        if vram_frac >= 0.70:
            score += 0.20
        elif vram_frac >= 0.55:
            score += 0.10
    gpu_temp = thermal.get("gpu_temp_c")
    if gpu_temp is not None:
        temp = float(gpu_temp)
        score += min(1.0, temp / 84.0) * 0.10
        if temp >= 60.0:
            score += 0.25
        elif temp >= 57.0:
            score += 0.16
        elif temp >= 54.0:
            score += 0.08
    sys_power = power.get("system_power_w")
    if sys_power is not None:
        score += min(1.0, float(sys_power) / 200.0) * 0.03
    avail_mb = mem.get("available_mb")
    if avail_mb is not None:
        avail = float(avail_mb)
        if avail <= 2200:
            score += 0.25
        elif avail <= 2800:
            score += 0.15
        elif avail <= 3400:
            score += 0.06
    pressure_mem = (mem.get("pressure") or {}).get("memory") or {}
    if isinstance(pressure_mem, dict):
        total = float(pressure_mem.get("total") or 0.0)
        if total > 0:
            score += min(0.18, total / 4_000_000_000.0)
    return round(min(1.0, score), 4)


def hardware_controller(hardware_state: dict[str, Any], requested_limit: int) -> dict[str, Any]:
    score = hardware_stress_score(hardware_state)
    cpu = hardware_state.get("cpu") or {}
    mem = hardware_state.get("memory") or {}
    gpu_blob = hardware_state.get("gpu") or {}
    primary = gpu_blob.get("primary") or {}
    gpu_mem_total = float(primary.get("memory_total_mb") or 0)
    gpu_mem_used = float(primary.get("memory_used_mb") or 0)
    vram_frac = (gpu_mem_used / gpu_mem_total) if gpu_mem_total else 0.0
    gpu_temp = float((hardware_state.get("thermal") or {}).get("gpu_temp_c") or 0.0)
    host_available = float(mem.get("available_mb") or 0.0)
    host_percent = float(mem.get("percent") or 0.0)

    if score >= 0.68 or vram_frac >= 0.68 or gpu_temp >= 60.0 or host_available <= 2200 or host_percent >= 82.0:
        limit = min(requested_limit, 2)
        include_activitywatch = False
        prefer_replication = False
        socket_timeout = 0.15
    elif score >= 0.52 or vram_frac >= 0.60 or gpu_temp >= 57.0 or host_available <= 2800 or host_percent >= 76.0:
        limit = min(requested_limit, 4)
        include_activitywatch = False
        prefer_replication = True
        socket_timeout = 0.25
    elif score >= 0.32 or vram_frac >= 0.50 or gpu_temp >= 54.0 or host_available <= 3400 or host_percent >= 68.0:
        limit = min(requested_limit, 8)
        include_activitywatch = False
        prefer_replication = True
        socket_timeout = 0.40
    elif score >= 0.22 or host_percent >= 55.0:
        limit = min(requested_limit, 16)
        include_activitywatch = True
        prefer_replication = True
        socket_timeout = 0.55
    else:
        limit = min(requested_limit, 24)
        include_activitywatch = True
        prefer_replication = True
        socket_timeout = 0.8
    return {
        "schema": "lucidota.bytewax.hardware_controller.v1",
        "hardware_stress_score": score,
        "tick_limit": limit,
        "include_activitywatch": include_activitywatch,
        "prefer_replication": prefer_replication,
        "unix_socket_timeout_seconds": socket_timeout,
        "request_limit": requested_limit,
        "needle_swarm_throttle_tok_per_sec": NEEDLE_SWARM_THROTTLE_TOK_PER_SEC,
        "vram_fraction": round(vram_frac, 4),
        "gpu_temperature_c": gpu_temp,
        "host_available_mb": round(host_available, 3),
    }


def ensure_schema() -> None:
    with psycopg.connect(STATE_DSN) as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()
    with psycopg.connect(STORAGE_DSN) as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()


def table_exists(conn: psycopg.Connection[Any], schema: str, table: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) IS NOT NULL AS exists", (f"{schema}.{table}",)).fetchone()
    try:
        return bool(row["exists"])
    except Exception:
        return bool(row[0])


def load_cursor(conn: psycopg.Connection[Any], name: str) -> tuple[str, str]:
    conn.execute("INSERT INTO lucidota_learning.bytewax_abductive_cursor(cursor_name) VALUES (%s) ON CONFLICT DO NOTHING", (name,))
    row = conn.execute("SELECT last_seen_at::text AS last_seen_at_text, last_seen_ref FROM lucidota_learning.bytewax_abductive_cursor WHERE cursor_name=%s", (name,)).fetchone()
    try:
        return str(row["last_seen_at_text"]), str(row["last_seen_ref"])
    except Exception:
        return str(row[0]), str(row[1])


def update_cursor(conn: psycopg.Connection[Any], name: str, last_seen_at: str, last_seen_ref: str, detail: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO lucidota_learning.bytewax_abductive_cursor(cursor_name,last_seen_at,last_seen_ref,detail,updated_at)
        VALUES (%s,%s,%s,%s::jsonb,now())
        ON CONFLICT(cursor_name) DO UPDATE SET last_seen_at=EXCLUDED.last_seen_at,
            last_seen_ref=EXCLUDED.last_seen_ref, detail=EXCLUDED.detail, updated_at=now()
        """,
        (name, last_seen_at, last_seen_ref, jdump(detail)),
    )


def record_replication_receipt(slot_name: str, plugin: str, status: str, events_seen: int, detail: dict[str, Any]) -> None:
    try:
        with psycopg.connect(STATE_DSN) as conn:
            conn.execute(
                """
                INSERT INTO lucidota_learning.bytewax_replication_receipt(slot_name, plugin, status, events_seen, detail)
                VALUES (%s,%s,%s,%s,%s::jsonb)
                """,
                (slot_name, plugin, status, events_seen, jdump(detail)),
            )
            conn.commit()
    except Exception:
        return


def ensure_replication_slot(slot_name: str, plugin: str = DEFAULT_REPLICATION_PLUGIN, *, create: bool = False) -> dict[str, Any]:
    if not shutil.which("pg_recvlogical"):
        return {"status": "unavailable", "reason": "pg_recvlogical not found"}
    if not create:
        return {"status": "not_created", "reason": "creation not requested", "slot_name": slot_name, "plugin": plugin}
    cp = subprocess.run(
        [
            "pg_recvlogical",
            "--dbname",
            STATE_DSN,
            "--slot",
            slot_name,
            "--plugin",
            plugin,
            "--create-slot",
            "--if-not-exists",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    status = "ready" if cp.returncode == 0 else "blocked"
    detail = {"returncode": cp.returncode, "stdout": cp.stdout[-2000:], "stderr": cp.stderr[-2000:], "slot_name": slot_name, "plugin": plugin}
    record_replication_receipt(slot_name, plugin, status, 0, detail)
    return {"status": status, **detail}


def fetch_pg_recvlogical_events(
    limit: int,
    *,
    slot_name: str = DEFAULT_REPLICATION_SLOT,
    plugin: str = DEFAULT_REPLICATION_PLUGIN,
    create_slot: bool = False,
    timeout_seconds: float = DEFAULT_REPLICATION_TIMEOUT_SECONDS,
) -> list[BlenderEvent]:
    """Read a bounded pulse from a Postgres logical replication slot.

    This is opt-in.  If the host is not configured for logical decoding, the
    function records a receipt and returns no events; normal cursor polling then
    continues.
    """
    slot_status = ensure_replication_slot(slot_name, plugin, create=create_slot)
    if slot_status.get("status") in {"unavailable", "blocked"}:
        record_replication_receipt(slot_name, plugin, "fallback_to_cursor_poll", 0, slot_status)
        return []
    if not shutil.which("pg_recvlogical"):
        return []
    cmd = [
        "pg_recvlogical",
        "--dbname",
        STATE_DSN,
        "--slot",
        slot_name,
        "--start",
        "--file",
        "-",
        "--no-loop",
        "--status-interval",
        "1",
    ]
    started = time.time()
    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = proc.communicate(timeout=max(0.5, timeout_seconds))
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    rows = lines[:limit]
    events: list[BlenderEvent] = []
    for line in rows:
        ref = hashlib.sha256(line.encode("utf-8", errors="replace")).hexdigest()
        payload = {
            "logical_replication_line": line,
            "slot_name": slot_name,
            "plugin": plugin,
            "source_transport": "pg_recvlogical",
            "returncode": proc.returncode,
        }
        events.append(BlenderEvent("pg_recvlogical", ref, now_z(), line, payload, {}))
    detail = {
        "slot_status": slot_status,
        "cmd": cmd[:1] + ["--dbname", "<STATE_DSN>", *cmd[3:]],
        "returncode": proc.returncode,
        "stderr": stderr[-2000:],
        "duration_seconds": round(time.time() - started, 3),
        "lines_seen": len(lines),
        "lines_used": len(events),
    }
    record_replication_receipt(slot_name, plugin, "stream_pulse" if events else "empty_pulse", len(events), detail)
    return events


def fetch_unix_socket_events(
    limit: int,
    *,
    socket_path: str = DEFAULT_UNIX_SOCKET,
    timeout_seconds: float = 1.0,
) -> list[BlenderEvent]:
    """Read newline-delimited JSON telemetry from an optional local service socket."""
    path = Path(socket_path)
    if not path.exists():
        return []
    events: list[BlenderEvent] = []
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(max(0.25, timeout_seconds))
        sock.connect(str(path))
        fh = sock.makefile("r", encoding="utf-8", newline="\n")
        started = time.time()
        while len(events) < limit and (time.time() - started) < max(0.5, timeout_seconds):
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                payload = {"raw_line": line}
            if not isinstance(payload, dict):
                payload = {"raw_line": line, "decoded": payload}
            source = str(payload.get("source") or payload.get("event_source") or "unix_socket_diogenes")
            source_ref = str(payload.get("event_id") or payload.get("uuid") or hashlib.sha256(line.encode("utf-8", errors="replace")).hexdigest())
            event_time = str(payload.get("timestamp") or payload.get("created_at") or payload.get("event_time") or now_z())
            text_surface = str(payload.get("text_surface") or payload.get("text") or payload.get("surface") or line)
            activity = payload.get("compressed_activity") if isinstance(payload.get("compressed_activity"), dict) else compress_activity(payload, payload)
            if "mouse_delta_sum" in payload or "keystroke_burst" in payload:
                activity = staple_activity({"source": source}, payload, payload)["compressed_activity"]
            events.append(
                BlenderEvent(
                    source=source,
                    source_ref=source_ref,
                    event_time=event_time,
                    text_surface=text_surface,
                    payload=payload,
                    compressed_activity=activity,
                )
            )
        fh.close()
        sock.close()
    except Exception:
        return []
    return events


def source_text(payload: dict[str, Any], *keys: str) -> str:
    bits: list[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            bits.append(value)
        elif value is not None:
            bits.append(jdump(value))
    return "\n".join(bits)


def _deterministic_event_uuid(source: str, source_ref: str, event_time: str, text_surface: str) -> uuid.UUID:
    return uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"lucidota.bytewax.workflow_event|{source}|{source_ref}|{event_time}|{text_surface}",
    )


def _workflow_status_for_epistemic(epistemic_flag: str) -> str:
    return "blocked" if epistemic_flag == "BULLSHIT" else "succeeded"


def fetch_conversation_commands(limit: int, live_cursor: bool) -> list[BlenderEvent]:
    with psycopg.connect(STATE_DSN, row_factory=dict_row) as conn:
        if not table_exists(conn, "lucidota_control", "conversation_command"):
            return []
        if live_cursor:
            last_at, last_ref = load_cursor(conn, "conversation_command")
            rows = conn.execute(
                """
                SELECT command_uuid::text, created_at::text, command_kind, plain_language_instruction,
                       command_envelope, source_surface_id, source_artifact_refs, target_refs,
                       evidence_refs, allowed_effect, authority_class, canonical_mutation_allowed,
                       status, detail
                FROM lucidota_control.conversation_command
                WHERE (created_at, command_uuid::text) > (%s::timestamptz, %s)
                ORDER BY created_at ASC, command_uuid ASC
                LIMIT %s
                """,
                (last_at, last_ref, limit),
            ).fetchall()
            if rows:
                update_cursor(conn, "conversation_command", rows[-1]["created_at"], rows[-1]["command_uuid"], {"rows": len(rows)})
        else:
            rows = conn.execute(
                """
                SELECT command_uuid::text, created_at::text, command_kind, plain_language_instruction,
                       command_envelope, source_surface_id, source_artifact_refs, target_refs,
                       evidence_refs, allowed_effect, authority_class, canonical_mutation_allowed,
                       status, detail
                FROM lucidota_control.conversation_command
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        conn.commit()
    out: list[BlenderEvent] = []
    for row in rows:
        d = dict(row)
        text = source_text(d, "command_kind", "plain_language_instruction", "command_envelope", "allowed_effect", "authority_class", "status")
        out.append(BlenderEvent("conversation_command", d["command_uuid"], d["created_at"], text, d, {}))
    return out


def fetch_workflow_events(limit: int, live_cursor: bool) -> list[BlenderEvent]:
    with psycopg.connect(STATE_DSN, row_factory=dict_row) as conn:
        if not table_exists(conn, "lucidota_control", "workflow_event"):
            return []
        if live_cursor:
            last_at, last_ref = load_cursor(conn, "workflow_event")
            rows = conn.execute(
                """
                SELECT event_id::text, created_at::text, workflow_id, run_id, phase, status, source, detail
                FROM lucidota_control.workflow_event
                WHERE (created_at, event_id::text) > (%s::timestamptz, %s)
                ORDER BY created_at ASC, event_id ASC
                LIMIT %s
                """,
                (last_at, last_ref, limit),
            ).fetchall()
            if rows:
                update_cursor(conn, "workflow_event", rows[-1]["created_at"], rows[-1]["event_id"], {"rows": len(rows)})
        else:
            rows = conn.execute(
                """
                SELECT event_id::text, created_at::text, workflow_id, run_id, phase, status, source, detail
                FROM lucidota_control.workflow_event
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        conn.commit()
    out: list[BlenderEvent] = []
    for row in rows:
        d = dict(row)
        text = source_text(d, "workflow_id", "run_id", "phase", "status", "source", "detail")
        out.append(BlenderEvent("workflow_event", d["event_id"], d["created_at"], text, d, {}))
    return out


def fetch_absurd_events(limit: int, live_cursor: bool) -> list[BlenderEvent]:
    with psycopg.connect(STATE_DSN, row_factory=dict_row) as conn:
        if not table_exists(conn, "lucidota_control", "absurd_workflow"):
            return []
        if live_cursor:
            last_at, last_ref = load_cursor(conn, "absurd_workflow")
            rows = conn.execute(
                """
                SELECT workflow_id::text, updated_at::text, workflow_name, status, state, payload, error_log
                FROM lucidota_control.absurd_workflow
                WHERE (updated_at, workflow_id::text) > (%s::timestamptz, %s)
                ORDER BY updated_at ASC, workflow_id ASC
                LIMIT %s
                """,
                (last_at, last_ref, limit),
            ).fetchall()
            if rows:
                update_cursor(conn, "absurd_workflow", rows[-1]["updated_at"], rows[-1]["workflow_id"], {"rows": len(rows)})
        else:
            rows = conn.execute(
                """
                SELECT workflow_id::text, updated_at::text, workflow_name, status, state, payload, error_log
                FROM lucidota_control.absurd_workflow
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        conn.commit()
    out: list[BlenderEvent] = []
    for row in rows:
        d = dict(row)
        text = source_text(d, "workflow_name", "status", "state", "payload", "error_log")
        out.append(BlenderEvent("absurd_workflow", d["workflow_id"], d["updated_at"], text, d, {}))
    return out


def activitywatch_db_candidates() -> list[Path]:
    home = Path.home()
    candidates = [
        home / ".local/share/activitywatch/aw-server/peewee-sqlite.v2.db",
        home / ".local/share/activitywatch/aw-server-rust/peewee-sqlite.v2.db",
        ROOT / "04_RUNTIME/activitywatch/events.sqlite",
    ]
    return [p for p in candidates if p.exists()]


def compress_activity_payload(events: list[dict[str, Any]], window_seconds: int = DEFAULT_ACTIVITY_WINDOW_SECONDS) -> list[BlenderEvent]:
    buckets: dict[int, dict[str, Any]] = {}
    for ev in events:
        ts = ev.get("timestamp") or ev.get("time") or ev.get("created_at") or now_z()
        try:
            epoch = int(datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp())
        except Exception:
            epoch = int(time.time())
        bucket = epoch - (epoch % window_seconds)
        b = buckets.setdefault(bucket, {"event_count": 0, "key_count": 0, "click_count": 0, "scroll_count": 0, "pointer_delta_total": 0.0, "sources": Counter(), "flow_friction_score": 0.0})
        data = ev.get("data") if isinstance(ev.get("data"), dict) else ev
        b["event_count"] += 1
        label = str(data.get("label") or data.get("app") or data.get("title") or data.get("event_type") or "activity")[:80]
        b["sources"][label] += 1
        if "key" in data or "keystroke" in label.lower(): b["key_count"] += 1
        if "click" in label.lower(): b["click_count"] += 1
        if "scroll" in label.lower(): b["scroll_count"] += 1
        if all(k in data for k in ("dx", "dy")):
            b["pointer_delta_total"] += math.sqrt(float(data.get("dx") or 0) ** 2 + float(data.get("dy") or 0) ** 2)
        b["flow_friction_score"] = float(b["key_count"] + b["click_count"] + b["scroll_count"]) / max(float(window_seconds), 1.0)
    out: list[BlenderEvent] = []
    for bucket, b in sorted(buckets.items()):
        top = b.pop("sources").most_common(5)
        compressed = {**b, "top_sources": top, "window_seconds": window_seconds}
        text = f"ActivityWatch compressed window events={compressed['event_count']} keys={compressed['key_count']} clicks={compressed['click_count']} pointer_delta_total={compressed['pointer_delta_total']:.2f}"
        out.append(BlenderEvent("activitywatch_compressed", str(bucket), datetime.fromtimestamp(bucket, timezone.utc).isoformat(), text, {"compressed": compressed}, compressed))
    return out


def fetch_activitywatch(limit: int, window_seconds: int = DEFAULT_ACTIVITY_WINDOW_SECONDS) -> list[BlenderEvent]:
    # Optional local source. No pixel-perfect event storage: immediately aggregate into windows.
    rows: list[dict[str, Any]] = []
    for db in activitywatch_db_candidates()[:1]:
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            cur = con.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            for table in tables:
                if "event" not in table.lower():
                    continue
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                if "timestamp" not in cols and "created_at" not in cols:
                    continue
                time_col = "timestamp" if "timestamp" in cols else "created_at"
                data_col = "data" if "data" in cols else None
                select = f"SELECT {time_col}{',' + data_col if data_col else ''} FROM {table} ORDER BY {time_col} DESC LIMIT ?"
                for r in cur.execute(select, (limit,)).fetchall():
                    data: dict[str, Any] = {}
                    if data_col and len(r) > 1:
                        try: data = json.loads(r[1]) if isinstance(r[1], str) else {}
                        except Exception: data = {"raw": str(r[1])[:500]}
                    rows.append({"timestamp": r[0], "data": data, "source_db": rel(db), "source_table": table})
                break
            con.close()
        except Exception:
            continue
    return compress_activity_payload(rows[:limit], window_seconds=window_seconds)


def ontology_terms(text: str) -> list[str]:
    low = text.lower()
    hits: list[str] = []
    synonyms = {
        "ACTION": ["action", "execute", "send", "file", "run", "submit"],
        "EVENT": ["event", "happened", "workflow", "command"],
        "HYPOTHESIS": ["hypothesis", "maybe", "could", "possible", "abduct"],
        "CLAIM": ["claim", "assert", "says"],
        "EVIDENCE": ["evidence", "receipt", "source", "hash", "custody"],
        "SIGNAL": ["signal", "hint", "anomaly", "telemetry"],
        "ALGORITHM": ["algorithm", "bytewax", "rete", "centrality", "temporal"],
        "COMMENT": ["comment", "note", "aside"],
    }
    for term in GO25:
        if term.lower() in low or any(s in low for s in synonyms.get(term, [])):
            hits.append(term)
    if not hits:
        hits = ["SIGNAL", "COMMENT"]
    return list(dict.fromkeys(hits))


def injection_findings(text: str) -> list[str]:
    return [p.pattern for p in INJECTION_PATTERNS if p.search(text)]


def epistemic_for_event(source: str, text: str, injection: bool, payload: dict[str, Any]) -> str:
    if injection:
        return "BULLSHIT"
    if source in {"workflow_event", "conversation_command", "absurd_workflow"}:
        return "FACT"
    if source == "pg_recvlogical":
        return "FACT"
    if source == "activitywatch_compressed":
        return "PROBABLE"
    if any(w in text.lower() for w in ["maybe", "possible", "hypothesis", "abduct"]):
        return "POSSIBLE"
    return "SURE_MAYBE"


def has_mtime_snapshot_tag(text: str, payload: dict[str, Any]) -> bool:
    if "mtime_snapshot_v1" in text:
        return True
    try:
        return "mtime_snapshot_v1" in jdump(payload)
    except Exception:
        return False


def temporal_lora_policy(text: str, payload: dict[str, Any]) -> dict[str, Any]:
    fragile_mtime = has_mtime_snapshot_tag(text, payload)
    threshold = MTIME_SNAPSHOT_LORA_SWAP_TOKEN_THRESHOLD if fragile_mtime else DEFAULT_LORA_SWAP_TOKEN_THRESHOLD
    return {
        "schema": "lucidota.bytewax.temporal_lora_policy.v1",
        "fragile_temporal_source": "mtime_snapshot_v1" if fragile_mtime else "",
        "lora_swap_token_threshold": threshold,
        "max_context_tokens": threshold,
        "deepseek_context_mode": "lean_mtime_snapshot_fast_path" if fragile_mtime else "default_context_path",
        "date_extraction_router": "treelite" if fragile_mtime else "bytewax_default",
        "treelite_artifact": TREELITE_DATE_ROUTER_ARTIFACT if fragile_mtime else "",
        "vram_policy": "do_not_expand_deep_context_for_low_trust_mtime_snapshot" if fragile_mtime else "normal_512_token_lora_window",
    }


def epistemic_trace(source: str, flag: str, terms: list[str], injection: bool, payload: dict[str, Any], text: str = "") -> dict[str, Any]:
    lora_policy = temporal_lora_policy(text, payload)
    return {
        "schema": "lucidota.bytewax.signal_packet_certainty.v1",
        "epistemic_flag": flag,
        "allowed_flags": EPISTEMIC_FLAGS,
        "source": source,
        "ontology_terms": terms,
        "prompt_injection_detected": injection,
        "classification_method": "deterministic_source_rules_plus_prompt_injection_regex",
        "transport": payload.get("source_transport") or "cursor_poll",
        "temporal_lora_policy": lora_policy,
        "lora_swap_token_threshold": lora_policy["lora_swap_token_threshold"],
        "treelite_date_router_enabled": bool(lora_policy["fragile_temporal_source"]),
        "generated_at": now_z(),
    }


def inject_epistemic_signal_packet(payload: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    packet = dict(payload or {})
    packet["epistemic_flag"] = trace["epistemic_flag"]
    packet["epistemic_certainty"] = trace
    packet["lora_swap_token_threshold"] = trace["lora_swap_token_threshold"]
    packet["temporal_lora_policy"] = trace["temporal_lora_policy"]
    packet["treelite_date_router_enabled"] = trace["treelite_date_router_enabled"]
    packet["validation_schema"] = "06_SCHEMA/bytewax_signal_packet.schema.json"
    return packet



_BANDIT_POLICY_HYDRATED = False

def hydrate_bandit_policy_from_db() -> None:
    global _BANDIT_POLICY_HYDRATED
    if _BANDIT_POLICY_HYDRATED:
        return
    try:
        with psycopg.connect(STATE_DSN, row_factory=dict_row) as conn:
            rows = conn.execute("""
                SELECT algorithm_id, total_reward, pulls
                FROM lucidota_learning.bytewax_bandit_policy
                WHERE pulls > 0
            """).fetchall()
        updates = []
        for row in rows:
            pulls = int(row["pulls"] or 0)
            if pulls <= 0:
                continue
            avg = float(row["total_reward"] or 0.0) / pulls
            updates.extend(BanditUpdate(context_id="persisted_policy", action_id=str(row["algorithm_id"]), reward=avg, propensity=1.0) for _ in range(min(pulls, 64)))
        if updates:
            update_policy(updates)
        _BANDIT_POLICY_HYDRATED = True
    except Exception:
        _BANDIT_POLICY_HYDRATED = True

def event_to_hint(event: BlenderEvent) -> dict[str, Any]:
    terms = ontology_terms(event.text_surface)
    findings = injection_findings(event.text_surface)
    inj = bool(findings)
    flag = epistemic_for_event(event.source, event.text_surface, inj, event.payload)
    trace = epistemic_trace(event.source, flag, terms, inj, event.payload, event.text_surface)
    injected_payload = inject_epistemic_signal_packet(event.payload, trace)
    injected_payload = staple_activity(injected_payload, event.payload, {"mouse_delta_sum": event.compressed_activity.get("pointer_delta_total", 0.0), "keystroke_burst": event.compressed_activity.get("key_count", 0), "click_count": event.compressed_activity.get("click_count", 0), "scroll_count": event.compressed_activity.get("scroll_count", 0)})
    rete_bandit = apply_rete_bandit({
        "source": event.source,
        "source_ref": event.source_ref,
        "event_time": event.event_time,
        "text_surface": event.text_surface,
        "payload": injected_payload,
        "compressed_activity": injected_payload.get("compressed_activity") or event.compressed_activity or compress_activity(event.payload, event.compressed_activity),
        "ontology_terms": terms,
        "epistemic_flag": flag,
        "injection_flag": inj,
    })
    injected_payload["rete_bandit_decision"] = rete_bandit
    trace["rete_bandit_decision"] = {k: rete_bandit.get(k) for k in ("context_id", "algorithm_pool", "selected_algorithm", "selected_engine", "parallel_engine_plan", "reward", "facts_yielded", "penalty_reason", "execution_status")}
    low = event.text_surface.lower()
    support = min(1.0, 0.15 * len(terms) + (0.25 if "evidence" in low or "receipt" in low else 0.0) + (0.2 if event.source in {"workflow_event", "conversation_command", "absurd_workflow"} else 0.0))
    contradiction = 1.0 if inj else min(1.0, low.count("contradiction") * 0.4 + low.count("failed") * 0.15 + low.count("dead_letter") * 0.4)
    centrality = min(1.0, (len(set(terms)) / len(GO25)) + (0.2 if "RELATIONSHIP" in terms else 0.0) + (0.2 if "LEVERAGE" in terms else 0.0))
    if inj:
        hypothesis = "Prompt-injection or unsafe execution string detected; force deterministic membrane and draft-only handling."
    elif event.source == "activitywatch_compressed":
        hypothesis = "Compressed user activity window indicates flow/friction cadence; no pixel-perfect movement retained."
    elif "failed" in low or "dead_letter" in low:
        hypothesis = "Failure/anomaly path requires Serpentina retry economics and contradiction review."
    elif "canonical_mutation_allowed" in low or "graph" in low:
        hypothesis = "Command envelope touches graph/mutation language; require authority and evidence gate before action."
    else:
        hypothesis = "Low-friction eventflow signal; route through ontology and keep as abductive hint."
    return {
        "source": event.source,
        "source_ref": event.source_ref,
        "event_time": event.event_time,
        "text_surface": event.text_surface[:20000],
        "payload": injected_payload,
        "compressed_activity": event.compressed_activity,
        "ontology_terms": terms,
        "epistemic_flag": flag,
        "certainty_trace": trace,
        "temporal_lora_policy": trace["temporal_lora_policy"],
        "rete_bandit_decision": rete_bandit,
        "injection_flag": inj,
        "injection_findings": findings,
        "hypothesis": hypothesis,
        "support_score": float(round(support, 6)),
        "contradiction_score": float(round(contradiction, 6)),
        "centrality_score": float(round(centrality, 6)),
    }


def run_bytewax_or_fallback(events: list[BlenderEvent]) -> tuple[list[dict[str, Any]], str]:
    hydrate_bandit_policy_from_db()
    try:
        from bytewax.dataflow import Dataflow
        import bytewax.operators as op
        from bytewax.testing import TestingSource, TestingSink, run_main
        flow = Dataflow("lucidota-abductive-blender")
        inp = op.input("events", flow, TestingSource(events))
        hints_stream = op.map("event-to-abductive-hint", inp, event_to_hint)
        out: list[dict[str, Any]] = []
        op.output("out", hints_stream, TestingSink(out))
        run_main(flow)
        return out, "bytewax"
    except Exception:
        return [event_to_hint(e) for e in events], "fallback_deterministic_map"


def persist_duckdb_footprint(hints: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import duckdb
        db_path = RUNTIME_DIR / "abductive_blender.duckdb"
        temp_dir = RUNTIME_DIR / "duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(db_path))
        con.execute(f"PRAGMA memory_limit='{DEFAULT_DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"PRAGMA temp_directory='{temp_dir}'")
        con.execute("PRAGMA preserve_insertion_order=false")
        con.execute("CREATE TABLE IF NOT EXISTS hints(run_id VARCHAR, source VARCHAR, source_ref VARCHAR, epistemic_flag VARCHAR, support DOUBLE, contradiction DOUBLE, centrality DOUBLE, created_at VARCHAR)")
        rows = [(run_id, h["source"], h["source_ref"], h["epistemic_flag"], h["support_score"], h["contradiction_score"], h["centrality_score"], now_z()) for h in hints]
        if rows:
            con.executemany(
                "INSERT INTO hints VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
        con.close()
        return {"backend": "duckdb", "path": rel(db_path), "rows": len(hints), "memory_limit": DEFAULT_DUCKDB_MEMORY_LIMIT, "temp_directory": rel(temp_dir)}
    except Exception as exc:
        path = RUNTIME_DIR / "async_duckdb_footprint.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            for h in hints:
                fh.write(jdump({"run_id": run_id, "duckdb_unavailable": str(exc), **h}) + "\n")
        return {"backend": "jsonl_fallback", "path": rel(path), "rows": len(hints), "duckdb_error": str(exc)}


def log_comment(summary: dict[str, Any]) -> None:
    try:
        with psycopg.connect(STORAGE_DSN) as conn:
            conn.execute(
                """
                INSERT INTO lucidota_go.graph_promotion_evidence_resolution(
                    evidence_ref, ref_kind, resolved, resolver, source_table, source_uuid, source_path, detail
                ) VALUES ('BYTEWAX_ABDUCTIVE_BLENDER', 'comment_primitive', true,
                          'bytewax_abductive_blender', 'lucidota_learning.bytewax_abductive_hint',
                          gen_random_uuid(), %s, %s::jsonb)
                """,
                (WORKER_SOURCE, jdump({"go_term": "COMMENT", "epistemic_flag": "SURE_MAYBE", **summary})),
            )
            conn.commit()
    except Exception:
        return


def persist_absurd_workflow_journal(
    conn: psycopg.Connection[Any],
    *,
    run_id: str,
    events: list[BlenderEvent],
    hints: list[dict[str, Any]],
) -> int:
    inserted = 0
    hint_index = {(h.get("source"), h.get("source_ref")): h for h in hints}
    for event in events:
        hint = hint_index.get((event.source, event.source_ref), {})
        epistemic_flag = str(hint.get("epistemic_flag") or "SURE_MAYBE")
        detail = {
            "schema": "lucidota.bytewax.absurd_workflow_journal.v1",
            "run_id": run_id,
            "source": event.source,
            "source_ref": event.source_ref,
            "event_time": event.event_time,
            "text_surface": event.text_surface,
            "compressed_activity": event.compressed_activity,
            "epistemic_flag": epistemic_flag,
            "hints_seen": 1 if hint else 0,
        }
        event_uuid = _deterministic_event_uuid(event.source, event.source_ref, event.event_time, event.text_surface)
        conn.execute(
            """
            INSERT INTO lucidota_control.workflow_event(
                event_id, workflow_id, run_id, phase, status, source, detail
            ) VALUES (%s::uuid, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (event_id) DO NOTHING
            """,
            (
                str(event_uuid),
                BYTEWAX_WORKFLOW_ID,
                run_id,
                event.source,
                _workflow_status_for_epistemic(epistemic_flag),
                WORKER_SOURCE,
                jdump(detail),
            ),
        )
        inserted += 1
    return inserted


def persist_hints(events: list[BlenderEvent], hints: list[dict[str, Any]], mode: str, flow_backend: str) -> dict[str, Any]:
    run_detail = {"mode": mode, "flow_backend": flow_backend, "tokio_channel_semantics": "nonblocking_tick_queue", "generated_at": now_z()}
    with psycopg.connect(STATE_DSN, row_factory=dict_row) as conn:
        run_row = conn.execute(
            """
            INSERT INTO lucidota_learning.bytewax_stream_run(status, events_in, hints_out, detail)
            VALUES ('succeeded', %s, %s, %s::jsonb)
            RETURNING run_id::text
            """,
            (len(events), len(hints), jdump(run_detail)),
        ).fetchone()
        run_id = str(run_row["run_id"])
        inserted_events = 0
        inserted_hints = 0
        for h in hints:
            erow = conn.execute(
                """
                INSERT INTO lucidota_learning.bytewax_abductive_event(
                    source, source_ref, event_time, text_surface, ontology_terms, epistemic_flag,
                    injection_flag, compressed_activity, certainty_trace, payload
                ) VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
                ON CONFLICT(source, source_ref) DO UPDATE SET
                    text_surface=EXCLUDED.text_surface,
                    ontology_terms=EXCLUDED.ontology_terms,
                    epistemic_flag=EXCLUDED.epistemic_flag,
                    injection_flag=EXCLUDED.injection_flag,
                    compressed_activity=EXCLUDED.compressed_activity,
                    certainty_trace=EXCLUDED.certainty_trace,
                    payload=EXCLUDED.payload
                RETURNING event_uuid::text
                """,
                (h["source"], h["source_ref"], h["event_time"], h["text_surface"], jdump(h["ontology_terms"]), h["epistemic_flag"], h["injection_flag"], jdump(h["compressed_activity"]), jdump(h["certainty_trace"]), jdump(h["payload"])),
            ).fetchone()
            event_uuid = str(erow["event_uuid"])
            inserted_events += 1
            row = conn.execute(
                """
                INSERT INTO lucidota_learning.bytewax_abductive_hint(
                    event_uuid, source, source_ref, epistemic_flag, hypothesis, support_score,
                    contradiction_score, centrality_score, injection_flag, ontology_terms, certainty_trace, detail
                ) VALUES (%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
                ON CONFLICT(source, source_ref, hypothesis) DO UPDATE SET
                    epistemic_flag=EXCLUDED.epistemic_flag,
                    support_score=EXCLUDED.support_score,
                    contradiction_score=EXCLUDED.contradiction_score,
                    centrality_score=EXCLUDED.centrality_score,
                    injection_flag=EXCLUDED.injection_flag,
                    ontology_terms=EXCLUDED.ontology_terms,
                    certainty_trace=EXCLUDED.certainty_trace,
                    detail=EXCLUDED.detail
                RETURNING hint_uuid::text
                """,
                (event_uuid, h["source"], h["source_ref"], h["epistemic_flag"], h["hypothesis"], h["support_score"], h["contradiction_score"], h["centrality_score"], h["injection_flag"], jdump(h["ontology_terms"]), jdump(h["certainty_trace"]), jdump({k: v for k, v in h.items() if k not in {"payload", "text_surface"}})),
            ).fetchone()
            inserted_hints += 1 if row else 0
            rb = h.get("rete_bandit_decision") or {}
            if rb:
                conn.execute(
                    """
                    INSERT INTO lucidota_learning.bytewax_rete_bandit_decision(
                        run_id, event_uuid, source, source_ref, context_id, algorithm_pool, selected_algorithm,
                        selected_engine, parallel_engine_plan, bandit_strategy, rule_hits, execution_status, runtime_ms, facts_yielded, reward,
                        regret_weights, penalty_reason, detail
                    ) VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s::jsonb,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb)
                    ON CONFLICT(source, source_ref, context_id, selected_algorithm) DO UPDATE SET
                        run_id=EXCLUDED.run_id, event_uuid=EXCLUDED.event_uuid, algorithm_pool=EXCLUDED.algorithm_pool,
                        selected_engine=EXCLUDED.selected_engine, parallel_engine_plan=EXCLUDED.parallel_engine_plan,
                        bandit_strategy=EXCLUDED.bandit_strategy, rule_hits=EXCLUDED.rule_hits,
                        execution_status=EXCLUDED.execution_status, runtime_ms=EXCLUDED.runtime_ms,
                        facts_yielded=EXCLUDED.facts_yielded, reward=EXCLUDED.reward,
                        regret_weights=EXCLUDED.regret_weights, penalty_reason=EXCLUDED.penalty_reason,
                        detail=EXCLUDED.detail, created_at=now()
                    """,
                    (run_id, event_uuid, h["source"], h["source_ref"], rb.get("context_id"), jdump(rb.get("algorithm_pool") or []), rb.get("selected_algorithm"), rb.get("selected_engine") or "cpu_fairyfuse_ternary", jdump(rb.get("parallel_engine_plan") or {}), rb.get("bandit_strategy"), jdump(rb.get("rule_hits") or []), rb.get("execution_status"), float(rb.get("runtime_ms") or 0.0), int(rb.get("facts_yielded") or 0), float(rb.get("reward") or 0.0), jdump(rb.get("regret_weights") or {}), rb.get("penalty_reason"), jdump(rb)),
                )
                conn.execute(
                    """
                    INSERT INTO lucidota_learning.bytewax_bandit_policy(algorithm_id,pulls,total_reward,last_reward,last_context_id,detail)
                    VALUES (%s,1,%s,%s,%s,%s::jsonb)
                    ON CONFLICT(algorithm_id) DO UPDATE SET
                        pulls=lucidota_learning.bytewax_bandit_policy.pulls + 1,
                        total_reward=lucidota_learning.bytewax_bandit_policy.total_reward + EXCLUDED.last_reward,
                        last_reward=EXCLUDED.last_reward,
                        last_context_id=EXCLUDED.last_context_id,
                        updated_at=now(),
                        detail=EXCLUDED.detail
                    """,
                    (rb.get("selected_algorithm"), float(rb.get("reward") or 0.0), float(rb.get("reward") or 0.0), rb.get("context_id"), jdump({"last_decision": rb})),
                )
            conn.execute(
                """
                INSERT INTO lucidota_learning.bytewax_hint(source, phase, status, hint, score, detail)
                VALUES (%s, 'abductive_blender', %s, %s, %s, %s::jsonb)
                """,
                (h["source"], h["epistemic_flag"], h["hypothesis"], int(100 * h["support_score"]), jdump(h)),
            )
        inserted_journal = persist_absurd_workflow_journal(conn, run_id=run_id, events=events, hints=hints)
        conn.commit()
    footprint = persist_duckdb_footprint(hints, run_id)
    summary = {"run_id": run_id, "events_in": len(events), "hints_out": len(hints), "inserted_events": inserted_events, "inserted_hints": inserted_hints, "journaled_workflow_events": inserted_journal, "mode": mode, "flow_backend": flow_backend, "footprint": footprint}
    log_comment(summary)
    return summary


def collect_events(
    limit: int,
    live_cursor: bool,
    include_activitywatch: bool,
    activity_window_seconds: int = DEFAULT_ACTIVITY_WINDOW_SECONDS,
    *,
    hardware_state: dict[str, Any] | None = None,
    prefer_replication: bool = False,
    replication_slot: str = DEFAULT_REPLICATION_SLOT,
    create_replication_slot: bool = False,
    replication_timeout_seconds: float = DEFAULT_REPLICATION_TIMEOUT_SECONDS,
    unix_socket: str | None = None,
    unix_socket_timeout_seconds: float = 1.0,
) -> list[BlenderEvent]:
    controller = hardware_controller(hardware_state or sample_hardware_telemetry(), limit)
    limit = min(limit, controller["tick_limit"])
    include_activitywatch = include_activitywatch and controller["include_activitywatch"]
    prefer_replication = prefer_replication or controller["prefer_replication"]
    unix_socket_timeout_seconds = min(unix_socket_timeout_seconds, controller["unix_socket_timeout_seconds"])
    per = max(1, limit // 4)
    events: list[BlenderEvent] = []
    if unix_socket:
        events.extend(
            fetch_unix_socket_events(
                max(1, per),
                socket_path=unix_socket,
                timeout_seconds=unix_socket_timeout_seconds,
            )
        )
    if prefer_replication:
        events.extend(
            fetch_pg_recvlogical_events(
                max(1, per),
                slot_name=replication_slot,
                create_slot=create_replication_slot,
                timeout_seconds=replication_timeout_seconds,
            )
        )
    events.extend(fetch_conversation_commands(per, live_cursor))
    events.extend(fetch_workflow_events(per, live_cursor))
    events.extend(fetch_absurd_events(per, live_cursor))
    if include_activitywatch:
        events.extend(fetch_activitywatch(per, window_seconds=activity_window_seconds))
    return events[:limit]


def tick(
    limit: int,
    live_cursor: bool = True,
    include_activitywatch: bool = True,
    activity_window_seconds: int = DEFAULT_ACTIVITY_WINDOW_SECONDS,
    *,
    prefer_replication: bool = False,
    replication_slot: str = DEFAULT_REPLICATION_SLOT,
    create_replication_slot: bool = False,
    replication_timeout_seconds: float = DEFAULT_REPLICATION_TIMEOUT_SECONDS,
    unix_socket: str | None = None,
    unix_socket_timeout_seconds: float = 1.0,
) -> dict[str, Any]:
    ensure_schema()
    hardware_state = sample_hardware_telemetry()
    controller = hardware_controller(hardware_state, limit)
    events = collect_events(
        limit=controller["tick_limit"],
        live_cursor=live_cursor,
        include_activitywatch=include_activitywatch and controller["include_activitywatch"],
        activity_window_seconds=activity_window_seconds,
        hardware_state=hardware_state,
        prefer_replication=prefer_replication,
        replication_slot=replication_slot,
        create_replication_slot=create_replication_slot,
        replication_timeout_seconds=replication_timeout_seconds,
        unix_socket=unix_socket,
        unix_socket_timeout_seconds=min(unix_socket_timeout_seconds, controller["unix_socket_timeout_seconds"]),
    )
    hints, backend = run_bytewax_or_fallback(events)
    summary = persist_hints(events, hints, "live_cursor" if live_cursor else "latest_window", backend)
    summary["unix_socket"] = unix_socket
    summary["activity_window_seconds"] = activity_window_seconds
    summary["prefer_replication"] = controller["prefer_replication"]
    summary["replication_slot"] = replication_slot if controller["prefer_replication"] else None
    summary["hardware_state"] = hardware_state
    summary["hardware_controller"] = controller
    summary["needle_swarm_throttle_tok_per_sec"] = NEEDLE_SWARM_THROTTLE_TOK_PER_SEC
    emit("BYTEWAX_ABDUCTIVE_TICK", summary)
    return summary


def status() -> dict[str, Any]:
    ensure_schema()
    with psycopg.connect(STATE_DSN, row_factory=dict_row) as conn:
        rows = conn.execute(
            """
            SELECT epistemic_flag, count(*) AS n, max(created_at)::text AS newest
            FROM lucidota_learning.bytewax_abductive_hint
            GROUP BY epistemic_flag ORDER BY epistemic_flag
            """
        ).fetchall()
        cursors = conn.execute("SELECT cursor_name,last_seen_at::text,last_seen_ref FROM lucidota_learning.bytewax_abductive_cursor ORDER BY cursor_name").fetchall()
    return {"schema": "lucidota.bytewax_abductive_blender.status.v1", "generated_at": now_z(), "hints": [dict(r) for r in rows], "cursors": [dict(r) for r in cursors]}


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}_{stamp()}.json"
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bytewax-compatible abductive hypothesis blender")
    p.add_argument("action", nargs="?", default="tick", choices=["init", "tick", "loop", "status"])
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--idle-sleep", type=float, default=5.0)
    p.add_argument("--max-cycles", type=int, default=1, help="loop action cycle cap; default is one ephemeral pass; values below 1 are clamped to 1")
    p.add_argument("--latest-window", action="store_true")
    p.add_argument("--no-activitywatch", action="store_true")
    p.add_argument("--activity-window-seconds", type=int, default=DEFAULT_ACTIVITY_WINDOW_SECONDS)
    p.add_argument("--unix-socket", default=None, help="optional AF_UNIX stream of Diogenes packets from a local service socket")
    p.add_argument("--unix-socket-timeout-seconds", type=float, default=1.0)
    p.add_argument("--prefer-replication", action="store_true", help="try pg_recvlogical pulse before cursor polling; falls back safely")
    p.add_argument("--replication-slot", default=DEFAULT_REPLICATION_SLOT)
    p.add_argument("--create-replication-slot", action="store_true")
    p.add_argument("--replication-timeout-seconds", type=float, default=DEFAULT_REPLICATION_TIMEOUT_SECONDS)
    args = p.parse_args(argv)
    args.limit = max(1, min(args.limit, MAX_TICK_LIMIT))
    args.max_cycles = max(1, min(args.max_cycles, MAX_LOOP_CYCLES))
    args.idle_sleep = max(0.0, min(float(args.idle_sleep), MAX_IDLE_SLEEP_SECONDS))
    args.activity_window_seconds = max(1, min(int(args.activity_window_seconds), 3600))
    args.unix_socket_timeout_seconds = max(0.05, min(float(args.unix_socket_timeout_seconds), 5.0))
    args.replication_timeout_seconds = max(0.1, min(float(args.replication_timeout_seconds), 30.0))
    if args.action == "init":
        ensure_schema(); report = status(); path = write_report("init", report); print("REPORT_PATH=" + rel(path)); return 0
    if args.action == "tick":
        report = tick(
            args.limit,
            live_cursor=not args.latest_window,
            include_activitywatch=not args.no_activitywatch,
            activity_window_seconds=args.activity_window_seconds,
            prefer_replication=args.prefer_replication,
            replication_slot=args.replication_slot,
            create_replication_slot=args.create_replication_slot,
            replication_timeout_seconds=args.replication_timeout_seconds,
            unix_socket=args.unix_socket,
            unix_socket_timeout_seconds=args.unix_socket_timeout_seconds,
        ); path = write_report("tick", report); print("REPORT_PATH=" + rel(path)); return 0
    if args.action == "status":
        report = status(); path = write_report("status", report); print(json.dumps(report, indent=2, default=str)); print("REPORT_PATH=" + rel(path)); return 0
    if args.action == "loop":
        ensure_schema(); emit("BYTEWAX_ABDUCTIVE_LOOP_ONLINE", {"limit": args.limit, "idle_sleep": args.idle_sleep, "max_cycles": args.max_cycles, "activitywatch": not args.no_activitywatch, "activity_window_seconds": args.activity_window_seconds, "duckdb_memory_limit": DEFAULT_DUCKDB_MEMORY_LIMIT, "prefer_replication": args.prefer_replication, "replication_slot": args.replication_slot if args.prefer_replication else None, "needle_swarm_throttle_tok_per_sec": NEEDLE_SWARM_THROTTLE_TOK_PER_SEC, "unix_socket": args.unix_socket})
        cycles = 0
        while cycles < args.max_cycles:
            try:
                cycles += 1
                tick(
                    args.limit,
                    live_cursor=not args.latest_window,
                    include_activitywatch=not args.no_activitywatch,
                    activity_window_seconds=args.activity_window_seconds,
                    prefer_replication=args.prefer_replication,
                    replication_slot=args.replication_slot,
                    create_replication_slot=args.create_replication_slot,
                    replication_timeout_seconds=args.replication_timeout_seconds,
                    unix_socket=args.unix_socket,
                    unix_socket_timeout_seconds=args.unix_socket_timeout_seconds,
                )
                if cycles < args.max_cycles:
                    time.sleep(args.idle_sleep)
            except KeyboardInterrupt:
                emit("BYTEWAX_ABDUCTIVE_LOOP_STOP", {"reason": "KeyboardInterrupt"}); return 0
            except Exception as exc:
                emit("BYTEWAX_ABDUCTIVE_LOOP_SURVIVED_EXCEPTION", {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
                if cycles < args.max_cycles:
                    time.sleep(max(5.0, args.idle_sleep))
        emit("BYTEWAX_ABDUCTIVE_LOOP_STOP", {"reason": "max_cycles", "cycles": cycles})
        return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
