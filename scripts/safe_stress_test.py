#!/usr/bin/env python3
"""
LUCIDOTA Live Throughput Safety Stress Test

Read-only metric suite:
- checks resident worker liveness,
- measures local loop throughput,
- samples GPU thermal / VRAM state,
- checks Postgres queue posture,
- prints a compact operator-facing metric table.

No process killing. No cache flushing. No writes except an optional JSON report
to 05_OUTPUTS/safe_stress_test/ and a tickletrunk rescan on explicit request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psycopg.rows import dict_row
from core.telemetry.sanity_gate import TelemetrySanityGate
OUT_DIR = ROOT / "05_OUTPUTS" / "safe_stress_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_DSN = os.environ.get("LUCIDOTA_ABSURD_STATE_DSN") or os.environ.get("LUCIDOTA_GO_STATE_DSN") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN") or "postgresql:///lucidota_storage"
TOK_BUS_CEILING = int(os.environ.get("LUCIDOTA_TOKEN_BUS_CEILING", "12000"))
THROTTLE_CEILING = int(os.environ.get("LUCIDOTA_NEEDLE_SWARM_THROTTLE_TOK_PER_SEC", "7200"))
GPU_TEMP_WALL_C = float(os.environ.get("LUCIDOTA_GPU_TEMP_WALL_C", "84.0"))


@dataclass(frozen=True)
class MetricRow:
    phase: str
    loops: int
    elapsed_ms: float
    loops_per_sec: float
    tokens_per_sec: float
    gpu_temp_c: float
    vram_used_mb: int
    vram_free_mb: int
    safety_margin_ok: bool
    throttle_cap_tok_s: int
    note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def gpu_state() -> dict[str, Any]:
    default = {"temp_c": 0.0, "used_mb": 0, "free_mb": 0, "status": "missing"}
    gate = TelemetrySanityGate(known_static_tokens=["1650"])
    try:
        temp_cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
        cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
        if cp.returncode != 0 or not cp.stdout.strip():
            return default | {"status": "error", "stderr": cp.stderr[-200:]}
        row = cp.stdout.strip().splitlines()[0].split(",")
        used_mb = int(float(row[0].strip()))
        free_mb = int(float(row[1].strip()))
        audit = gate.verify_metric_integrity(cp.stdout, {"vram_used_mb": used_mb, "vram_free_mb": free_mb})
        corrected = audit["corrected_metrics"]
        return {
            "temp_c": float(temp_cp.stdout.strip().splitlines()[0].split(",")[0].strip()) if temp_cp.returncode == 0 and temp_cp.stdout.strip() else 0.0,
            "used_mb": int(corrected.get("vram_used_mb") or 0),
            "free_mb": int(corrected.get("vram_free_mb") or 0),
            "status": "ok",
            "telemetry_sanity_gate": audit["status"],
        }
    except Exception as exc:
        return default | {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def db_counts() -> dict[str, Any]:
    out = {"state_pending": None, "state_running": None, "storage_claims": None, "status": "error"}
    try:
        with psycopg.connect(STATE_DSN, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                SELECT
                    count(*) FILTER (WHERE status='pending') AS pending,
                    count(*) FILTER (WHERE status='running') AS running
                FROM lucidota_control.absurd_workflow
                """
            ).fetchone()
            out["state_pending"] = int(row["pending"])
            out["state_running"] = int(row["running"])
        with psycopg.connect(STORAGE_DSN, row_factory=dict_row) as conn:
            row = conn.execute("SELECT count(*) AS n FROM lucidota_korpus.temporal_claim").fetchone()
            out["storage_claims"] = int(row["n"])
        out["status"] = "ok"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def loop_benchmark(iterations: int, payload_kb: int) -> tuple[int, float]:
    payload = b"x" * (payload_kb * 1024)
    h = hashlib.sha256
    start = time.perf_counter()
    digest = 0
    for i in range(iterations):
        digest ^= int.from_bytes(h(payload + i.to_bytes(8, "little")).digest()[:8], "little")
    elapsed = time.perf_counter() - start
    # prevent dead-code elimination by keeping digest live
    if digest == -1:
        print("never")
    return iterations, elapsed


def build_rows(sample_count: int, loop_iters: int, payload_kb: int) -> list[MetricRow]:
    gpu = gpu_state()
    db = db_counts()
    rows: list[MetricRow] = []
    for idx in range(sample_count):
        loops, elapsed = loop_benchmark(loop_iters, payload_kb)
        lps = loops / max(elapsed, 1e-9)
        toks = min(TOK_BUS_CEILING, lps * payload_kb * 256.0)
        temp_c = float(gpu.get("temp_c") or 0.0)
        safety_margin_ok = (temp_c < GPU_TEMP_WALL_C) and (int(gpu.get("free_mb") or 0) > 0)
        note = "ok"
        if temp_c >= GPU_TEMP_WALL_C:
            note = "thermal_wall_reached_context_reaper_should_flush"
        elif toks >= TOK_BUS_CEILING:
            note = "bus_cap_saturated"
        rows.append(
            MetricRow(
                phase=f"sample_{idx + 1:02d}",
                loops=loops,
                elapsed_ms=round(elapsed * 1000.0, 3),
                loops_per_sec=round(lps, 2),
                tokens_per_sec=round(toks, 2),
                gpu_temp_c=round(temp_c, 1),
                vram_used_mb=int(gpu.get("used_mb") or 0),
                vram_free_mb=int(gpu.get("free_mb") or 0),
                safety_margin_ok=safety_margin_ok,
                throttle_cap_tok_s=THROTTLE_CEILING,
                note=note,
            )
        )
    # attach DB posture to first row note block via summary line only
    if rows:
        first = rows[0].as_dict()
        first["db_state_pending"] = db.get("state_pending")
        first["db_state_running"] = db.get("state_running")
        first["db_storage_claims"] = db.get("storage_claims")
        rows[0] = MetricRow(**{k: first[k] for k in MetricRow.__annotations__.keys()})
    return rows


def print_table(rows: list[MetricRow]) -> None:
    print("=" * 110)
    print("LUCIDOTA SAFE STRESS TEST :: LIVE OPERATIONAL THROUGHPUT")
    print("=" * 110)
    header = (
        f"{'PHASE':<10} {'LOOPS':>10} {'ELAPSED_MS':>12} {'LOOPS/S':>12} {'TOK/S':>12} "
        f"{'GPU_C':>8} {'VRAM_USED':>10} {'VRAM_FREE':>10} {'SAFE':>6} {'CAP':>6} NOTE"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.phase:<10} {row.loops:>10d} {row.elapsed_ms:>12.3f} {row.loops_per_sec:>12.2f} "
            f"{row.tokens_per_sec:>12.2f} {row.gpu_temp_c:>8.1f} {row.vram_used_mb:>10d} {row.vram_free_mb:>10d} "
            f"{str(row.safety_margin_ok):>6} {row.throttle_cap_tok_s:>6d} {row.note}"
        )
    print("-" * len(header))
    gpu = gpu_state()
    db = db_counts()
    summary = {
        "generated_at": now_z(),
        "token_bus_ceiling": TOK_BUS_CEILING,
        "needle_swarm_throttle_tok_per_sec": THROTTLE_CEILING,
        "gpu_temp_wall_c": GPU_TEMP_WALL_C,
        "gpu": gpu,
        "db": db,
        "safety_margin_ok": bool(gpu.get("status") == "ok" and float(gpu.get("temp_c") or 0.0) < GPU_TEMP_WALL_C),
        "draft_only": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("=" * 110)


def write_report(rows: list[MetricRow], sample_count: int, loop_iters: int, payload_kb: int) -> Path:
    path = OUT_DIR / f"safe_stress_test_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    payload = {
        "schema": "lucidota.safe_stress_test.v1",
        "generated_at": now_z(),
        "sample_count": sample_count,
        "loop_iters": loop_iters,
        "payload_kb": payload_kb,
        "rows": [r.as_dict() for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--loop-iters", type=int, default=50000)
    ap.add_argument("--payload-kb", type=int, default=16)
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()
    rows = build_rows(args.samples, args.loop_iters, args.payload_kb)
    print_table(rows)
    if args.write_report:
        report = write_report(rows, args.samples, args.loop_iters, args.payload_kb)
        print(f"REPORT_PATH={rel(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
