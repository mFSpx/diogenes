#!/usr/bin/env python3
"""WO-3 DIFFUSION 4-PACK: Koopman wire — real DB time-series → fit → forecast → receipt.

Signal priority (richest first):
  1. Governor PSI telemetry (5-D: cpu_psi, mem_psi, io_psi, mem_avail_norm, fleet_width)
     from lucidota_control.governor_action (lucidota_state).  Forecasts system pressure.
  2. Staging count-rate (1-D) from lucidota_go.staging_packet (last 40 min).
  3. Corpus chunk count-rate (1-D) fallback.

Registry seam: the capability entry is appended to BOOKS/ontology_packs/diffusion_pack/
registry.json so scripts/capability_pack_registry.py can absorb it.
"""
from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ALGOS"))

from koopman_operator import fit_koopman, koopman_forecast, reconstruction_error  # noqa: E402

try:
    import psycopg  # type: ignore
except ImportError:
    print("ERROR: psycopg not available", file=sys.stderr)
    sys.exit(1)

import numpy as np  # noqa: E402

OUT_DIR = ROOT / "05_OUTPUTS" / "runtime"
PACK_DIR = ROOT / "BOOKS" / "ontology_packs" / "diffusion_pack"

STAGING_SQL = """
SELECT date_trunc('minute', created_at) AS bucket, count(*) AS n
FROM lucidota_go.staging_packet
WHERE created_at > now() - interval '40 min'
GROUP BY 1 ORDER BY 1
"""

CORPUS_SQL = """
SELECT date_trunc('minute', created_at) AS bucket, count(*) AS n
FROM lucidota_go.corpus_chunk
WHERE created_at > now() - interval '120 min'
GROUP BY 1 ORDER BY 1
"""

STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
STATE_DSN   = os.environ.get("LUCIDOTA_GO_STATE_DSN",   "postgresql:///lucidota_state")

# Governor PSI signal: 5-D vector per tick
# [cpu_psi_some_avg10, mem_psi_some_avg10, io_psi_some_avg10, mem_avail_norm, fleet_width_norm]
GOVERNOR_SQL = """
SELECT ts,
       (telemetry_snapshot->'cpu_psi'->>'some_avg10')::float AS cpu_psi,
       (telemetry_snapshot->'mem_psi'->>'some_avg10')::float AS mem_psi,
       (telemetry_snapshot->'io_psi'->>'some_avg10')::float  AS io_psi,
       LEAST(1.0, (telemetry_snapshot->>'mem_avail_mb')::float / 8000.0) AS mem_avail_norm,
       LEAST(1.0, (telemetry_snapshot->>'bge_fleet_width')::float / 8.0) AS fleet_width_norm
FROM lucidota_control.governor_action
WHERE ts > now() - interval '2 hours'
  AND dry_run = false
ORDER BY ts
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fetch_series(conn, sql: str) -> list[tuple[str, int]]:
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [(str(r[0]), int(r[1])) for r in rows]


def series_to_trajectory(rows: list[tuple[str, int]]) -> np.ndarray:
    """Convert [(bucket_str, count)] -> (T, 1) trajectory array."""
    counts = np.array([r[1] for r in rows], dtype=float)
    return counts.reshape(-1, 1)


def fetch_governor_signal(state_dsn: str) -> tuple[list[str], np.ndarray] | tuple[None, None]:
    """Pull 5-D governor PSI telemetry from governor_action table.

    Returns (timestamps_list, trajectory_array shape (T, 5)) or (None, None) if too short.
    Dimensions: cpu_psi, mem_psi, io_psi, mem_avail_norm, fleet_width_norm.
    """
    try:
        with psycopg.connect(state_dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(GOVERNOR_SQL)
                rows = cur.fetchall()
    except Exception as e:
        print(f"[diffusion_wire] governor telemetry fetch failed: {e}", file=sys.stderr)
        return None, None

    if len(rows) < 5:
        return None, None

    timestamps = [str(r[0]) for r in rows]
    traj = np.array(
        [[float(r[i] or 0.0) for i in range(1, 6)] for r in rows],
        dtype=float,
    )
    return timestamps, traj


def ensure_pack_registry() -> None:
    """Write/update BOOKS/ontology_packs/diffusion_pack/registry.json."""
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    registry_path = PACK_DIR / "registry.json"
    payload = {
        "pack_id": "diffusion_pack",
        "pack_name": "Diffusion 4-Pack (Koopman wire)",
        "version": "0.1-wired",
        "status": "staged_not_promoted",
        "primary_chain": ["koopman_operator", "koopman_forecast", "reconstruction_error"],
        "graph_safety_rule": "Read-only DB pull; writes receipt JSON only; no canonical graph mutation.",
        "pillars": [
            {
                "index": 1,
                "pillar": "KOOPMAN",
                "operational_name": "KOOPMAN_WIRE",
                "technical_implementation": "ALGOS/koopman_operator.py",
                "function": "Fit Koopman operator via DMD on live DB time-series; forecast + reconstruction error.",
            }
        ],
        "recommended_first_test": {
            "command": "python3 scripts/diffusion_pack_wire.py",
            "expected_output": "05_OUTPUTS/runtime/koopman_forecast_<ts>.json",
        },
    }
    registry_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ts = stamp()
    out_path = OUT_DIR / f"koopman_forecast_{ts}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Signal priority: governor PSI (5-D) > staging count (1-D) > corpus count (1-D)
    timestamps, traj = fetch_governor_signal(STATE_DSN)
    source = "lucidota_control.governor_action (psi_5d)"
    input_series_repr: list = []

    if traj is None:
        # Fallback to count-rate series
        with psycopg.connect(STORAGE_DSN, connect_timeout=5) as conn:
            rows = fetch_series(conn, STAGING_SQL)
            source = "lucidota_go.staging_packet"
            if len(rows) < 5:
                print(f"staging_packet series too short ({len(rows)} pts), falling back to corpus_chunk", file=sys.stderr)
                rows = fetch_series(conn, CORPUS_SQL)
                source = "lucidota_go.corpus_chunk"
        if len(rows) < 2:
            print(f"ERROR: series still too short ({len(rows)} pts) — cannot fit Koopman", file=sys.stderr)
            return 1
        traj = series_to_trajectory(rows)
        input_series_repr = [{"bucket": r[0], "count": r[1]} for r in rows]
    else:
        input_series_repr = [
            {"ts": timestamps[i],
             "cpu_psi": float(traj[i, 0]), "mem_psi": float(traj[i, 1]),
             "io_psi": float(traj[i, 2]),  "mem_avail_norm": float(traj[i, 3]),
             "fleet_width_norm": float(traj[i, 4])}
            for i in range(len(timestamps))
        ]

    T = len(traj)
    model = fit_koopman(traj, rank=min(10, T - 1))
    forecast_steps = max(3, T // 4)
    forecast = koopman_forecast(traj[0], model, forecast_steps)
    recon_err = reconstruction_error(traj, model)

    receipt = {
        "schema": "lucidota.diffusion_pack.koopman_wire.v2",
        "generated_at": now_utc(),
        "source": source,
        "signal_dims": traj.shape[1] if traj.ndim > 1 else 1,
        "series_length": T,
        "koopman_rank": model["rank"],
        "forecast_steps": forecast_steps,
        "reconstruction_error": recon_err,
        "input_series": input_series_repr,
        "forecast": forecast.tolist(),
        "receipt_path": f"05_OUTPUTS/runtime/koopman_forecast_{ts}.json",
    }

    out_path.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, default=str))

    # Keep capability pack registry up to date.
    ensure_pack_registry()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
