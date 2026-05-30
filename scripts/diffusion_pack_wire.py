#!/usr/bin/env python3
"""WO-3 DIFFUSION 4-PACK: Koopman wire — real DB time-series → fit → forecast → receipt.

Pulls a per-minute event-count series from lucidota_go.staging_packet (last 40 min),
feeds it to ALGOS/koopman_operator.py (fit_koopman + koopman_forecast +
reconstruction_error), and writes a receipt JSON to 05_OUTPUTS/runtime/.

Falls back to corpus_chunk.created_at if staging_packet series is too short (<5 pts).

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

    traj = series_to_trajectory(rows)  # (T, 1)
    T = len(traj)

    model = fit_koopman(traj, rank=min(10, T - 1))
    forecast_steps = max(3, T // 4)
    forecast = koopman_forecast(traj[0], model, forecast_steps)
    recon_err = reconstruction_error(traj, model)

    receipt = {
        "schema": "lucidota.diffusion_pack.koopman_wire.v1",
        "generated_at": now_utc(),
        "source_table": source,
        "series_length": T,
        "koopman_rank": model["rank"],
        "forecast_steps": forecast_steps,
        "reconstruction_error": recon_err,
        "input_series": [{"bucket": r[0], "count": r[1]} for r in rows],
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
