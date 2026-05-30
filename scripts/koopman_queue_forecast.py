#!/usr/bin/env python3
"""
koopman_queue_forecast.py — Apply Koopman operator (DMD) to ABSURD queue
time series to forecast future queue load linearly.

Mutation class: read_only
Receipt scope:  LOCAL_FILE_PRODUCT

Usage
-----
    python3 scripts/koopman_queue_forecast.py --dry-run
    python3 scripts/koopman_queue_forecast.py --execute
    python3 scripts/koopman_queue_forecast.py --execute --steps 20 --rank 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ALGOS_DIR = ROOT / "ALGOS"
OUT_DIR = ROOT / "05_OUTPUTS" / "absurd"

sys.path.insert(0, str(ALGOS_DIR))
from koopman_operator import (  # noqa: E402
    fit_koopman,
    koopman_forecast,
    observable_lift,
    reconstruction_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dominant_frequencies(eigenvalues: np.ndarray) -> list[dict]:
    """Convert complex Koopman eigenvalues to frequency/growth descriptors."""
    out = []
    for ev in eigenvalues:
        mag = float(abs(ev))
        angle = float(np.angle(ev))
        freq_per_step = angle / (2 * np.pi)
        out.append({
            "real": float(ev.real),
            "imag": float(ev.imag),
            "magnitude": round(mag, 6),
            "frequency_per_step": round(freq_per_step, 6),
            "growth_per_step": round(mag, 6),
        })
    return out


def build_lifted_trajectory(raw: np.ndarray, degree: int = 2) -> np.ndarray:
    """Apply observable_lift row-by-row; return (T, lifted_dim)."""
    lifted_rows = [observable_lift(row, degree=degree) for row in raw]
    return np.array(lifted_rows, dtype=float)


def synthetic_trajectory(n_steps: int = 30) -> np.ndarray:
    """Generate a plausible synthetic ABSURD queue trajectory.

    Models a queue that fills (pending rising), processes (running peak),
    then drains (completed rising) — a realistic work-burst pattern.
    Returns shape (n_steps, 3): columns = [pending, running, completed].
    """
    rng = np.random.default_rng(42)
    t = np.arange(n_steps)
    # pending: Gaussian burst early then decay
    pending = np.clip(
        20 * np.exp(-((t - 8) ** 2) / 30) + rng.normal(0, 1, n_steps), 0, None
    )
    # running: lags pending by ~4 steps
    running = np.clip(
        12 * np.exp(-((t - 12) ** 2) / 25) + rng.normal(0, 0.5, n_steps), 0, None
    )
    # completed: monotone ramp
    completed = np.clip(t * 1.2 + rng.normal(0, 2, n_steps), 0, None)
    return np.stack([pending, running, completed], axis=1)


# ---------------------------------------------------------------------------
# DB query
# ---------------------------------------------------------------------------

def query_queue_timeseries(dsn: str) -> tuple[list[dict], bool]:
    """Query ABSURD queue job table for per-minute status counts.

    Returns (rows, is_synthetic).
    """
    try:
        import psycopg  # type: ignore
    except ImportError:
        import psycopg2 as psycopg  # type: ignore

    sql = """
        SELECT
            date_trunc('minute', created_at) AS t,
            COUNT(*) FILTER (WHERE status = 'pending')   AS pending,
            COUNT(*) FILTER (WHERE status = 'running')   AS running,
            COUNT(*) FILTER (WHERE status = 'completed') AS completed
        FROM lucidota_control.absurd_queue_job
        GROUP BY 1
        ORDER BY 1
    """
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
    except Exception as exc:
        print(f"[koopman_queue_forecast] DB query failed: {exc}", file=sys.stderr)
        return [], True  # caller will use synthetic

    result = [
        {
            "t": str(r[0]),
            "pending": int(r[1]),
            "running": int(r[2]),
            "completed": int(r[3]),
        }
        for r in rows
    ]
    return result, False


# ---------------------------------------------------------------------------
# Core forecast
# ---------------------------------------------------------------------------

def run_forecast(steps: int, rank: int, dry_run: bool, dsn: str) -> dict:
    """Full pipeline: load data → lift → fit → forecast → report."""

    # 1. Load data
    rows, forced_synthetic = query_queue_timeseries(dsn)

    is_synthetic = forced_synthetic or len(rows) < 10
    if is_synthetic:
        print(
            "[koopman_queue_forecast] Fewer than 10 real timesteps — "
            "using synthetic trajectory.",
            file=sys.stderr,
        )
        raw = synthetic_trajectory(n_steps=30)
        data_source = "synthetic"
        db_rows = []
    else:
        raw = np.array(
            [[r["pending"], r["running"], r["completed"]] for r in rows],
            dtype=float,
        )
        data_source = "live_db"
        db_rows = rows

    T, d = raw.shape  # d == 3

    # 2. Lift observables
    lifted = build_lifted_trajectory(raw, degree=2)
    lifted_dim = lifted.shape[1]

    # 3. Fit Koopman (clamp rank to feasible)
    effective_rank = min(rank, lifted_dim, T - 1)
    model = fit_koopman(lifted, rank=effective_rank)

    # 4. Reconstruction error on lifted trajectory
    recon_err = reconstruction_error(lifted, model)

    # 5. Forecast from last known lifted state
    x0_lifted = observable_lift(raw[-1], degree=2)
    fc_lifted = koopman_forecast(x0_lifted, model, steps=steps)  # (steps, lifted_dim)
    # Extract only the first 3 dimensions (original state components)
    fc_state = fc_lifted[:, :d]

    # 6. Dominant eigenvalues
    eigs = dominant_frequencies(model["eigenvalues"])

    # 7. Build report
    forecast_records = []
    for i, row in enumerate(fc_state):
        forecast_records.append({
            "step": i + 1,
            "pending": round(float(row[0]), 3),
            "running": round(float(row[1]), 3),
            "completed": round(float(row[2]), 3),
        })

    report = {
        "generated_at": now_iso(),
        "mutation_class": "read_only",
        "data_source": data_source,
        "is_synthetic": is_synthetic,
        "timesteps_observed": T,
        "lifted_dim": lifted_dim,
        "effective_rank": effective_rank,
        "reconstruction_error_mse": round(recon_err, 6),
        "dominant_eigenvalues": eigs,
        "forecast_steps": steps,
        "forecast": forecast_records,
        "dry_run": dry_run,
    }

    if not dry_run:
        ts = stamp()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"koopman_forecast_{ts}.json"
        out_path.write_text(json.dumps(report, indent=2))
        report["receipt_path"] = str(out_path)
        print(f"[koopman_queue_forecast] Receipt written: {out_path}", file=sys.stderr)
    else:
        report["receipt_path"] = None

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Koopman/DMD forecast of ABSURD queue load.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--steps", type=int, default=10, help="Forecast horizon (steps ahead)")
    ap.add_argument("--rank", type=int, default=5, help="DMD truncation rank")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run full pipeline but do not write receipt file.",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Write receipt to 05_OUTPUTS/absurd/.",
    )
    ap.add_argument(
        "--database-url",
        default=(
            os.environ.get("LUCIDOTA_GO_STATE_DSN")
            or os.environ.get("DATABASE_URL")
            or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
            or "postgresql:///lucidota_state"
        ),
        help="Postgres DSN for lucidota_state.",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout.")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.execute and args.dry_run:
        ap.error("--execute and --dry-run are mutually exclusive")

    dry_run = not args.execute  # default is dry-run

    report = run_forecast(
        steps=args.steps,
        rank=args.rank,
        dry_run=dry_run,
        dsn=args.database_url,
    )

    if args.json:
        print(json.dumps(report, indent=2))
        return

    # Human-readable summary
    print("=" * 60)
    print("KOOPMAN QUEUE FORECAST")
    print("=" * 60)
    print(f"  data_source        : {report['data_source']}")
    print(f"  is_synthetic       : {report['is_synthetic']}")
    print(f"  timesteps_observed : {report['timesteps_observed']}")
    print(f"  lifted_dim         : {report['lifted_dim']}")
    print(f"  effective_rank     : {report['effective_rank']}")
    print(f"  recon_error_mse    : {report['reconstruction_error_mse']}")
    print()
    print("  Dominant eigenvalues (as frequencies):")
    for i, ev in enumerate(report["dominant_eigenvalues"][:5]):
        print(
            f"    [{i}] mag={ev['magnitude']:.4f}  "
            f"freq/step={ev['frequency_per_step']:.4f}  "
            f"growth/step={ev['growth_per_step']:.4f}"
        )
    print()
    print(f"  10-step forecast (pending | running | completed):")
    for fc in report["forecast"]:
        print(
            f"    step {fc['step']:>2}: "
            f"pending={fc['pending']:>8.2f}  "
            f"running={fc['running']:>8.2f}  "
            f"completed={fc['completed']:>8.2f}"
        )
    print()
    if report["receipt_path"]:
        print(f"  receipt: {report['receipt_path']}")
    else:
        print("  dry-run: no receipt written")
    print("=" * 60)


if __name__ == "__main__":
    main()
