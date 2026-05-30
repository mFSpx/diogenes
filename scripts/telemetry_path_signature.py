#!/usr/bin/env python3
"""Apply path signatures to LUCIDOTA hardware telemetry stream for invariant feature extraction.

Mutation class: read_only

Loads up to 500 lines from 05_OUTPUTS/hw_telemetry.jsonl (or generates 100-step
synthetic 3-channel data if the file is absent), then runs the ALGOS/path_signature
pipeline:
  normalize_path -> lead_lag_transform -> signature_flat(depth=N)

Also computes logsig_level2 (Levy area matrix) and reports top channel-pair areas.
Writes a JSON receipt to 05_OUTPUTS/telemetry/path_sig_{ts}.json.
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
# Path setup — add ALGOS/ so path_signature is importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "ALGOS"))

from path_signature import (  # noqa: E402
    lead_lag_transform,
    logsig_level2,
    normalize_path,
    signature_flat,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INPUT = _REPO_ROOT / "05_OUTPUTS" / "hw_telemetry.jsonl"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "05_OUTPUTS" / "telemetry"
CHANNEL_NAMES = ["cpu_pct", "mem_mb", "gpu_util"]
MAX_LINES = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, max_lines: int = MAX_LINES) -> list[dict]:
    """Read up to max_lines JSON lines from path."""
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_channels(rows: list[dict]) -> np.ndarray:
    """Extract (cpu_pct, mem_mb, gpu_util) from hw_telemetry rows.

    hw_telemetry keys used:
      cpu_load_per_core  -> cpu_pct   (already 0-100 range approx)
      ram_avail_mb       -> mem_mb    (available MB; invert later if desired — kept raw)
      vram_used_pct      -> gpu_util
    """
    matrix = []
    for r in rows:
        cpu = float(r.get("cpu_load_per_core", r.get("cpu_pct", 0.0)))
        mem = float(r.get("ram_avail_mb", r.get("mem_mb", 0.0)))
        gpu = float(r.get("vram_used_pct", r.get("gpu_util", 0.0)))
        matrix.append([cpu, mem, gpu])
    return np.array(matrix, dtype=float)


def _synthetic_path(T: int = 100) -> np.ndarray:
    """Generate synthetic 3-channel telemetry over T steps."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, T)
    cpu = 30 + 20 * np.sin(t) + rng.normal(0, 2, T)
    mem = 4000 + 500 * np.cos(t * 0.7) + rng.normal(0, 50, T)
    gpu = 10 + 5 * np.sin(t * 1.3 + 1) + rng.normal(0, 1, T)
    return np.column_stack([cpu, mem, gpu])


def _area_matrix_to_list(mat: np.ndarray) -> list[list[float]]:
    return mat.tolist()


def _top_pairs(area_mat: np.ndarray, channel_names: list[str], n: int = 3) -> list[dict]:
    """Return top-n channel pairs by absolute Levy area (off-diagonal upper triangle)."""
    d = area_mat.shape[0]
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append({
                "channels": [channel_names[i], channel_names[j]],
                "area": float(area_mat[i, j]),
                "abs_area": float(abs(area_mat[i, j])),
            })
    pairs.sort(key=lambda x: x["abs_area"], reverse=True)
    return pairs[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    depth = args.depth

    # --- Load data ---
    if input_path.exists():
        rows = _load_jsonl(input_path, max_lines=MAX_LINES)
        path_matrix = _extract_channels(rows)
        data_source = str(input_path)
        print(f"[telemetry_path_signature] Loaded {len(rows)} rows from {input_path}")
    else:
        print(f"[telemetry_path_signature] {input_path} not found — using synthetic 3-channel data (T=100)")
        path_matrix = _synthetic_path(100)
        data_source = "synthetic:3ch_cpu_mem_gpu_T100"

    T, d = path_matrix.shape
    print(f"[telemetry_path_signature] path matrix shape: ({T}, {d})  channels: {CHANNEL_NAMES}")

    if T < 2:
        print("[telemetry_path_signature] ERROR: need at least 2 time steps.", file=sys.stderr)
        sys.exit(1)

    # --- Pipeline ---
    normed = normalize_path(path_matrix)            # translate+scale
    ll = lead_lag_transform(normed)                 # (2T-1, 2d)
    flat = signature_flat(ll, depth=depth)          # 1D feature vector

    # logsig on the original (non-lead-lag) normalized path for interpretability
    area_mat = logsig_level2(normed)                # (d, d)  Levy area
    top = _top_pairs(area_mat, CHANNEL_NAMES)

    print(f"[telemetry_path_signature] flat_sig length: {len(flat)}  (lead-lag d={2*d}, depth={depth})")
    print(f"[telemetry_path_signature] top channel pairs by |area|:")
    for p in top:
        print(f"  {p['channels'][0]} x {p['channels'][1]}: area={p['area']:.6f}")

    # --- Receipt ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt = {
        "receipt_ts": ts,
        "mutation_class": "read_only",
        "data_source": data_source,
        "T": T,
        "d": d,
        "channel_names": CHANNEL_NAMES,
        "depth": depth,
        "lead_lag_shape": list(ll.shape),
        "flat_sig_length": len(flat),
        "flat_sig_preview": flat[:8].tolist(),
        "area_matrix": _area_matrix_to_list(area_mat),
        "top_pairs_by_abs_area": top,
    }

    if args.dry_run:
        print(f"[telemetry_path_signature] --dry-run: receipt not written.")
        print(json.dumps(receipt, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"path_sig_{ts}.json"
    out_path.write_text(json.dumps(receipt, indent=2))
    print(f"[telemetry_path_signature] receipt -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Path signature feature extraction on LUCIDOTA hardware telemetry."
    )
    p.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to hw_telemetry.jsonl (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Signature truncation depth (default: 3)",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for receipt JSON (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print receipt but do not write to disk.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
