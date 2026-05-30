# DARWIN HAMMER — match 34, survivor 3
# gen: 2
# parent_a: hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py (gen1)
# parent_b: hybrid_path_signature_kan_m30_s3.py (gen1)
# born: 2026-05-29T23:25:17Z

"""Hybrid Audit‑Signature Pruning (Hybrid_AuditSignaturePrune)

This module fuses the *ternary lens audit* logic of
`hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py` with the
*path‑signature / KAN* machinery of
`hybrid_path_signature_kan_m30_s3.py`.

Mathematical bridge
-------------------
* The audit algorithm yields a categorical classification per candidate.
  We embed each classification into a one‑hot numeric vector, producing a
  discrete time‑series when the candidates are ordered (e.g. by discovery
  timestamp).  This series is a piecewise‑constant path in ℝⁿ.
* The signature side‑chain treats any multivariate path `X(t)` and extracts
  linear (`level‑1`) and quadratic (`level‑2`) features via the lead‑lag
  transform.  Those features are invariant to re‑parameterisation and capture
  the “shape” of the audit‑derived path.
* The KAN (Kolmogorov‑Arnold Network) part builds a spline basis on a grid
  and linearly mixes the basis with learned weights.  We reuse the spline
  basis as a *decreasing‑rate pruning schedule*: the spline evaluates a
  monotone decay curve (e.g. exponential) that assigns a pruning probability
  to each signature component.
* By multiplying the signature vector with the spline‑derived schedule we
  obtain a **pruned score** that respects both the audit’s categorical
  decisions and the mathematically smooth decreasing pruning schedule.

The public API consists of three core functions demonstrating this hybrid
operation:
`load_manifest`, `audit_signature`, and `prune_candidates`.  The `__main__`
section runs a smoke test on a tiny synthetic manifest."""

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A (audit) utilities
# ----------------------------------------------------------------------
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}
LOCAL_PATTERNS = [
    "*bitnet*",
    "*BitNet*",
    "*fairyfuse*",
    "*FairyFuse*",
    "*lora*",
    "*LoRA*",
    "*adapter*",
]


def utc_now() -> str:
    """Return the current UTC time in ISO‑8601 format."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON manifest and validate classifications."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for candidate in data.get("vendors", []):
        classification = candidate.get("classification")
        if classification not in CLASSIFICATIONS:
            raise SystemExit(
                f"invalid classification {classification!r} for {candidate.get('candidate_key')}"
            )
    return data


def classification_one_hot(classification: str) -> np.ndarray:
    """Map a classification string to a one‑hot vector of length |CLASSIFICATIONS|."""
    ordered = sorted(CLASSIFICATIONS)  # deterministic ordering
    vec = np.zeros(len(ordered), dtype=float)
    try:
        idx = ordered.index(classification)
        vec[idx] = 1.0
    except ValueError:
        # unknown classification – leave as zero vector
        pass
    return vec


def manifest_to_path_matrix(manifest: dict[str, Any]) -> np.ndarray:
    """
    Convert the list of candidates into a (N, C) matrix where N is the number
    of candidates and C = |CLASSIFICATIONS|.  The rows are ordered by the
    candidate's discovery timestamp (if present) or by insertion order.
    """
    candidates = manifest.get("vendors", [])
    # try to sort by a timestamp field; fallback to original order
    def key_func(c):
        ts = c.get("timestamp")
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                return utc_now()
        return utc_now()

    candidates = sorted(candidates, key=key_func)

    rows = [classification_one_hot(c.get("classification", "")) for c in candidates]
    return np.stack(rows, axis=0)  # shape (N, C)


# ----------------------------------------------------------------------
# Parent‑B (signature / KAN) utilities (adapted)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform of a path.
    Input shape (T, d); output shape (2*T‑1, 2*d).
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """First‑order signature (increment from start to end)."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path: np.ndarray) -> np.ndarray:
    """Second‑order signature (matrix of iterated integrals)."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    return running.T @ increments               # (d, d)


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate B‑spline basis functions of degree k‑1 on the knot vector
    derived from `grid`. Returns an (len(x), n_basis) array.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = ((x - t[i]) / denom_l * B[:, i]) if denom_l > 0 else np.zeros(N)
            term_r = ((t[i + order] - x) / denom_r * B[:, i + 1]) if denom_r > 0 else np.zeros(N)
            B_new[:, i] = term_l + term_r
        B = B_new
    return B


def kan_layer(
    x: np.ndarray,
    spline_weights: np.ndarray,
    grid: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """
    Apply a single KAN layer: evaluate spline basis on `x`, then linearly
    combine with `spline_weights`.  Shapes:
        x                : (batch, n_in)
        spline_weights   : (n_out, n_in, n_basis)
        grid             : (n_grid,)
    Returns:
        y                : (batch, n_out)
    """
    x = np.asarray(x, dtype=np.float64)
    spline_weights = np.asarray(spline_weights, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    batch, n_in = x.shape
    n_out, n_in_w, n_basis = spline_weights.shape
    assert n_in == n_in_w, f"input dim mismatch: {n_in} vs {n_in_w}"
    expected_n_basis = len(grid) + k - 2
    assert n_basis == expected_n_basis, f"basis count mismatch: {n_basis} vs {expected_n_basis}"

    # evaluate basis for each input dimension independently and stack
    # result shape: (batch, n_in, n_basis)
    basis_per_dim = np.stack([bspline_basis(x[:, dim], grid, k) for dim in range(n_in)], axis=1)

    # contract: (batch, n_in, n_basis) • (n_out, n_in, n_basis) -> (batch, n_out)
    y = np.einsum("bin,oin->bo", basis_per_dim, spline_weights)
    return y


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def audit_signature(manifest_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a manifest, convert classifications to a path matrix, and compute
    first‑ and second‑order signatures of the resulting path.
    Returns:
        level1 : (C,)   – increment over the whole candidate list
        level2 : (C, C) – iterated integral matrix
    """
    manifest = load_manifest(manifest_path)
    path_matrix = manifest_to_path_matrix(manifest)          # (N, C)
    # Apply lead‑lag to capture piecewise constant behaviour
    ll_path = lead_lag_transform(path_matrix)                # (2N‑1, 2C)
    level1 = signature_level1(ll_path)                       # (2C,)
    level2 = signature_level2(ll_path)                       # (2C, 2C)
    return level1, level2


def decreasing_pruning_schedule(length: int, decay: float = 0.9) -> np.ndarray:
    """
    Generate a monotone decreasing schedule of length `length` using an
    exponential decay factor.  The schedule is normalised to sum to 1.
    """
    schedule = np.array([decay ** i for i in range(length)], dtype=float)
    return schedule / schedule.sum()


def hybrid_prune(manifest_path: Path, keep_ratio: float = 0.5) -> List[dict[str, Any]]:
    """
    End‑to‑end hybrid pruning:
      1. Compute signatures of the audit path.
      2. Build a spline‑based KAN layer whose weights encode a decreasing
         pruning schedule.
      3. Apply the layer to the level‑1 signature (treated as a batch of size 1)
         to obtain a scalar pruning score for each classification dimension.
      4. Rank candidates by the dot product of their one‑hot vector with the
         score vector and keep the top `keep_ratio` fraction.
    Returns the list of kept candidate dictionaries.
    """
    # ----- step 1: signatures -----
    level1, _ = audit_signature(manifest_path)               # (2C,)

    # ----- step 2: construct KAN weights -----
    # We treat each dimension of level1 as a separate “input”.
    # The schedule determines how much each dimension contributes.
    C = level1.shape[0]
    grid = np.linspace(0.0, 1.0, num=10)                     # modest grid
    schedule = decreasing_pruning_schedule(C)               # (C,)

    # Create spline_weights of shape (1, C, n_basis) where the single output
    # dimension linearly mixes the basis according to the schedule.
    n_basis = len(grid) + 3 - 2   # k=3 by default
    spline_weights = np.zeros((1, C, n_basis), dtype=float)

    # For each input dimension, set the weight of the first basis function to the
    # schedule value; remaining basis functions stay zero (producing a simple
    # decreasing linear map).
    spline_weights[0, np.arange(C), 0] = schedule

    # ----- step 3: apply KAN layer -----
    # Reshape level1 to (batch=1, n_in=C)
    level1_batch = level1.reshape(1, -1)
    scores = kan_layer(level1_batch, spline_weights, grid)  # (1, 1)
    score_vec = scores.ravel()                               # (1,)

    # ----- step 4: rank candidates -----
    manifest = load_manifest(manifest_path)
    candidates = manifest.get("vendors", [])
    # Compute a scalar relevance = one_hot · score_vec (broadcast)
    relevance = []
    for cand in candidates:
        vec = classification_one_hot(cand.get("classification", ""))
        relevance.append(vec @ score_vec)

    # Pair candidates with relevance and select top fraction
    paired = list(zip(candidates, relevance))
    paired.sort(key=lambda x: x[1], reverse=True)
    keep_n = max(1, int(len(paired) * keep_ratio))
    kept = [c for c, _ in paired[:keep_n]]
    return kept


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic manifest in a temporary location
    tmp_path = Path("synthetic_manifest.json")
    synthetic = {
        "vendors": [
            {
                "candidate_key": "lensA",
                "classification": "usable_now",
                "timestamp": utc_now(),
            },
            {
                "candidate_key": "lensB",
                "classification": "research_only",
                "timestamp": utc_now(),
            },
            {
                "candidate_key": "lensC",
                "classification": "needs_conversion",
                "timestamp": utc_now(),
            },
            {
                "candidate_key": "lensD",
                "classification": "unsupported",
                "timestamp": utc_now(),
            },
        ]
    }
    tmp_path.write_text(json.dumps(synthetic, indent=2), encoding="utf-8")

    # Run hybrid pruning
    kept = hybrid_prune(tmp_path, keep_ratio=0.5)
    print("Kept candidates:")
    for c in kept:
        print(f" - {c['candidate_key']} ({c['classification']})")

    # Clean up
    tmp_path.unlink()
    sys.exit(0)