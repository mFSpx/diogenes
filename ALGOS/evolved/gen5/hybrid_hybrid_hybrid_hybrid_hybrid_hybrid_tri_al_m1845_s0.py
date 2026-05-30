# DARWIN HAMMER — match 1845, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m498_s0.py (gen4)
# parent_b: hybrid_hybrid_tri_algo_cond_hybrid_hard_truth_ma_m755_s0.py (gen3)
# born: 2026-05-29T23:39:23Z

"""
Hybrid Algorithm: lead‑lag path signatures weighted by entropy‑based stylometry.

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_path_signature_m56_s0 (lead‑lag transform + B‑spline basis)
- hybrid_hybrid_tri_algo_cond_hybrid_hard_truth_ma_m755_s0 (entropy scoring, stylometry weighting, edge‑weight concepts)

Mathematical Bridge:
The lead‑lag transformed path is projected onto a B‑spline basis, yielding a feature matrix **Φ** ∈ ℝ^{N×M}.
From the textual side we compute a normalized Shannon entropy **H∈[0,1]** for each byte block; this entropy vector is interpreted as a diagonal weighting matrix **W = diag(H)**.
The hybrid representation is the bilinear form **Z = Φᵀ W Φ**, i.e. the path features are weighted by the information content of the accompanying text.
The resulting matrix is then compressed with a Count‑Min Sketch to obtain a low‑dimensional, query‑able summary.
A final decision object combines the sketch estimate with entropy‑derived confidence scores.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ConduitDecision:
    """Immutable container for a decision made by the hybrid system."""
    action: str
    confidence_gap: float
    epsilon: float
    signal_score: float
    noise_score: float
    dormancy_probability: float
    recovery_priority: float
    reason: str

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    Input shape: (T, d)
    Output shape: (2T‑1, 2d)
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dimension)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate B‑spline basis functions of order k at positions x.
    Returns a matrix B where B[i, j] = B_j(x_i).
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Knot vector with clamped ends
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    N = len(x)
    K = len(t) - 1
    B = np.zeros((N, K), dtype=np.float64)

    # Order‑1 (piecewise constant) basis
    for i in range(K):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # Recurrence for higher orders
    for order in range(2, k + 1):
        B_new = np.zeros((N, K - order + 1), dtype=np.float64)
        for i in range(K - order + 1):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = ((x - t[i]) / denom_l) * B[:, i] if denom_l > 0 else 0.0
            term_r = ((t[i + order] - x) / denom_r) * B[:, i + 1] if denom_r > 0 else 0.0

            B_new[:, i] = term_l + term_r
        B = B_new
    return B

# ----------------------------------------------------------------------
# Parent B components (entropy utilities)
# ----------------------------------------------------------------------
def shannon_entropy(sequence: bytes) -> float:
    """Classic Shannon entropy (bits) for a byte sequence."""
    if not sequence:
        return 0.0
    freq = np.bincount(np.frombuffer(sequence, dtype=np.uint8), minlength=256)
    prob = freq[freq > 0] / len(sequence)
    return -np.sum(prob * np.log2(prob))

def normalized_byte_entropy(data: bytes, sample: int = 8192) -> float:
    """Normalized Shannon entropy in [0,1] (8 bits max)."""
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(chunk) / 8.0

def entropy_weight_vector(text: bytes, dim: int) -> np.ndarray:
    """
    Produce a length‑dim weight vector from text entropy.
    The single scalar entropy is broadcast, but we add a small random jitter
    to avoid singularities when dim > 1.
    """
    base = normalized_byte_entropy(text)
    rng = np.random.default_rng(seed=42)
    jitter = rng.uniform(-0.01, 0.01, size=dim)
    w = np.clip(base + jitter, 0.0, 1.0)
    return w

# ----------------------------------------------------------------------
# Count‑Min Sketch (simple implementation)
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    Very lightweight Count‑Min Sketch with d hash functions and width w.
    """
    def __init__(self, depth: int = 5, width: int = 2 ** 12):
        self.depth = depth
        self.width = width
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = np.random.default_rng(seed=123)
        # Generate pairwise‑independent hash parameters a, b for each row
        self._hash_a = rng.integers(1, 2 ** 31 - 1, size=depth, dtype=np.int64)
        self._hash_b = rng.integers(0, 2 ** 31 - 1, size=depth, dtype=np.int64)
        self._prime = 2 ** 31 - 1

    def _hash(self, x: int, row: int) -> int:
        return ((self._hash_a[row] * x + self._hash_b[row]) % self._prime) % self.width

    def update(self, key: int, increment: int = 1) -> None:
        for r in range(self.depth):
            idx = self._hash(key, r)
            self.tables[r, idx] += increment

    def estimate(self, key: int) -> int:
        estimates = [self.tables[r, self._hash(key, r)] for r in range(self.depth)]
        return min(estimates)

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_feature_matrix(path: np.ndarray, grid_points: int = 20) -> np.ndarray:
    """
    Apply lead‑lag transform, then project onto a cubic B‑spline basis.
    Returns Φ ∈ ℝ^{(2T‑1) × (grid_points + k - 2)}.
    """
    lead_lag = lead_lag_transform(path)                      # (2T‑1, 2d)
    # Use the first coordinate of each 2d vector as the scalar argument for splines
    scalar_series = lead_lag[:, 0]
    grid = np.linspace(scalar_series.min(), scalar_series.max(), grid_points)
    Φ = bspline_basis(scalar_series, grid, k=3)             # (2T‑1, M)
    return Φ

def entropy_weighted_bilinear(Φ: np.ndarray, text: bytes) -> np.ndarray:
    """
    Compute Z = Φᵀ W Φ where W = diag(w) and w is derived from text entropy.
    The result Z ∈ ℝ^{M×M} captures interaction of path features modulated by
    information content of the accompanying text.
    """
    M = Φ.shape[1]
    w = entropy_weight_vector(text, M)
    W = np.diag(w)
    Z = Φ.T @ W @ Φ
    return Z

def sketch_from_matrix(Z: np.ndarray, cms: CountMinSketch) -> List[Tuple[int, int]]:
    """
    Feed each element of Z (rounded to int) into the Count‑Min Sketch.
    Returns a list of (key, estimate) pairs for the non‑zero entries.
    """
    rows, cols = Z.shape
    for i in range(rows):
        for j in range(cols):
            val = int(round(Z[i, j]))
            if val != 0:
                # Combine i and j into a single integer key (Cantor pairing)
                key = (i + j) * (i + j + 1) // 2 + j
                cms.update(key, val)
    # Extract estimates for the same keys
    result = []
    for i in range(rows):
        for j in range(cols):
            key = (i + j) * (i + j + 1) // 2 + j
            est = cms.estimate(key)
            if est > 0:
                result.append((key, est))
    return result

def hybrid_decision(path: np.ndarray, text: bytes) -> ConduitDecision:
    """
    End‑to‑end hybrid pipeline producing a ConduitDecision.
    """
    Φ = hybrid_feature_matrix(path)
    Z = entropy_weighted_bilinear(Φ, text)

    cms = CountMinSketch(depth=4, width=4096)
    sketch_entries = sketch_from_matrix(Z, cms)

    # Simple scoring: signal = sum of sketch counts, noise = number of zero entries
    signal_score = sum(cnt for _, cnt in sketch_entries)
    total_elements = Z.size
    nonzero = len(sketch_entries)
    noise_score = total_elements - nonzero

    # Confidence derived from normalized entropy
    entropy_norm = normalized_byte_entropy(text)
    confidence_gap = entropy_norm * (signal_score / (signal_score + noise_score + 1e-9))

    # Assemble decision fields
    return ConduitDecision(
        action="accept" if confidence_gap > 0.5 else "reject",
        confidence_gap=confidence_gap,
        epsilon=1.0 - entropy_norm,
        signal_score=float(signal_score),
        noise_score=float(noise_score),
        dormancy_probability=1.0 - entropy_norm,
        recovery_priority=entropy_norm,
        reason=f"Hybrid score={signal_score:.1f}, entropy={entropy_norm:.3f}"
    )

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic time series: 8 timesteps, 2‑dimensional
    np.random.seed(0)
    synthetic_path = np.cumsum(np.random.randn(8, 2), axis=0)

    # Synthetic text: random ASCII bytes
    synthetic_text = bytes([random.randint(32, 126) for _ in range(1024)])

    decision = hybrid_decision(synthetic_path, synthetic_text)
    print(decision)