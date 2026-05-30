# DARWIN HAMMER — match 5580, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1321_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m1969_s0.py (gen4)
# born: 2026-05-30T00:02:59Z

"""Hybrid Fusion of Algorithm A (weekday weighting, B‑spline, SSIM) and Algorithm B (TTT linear compression and RBF surrogate).

Mathematical bridge:
- The TTT weight matrix **W** compresses the raw input vector **x** (Algorithm B).
- The compressed vector is projected onto a B‑spline basis **B(x;grid)** (Algorithm A) to obtain a smooth feature representation.
- An RBF surrogate model evaluates these features, producing a prediction **ŷ** (Algorithm B).
- The weekday weight vector **w_dow** (Algorithm A) modulates the surrogate output, encoding temporal context.
- Finally, the Structural Similarity Index (SSIM) quantifies the agreement between the original input **x** and the reconstructed signal **x̂ = Wᵀ ŷ**, closing the loop between the two parent topologies.

The module implements this unified pipeline with three core functions and a simple smoke test.
"""

import sys
import random
import math
import json
from pathlib import Path
from datetime import date, datetime, timezone
from dataclasses import dataclass
from typing import List, Any

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index 0‑6 where 0 = Sunday."""
    return (date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
    """Cyclic sinusoidal weighting of *groups* based on day‑of‑week."""
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3, weights: np.ndarray | None = None) -> np.ndarray:
    """Cubic (or order‑k) B‑spline basis matrix evaluated at points *x*."""
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # knot vector with clamped ends
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    # initial piecewise constant functions
    B = np.zeros((x.size, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    if weights is not None:
        B *= weights

    # Cox‑de Boor recursion
    for order in range(2, k + 1):
        B_new = np.zeros((x.size, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = np.where(denom_l != 0,
                              (x - t[i]) / denom_l * B[:, i],
                              0.0)
            term_r = np.where(denom_r != 0,
                              (t[i + order] - x) / denom_r * B[:, i + 1],
                              0.0)
            B_new[:, i] = term_l + term_r
        B = B_new

    return B


def ssim(x: np.ndarray, y: np.ndarray, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> float:
    """Simplified 1‑D Structural Similarity Index."""
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_context(text: str | None) -> dict[str, Any]:
    """Parse optional JSON context string."""
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value


def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize TTT‑Linear weight matrix."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, targ: np.ndarray) -> float:
    """Mean‑squared error of a linear TTT mapping."""
    return np.mean((W @ x - targ) ** 2)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial basis Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass
class RBFSurrogate:
    centers: List[List[float]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: np.ndarray) -> float:
        """RBF surrogate prediction for a single feature vector."""
        if len(self.centers) != len(self.weights):
            raise ValueError("centers and weights must have the same length")
        total = 0.0
        for c, w in zip(self.centers, self.weights):
            r = euclidean(c, x.tolist())
            total += w * gaussian(r, self.epsilon)
        return total


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def hybrid_feature_extraction(raw_input: np.ndarray, W_ttt: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    1. Compress raw_input with TTT matrix W_ttt  (Algorithm B).
    2. Project the compressed vector onto a B‑spline basis defined by *grid* (Algorithm A).
    Returns the resulting feature matrix (N_samples × N_basis).
    """
    # Step 1 – linear compression
    compressed = W_ttt @ raw_input  # shape (d_out,)

    # Step 2 – B‑spline expansion (treat each component as a separate sample)
    # For simplicity we treat the compressed vector elements as evaluation points.
    return bspline_basis(compressed, grid)


def hybrid_surrogate_evaluation(features: np.ndarray, surrogate: RBFSurrogate) -> np.ndarray:
    """
    Evaluate the RBF surrogate on each row of *features*.
    Returns a column vector of predictions.
    """
    preds = np.apply_along_axis(surrogate.predict, 1, features)
    return preds[:, np.newaxis]  # shape (N_samples, 1)


def hybrid_predict(
    date_tuple: tuple[int, int, int],
    groups: List[str],
    raw_input: np.ndarray,
    W_ttt: np.ndarray,
    surrogate: RBFSurrogate,
    grid: np.ndarray,
) -> dict[str, Any]:
    """
    End‑to‑end hybrid inference:

    * Compute day‑of‑week weight vector w_dow.
    * Extract B‑spline features from TTT‑compressed input.
    * Obtain surrogate prediction ŷ.
    * Modulate ŷ by w_dow (temporal gating).
    * Reconstruct an approximation of the original input via W_tttᵀ.
    * Report SSIM between raw_input and reconstruction.

    Returns a dictionary with intermediate results for inspection.
    """
    year, month, day = date_tuple
    dow = doomsday(year, month, day)

    # Temporal weighting
    w_dow = weekday_weight_vector(groups, dow)  # shape (n_groups,)

    # Feature extraction
    features = hybrid_feature_extraction(raw_input, W_ttt, grid)  # (d_out, n_basis)

    # Surrogate prediction (scalar per basis row)
    y_hat = hybrid_surrogate_evaluation(features, surrogate)  # (d_out, 1)

    # Temporal modulation – broadcast w_dow over the prediction dimension
    if w_dow.size != y_hat.shape[0]:
        # If dimensions differ, repeat or truncate to match
        if w_dow.size < y_hat.shape[0]:
            w_mod = np.pad(w_dow, (0, y_hat.shape[0] - w_dow.size), mode='wrap')
        else:
            w_mod = w_dow[: y_hat.shape[0]]
    else:
        w_mod = w_dow

    y_mod = y_hat.ravel() * w_mod

    # Reconstruct approximation of the original input
    reconstruction = W_ttt.T @ y_mod

    # Similarity metric
    similarity = ssim(raw_input, reconstruction)

    return {
        "dow": dow,
        "weekday_weights": w_dow,
        "features": features,
        "surrogate_output": y_hat.ravel(),
        "temporal_modulated": y_mod,
        "reconstruction": reconstruction,
        "ssim": similarity,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic random seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Dummy temporal context
    groups = ["alpha", "beta", "gamma", "delta", "epsilon"]
    today = (2026, 5, 30)  # year, month, day

    # Raw input vector (e.g., sensor readings)
    d_in = 7
    raw = np.random.rand(d_in)

    # TTT matrix (compress to d_out = 5)
    d_out = 5
    W = init_ttt(d_in, d_out, scale=0.05, seed=42)

    # B‑spline grid (uniform over compressed range)
    grid = np.linspace(-1.0, 1.0, num=10)

    # RBF surrogate – random centers in the feature space, random weights
    n_centers = 8
    centers = [np.random.uniform(-1, 1, size=W.shape[0]).tolist() for _ in range(n_centers)]
    weights = np.random.randn(n_centers).tolist()
    surrogate = RBFSurrogate(centers=centers, weights=weights, epsilon=1.5)

    # Run the hybrid pipeline
    result = hybrid_predict(
        date_tuple=today,
        groups=groups,
        raw_input=raw,
        W_ttt=W,
        surrogate=surrogate,
        grid=grid,
    )

    # Simple sanity check output
    print("Day of week (0=Sun):", result["dow"])
    print("SSIM between input and reconstruction:", result["ssim"])
    print("Temporal‑modulated surrogate output:", result["temporal_modulated"])
    print("Reconstruction vector (first 3 values):", result["reconstruction"][:3])