# DARWIN HAMMER — match 498, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_hoeffding_tre_m16_s0.py (gen3)
# born: 2026-05-29T23:29:26Z

import numpy as np
import math
import hashlib
import struct
from collections import defaultdict
from typing import Iterable, Tuple, List


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform of a multivariate path.
    For a path of shape (T, d) returns an array of shape (2*T‑1, 2*d)
    where consecutive rows encode the “lead’’ and “lag’’ channels.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dimension)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)

    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])          # lead
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])  # lag
    out[-1] = np.concatenate([path[-1], path[-1]])               # final lead
    return out


def bspline_basis(x: np.ndarray, knots: np.ndarray, order: int = 3) -> np.ndarray:
    """
    Evaluate B‑spline basis functions of given order at points x.
    Returns a matrix B of shape (len(x), n_basis) where each column
    corresponds to one basis function.
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)

    # Augment knot vector with clamped ends
    t = np.concatenate((
        np.full(order, knots[0]),
        knots,
        np.full(order, knots[-1])
    ))
    n_basis = len(knots) + order - 1
    N = x.shape[0]

    # Zeroth order (piecewise constant) basis
    B = np.zeros((N, len(t) - 1), dtype=float)
    for i in range(len(t) - 1):
        B[:, i] = ((x >= t[i]) & (x < t[i + 1])).astype(float)
    # Include right‑most knot
    B[x == t[-1], -1] = 1.0

    # Recursion for higher orders
    for k in range(2, order + 1):
        B_next = np.zeros((N, len(t) - k), dtype=float)
        for i in range(len(t) - k):
            left_den = t[i + k - 1] - t[i]
            right_den = t[i + k] - t[i + 1]

            left = ((x - t[i]) / left_den) * B[:, i] if left_den > 0 else 0.0
            right = ((t[i + k] - x) / right_den) * B[:, i + 1] if right_den > 0 else 0.0
            B_next[:, i] = left + right
        B = B_next

    if B.shape[1] != n_basis:
        raise RuntimeError(
            f"B‑spline basis shape mismatch: got {B.shape[1]}, expected {n_basis}"
        )
    return B


def hybrid_banded_path_signature(
    path: np.ndarray,
    knots: np.ndarray,
    order: int = 3,
    band_width: int = 5
) -> np.ndarray:
    """
    Compute a banded path signature using B‑spline basis functions.
    The path is first windowed with a sliding band of width `band_width`.
    For each time step the signature is the inner product of the
    B‑spline basis evaluated on the window and the window itself.
    Returns a 1‑D array of length `len(path) - band_width + 1`.
    """
    path = np.asarray(path, dtype=float)
    if path.shape[0] < band_width:
        raise ValueError("path length must be at least band_width")
    windows = np.lib.stride_tricks.sliding_window_view(path, (band_width, path.shape[1]))
    windows = windows.reshape(-1, band_width * path.shape[1])  # (n_windows, band_width*d)

    # Normalise each window to [0, 1] for stable spline evaluation
    mins = windows.min(axis=1, keepdims=True)
    maxs = windows.max(axis=1, keepdims=True)
    norm_windows = (windows - mins) / np.where(maxs - mins == 0, 1, maxs - mins)

    B = bspline_basis(norm_windows.ravel(), knots, order=order)
    B = B.reshape(windows.shape[0], -1, B.shape[1])  # (n_windows, band_width*d, n_basis)

    # Weighted sum across the band dimension
    signature = np.einsum("wbd,wbd->w", B, windows[:, :, None] * B).sum(axis=1)
    return signature


def _hash_item(item: np.ndarray, depth: int, seed: int) -> int:
    """
    Deterministic hash of a numeric vector for count‑min sketch.
    The vector is packed as 64‑bit floats and mixed with depth‑specific seed.
    """
    # Ensure contiguous memory layout
    data = np.ascontiguousarray(item, dtype=np.float64).tobytes()
    h = hashlib.blake2b(data, digest_size=8, person=bytes([depth, seed]))
    return int.from_bytes(h.digest(), "little")


def count_min_sketch(
    items: Iterable[np.ndarray],
    width: int = 64,
    depth: int = 4
) -> List[np.ndarray]:
    """
    Classic count‑min sketch. Returns a list of `depth` arrays of length `width`.
    """
    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive integers")
    tables = [np.zeros(width, dtype=np.int64) for _ in range(depth)]

    for item in items:
        vec = np.asarray(item, dtype=np.float64)
        for d in range(depth):
            idx = _hash_item(vec, d, seed=42) % width
            tables[d][idx] += 1
    return tables


def estimate_frequencies(
    sketch: List[np.ndarray]
) -> np.ndarray:
    """
    Return the pointwise minimum across depth tables – the standard
    unbiased estimator for count‑min sketches.
    """
    return np.min(np.stack(sketch, axis=0), axis=0)


def hybrid_hoeffding_tree(
    features: np.ndarray,
    sketch: List[np.ndarray],
    delta: float = 0.1
) -> np.ndarray:
    """
    Build a very shallow Hoeffding decision rule using the sketch
    frequencies as surrogate counts for each feature dimension.
    The rule is:
        predict 1 if estimated count > bound else 0
    where the bound is the Hoeffding inequality for the observed sample size.
    """
    if features.ndim != 1:
        raise ValueError("features must be a 1‑D array")
    n = features.shape[0]
    if n == 0:
        raise ValueError("empty feature vector")

    # Hoeffding bound for a Bernoulli variable in [0,1]
    hoeffding_bound = math.sqrt(math.log(2 / delta) / (2 * n))

    # Map each feature to a sketch cell (same hashing as in sketch construction)
    cell_counts = np.empty_like(features, dtype=np.float64)
    for i, val in enumerate(features):
        # Use the same hashing scheme as count_min_sketch for a single scalar
        idxs = [_hash_item(np.array([val]), d, seed=42) % len(sketch[d]) for d in range(len(sketch))]
        cell_counts[i] = min(sketch[d][idx] for d, idx in enumerate(idxs))

    # Decision: 1 if estimated frequency exceeds bound, else 0
    return (cell_counts / n > hoeffding_bound).astype(int)


def hybrid_pipeline(
    raw_path: np.ndarray,
    knots: np.ndarray,
    order: int = 3,
    band_width: int = 5,
    sketch_width: int = 64,
    sketch_depth: int = 4,
    delta: float = 0.1
) -> np.ndarray:
    """
    End‑to‑end hybrid algorithm:
      1. Lead‑lag transform of the raw path.
      2. Banded path‑signature extraction.
      3. Count‑min sketch of the signature vector.
      4. Hoeffding‑tree‑style binary decision based on sketch frequencies.
    Returns a binary prediction array of the same length as the signature.
    """
    # 1. Lead‑lag encoding
    transformed = lead_lag_transform(raw_path)

    # 2. Signature extraction
    signature = hybrid_banded_path_signature(
        transformed, knots, order=order, band_width=band_width
    )

    # 3. Sketch the signature (treat each scalar as an item)
    sketch = count_min_sketch(
        (np.array([v]) for v in signature), width=sketch_width, depth=sketch_depth
    )

    # 4. Decision rule
    prediction = hybrid_hoeffding_tree(signature, sketch, delta=delta)
    return prediction


if __name__ == "__main__":
    # Simple sanity check
    np.random.seed(0)
    path = np.random.rand(20, 3)               # 20 time steps, 3 dimensions
    knots = np.linspace(0, 1, 8)                # knot vector for splines
    pred = hybrid_pipeline(
        raw_path=path,
        knots=knots,
        order=3,
        band_width=4,
        sketch_width=128,
        sketch_depth=5,
        delta=0.05
    )
    print("Prediction shape:", pred.shape)
    print("Predicted labels:", pred)