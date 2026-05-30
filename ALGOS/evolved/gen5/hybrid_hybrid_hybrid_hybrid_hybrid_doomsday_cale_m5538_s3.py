# DARWIN HAMMER — match 5538, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_korpus_hybrid_hybrid_hybrid_m1925_s1.py (gen4)
# parent_b: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s2.py (gen3)
# born: 2026-05-30T00:02:35Z

"""Hybrid module combining:
- Parent A: minhash text representation, count‑min sketch weighting, Voronoi region partitioning.
- Parent B: doomsday‑derived learning‑rate seed for NLMS weight adaptation and BIC evaluation.

Mathematical bridge:
The count‑min sketch table (a frequency vector) is treated as the NLMS input vector **x**.
The minhash signature of a text provides the NLMS weight vector **w**.
The NLMS learning rate **μ** is modulated by the Doomsday algorithm output for a given date,
thereby injecting a cyclic temporal bias into the adaptation of the sketch‑derived frequencies.
The adapted weights are finally used to weight Voronoi region edges, producing a unified
hybrid representation that blends textual compactness, streaming frequency estimation,
spatial partitioning, and temporally‑aware adaptive learning.
"""

import numpy as np
import math
import random
import sys
import pathlib
import re
import hashlib
from collections import defaultdict
from datetime import date

# -------------------- Parent A components --------------------

def minhash_for_text(text: str, k: int = 64) -> list[int]:
    """Compact MinHash signature of a string."""
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    shingles = [text[i:i + 5] for i in range(len(text) - 4)]
    signature = np.full(k, 2 ** 31 - 1, dtype=np.int64)
    for s in shingles:
        h = hash(s)
        idx = h % k
        signature[idx] = min(signature[idx], h % 1000000)
    return signature.tolist()


def count_min_sketch(items, width: int = 64, depth: int = 4) -> list[list[int]]:
    """Streaming frequency estimator."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16)
            col = h % width
            table[d][col] += 1
    return table


def assign_points_to_regions(
    points: list[tuple[float, float]],
    seeds: list[tuple[float, float]],
) -> dict[int, list[tuple[float, float]]]:
    """Voronoi assignment of 2‑D points to nearest seed."""
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest_point(p, seeds)].append(p)
    return regions


def nearest_point(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError("seeds required")
    return min(
        range(len(seeds)),
        key=lambda i: (distance(point, seeds[i]), i),
    )


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# -------------------- Parent B components --------------------

def doomsday(year: int, month: int, day: int) -> int:
    """Doomsday algorithm – weekday index (0=Monday … 6=Sunday)."""
    return (date(year, month, day).weekday() + 1) % 7


def bayesian_information_criterion(
    log_likelihood: float, n_params: int, n_samples: int
) -> float:
    """Standard BIC = -2*LL + p*log(N)."""
    return -2 * log_likelihood + n_params * math.log(n_samples)


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """
    Normalized LMS weight update.

    w_{new} = w + (mu / (||x||^2 + eps)) * e * x
    where e = target - w·x
    """
    norm_sq = float(x @ x) + eps
    error = target - nlms_predict(weights, x)
    step = (mu / norm_sq) * error
    new_weights = weights + step * x
    return new_weights, error


# -------------------- Hybrid operations --------------------

def flatten_sketch(sketch: list[list[int]]) -> np.ndarray:
    """Convert 2‑D sketch table into a 1‑D float vector."""
    return np.array([float(v) for row in sketch for v in row], dtype=np.float64)


def compute_region_weights(
    points: list[tuple[float, float]],
    seeds: list[tuple[float, float]],
    sketch: list[list[int]],
) -> dict[int, float]:
    """
    For each Voronoi region, compute a weight as the dot product between the
    region's centroid vector (x, y) and the flattened sketch vector (truncated
    to length 2). This demonstrates a concrete mathematical coupling.
    """
    regions = assign_points_to_regions(points, seeds)
    flat = flatten_sketch(sketch)
    # Use only first two components of the sketch as a 2‑D direction vector
    direction = flat[:2] if flat.size >= 2 else np.array([1.0, 1.0])
    region_weights = {}
    for idx, pts in regions.items():
        if not pts:
            region_weights[idx] = 0.0
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        region_weights[idx] = float(np.dot(direction, np.array([cx, cy])))
    return region_weights


def hybrid_nlms_adapt(
    signature: list[int],
    sketch: list[list[int]],
    target: float,
    year: int,
    month: int,
    day: int,
) -> tuple[np.ndarray, float]:
    """
    Treat the MinHash signature as initial NLMS weights and the flattened
    sketch as the input vector. The learning rate μ is derived from the
    Doomsday weekday (cyclical factor in [0,1]).
    """
    w = np.array(signature, dtype=np.float64)
    x = flatten_sketch(sketch)
    # Ensure matching dimensions
    if w.size != x.size:
        # Truncate or pad the shorter vector
        min_len = min(w.size, x.size)
        w = w[:min_len]
        x = x[:min_len]
    # Doomsday returns 0‑6; map to μ ∈ [0.05, 0.5]
    weekday = doomsday(year, month, day)
    mu = 0.05 + (weekday / 6) * 0.45
    new_w, err = nlms_update(w, x, target, mu=mu)
    return new_w, err


def hybrid_process(
    text: str,
    items,
    points: list[tuple[float, float]],
    seeds: list[tuple[float, float]],
    target: float,
    year: int,
    month: int,
    day: int,
) -> dict:
    """
    End‑to‑end hybrid pipeline:
    1. MinHash the text → signature.
    2. Build Count‑Min sketch from items → sketch.
    3. Adapt signature with NLMS using Doomsday‑scaled μ.
    4. Compute Voronoi region weights using the adapted signature and sketch.
    5. Return a dictionary summarising the steps plus a BIC score for a dummy model.
    """
    signature = minhash_for_text(text)
    sketch = count_min_sketch(items)

    adapted_weights, nlms_error = hybrid_nlms_adapt(
        signature, sketch, target, year, month, day
    )

    region_weights = compute_region_weights(points, seeds, sketch)

    # Dummy log‑likelihood for illustration (higher is better)
    log_likelihood = -abs(nlms_error)
    bic = bayesian_information_criterion(
        log_likelihood, n_params=adapted_weights.size, n_samples=len(items)
    )

    return {
        "original_signature": signature,
        "adapted_weights": adapted_weights.tolist(),
        "nlms_error": nlms_error,
        "region_weights": region_weights,
        "bic": bic,
    }


# -------------------- Smoke test --------------------

if __name__ == "__main__":
    # Sample data
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_items = ["apple", "banana", "apple", "cherry", "date", "banana", "apple"]
    sample_points = [(random.random(), random.random()) for _ in range(30)]
    sample_seeds = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    target_value = 42.0
    today = date.today()
    result = hybrid_process(
        text=sample_text,
        items=sample_items,
        points=sample_points,
        seeds=sample_seeds,
        target=target_value,
        year=today.year,
        month=today.month,
        day=today.day,
    )
    print("Hybrid process result summary:")
    for k, v in result.items():
        print(f"{k}: {v}")