# DARWIN HAMMER — match 1420, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m587_s3.py (gen5)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s0.py (gen1)
# born: 2026-05-29T23:36:18Z

"""
Hybrid Algorithm: Fisher‑Gini‑Hoeffding Fusion

Parents
-------
* hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m587_s3.py (Algorithm A)
* hybrid_hoeffding_tree_gini_coefficient_m13_s0.py (Algorithm B)

Mathematical Bridge
-------------------
Algorithm A provides a *Fisher information* I(θ) that quantifies the
sensitivity of a Gaussian beam intensity to its angle θ.  Algorithm B
offers a *Gini coefficient* G that measures inequality of a set of
values and a *Hoeffding bound* ε that gives a confidence interval for
the difference of two empirical gains.

The fusion treats the Fisher information of each sketch bucket as a
precision (inverse variance) of a Gaussian prior on a graph edge.
The distribution of these precisions across the sketch is summarised by
the Gini coefficient, yielding a scalar measure of heterogeneity.
Finally, the Hoeffding bound is added to this heterogeneity measure,
producing a *hybrid bound*:

    B_hybrid = G(Fisher‑precisions) + ε_Hoeffding

This bound can be used as a decision criterion (e.g. to split a node in a
streaming decision tree) while simultaneously encoding information
about the underlying energy landscape of the model.

The module below implements three core functions that realise this
fusion.
"""

import math
import random
import hashlib
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def count_min_sketch(items: Iterable[float], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Standard count‑min sketch counting hash collisions."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound used in streaming decision trees."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini inequality coefficient (non‑negative, ignores a leading negative)."""
    xs = sorted((float(x) for x in values))
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        # Drop the first negative entry as prescribed by the parent.
        for i, x in enumerate(xs):
            if x >= 0:
                xs = xs[i:]
                break
        if not xs or sum(xs) == 0:
            return 0.0
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def fisher_sketch_precisions(
    angles: Iterable[float],
    center: float,
    width: float,
    sketch_width: int = 64,
    sketch_depth: int = 4,
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Compute a count‑min sketch of *angles* and the aggregated Fisher
    information per bucket.  The returned ``precisions`` array contains,
    for each depth, the sum of Fisher scores of items that landed in the
    corresponding bucket.
    """
    # Build the sketch
    sketch = count_min_sketch(angles, width=sketch_width, depth=sketch_depth)

    # Aggregate Fisher scores per bucket
    precisions = np.zeros((sketch_depth, sketch_width), dtype=np.float64)
    for theta in angles:
        f = fisher_score(theta, center, width)
        for d in range(sketch_depth):
            idx = int(hashlib.sha256(f"{d}:{theta}".encode()).hexdigest(), 16) % sketch_width
            precisions[d, idx] += f
    return sketch, precisions


def hybrid_gini_hoeffding_bound(
    angles: Iterable[float],
    center: float,
    width: float,
    r: float,
    delta: float,
    n: int,
    sketch_width: int = 64,
    sketch_depth: int = 4,
) -> float:
    """
    Compute the hybrid bound B_hybrid = G + ε where

    * G  – Gini coefficient of the flattened Fisher‑precision matrix.
    * ε  – Hoeffding bound for the given (r, delta, n).

    The Fisher precisions act as a surrogate for a Gaussian prior
    precision on graph edges; their inequality is captured by G.
    """
    _, precisions = fisher_sketch_precisions(
        angles, center, width, sketch_width=sketch_width, sketch_depth=sketch_depth
    )
    # Flatten to a 1‑D list for Gini computation
    flat_precisions = precisions.ravel()
    # Gini expects an iterable of non‑negative numbers; very small negatives can appear due to
    # floating‑point noise – clip them.
    flat_precisions = np.clip(flat_precisions, a_min=0.0, a_max=None)
    gini = gini_coefficient(flat_precisions)
    hoeff = hoeffding_bound(r, delta, n)
    return gini + hoeff


def should_hybrid_split(
    angles: Iterable[float],
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    center: float,
    width: float,
) -> Tuple[bool, float]:
    """
    Decision rule for splitting a streaming node.

    * Compute the hybrid bound B_hybrid.
    * If the observed gain gap (best_gain - second_best_gain) exceeds B_hybrid,
      we deem the split statistically significant.

    Returns a tuple (should_split, hybrid_bound).
    """
    gain_gap = best_gain - second_best_gain
    hybrid_bound = hybrid_gini_hoeffding_bound(
        angles, center, width, r, delta, n
    )
    return gain_gap > hybrid_bound, hybrid_bound


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data: random angles drawn from a mixture of Gaussians
    random.seed(42)
    angles = [random.gauss(0.0, 1.0) for _ in range(500)] + [
        random.gauss(3.0, 0.5) for _ in range(300)
    ]

    # Parameters for the Gaussian beam (parent A)
    beam_center = 0.0
    beam_width = 1.5

    # Parameters for Hoeffding bound (parent B)
    r = 0.5          # range of gain differences
    delta = 0.05     # confidence level
    n = len(angles) # number of observations

    # Simulated gains for split candidates
    best_gain = 0.42
    second_best_gain = 0.30

    # Compute hybrid bound and decision
    split, bound = should_hybrid_split(
        angles,
        best_gain,
        second_best_gain,
        r,
        delta,
        n,
        beam_center,
        beam_width,
    )

    print(f"Hybrid bound = {bound:.6f}")
    print(f"Gain gap = {best_gain - second_best_gain:.6f}")
    print(f"Should split? {'YES' if split else 'NO'}")