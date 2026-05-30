# DARWIN HAMMER — match 1420, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m587_s3.py (gen5)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s0.py (gen1)
# born: 2026-05-29T23:36:18Z

"""
Hybrid algorithm merging:

- Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m587_s3.py):
  * Gaussian beam model and Fisher information as a sensitivity/precision measure.
  * Count‑Min Sketch to compress a multiset of Fisher scores.

- Parent B (hybrid_hoeffding_tree_gini_coefficient_m13_s0.py):
  * Hoeffding bound for streaming decision‑tree split confidence.
  * Gini coefficient as an inequality (heterogeneity) measure.

Mathematical bridge:
The Fisher score provides a per‑sample variance‑like quantity.  When many
scores are aggregated in a Count‑Min Sketch the resulting bucket counts form
a discrete distribution.  The Gini coefficient evaluates the inequality of
that distribution, i.e. how concentrated the Fisher information is in a few
buckets.  The Hoeffding bound supplies a statistical confidence term that
depends on the number of observed samples.  By adding the Gini coefficient
to the Hoeffding bound we obtain a *hybrid bound* that measures both
heterogeneity of information (via Gini) and sampling uncertainty
(via Hoeffding).  This bound can be used as a split criterion in a streaming
tree that decides whether to partition the angular domain based on the
observed Fisher information.

The implementation below provides three core functions that realise this
fusion and a tiny smoke test.
"""

import math
import random
import hashlib
import sys
import pathlib
from typing import Iterable, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Parent A components
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


def count_min_sketch(items: Iterable[float], width: int = 64, depth: int = 4) -> np.ndarray:
    """
    Classic Count‑Min Sketch.
    Returns a NumPy array of shape (depth, width) with integer counters.
    """
    table = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16)
            idx = h % width
            table[d, idx] += 1
    return table


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound used in streaming decision trees."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini inequality coefficient."""
    xs = sorted((float(x) for x in values))
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        # Drop leading negatives (as per original implementation)
        for i, x in enumerate(xs):
            if x >= 0:
                xs = xs[i:]
                break
        if not xs or sum(xs) == 0.0:
            return 0.0
    n = len(xs)
    numerator = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return numerator / (n * sum(xs))


def hybrid_bound(values: Iterable[float], r: float, delta: float, n: int) -> float:
    """
    Hybrid bound = Gini coefficient (heterogeneity of sketch counts)
    + Hoeffding bound (statistical confidence).
    """
    gini = gini_coefficient(values)
    eps = hoeffding_bound(r, delta, n)
    return gini + eps


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_fisher_sketch(
    thetas: Iterable[float],
    center: float,
    width: float,
    sketch_width: int = 64,
    sketch_depth: int = 4,
) -> np.ndarray:
    """
    Compute Fisher scores for a stream of angles and compress them with a
    Count‑Min Sketch.
    Returns the sketch matrix.
    """
    scores = (fisher_score(theta, center, width) for theta in thetas)
    return count_min_sketch(scores, width=sketch_width, depth=sketch_depth)


def sketch_gini_over_depth(sketch: np.ndarray) -> List[float]:
    """
    Evaluate the Gini coefficient for each depth (row) of the sketch.
    Each row is a discrete distribution of bucket frequencies.
    """
    return [gini_coefficient(row) for row in sketch]


def hybrid_split_decision(
    sketch: np.ndarray,
    r: float,
    delta: float,
    n_samples: int,
    best_gain: float,
    second_best_gain: float,
) -> Tuple[bool, float]:
    """
    Decide whether to split a node in a streaming tree using the hybrid bound.

    Parameters
    ----------
    sketch : np.ndarray
        Count‑Min Sketch matrix summarising Fisher scores.
    r, delta, n_samples : float, float, int
        Parameters for the Hoeffding bound.
    best_gain, second_best_gain : float
        Information‑gain estimates for the best and runner‑up split candidates.

    Returns
    -------
    should_split : bool
        True if the hybrid bound exceeds the observed gain gap.
    hybrid_eps : float
        The computed hybrid bound value.
    """
    # Collapse the sketch to a single 1‑D distribution (minimum count per bucket)
    # This mimics the Count‑Min over‑estimate correction.
    aggregated = np.min(sketch, axis=0).astype(np.float64)
    hybrid_eps = hybrid_bound(aggregated, r, delta, n_samples)
    gain_gap = best_gain - second_best_gain
    should_split = gain_gap > hybrid_eps
    return should_split, hybrid_eps


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a synthetic stream of angles drawn from two Gaussian beams
    random.seed(42)
    n_total = 10_000
    center1, width1 = 0.0, 0.1
    center2, width2 = 1.0, 0.05
    thetas = (
        random.gauss(center1, width1) if random.random() < 0.6 else random.gauss(center2, width2)
        for _ in range(n_total)
    )

    # 1) Build the Fisher sketch
    sketch = compute_fisher_sketch(
        thetas,
        center=0.5,          # a neutral reference centre for Fisher computation
        width=0.2,
        sketch_width=128,
        sketch_depth=5,
    )

    # 2) Gini per depth (just to demonstrate the function)
    gini_per_depth = sketch_gini_over_depth(sketch)
    print("Gini per sketch depth:", gini_per_depth)

    # 3) Hybrid split decision
    # Dummy gain values (in a real tree they would be computed from class statistics)
    best_gain = 0.12
    second_gain = 0.07
    r = 1.0          # range of the gain (max - min)
    delta = 0.05     # confidence parameter
    n_samples = n_total

    split, eps = hybrid_split_decision(
        sketch,
        r=r,
        delta=delta,
        n_samples=n_samples,
        best_gain=best_gain,
        second_best_gain=second_gain,
    )
    print(f"Hybrid bound (eps): {eps:.6f}")
    print(f"Gain gap: {best_gain - second_gain:.6f}")
    print("Should split node?", split)