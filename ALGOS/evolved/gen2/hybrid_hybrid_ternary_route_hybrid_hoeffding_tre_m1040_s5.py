# DARWIN HAMMER — match 1040, survivor 5
# gen: 2
# parent_a: hybrid_ternary_router_ssim_m1_s2.py (gen1)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s4.py (gen1)
# born: 2026-05-29T23:32:29Z

"""Hybrid SSIM‑Hoeffding‑Gini algorithm.

Parent A (hybrid_ternary_router_ssim_m1_s2) supplies a Structural Similarity
Index (SSIM) that measures the similarity between a numeric payload and a
prototype vector.  Parent B (hybrid_hoeffding_tree_gini_coefficient_m13_s4)
provides a Hoeffding bound for streaming gain estimation and a Gini
coefficient that quantifies inequality in a distribution.

Mathematical bridge:
* The SSIM scores of recent packets form a bounded random variable in [0,1].
* Its range `r = max(ssim) - min(ssim)` (or `1‑mean(ssim)`) is a natural
  choice for the Hoeffding bound’s range parameter.
* The Gini coefficient computed on the same SSIM score set measures how
  uniformly the similarity scores are distributed.
* By feeding the SSIM‑derived range into the Hoeffding bound and coupling the
  bound with the Gini inequality, we obtain a unified split‑decision criterion
  that balances statistical confidence (Hoeffding) with distributional
  heterogeneity (Gini) while being directly driven by content similarity
  (SSIM)."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core primitives from the two parents
# ----------------------------------------------------------------------


def compute_ssim(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """Return the Structural Similarity Index between two equal‑length vectors."""
    if x.shape != y.shape:
        raise ValueError("Input vectors must have the same shape")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x2 = np.var(x)
    sigma_y2 = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(numerator / denominator)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a bounded random variable with range r."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini inequality coefficient for a non‑negative value collection."""
    xs = sorted(float(v) for v in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SplitDecision:
    """Result of a hybrid split evaluation."""
    should_split: bool
    hoeffding_eps: float
    gini: float
    gain_gap: float
    reason: str


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------


def hybrid_ssim_scores(
    payloads: Sequence[Sequence[float]],
    prototype: Sequence[float],
) -> List[float]:
    """
    Compute SSIM scores for a batch of payload vectors against a prototype.
    Returns a list of floats in the interval [0, 1].
    """
    proto_arr = np.asarray(prototype, dtype=np.float64)
    scores: List[float] = []
    for p in payloads:
        vec = np.asarray(p, dtype=np.float64)
        # Pad or truncate to match prototype length
        if vec.size < proto_arr.size:
            vec = np.pad(vec, (0, proto_arr.size - vec.size), mode="constant")
        elif vec.size > proto_arr.size:
            vec = vec[: proto_arr.size]
        scores.append(compute_ssim(vec, proto_arr))
    return scores


def hybrid_hoeffding_gini_bound(
    ssim_scores: Iterable[float],
    delta: float,
    n: int,
) -> Tuple[float, float]:
    """
    Derive a Hoeffding bound using the range of SSIM scores and compute the
    Gini coefficient of the same score set.

    Returns:
        (epsilon, gini) where epsilon is the Hoeffding bound and gini the
        inequality measure.
    """
    scores = list(ssim_scores)
    if not scores:
        raise ValueError("ssim_scores must contain at least one element")
    r = max(scores) - min(scores)  # range within [0,1]
    # Guard against zero range (identical scores) – use a small epsilon.
    r = r if r > 1e-12 else 1e-12
    epsilon = hoeffding_bound(r, delta, n)
    gini = gini_coefficient(scores)
    return epsilon, gini


def hybrid_split_decision(
    ssim_scores: Iterable[float],
    best_gain: float,
    second_best_gain: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
    gini_split_threshold: float = 0.5,
) -> SplitDecision:
    """
    Combine SSIM‑derived Hoeffding bound and Gini coefficient to decide
    whether a streaming split should occur.

    Decision logic:
        * Compute epsilon via Hoeffding bound on the SSIM score range.
        * Compute Gini on the SSIM scores.
        * Split if (gain_gap > epsilon) AND (gini > gini_split_threshold)
          OR if epsilon is below a tie threshold (indicating high confidence).
    """
    epsilon, gini = hybrid_hoeffding_gini_bound(ssim_scores, delta, n)
    gain_gap = best_gain - second_best_gain
    split = (gain_gap > epsilon and gini > gini_split_threshold) or (epsilon < tie_threshold)
    if split:
        reason = "gap_exceeds_bound_and_high_gini" if (gain_gap > epsilon and gini > gini_split_threshold) else "tight_hoeffding_bound"
    else:
        reason = "wait_for_more_data"
    return SplitDecision(split, epsilon, gini, gain_gap, reason)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Prototype vector (same as in parent A)
    PROTOTYPE = [0.2, 0.5, 0.3, 0.7, 0.1]

    # Generate synthetic payloads
    random.seed(42)
    payload_batch = [
        [random.random() for _ in range(5)] for _ in range(20)
    ]

    # Compute SSIM scores
    scores = hybrid_ssim_scores(payload_batch, PROTOTYPE)
    print(f"SSIM scores (first 5): {scores[:5]}")

    # Dummy gain values for split decision
    best_gain = 0.12
    second_best_gain = 0.07
    delta = 0.05
    n = len(scores)

    decision = hybrid_split_decision(
        ssim_scores=scores,
        best_gain=best_gain,
        second_best_gain=second_best_gain,
        delta=delta,
        n=n,
    )

    print(f"Hybrid split decision: {decision}")
    sys.exit(0)