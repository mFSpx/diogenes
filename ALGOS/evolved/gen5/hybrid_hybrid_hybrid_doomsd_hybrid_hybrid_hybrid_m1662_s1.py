# DARWIN HAMMER — match 1662, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_hybrid_bandit_m121_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_voronoi_parti_m53_s1.py (gen4)
# born: 2026-05-29T23:38:13Z

"""Hybrid algorithm merging Doomsday‑Calendar Gini analysis (Parent A) with
Bayesian‑Voronoi similarity scoring (Parent B).

Mathematical bridge
-------------------
* Parent A yields a 7‑element weekday count vector **c** and builds the
  weighted‑difference matrix **W = outer(c, c) * |i‑j|**.  The flattened,
  L2‑normalised version **x = vec(W) / ‖vec(W)‖** is a high‑dimensional context.
* Parent B provides a Structural‑Similarity (SSIM) function and a prototype
  vector **p** (size 5).  We compute **s = SSIM(x[:5], p)** – the similarity of the
  first five context components with the prototype.
* The final reward combines the uniformity reward from Parent A,
  **r₁ = 1 – Gini(c)**, with the similarity reward from Parent B,
  **r₂ = (1 + s)/2**.  The hybrid reward is **R = r₁·r₂**.
* A simple Beta‑Bernoulli Bayesian posterior is updated with **R** and then
  used inside a LinUCB‑style action selector whose confidence term is scaled
  by the context norm ‖x‖.

The module therefore fuses matrix‑based statistics, similarity metrics and
Bayesian updating into a single learning loop."""
from __future__ import annotations

import datetime as dt
import math
import random
import sys
from pathlib import Path
from typing import Iterable, Tuple, Union, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Doomsday calendar utilities (adapted)
# ----------------------------------------------------------------------


def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """Return weekday numbers (Mon=0 … Sun=6) for vectorised dates."""
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            dt.datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # shift so that Sunday becomes 0, Monday 1, … Saturday 6 (as in parent)
    return (py_weekday + 1) % 7


def weekday_counts(
    dates: Iterable[Union[dt.date, Tuple[int, int, int]]],
) -> np.ndarray:
    """Count occurrences of each weekday for an iterable of dates."""
    counts = np.zeros(7, dtype=int)
    for d in dates:
        if isinstance(d, dt.date):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        w = (dt.date(y, m, day).weekday() + 1) % 7
        counts[w] += 1
    return counts


def gini_coefficient(vec: np.ndarray) -> float:
    """Gini coefficient of a non‑negative vector."""
    if vec.size == 0:
        return 0.0
    if np.any(vec < 0):
        raise ValueError("Gini undefined for negative values")
    sorted_vec = np.sort(vec.astype(float))
    n = vec.size
    cumulative = np.cumsum(sorted_vec)
    sum_y = cumulative[-1]
    if sum_y == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_y) / n
    return float(gini)


def weighted_difference_matrix(counts: np.ndarray) -> np.ndarray:
    """W = outer(c, c) * |i‑j|  (7×7 matrix)."""
    c = counts.astype(float)
    outer = np.outer(c, c)
    idx = np.arange(7)
    diff = np.abs(idx[:, None] - idx[None, :])
    return outer * diff


def flatten_normalise_context(W: np.ndarray) -> np.ndarray:
    """Flatten W and L2‑normalise to obtain a unit‑norm context vector."""
    vec = W.ravel().astype(float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ----------------------------------------------------------------------
# Parent B – Bayesian‑Voronoi utilities (adapted)
# ----------------------------------------------------------------------


PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index between two 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


def hybrid_score(packet: Dict[str, List[float]]) -> float:
    """Score a packet by cosine similarity to the prototype vector."""
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    vec = np.asarray(payload, dtype=np.float64)
    if vec.size < PROTOTYPE_VECTOR.size:
        vec = np.pad(vec, (0, PROTOTYPE_VECTOR.size - vec.size))
    elif vec.size > PROTOTYPE_VECTOR.size:
        vec = vec[: PROTOTYPE_VECTOR.size]
    dot = float(np.dot(vec, PROTOTYPE_VECTOR))
    norm_prod = float(np.linalg.norm(vec) * np.linalg.norm(PROTOTYPE_VECTOR))
    return dot / norm_prod if norm_prod != 0 else 0.0


class BetaPosterior:
    """Simple Beta‑Bernoulli posterior updated with continuous rewards."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def update(self, reward: float) -> None:
        """Treat reward ∈[0,1] as fractional success."""
        self.alpha += reward
        self.beta += 1.0 - reward

    def mean(self) -> float:
        """Posterior mean."""
        total = self.alpha + self.beta
        return self.alpha / total if total != 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid core functions (fusion of A & B)
# ----------------------------------------------------------------------


def hybrid_reward(counts: np.ndarray) -> float:
    """
    Compute the fused reward:

    * Uniformity part   r₁ = 1 – Gini(counts)
    * Similarity part   r₂ = (1 + SSIM(context[:5], prototype)) / 2
    * Combined reward   R  = r₁·r₂
    """
    # Uniformity reward
    gini = gini_coefficient(counts)
    r1 = 1.0 - gini

    # Context from matrix W
    W = weighted_difference_matrix(counts)
    ctx = flatten_normalise_context(W)

    # SSIM on the first five components (same length as prototype)
    ssim_val = compute_ssim(ctx[: PROTOTYPE_VECTOR.size].tolist(),
                            PROTOTYPE_VECTOR.tolist(),
                            dynamic_range=1.0)

    r2 = (1.0 + ssim_val) / 2.0
    return float(r1 * r2)


def linucb_estimate(context: np.ndarray, posterior: BetaPosterior, alpha: float = 1.0) -> float:
    """
    LinUCB‑style estimate using the Bayesian mean as the base value and an
    exploration term proportional to sqrt(||context||).
    """
    base = posterior.mean()
    exploration = alpha * math.sqrt(np.linalg.norm(context))
    return base + exploration


def hybrid_step(dates: Iterable[Union[dt.date, Tuple[int, int, int]]],
                posterior: BetaPosterior) -> Tuple[float, float]:
    """
    Perform one learning step:

    1. Convert dates → weekday counts.
    2. Compute hybrid reward R.
    3. Update Bayesian posterior with R.
    4. Build context vector from the weighted‑difference matrix.
    5. Return (R, LinUCB estimate).
    """
    counts = weekday_counts(dates)
    R = hybrid_reward(counts)
    posterior.update(R)

    W = weighted_difference_matrix(counts)
    ctx = flatten_normalise_context(W)

    estimate = linucb_estimate(ctx, posterior, alpha=0.5)
    return R, estimate


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Generate a random set of dates within a year
    random.seed(42)
    base = dt.date(2023, 1, 1)
    dates = []
    for _ in range(50):
        offset = random.randint(0, 364)
        dates.append(base + dt.timedelta(days=offset))

    posterior = BetaPosterior(alpha=1.0, beta=1.0)

    # Run a few hybrid steps
    for i in range(5):
        reward, est = hybrid_step(dates, posterior)
        print(f"Step {i+1}: reward={reward:.4f}, LinUCB estimate={est:.4f}, posterior mean={posterior.mean():.4f}")

    # Demonstrate hybrid_score on a dummy packet
    packet = {"payload": [0.25, 0.55, 0.33, 0.68, 0.12]}
    print("Hybrid packet score:", hybrid_score(packet))