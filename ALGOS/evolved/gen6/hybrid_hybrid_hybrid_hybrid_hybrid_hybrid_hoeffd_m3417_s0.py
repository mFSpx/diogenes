# DARWIN HAMMER — match 3417, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_distri_m145_s2.py (gen4)
# parent_b: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m979_s1.py (gen5)
# born: 2026-05-29T23:49:58Z

"""Hybrid Endpoint‑SSIM‑Leader‑Chelydrid with Hoeffding‑Gini Decision.

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – provides physical morphology descriptors (sphericity, flatness) and
  the notion of an *endpoint circuit breaker* that should fire when a similarity‑based
  distributed leader election becomes too heterogeneous.
* **Parent B** – supplies the Hoeffding bound for statistically‑driven split decisions
  and the Gini coefficient to quantify inequality (here applied to similarity scores).

**Mathematical bridge** – the similarity matrix (computed with an SSIM‑like formula)
between morphology feature vectors is treated as a distribution of “regrets”.  The
Gini coefficient measures the inequality of that distribution; the Hoeffding bound
uses the Gini‑derived range to decide, with a configurable confidence, whether the
heterogeneity exceeds a threshold and therefore triggers the endpoint circuit‑breaker
(split).  The Chelydrid ambush‑strike kinematics primitive uses the sphericity index
as a burst‑strength factor, completing the hybrid loop.

The resulting algorithm, `hybrid_endpoint_ssim_leader_chelydrid`, can be used in
distributed sensor‑fusion or adaptive streaming scenarios where geometric similarity,
statistical confidence, and physical burst dynamics must be jointly considered.
"""

import math
import random
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import List, Iterable, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology and geometric descriptors
# ----------------------------------------------------------------------


class Morphology:
    """Physical object morphology."""

    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All dimensions and mass must be positive")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

    def as_vector(self) -> np.ndarray:
        """Return a normalized feature vector (length, width, height, mass)."""
        vec = np.array([self.length, self.width, self.height, self.mass], dtype=float)
        norm = np.linalg.norm(vec)
        return vec / norm if norm != 0 else vec


def sphericity_index(length: float, width: float, height: float) -> float:
    """Sphericity index = (volume)^(1/3) / longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    volume = length * width * height
    longest = max(length, width, height)
    return volume ** (1.0 / 3.0) / longest


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness index = shortest dimension / longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    shortest = min(length, width, height)
    longest = max(length, width, height)
    return shortest / longest


# ----------------------------------------------------------------------
# Parent B – Hoeffding bound and Gini‑based statistics
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑Gini split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound ε for range r, confidence 1‑δ and sample size n."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini inequality coefficient for a non‑negative iterable."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


# ----------------------------------------------------------------------
# Hybrid building blocks
# ----------------------------------------------------------------------


def ssim_like_similarity(u: np.ndarray, v: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """
    A lightweight SSIM‑like similarity for 1‑D normalized vectors.
    Returns a value in [0, 1] where 1 means identical.
    """
    mu_u, mu_v = u.mean(), v.mean()
    sigma_u2, sigma_v2 = u.var(), v.var()
    sigma_uv = ((u - mu_u) * (v - mu_v)).mean()

    numerator = (2 * mu_u * mu_v + C1) * (2 * sigma_uv + C2)
    denominator = (mu_u ** 2 + mu_v ** 2 + C1) * (sigma_u2 + sigma_v2 + C2)
    return numerator / denominator if denominator != 0 else 0.0


def similarity_matrix(morphologies: List[Morphology]) -> np.ndarray:
    """
    Compute the pairwise SSIM‑like similarity matrix for a list of Morphology objects.
    Diagonal entries are 1.0.
    """
    n = len(morphologies)
    mat = np.eye(n, dtype=float)
    vectors = [m.as_vector() for m in morphologies]

    for i in range(n):
        for j in range(i + 1, n):
            sim = ssim_like_similarity(vectors[i], vectors[j])
            mat[i, j] = mat[j, i] = sim
    return mat


def chelydrid_burst_velocity(base_velocity: float, sphericity: float) -> float:
    """
    Chelydrid ambush‑strike primitive.
    The burst factor is proportional to the sphericity index (more spherical → stronger burst).
    """
    if base_velocity <= 0:
        raise ValueError("base_velocity must be positive")
    burst_factor = 1.0 + 2.0 * sphericity  # arbitrary scaling
    return base_velocity * burst_factor


def hybrid_split_decision(
    morphologies: List[Morphology],
    delta: float = 0.05,
    min_samples: int = 30,
) -> SplitDecision:
    """
    Fuse the SSIM similarity distribution (Parent A) with the Hoeffding‑Gini
    statistical test (Parent B) to decide whether the endpoint circuit breaker
    should fire (i.e., a split/leader change is needed).

    Steps:
    1. Build the similarity matrix and flatten the upper triangle → similarity scores.
    2. Compute the Gini coefficient of these scores (inequality of similarity).
    3. Use the Gini coefficient as the *range* r in the Hoeffding bound.
    4. If ε < gain_gap (here taken as Gini * 0.1) and enough samples have been seen,
       trigger a split.
    """
    if len(morphologies) < 2:
        return SplitDecision(False, 0.0, 0.0, "Insufficient objects for similarity analysis")

    sim_mat = similarity_matrix(morphologies)
    # Extract upper‑triangular (excluding diagonal) similarity values
    idx = np.triu_indices_from(sim_mat, k=1)
    sim_scores = sim_mat[idx]

    # Gini of similarity scores quantifies heterogeneity
    gini = gini_coefficient(sim_scores)

    n_samples = len(sim_scores)
    if n_samples < min_samples:
        return SplitDecision(False, 0.0, 0.0, f"Sample size {n_samples} < min_samples {min_samples}")

    # Use Gini as the effective range (max‑min) for Hoeffding bound
    r = max(sim_scores) - min(sim_scores)
    epsilon = hoeffding_bound(r, delta, n_samples)

    # Define a synthetic gain gap proportional to Gini (larger inequality → larger gap)
    gain_gap = gini * 0.1

    should_split = gain_gap > epsilon
    reason = (
        f"Gini={gini:.4f}, ε={epsilon:.4f}, gain_gap={gain_gap:.4f} → "
        f"{'split' if should_split else 'no split'}"
    )
    return SplitDecision(should_split, epsilon, gain_gap, reason)


def hybrid_endpoint_ssim_leader_chelydrid(
    morphologies: List[Morphology],
    base_velocity: float = 10.0,
) -> Tuple[SplitDecision, List[float]]:
    """
    High‑level hybrid operation.

    Returns:
        - SplitDecision describing whether the endpoint circuit breaker fires.
        - List of burst velocities for each object, computed from its sphericity.
    """
    decision = hybrid_split_decision(morphologies)
    burst_velocities = [
        chelydrid_burst_velocity(base_velocity, sphericity_index(m.length, m.width, m.height))
        for m in morphologies
    ]
    return decision, burst_velocities


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small heterogeneous set of morphologies
    random.seed(42)
    objs = []
    for _ in range(8):
        l = random.uniform(0.5, 2.0)
        w = random.uniform(0.5, 2.0)
        h = random.uniform(0.5, 2.0)
        m = random.uniform(1.0, 5.0)
        objs.append(Morphology(l, w, h, m))

    split_dec, bursts = hybrid_endpoint_ssim_leader_chelydrid(objs, base_velocity=12.0)

    print("Hybrid Split Decision:")
    print(f"  should_split: {split_dec.should_split}")
    print(f"  epsilon    : {split_dec.epsilon:.6f}")
    print(f"  gain_gap   : {split_dec.gain_gap:.6f}")
    print(f"  reason     : {split_dec.reason}")

    print("\nBurst velocities per object (m/s):")
    for i, v in enumerate(bursts, 1):
        print(f"  Object {i}: {v:.3f}")