# DARWIN HAMMER — match 3503, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m1653_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_hybrid_hdc_se_m2427_s0.py (gen4)
# born: 2026-05-29T23:50:25Z

"""Hybrid Hoeffding‑Gini Bandit + CMS‑HDC Risk‑Adjusted Selector

Parents:
- hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m1653_s1.py
- hybrid_hybrid_hybrid_privac_hybrid_hdc_se_m2427_s0.py

Mathematical bridge:
Both parents employ a Count‑Min Sketch (CMS).  In the Hoeffding‑Gini bandit the
CMS stores reward statistics; the Gini coefficient of the estimated reward
distribution regularises the Hoeffding confidence bound.  In the CMS‑HDC
module the sketch matrix is interpreted as a weighted collection of tokens,
each token being mapped to a random unit‑magnitude complex hyper‑vector.
By converting the same CMS into a hyper‑vector we obtain a *risk* representation
that can be bound to a causal arm‑vector and modulated by a fractional‑power
operator.  The similarity between this risk hyper‑vector and the arm‑vector is
used to bias the UCB score, yielding a unified decision rule that simultaneously
exploits statistical confidence, reward inequality and high‑dimensional
privacy‑risk encoding.
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Core primitives from Parent A
# ----------------------------------------------------------------------


def hoeffding_bound_with_gini(r: float, delta: float, n: int, gini_coeff: float) -> float:
    """Hoeffding bound regularised by a Gini term."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("Invalid arguments for Hoeffding bound")
    regularization_term = gini_coeff * math.pi / 6.0
    return math.sqrt((r * r * math.log(1.0 / delta) + regularization_term) / (2.0 * n))


def gini_coefficient(values: List[float]) -> float:
    """Standard Gini coefficient for a list of non‑negative numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(values)
    cumulative = 0.0
    for i, v in enumerate(sorted_vals, 1):
        cumulative += i * v
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini = (2.0 * cumulative) / (n * total) - (n + 1) / n
    return gini


# ----------------------------------------------------------------------
# Core primitives from Parent B (CMS + HDC)
# ----------------------------------------------------------------------


def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Hash an item into `depth` column indices (one per row)."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]


class CountMinSketch:
    """Simple Count‑Min Sketch with integer counters."""

    def __init__(self, depth: int = 5, width: int = 2000, seed: int = 0):
        self.depth = depth
        self.width = width
        self.seed = seed
        self.table = np.zeros((depth, width), dtype=np.int64)

    def add(self, item: str, count: int = 1) -> None:
        """Increment the counters for `item`."""
        for row, col in enumerate(_cms_hash(item, self.depth, self.width)):
            self.table[row, col] += count

    def estimate(self, item: str) -> int:
        """Return the minimum count across rows (CMS estimate)."""
        cols = _cms_hash(item, self.depth, self.width)
        return min(self.table[row, col] for row, col in enumerate(cols))

    def matrix(self) -> np.ndarray:
        """Expose the raw CMS matrix (read‑only)."""
        return self.table.copy()


def random_complex_hv(dim: int, seed: int | str | None = None) -> np.ndarray:
    """Generate a unit‑magnitude complex hyper‑vector."""
    rng = random.Random(seed)
    angles = np.array([rng.random() for _ in range(dim)]) * 2.0 * math.pi
    return np.exp(1j * angles)


def cms_to_complex_hv(cms: CountMinSketch, dim: int = 10000, seed: int = 0) -> np.ndarray:
    """
    Convert a CMS matrix into a single complex hyper‑vector.
    Each cell (i, j) with count c contributes c * hv(i,j) where hv(i,j) is a
    deterministic random complex vector derived from the cell coordinates.
    The contributions are summed and finally normalised to unit magnitude.
    """
    rng = random.Random(seed)
    hv = np.zeros(dim, dtype=np.complex128)

    matrix = cms.matrix()
    depth, width = matrix.shape
    for i in range(depth):
        for j in range(width):
            c = matrix[i, j]
            if c == 0:
                continue
            # deterministic seed per cell
            cell_seed = int(hashlib.sha256(f"cell:{i}:{j}".encode()).hexdigest(), 16) % (2**32)
            cell_hv = random_complex_hv(dim, seed=cell_seed)
            hv += c * cell_hv

    # Normalise to unit magnitude (avoid division by zero)
    norm = np.linalg.norm(hv)
    if norm == 0:
        return random_complex_hv(dim, seed=seed)
    return hv / norm


def bind_hv(hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
    """Element‑wise multiplication (binding) of two complex hyper‑vectors."""
    return hv1 * hv2


def fractional_power_binding(hv: np.ndarray, exponent: float) -> np.ndarray:
    """
    Raise each complex component to a real exponent.
    This modulates the phase while preserving magnitude.
    """
    magnitude = np.abs(hv)
    phase = np.angle(hv)
    new_phase = phase * exponent
    return magnitude * np.exp(1j * new_phase)


def cosine_similarity(hv1: np.ndarray, hv2: np.ndarray) -> float:
    """Cosine similarity for complex vectors (real part of inner product)."""
    num = np.real(np.vdot(hv1, hv2))
    den = np.linalg.norm(hv1) * np.linalg.norm(hv2)
    return num / den if den != 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------


def hybrid_ucb_selection(
    arms: List[str],
    cms: CountMinSketch,
    total_observations: int,
    delta: float = 0.05,
    reward_range: float = 1.0,
    hv_dim: int = 10000,
    seed: int = 0,
) -> Tuple[str, float]:
    """
    Select an arm using a Gini‑regularised Hoeffding UCB that is further
    adjusted by a risk hyper‑vector derived from the same CMS.

    Returns a tuple (selected_arm, adjusted_score).
    """
    # 1. Estimate mean reward for each arm from the CMS
    reward_estimates = [cms.estimate(arm) / total_observations for arm in arms]

    # 2. Compute Gini over the reward estimates
    gini = gini_coefficient(reward_estimates)

    # 3. Compute Hoeffding bound for each arm
    bounds = [
        hoeffding_bound_with_gini(reward_range, delta, total_observations, gini)
        for _ in arms
    ]

    # 4. Raw UCB scores
    ucb_scores = [mu + b for mu, b in zip(reward_estimates, bounds)]

    # 5. Build a global risk hyper‑vector from the CMS
    risk_hv = cms_to_complex_hv(cms, dim=hv_dim, seed=seed)

    # 6. For each arm create a causal hyper‑vector and compute similarity‑based penalty
    adjusted_scores = []
    for arm, raw_score in zip(arms, ucb_scores):
        causal_hv = random_complex_hv(hv_dim, seed=f"arm:{arm}")
        bound_hv = bind_hv(risk_hv, causal_hv)
        # Modulate by (1 - gini) so that high inequality reduces confidence
        mod_hv = fractional_power_binding(bound_hv, exponent=1.0 - gini)
        sim = cosine_similarity(risk_hv, mod_hv)  # in [-1,1]
        # Penalise the raw UCB proportionally to similarity (more risk → lower score)
        penalty = (sim + 1.0) / 2.0  # map to [0,1]
        adjusted = raw_score * (1.0 - 0.5 * penalty)  # up to 50% reduction
        adjusted_scores.append(adjusted)

    # 7. Choose the arm with the highest adjusted score
    best_idx = int(np.argmax(adjusted_scores))
    return arms[best_idx], adjusted_scores[best_idx]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny problem with three arms
    arms = ["A", "B", "C"]
    cms = CountMinSketch(depth=4, width=500, seed=42)

    # Simulate some reward observations
    random.seed(123)
    total_obs = 0
    for _ in range(200):
        chosen = random.choice(arms)
        reward = random.random()  # reward in [0,1)
        # Encode reward as count (scaled to integer for CMS)
        count = int(reward * 1000) + 1
        cms.add(chosen, count)
        total_obs += count

    # Perform hybrid selection
    selected_arm, score = hybrid_ucb_selection(
        arms=arms,
        cms=cms,
        total_observations=total_obs,
        delta=0.05,
        reward_range=1.0,
        hv_dim=2048,
        seed=99,
    )
    print(f"Selected arm: {selected_arm}, adjusted UCB score: {score:.4f}")