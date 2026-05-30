# DARWIN HAMMER — match 3933, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1088_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1203_s4.py (gen6)
# born: 2026-05-29T23:52:35Z

"""Hybrid Fusion of Voronoi‑based Probabilistic Weighting (Parent A) and
Text‑Feature Projection with Gini‑Modulated Bandit Confidence (Parent B).

Mathematical Bridge
-------------------
* Parent A* produces a probability distribution **p** over seed regions by
  counting points in each Voronoi cell and then applying a sinusoidal
  weekday rotation `weekday_weight_vector`.
* *Parent B* projects a high‑dimensional feature vector **f** onto a low‑dimensional
  space using a bilinear matrix **P**, yielding **v = f·P**. The inequality of **v**
  is measured by the Gini coefficient **G(v)**, which modulates the regret‑weighted
  confidence bound **ε** of a bandit arm.

The fusion treats the rotated Voronoi probabilities **p̂** as a *mask* that
weights the projected vector **v** element‑wise, producing **v̂ = v ⊙ p̂**.
The Gini coefficient of **v̂** drives the confidence‑bound scaling, while the
temperature‑dependent rate **ρ(T)** scales the Hoeffding‑tree gain estimate.
Thus the two topologies are mathematically intertwined:

    v      = f · P
    p̂     = weekday_weight_vector(groups, dow)          # ∑ p̂ = 1
    v̂     = v * p̂                                      # element‑wise
    G      = gini_coefficient(v̂)
    ε      = ε0 * (1 + λ_g * G)
    gain   = ρ(T) * (max_gain – ε)

The final bandit arm is selected using the adjusted confidence bound.

The module implements this pipeline and provides three public functions that
demonstrate the hybrid operation.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Parent A – Voronoi utilities
# ----------------------------------------------------------------------
def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed closest to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to its nearest seed, returning a region dict."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
    """
    Normalised sinusoidal weight vector for *groups* based on weekday ``dow``.
    ``dow`` ∈ {0,…,6} where 0 = Monday.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow / 7.0)
    weight_vec = 1.0 + 0.5 * np.sin(base_angles + phase)
    return weight_vec / np.sum(weight_vec)


# ----------------------------------------------------------------------
# Parent B – Projection, Gini, temperature
# ----------------------------------------------------------------------
def project_features(f: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Bilinear projection of high‑dimensional feature vector ``f`` onto
    low‑dimensional space using matrix ``P``.
    Returns ``v = f·P`` (shape (m,)).
    """
    if f.ndim != 1 or P.ndim != 2:
        raise ValueError("f must be 1‑D and P must be 2‑D")
    if f.shape[0] != P.shape[0]:
        raise ValueError("inner dimensions of f and P must match")
    return f @ P


def gini_coefficient(x: np.ndarray) -> float:
    """
    Gini coefficient of a non‑negative 1‑D array.
    Returns a value in [0, 1] where 0 = perfect equality.
    """
    if x.ndim != 1:
        raise ValueError("x must be 1‑D")
    if np.any(x < 0):
        raise ValueError("x must be non‑negative")
    if np.all(x == 0):
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    cumvals = np.cumsum(sorted_x, dtype=float)
    gini = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
    return float(gini)


def temperature_rate(T: float) -> float:
    """
    Developmental rate ρ(T) – a smooth monotonic mapping from temperature
    to (0, 1).  Chosen as a logistic function centred at T=0.5.
    """
    return 1.0 / (1.0 + math.exp(-10.0 * (T - 0.5)))


# ----------------------------------------------------------------------
# Data structures for the bandit side
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action used in the regret‑weighted Hoeffding tree."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class BanditAction:
    """Bandit arm with propensity‑adjusted confidence bound."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_voronoi_distribution(
    seeds: List[Point],
    points: List[Point],
    dow: int,
) -> np.ndarray:
    """
    Returns a probability vector **p̂** of length ``len(seeds)``.
    Steps:
        1. Build Voronoi regions.
        2. Convert region cardinalities to raw probabilities.
        3. Apply sinusoidal weekday rotation via ``weekday_weight_vector``.
        4. Normalise to sum to 1.
    """
    regions = assign(points, seeds)
    raw_counts = np.array([len(regions[i]) for i in range(len(seeds))], dtype=float)
    # Avoid division‑by‑zero when no points fall into a region
    if raw_counts.sum() == 0:
        raw_counts = np.ones_like(raw_counts)
    raw_probs = raw_counts / raw_counts.sum()
    groups = [f"seed_{i}" for i in range(len(seeds))]
    weekday_weights = weekday_weight_vector(groups, dow)
    rotated = raw_probs * weekday_weights
    return rotated / rotated.sum()


def compute_rotated_projection(
    f: np.ndarray,
    P: np.ndarray,
    voronoi_weights: np.ndarray,
) -> np.ndarray:
    """
    Projects ``f`` using ``P`` and then masks the result with the
    Voronoi‑derived weight vector.
    The element‑wise product is returned (shape ``(m,)``).
    """
    v = project_features(f, P)
    if voronoi_weights.shape[0] != v.shape[0]:
        # If dimensions differ, broadcast by linear interpolation.
        # This keeps the fusion mathematically meaningful without external libs.
        idx = np.linspace(0, len(voronoi_weights) - 1, num=v.shape[0])
        interpolated = np.interp(idx, np.arange(len(voronoi_weights)), voronoi_weights)
        mask = interpolated
    else:
        mask = voronoi_weights
    return v * mask


def select_bandit_action(
    projected_masked: np.ndarray,
    actions: List[MathAction],
    epsilon_0: float = 0.1,
    lambda_g: float = 0.5,
    max_gain: float = 1.0,
    temperature: float = 0.5,
) -> BanditAction:
    """
    Chooses a bandit arm using the Gini‑modulated confidence bound.

    1. Compute Gini of the masked projection.
    2. Modulate the base confidence ``ε₀`` with ``λ_g·G``.
    3. Apply temperature scaling to the gain gap.
    4. For each candidate ``MathAction`` compute a confidence‑adjusted score:
           score = expected_value + confidence_bound
       where ``confidence_bound = ε`` scaled by the gain gap.
    5. Return the ``BanditAction`` with the highest score.
    """
    G = gini_coefficient(projected_masked)
    epsilon = epsilon_0 * (1.0 + lambda_g * G)
    rho = temperature_rate(temperature)
    gain_gap = rho * (max_gain - epsilon)

    best_score = -math.inf
    best_action: BanditAction | None = None

    for a in actions:
        # Propensity is a softmax over expected values to keep it in (0,1)
        prop = math.exp(a.expected_value) / sum(math.exp(b.expected_value) for b in actions)
        confidence = epsilon + gain_gap
        score = a.expected_value + confidence
        if score > best_score:
            best_score = score
            best_action = BanditAction(
                action_id=a.id,
                propensity=prop,
                expected_reward=a.expected_value,
                confidence_bound=confidence,
            )
    if best_action is None:
        raise RuntimeError("No actions provided")
    return best_action


# ----------------------------------------------------------------------
# Demonstration API
# ----------------------------------------------------------------------
def hybrid_pipeline(
    seeds: List[Point],
    points: List[Point],
    dow: int,
    f: np.ndarray,
    P: np.ndarray,
    actions: List[MathAction],
    temperature: float = 0.5,
) -> BanditAction:
    """
    End‑to‑end hybrid operation:
        * Voronoi → weekday‑rotated probabilities
        * Feature projection → masked vector
        * Gini‑modulated bandit selection
    Returns the selected ``BanditAction``.
    """
    voronoi_weights = compute_voronoi_distribution(seeds, points, dow)
    masked_proj = compute_rotated_projection(f, P, voronoi_weights)
    return select_bandit_action(
        projected_masked=masked_proj,
        actions=actions,
        temperature=temperature,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic data for a quick sanity check
    random.seed(0)
    np.random.seed(0)

    # Voronoi seeds and random points
    seeds = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    points = [(random.random() * 10, random.random() * 5) for _ in range(200)]
    dow = 2  # Wednesday

    # High‑dimensional feature vector and projection matrix
    f = np.random.rand(12)          # 12‑dimensional feature
    P = np.random.rand(12, 3)       # Project down to 3 dimensions (matches seed count)

    # Define a few MathActions
    actions = [
        MathAction(id="A", expected_value=0.2),
        MathAction(id="B", expected_value=0.5),
        MathAction(id="C", expected_value=0.1),
    ]

    selected = hybrid_pipeline(
        seeds=seeds,
        points=points,
        dow=dow,
        f=f,
        P=P,
        actions=actions,
        temperature=0.7,
    )
    print("Selected Bandit Action:")
    print(selected)