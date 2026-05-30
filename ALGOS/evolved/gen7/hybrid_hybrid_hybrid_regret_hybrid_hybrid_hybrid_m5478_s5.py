# DARWIN HAMMER — match 5478, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_regret_engine_hybrid_hybrid_bandit_m83_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1207_s0.py (gen6)
# born: 2026-05-30T00:02:11Z

"""Hybrid Regret‑Weighted Voronoi Engine
Parents:
    - hybrid_hybrid_regret_engine_hybrid_doomsday_cale_m19_s3.py (Regret‑weighted strategy + Gini)
    - hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1207_s0.py (Voronoi partition + regret‑weighted curvature)

Mathematical bridge:
Both parents expose a *regret* quantity.  The first parent converts actions
and counter‑factuals into a probability distribution 𝑝 over actions.
The second parent consumes a vector of regrets 𝑟 (one per Voronoi seed) to
modulate a curvature matrix.  The hybrid therefore:
    1. Builds 𝑝 via the regret‑weighted soft‑max of the first parent.
    2. Forms a regret vector 𝑟 = 1 − 𝑝 (aligned with the Voronoi seeds).
    3. Computes the Gini coefficient G(𝑝) as a scalar measure of
       distribution inequality.
    4. Generates a Voronoi partition of the data points and evaluates a
       curvature value for each point using 𝑟.
    5. Scales the curvature by G(𝑝) to obtain the final hybrid output.

The resulting system can be used for adaptive sampling, workload
allocation, or any scenario where an unevenness‑aware spatial
regularisation is required.
"""

from __future__ import annotations
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------


def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """
    Soft‑max over (expected_value – cost – risk + counterfactual contribution).
    Returns a probability distribution over action IDs.
    """
    if not actions:
        return {}

    # aggregate counterfactual contributions per action
    cf: Dict[str, float] = {
        c.action_id: c.outcome_value * c.probability for c in counterfactuals
    }

    # raw utility per action
    utilities: Dict[str, float] = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions
    }

    # soft‑max with numerical stability
    best = max(utilities.values())
    exp_vals = {k: math.exp(v - best) for k, v in utilities.items()}
    total = sum(exp_vals.values()) or 1.0
    return {k: v / total for k, v in exp_vals.items()}


def gini_coefficient(values: Iterable[float]) -> float:
    """
    Classic Gini coefficient for a non‑negative vector.
    """
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = 0.0
    for i, x in enumerate(xs, 1):
        cumulative += (2 * i - n - 1) * x
    return cumulative / (n * sum(xs))


# ----------------------------------------------------------------------
# Parent‑B building blocks (Voronoi + curvature)
# ----------------------------------------------------------------------


def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    """
    Return the index of the seed closest to *point* (Euclidean distance).
    """
    if seeds.size == 0:
        raise ValueError("seed list cannot be empty")
    distances = np.linalg.norm(seeds - point, axis=1)
    return int(np.argmin(distances))


def assign_voronoi(points: np.ndarray, seeds: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Partition *points* into Voronoi regions defined by *seeds*.
    Returns a mapping seed_index → array of point indices belonging to that region.
    """
    regions: Dict[int, List[int]] = {i: [] for i in range(len(seeds))}
    for idx, p in enumerate(points):
        region = nearest(p, seeds)
        regions[region].append(idx)
    # convert lists to numpy arrays for efficient indexing
    return {k: np.array(v, dtype=int) for k, v in regions.items()}


def regret_weighted_curvature(feature_vector: np.ndarray, regret: float) -> np.ndarray:
    """
    Simple curvature model: scale a feature vector by (1 – regret).
    """
    return feature_vector * (1.0 - regret)


def compute_voronoi_curvature(
    points: np.ndarray,
    seeds: np.ndarray,
    regrets: np.ndarray,
) -> np.ndarray:
    """
    For each Voronoi region, compute a curvature value for the points inside.
    The curvature for a point is the norm of a placeholder feature vector
    scaled by the region's regret.
    Returns a vector `curv` of length len(points).
    """
    if len(seeds) != len(regrets):
        raise ValueError("Number of seeds must match length of regrets vector")

    regions = assign_voronoi(points, seeds)
    curv = np.zeros(len(points))

    # placeholder feature vector – could be enriched with domain‑specific data
    placeholder = np.array([1.0, 0.0, 0.0, 0.0])

    for seed_idx, point_indices in regions.items():
        if point_indices.size == 0:
            continue
        regret = regrets[seed_idx]
        # curvature for all points in the region
        region_curv = regret_weighted_curvature(placeholder, regret)
        # use the L2 norm of the scaled feature as a scalar curvature value
        scalar = float(np.linalg.norm(region_curv))
        curv[point_indices] = scalar

    return curv


# ----------------------------------------------------------------------
# Hybrid functions (fusion of both parents)
# ----------------------------------------------------------------------


def hybrid_regret_distribution(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Tuple[np.ndarray, float]:
    """
    Compute the regret‑weighted probability vector *p* and its Gini coefficient *G*.
    Returns (p, G) where p is a NumPy array ordered by the original actions list.
    """
    prob_dict = compute_regret_weighted_strategy(actions, counterfactuals)
    p = np.array([prob_dict.get(a.id, 0.0) for a in actions], dtype=float)
    G = gini_coefficient(p)
    return p, G


def map_actions_to_seeds(
    actions: List[MathAction],
    seeds: np.ndarray,
) -> np.ndarray:
    """
    Produce a regret vector aligned with *seeds*.
    If there are more seeds than actions, excess seeds receive zero regret.
    If fewer, the first |seeds| actions are used.
    """
    probs, _ = hybrid_regret_distribution(actions, [])
    n_seeds = len(seeds)
    if len(probs) >= n_seeds:
        selected = probs[:n_seeds]
    else:
        # pad with uniform low probability for missing actions
        pad = np.full(n_seeds - len(probs), 1e-6)
        selected = np.concatenate([probs, pad])
    # regret = 1 - probability (bounded in [0,1])
    return 1.0 - selected


def hybrid_regret_voronoi(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    points: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """
    Core hybrid operation:
        1. Build regret‑weighted distribution p and Gini G.
        2. Derive regret vector r = 1 – p (aligned with seeds).
        3. Compute Voronoi‑based curvature c for each point.
        4. Return G‑scaled curvature:  ĉ = G * c
    The output is a 1‑D array of length len(points) suitable for downstream
    allocation or sampling decisions.
    """
    # Step 1 – distribution & inequality measure
    p, G = hybrid_regret_distribution(actions, counterfactuals)

    # Step 2 – regret vector aligned with seeds
    if len(seeds) != len(p):
        # Align by truncation/padding
        r = map_actions_to_seeds(actions, seeds)
    else:
        r = 1.0 - p

    # Step 3 – Voronoi curvature
    c = compute_voronoi_curvature(points, seeds, r)

    # Step 4 – Gini scaling
    return G * c


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Dummy actions & counterfactuals
    actions = [
        MathAction(id="a1", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="a2", expected_value=8.0, cost=1.5, risk=0.5),
        MathAction(id="a3", expected_value=6.0, cost=1.0, risk=0.2),
    ]
    counterfactuals = [
        MathCounterfactual(action_id="a1", outcome_value=1.0, probability=0.8),
        MathCounterfactual(action_id="a2", outcome_value=-0.5, probability=0.5),
    ]

    # Random points in 2‑D and three seeds (same count as actions)
    points = np.random.rand(15, 2) * 10.0
    seeds = np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]])

    # Run the hybrid engine
    curvature = hybrid_regret_voronoi(actions, counterfactuals, points, seeds)

    # Simple sanity checks
    assert curvature.shape == (len(points),)
    print("Hybrid curvature vector:", curvature)
    print("All checks passed.")