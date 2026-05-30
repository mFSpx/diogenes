# DARWIN HAMMER — match 3294, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_regret_engine_m2_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2439_s0.py (gen6)
# born: 2026-05-29T23:49:15Z

"""Hybrid Algorithm combining Decision‑Regret (Parent A) and Voronoi‑Entropy (Parent B)

Parent A (hybrid_hybrid_hybrid_decisi_hybrid_regret_engine_m2_s3) introduces a
*regret‑weighted strategy* for a set of actions and evaluates inequality of the
resulting weight vector with the **Gini coefficient**.

Parent B (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2439_s0) builds a
spatial model using **Voronoi partitioning** whose seeds are weighted by the
**Shannon entropy** of feature importance; it also employs a lightweight
*Count‑Min sketch* to compress high‑dimensional action descriptors.

The mathematical bridge is the **probability distribution over actions** that
appears in both worlds:

* In Parent A the distribution is the regret‑weighted strategy **w**.
* In Parent B the same distribution is used to compute an **entropy H(w)**,
  which then modulates the Voronoi seed locations.

The hybrid algorithm therefore

1. builds a regret‑weighted distribution **w**,
2. measures its inequality with **Gini(w)**,
3. measures its uncertainty with **H(w)**,
4. uses **H(w)** to bias the Voronoi seeds,
5. evaluates the spatial inequality of the resulting regions with **Gini(region_sizes)**,
6. combines the two Gini terms into a single *hybrid regret* that can be
   minimised.

The code below implements this fusion and provides three public functions that
demonstrate the hybrid operation."""


import math
import random
import sys
import pathlib
import numpy as np
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Utility – Count‑Min Sketch (dimensionality reduction)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Very small Count‑Min sketch for integer‑valued vectors."""
    def __init__(self, width: int = 100, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        rng = random.Random(seed)
        self.hashes = [rng.randint(1, 2**31 - 1) for _ in range(depth)]
        self.table = np.zeros((depth, width), dtype=np.int64)

    def _hash(self, i: int, x: int) -> int:
        return (x * self.hashes[i]) % self.width

    def add(self, vector: List[int]) -> None:
        for i in range(self.depth):
            for x in vector:
                idx = self._hash(i, x)
                self.table[i, idx] += 1

    def estimate(self, vector: List[int]) -> int:
        """Return an upper‑bound estimate of the frequency of the whole vector."""
        mins = []
        for i in range(self.depth):
            vals = [self.table[i, self._hash(i, x)] for x in vector]
            mins.append(min(vals))
        return max(mins)


# ----------------------------------------------------------------------
# Decision‑Regret core (Parent A)
# ----------------------------------------------------------------------
def regret_weights(costs: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """
    Compute a regret‑weighted probability distribution over actions.
    Regret for action i is defined as cost_i * risk_i.
    The vector is normalised to sum to 1.
    """
    if costs.shape != risks.shape:
        raise ValueError("costs and risks must have the same shape")
    raw = costs * risks
    total = raw.sum()
    if total == 0:
        # avoid division by zero – fall back to uniform
        return np.full_like(raw, 1.0 / raw.size, dtype=float)
    return raw / total


def gini_coefficient(weights: np.ndarray) -> float:
    """Gini coefficient of a non‑negative weight vector (sum may be 1)."""
    if np.any(weights < 0):
        raise ValueError("weights must be non‑negative")
    sorted_w = np.sort(weights)  # ascending
    n = weights.size
    cumulative = np.cumsum(sorted_w)
    # The classic formula: G = 1 - 2 * sum_i (n - i + 0.5) * w_i / (n * sum w)
    numerator = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_w)
    denominator = n * sorted_w.sum()
    return numerator / denominator if denominator != 0 else 0.0


def shannon_entropy(weights: np.ndarray) -> float:
    """Shannon entropy of a probability distribution."""
    eps = np.finfo(float).eps
    probs = np.clip(weights, eps, 1.0)
    return -np.sum(probs * np.log(probs))


# ----------------------------------------------------------------------
# Voronoi core (Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_seed(point: Point, seeds: List[Point]) -> int:
    """Return index of the nearest seed (ties broken by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean(point, seeds[i]), i))


def assign_voronoi(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the region of its nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        idx = nearest_seed(p, seeds)
        regions[idx].append(p)
    return regions


def region_sizes(regions: Dict[int, List[Point]]) -> np.ndarray:
    """Return an array of region cardinalities."""
    return np.array([len(v) for v in regions.values()], dtype=float)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_hybrid_strategy(
    costs: np.ndarray,
    risks: np.ndarray,
    points: List[Point],
    seed_count: int = 5,
    sketch_width: int = 200,
    sketch_depth: int = 7,
) -> Dict[str, Any]:
    """
    Build the hybrid representation:

    1. Reduce the (cost, risk) vectors with a Count‑Min sketch.
    2. Compute regret‑weighted distribution w.
    3. Compute entropy H(w) and Gini G_regret = Gini(w).
    4. Initialise Voronoi seeds randomly inside the bounding box of points.
    5. Shift each seed by a factor proportional to H(w) * w_i (i‑th weight).
    6. Assign points to Voronoi regions.
    7. Compute spatial Gini G_spatial = Gini(region_sizes).

    Returns a dictionary containing all intermediate artefacts.
    """
    # ---- 1. Sketch reduction (illustrative – we only store a scalar estimate) ----
    cms = CountMinSketch(width=sketch_width, depth=sketch_depth, seed=42)
    for c, r in zip(costs.astype(int), risks.astype(int)):
        cms.add([int(c), int(r)])
    sketch_estimate = cms.estimate([int(c) for c in costs] + [int(r) for r in risks])

    # ---- 2. Regret‑weighted distribution ----
    w = regret_weights(costs, risks)

    # ---- 3. Information‑theoretic measures ----
    entropy = shannon_entropy(w)
    gini_regret = gini_coefficient(w)

    # ---- 4. Random seed initialisation ----
    if not points:
        raise ValueError("points list cannot be empty")
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rng = random.Random(1234)
    seeds: List[Point] = [
        (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
        for _ in range(seed_count)
    ]

    # ---- 5. Entropy‑biased seed displacement ----
    # Map each weight to a seed (if more seeds than actions, repeat cyclically)
    displaced_seeds: List[Point] = []
    for i, (sx, sy) in enumerate(seeds):
        weight = w[i % w.size]  # cyclic assignment
        # displacement magnitude proportional to entropy * weight
        delta = entropy * weight
        # random direction
        angle = rng.uniform(0, 2 * math.pi)
        dx = delta * math.cos(angle)
        dy = delta * math.sin(angle)
        displaced_seeds.append((sx + dx, sy + dy))

    # ---- 6. Voronoi assignment ----
    regions = assign_voronoi(points, displaced_seeds)

    # ---- 7. Spatial Gini ----
    sizes = region_sizes(regions)
    gini_spatial = gini_coefficient(sizes)

    # Combine the two Gini terms into a hybrid regret metric
    hybrid_regret = 0.5 * gini_regret + 0.5 * gini_spatial

    return {
        "sketch_estimate": sketch_estimate,
        "regret_weights": w,
        "entropy": entropy,
        "gini_regret": gini_regret,
        "seeds_original": seeds,
        "seeds_displaced": displaced_seeds,
        "regions": regions,
        "region_sizes": sizes,
        "gini_spatial": gini_spatial,
        "hybrid_regret": hybrid_regret,
    }


def rank_actions_by_hybrid_ev(
    costs: np.ndarray,
    risks: np.ndarray,
    points: List[Point],
    seed_count: int = 5,
) -> List[Tuple[int, float]]:
    """
    Rank actions by an expected‑value proxy that blends regret and spatial
    influence. For each action i we compute:

        EV_i = (1 - w_i) * (1 - region_density_i)

    where w_i is the regret weight and region_density_i is the proportion of
    points that fall into the Voronoi region associated with seed i
    (after entropy‑biased displacement). Higher EV indicates a more
    attractive action.
    """
    hybrid = compute_hybrid_strategy(costs, risks, points, seed_count=seed_count)
    w = hybrid["regret_weights"]
    regions = hybrid["regions"]
    total_points = len(points)

    # Map each seed to its region density
    densities = np.array(
        [len(regions[i]) / total_points if total_points > 0 else 0.0 for i in range(seed_count)]
    )

    ev = (1.0 - w[:seed_count]) * (1.0 - densities)
    ranking = sorted(enumerate(ev.tolist()), key=lambda kv: kv[1], reverse=True)
    return ranking


def optimize_decision_making(
    costs: np.ndarray,
    risks: np.ndarray,
    points: List[Point],
    seed_count: int = 5,
    iterations: int = 30,
    step_scale: float = 0.1,
) -> Dict[str, Any]:
    """
    Simple stochastic optimisation that perturbs Voronoi seeds to minimise the
    hybrid regret metric. At each iteration a random seed is moved by a small
    Gaussian offset; the move is kept if it reduces `hybrid_regret`.
    """
    # Initialise with the baseline hybrid strategy
    state = compute_hybrid_strategy(costs, risks, points, seed_count=seed_count)
    best_regret = state["hybrid_regret"]
    best_state = state

    rng = random.Random(999)

    for _ in range(iterations):
        # copy current displaced seeds
        new_seeds = [s for s in state["seeds_displaced"]]
        # pick a random seed to jitter
        idx = rng.randrange(seed_count)
        sx, sy = new_seeds[idx]
        jitter = (
            rng.gauss(0, step_scale),
            rng.gauss(0, step_scale),
        )
        new_seeds[idx] = (sx + jitter[0], sy + jitter[1])

        # recompute regions with the perturbed seeds
        regions = assign_voronoi(points, new_seeds)
        sizes = region_sizes(regions)
        gini_spatial = gini_coefficient(sizes)

        # hybrid regret uses the same regret weights and entropy as before
        hybrid_regret = 0.5 * state["gini_regret"] + 0.5 * gini_spatial

        if hybrid_regret < best_regret:
            # accept improvement
            best_regret = hybrid_regret
            best_state = {
                **state,
                "seeds_displaced": new_seeds,
                "regions": regions,
                "region_sizes": sizes,
                "gini_spatial": gini_spatial,
                "hybrid_regret": hybrid_regret,
            }
            state = best_state  # continue from improved state

    return best_state


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data generation
    ACTION_COUNT = 12
    POINT_COUNT = 250

    rng = np.random.default_rng(2026)

    # Random costs in [1, 100], risks in [0, 1]
    costs = rng.integers(1, 101, size=ACTION_COUNT).astype(float)
    risks = rng.random(ACTION_COUNT)

    # Random 2‑D points inside a unit square
    points = [(float(x), float(y)) for x, y in rng.random((POINT_COUNT, 2))]

    # Run hybrid strategy computation
    hybrid_info = compute_hybrid_strategy(costs, risks, points, seed_count=6)

    print("Hybrid regret metric:", hybrid_info["hybrid_regret"])
    print("Regret Gini:", hybrid_info["gini_regret"])
    print("Spatial Gini:", hybrid_info["gini_spatial"])
    print("Entropy of regret distribution:", hybrid_info["entropy"])

    # Rank actions
    ranking = rank_actions_by_hybrid_ev(costs, risks, points, seed_count=6)
    print("\nAction ranking (index, expected value):")
    for idx, ev in ranking:
        print(f"  Action {idx}: EV = {ev:.4f}")

    # Optimise seeds
    optimized = optimize_decision_making(costs, risks, points, seed_count=6, iterations=50)
    print("\nOptimised hybrid regret:", optimized["hybrid_regret"])
    print("Improvement:", hybrid_info["hybrid_regret"] - optimized["hybrid_regret"])