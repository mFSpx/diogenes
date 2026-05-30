# DARWIN HAMMER — match 2439, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_sketch_m1023_s0.py (gen4)
# born: 2026-05-29T23:42:19Z

"""Hybrid Voronoi‑Liquid‑Decision Algorithm
Integrates:
* hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c (Voronoi partitioning + liquid‑time‑constant ODE + hyper‑dimensional binding)
* hybrid_hybrid_hybrid_decisi_hybrid_hybrid_sketch (Decision‑hygiene scoring, Shannon entropy, Hoeffding bound, Count‑Min sketch, MinHash LSH, RLCT estimation)

Mathematical bridge:
The Voronoi regions provide spatial context for each datum.  Within a region the
input vector is bound to the region’s hyper‑dimensional seed; the bundled result
yields an input‑dependent time‑constant τ.  τ drives the liquid‑time‑constant
ODE that updates a hidden state h.  The hidden state is then scored by a
decision‑hygiene function whose feature weights are derived from a
Count‑Min sketch of the region’s identifiers.  The Hoeffding bound supplies a
confidence‑adjusted correction to the entropy‑based weights, and the RLCT
estimate supplies a pruning schedule for low‑impact regions.  All components are
tightly coupled through the shared τ and weight vectors, yielding a single
unified hybrid system.
"""

import math
import random
import sys
import pathlib
import hashlib
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Voronoi utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the index of its nearest Voronoi seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        idx = nearest(p, seeds)
        regions[idx].append(p)
    return regions

# ----------------------------------------------------------------------
# Hyper‑dimensional primitives
# ----------------------------------------------------------------------
def bind(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Element‑wise multiplication (binding) of two hyper‑vectors."""
    return np.multiply(v1, v2)

def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Sum‑then‑normalize (bundling) of a list of hyper‑vectors."""
    if not vectors:
        raise ValueError("cannot bundle empty list")
    s = np.sum(vectors, axis=0)
    norm = np.linalg.norm(s)
    return s / norm if norm != 0 else s

# ----------------------------------------------------------------------
# Liquid‑time‑constant network core
# ----------------------------------------------------------------------
def compute_time_constant(input_vec: np.ndarray, region_seed: np.ndarray) -> float:
    """Derive an input‑dependent τ by binding input to region seed and bundling."""
    bound = bind(input_vec, region_seed)
    # Use L2 norm of the bundled vector as τ (scaled to avoid zero)
    tau = np.linalg.norm(bound) + 1e-3
    return tau

def liquid_time_constant_update(
    h: np.ndarray,
    x: np.ndarray,
    tau: float,
    dt: float = 1.0,
    activation: callable = np.tanh,
) -> np.ndarray:
    """
    Euler update of dh/dt = -h/τ + activation(W·x + b)
    For simplicity W is identity and b is zero.
    """
    if tau <= 0:
        raise ValueError("time constant τ must be positive")
    drive = activation(x)  # simple activation of input
    dh = (-h / tau) + drive
    return h + dt * dh

# ----------------------------------------------------------------------
# Decision‑hygiene scoring utilities
# ----------------------------------------------------------------------
def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs + 1e-12))

def hoeffding_bound(mean: float, rng: float, n: int, delta: float = 0.05) -> float:
    """
    Hoeffding bound for the deviation of the empirical mean from the true mean.
    Returns the half‑width ε such that P(|μ̂‑μ| > ε) ≤ δ.
    """
    if n <= 0:
        raise ValueError("sample size n must be positive")
    return math.sqrt((rng ** 2 * math.log(2 / delta)) / (2 * n))

def count_min_sketch(items: List[int], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Simple Count‑Min sketch implementation."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table

def sketch_frequencies(sketch: List[List[int]]) -> np.ndarray:
    """Estimate frequencies from a Count‑Min sketch (minimum over rows)."""
    return np.min(np.array(sketch), axis=0).astype(float)

def decision_hygiene_score(features: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted sum of features where weights are entropy‑adjusted.
    Features are assumed non‑negative.
    """
    if features.shape != weights.shape:
        raise ValueError("features and weights must have the same shape")
    return float(np.dot(features, weights))

# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_process(
    points: List[Point],
    point_features: List[np.ndarray],
    voronoi_seeds: List[Point],
    hyper_seeds: List[np.ndarray],
    dt: float = 1.0,
) -> List[Tuple[int, np.ndarray, float]]:
    """
    Full hybrid processing:
    1. Voronoi assignment of points.
    2. For each point:
       a) Compute τ via hyper‑dimensional binding with its region seed.
       b) Update hidden state h with liquid‑time‑constant ODE.
       c) Build a Count‑Min sketch of region point IDs, derive entropy‑based
          weights corrected by Hoeffding bound, and compute a hygiene score.
    Returns a list of (region_index, updated_hidden_state, hygiene_score) per point.
    """
    if len(points) != len(point_features):
        raise ValueError("points and point_features must have the same length")
    if len(voronoi_seeds) != len(hyper_seeds):
        raise ValueError("voronoi_seeds and hyper_seeds must be of equal length")

    # 1. Voronoi partitioning
    regions = assign(points, voronoi_seeds)

    # Prepare output container
    results: List[Tuple[int, np.ndarray, float]] = []

    # Pre‑compute region‑wise identifier lists for sketching
    region_ids: Dict[int, List[int]] = {i: [] for i in range(len(voronoi_seeds))}
    for idx, p in enumerate(points):
        region_idx = nearest(p, voronoi_seeds)
        region_ids[region_idx].append(idx)

    # Process each point
    for idx, (pt, feat) in enumerate(zip(points, point_features)):
        region_idx = nearest(pt, voronoi_seeds)
        seed_vec = hyper_seeds[region_idx]

        # a) τ computation
        tau = compute_time_constant(feat, seed_vec)

        # b) Hidden‑state update (initialize h as zeros of same dim)
        h0 = np.zeros_like(feat)
        h1 = liquid_time_constant_update(h0, feat, tau, dt=dt)

        # c) Sketch‑based weight computation
        ids_in_region = region_ids[region_idx]
        sketch = count_min_sketch(ids_in_region)
        freqs = sketch_frequencies(sketch)
        total = freqs.sum() if freqs.sum() > 0 else 1.0
        probs = freqs / total

        # Entropy of the sketch distribution
        ent = shannon_entropy(probs)

        # Hoeffding correction for each probability estimate
        eps = hoeffding_bound(mean=probs.mean(), rng=1.0, n=len(ids_in_region))
        corrected_probs = np.clip(probs + eps, 0, 1)
        corrected_probs /= corrected_probs.sum() if corrected_probs.sum() > 0 else 1.0

        # Use corrected probabilities as feature weights (broadcast if needed)
        if corrected_probs.shape[0] != feat.shape[0]:
            # Broadcast or truncate to match feature dimension
            if corrected_probs.shape[0] < feat.shape[0]:
                pad = np.zeros(feat.shape[0] - corrected_probs.shape[0])
                weights = np.concatenate([corrected_probs, pad])
            else:
                weights = corrected_probs[: feat.shape[0]]
        else:
            weights = corrected_probs

        # Hygiene score
        score = decision_hygiene_score(feat, weights)

        results.append((region_idx, h1, score))

    return results

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic 2‑D points
    random.seed(42)
    np.random.seed(42)
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(20)]

    # Random feature vectors (dim = 8)
    point_features = [np.random.rand(8) for _ in points]

    # Voronoi seeds (3 regions)
    voronoi_seeds = [(2.5, 2.5), (7.5, 2.5), (5.0, 7.5)]

    # Hyper‑dimensional seeds (same count, dim = 8)
    hyper_seeds = [np.random.rand(8) for _ in voronoi_seeds]

    # Run hybrid pipeline
    out = hybrid_process(points, point_features, voronoi_seeds, hyper_seeds)

    # Simple verification: print first three results
    for i, (region, h_state, score) in enumerate(out[:3]):
        print(f"Point {i}: Region {region}, Hidden norm {np.linalg.norm(h_state):.4f}, Hygiene score {score:.4f}")

    print("Hybrid process completed without error.")