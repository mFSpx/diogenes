# DARWIN HAMMER — match 1209, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m614_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_counterfactual_effec_m316_s0.py (gen4)
# born: 2026-05-29T23:34:28Z

"""Hybrid algorithm combining geometric/Voronoi/Ollivier‑Ricci curvature and pheromone‑entropy
with reconstruction‑risk‑adjusted causal effect estimation.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m614_s0.py (geometric product, Voronoi partitioning,
  Ollivier‑Ricci curvature, pheromone surface usage, Shannon entropy, probabilistic labeling)
- hybrid_hybrid_hybrid_hybrid_counterfactual_effec_m316_s0.py (reconstruction risk scoring,
  causal effect estimation, weighted average treatment effect)

Mathematical bridge:
The Ollivier‑Ricci curvature 𝜅(i,j) quantifies the connectivity between neighboring
Voronoi regions i and j.  We treat the curvature as a similarity weight for the
distribution of pheromone‑derived probabilities (entropy‑regularized labels) and
use the reconstruction risk score ρ(i) of each region to re‑weight causal effect
estimates.  The final weighted average treatment effect (WATE) for the whole system
is

    WATE = Σ_i ρ(i)·(1+𝜅_i)·ATE_i / Σ_i ρ(i)·(1+𝜅_i)

where 𝜅_i is the mean curvature of region i with its Voronoi neighbours and
ATE_i is the average treatment effect estimated from data localized to region i.
"""

import math
import random
import sys
import pathlib
import statistics
from collections import Counter
from typing import List, Tuple, Dict, Iterable

import numpy as np

Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Geometric / Voronoi utilities (from Parent A)
# ----------------------------------------------------------------------
def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the nearest seed → Voronoi regions."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature approximation (simplified)
# ----------------------------------------------------------------------
def _probability_measure(region: List[Point], seed: Point) -> Dict[Point, float]:
    """Return a discrete probability measure over points of a region,
    weighted by inverse distance to the seed (simulating pheromone diffusion)."""
    if not region:
        return {}
    weights = np.array([1.0 / (distance(p, seed) + 1e-9) for p in region])
    total = weights.sum()
    return {p: w / total for p, w in zip(region, weights)}

def wasserstein_distance(mu: Dict[Point, float], nu: Dict[Point, float]) -> float:
    """A very cheap 1‑Wasserstein distance using Euclidean ground metric."""
    # Convert to arrays for speed
    pts_mu = np.array(list(mu.keys()))
    probs_mu = np.array(list(mu.values()))
    pts_nu = np.array(list(nu.keys()))
    probs_nu = np.array(list(nu.values()))

    # Compute pairwise distance matrix
    dmat = np.linalg.norm(pts_mu[:, None, :] - pts_nu[None, :, :], axis=2)
    # Solve transport problem with a greedy algorithm (acceptable for small sets)
    i, j = 0, 0
    dist = 0.0
    while i < len(probs_mu) and j < len(probs_nu):
        mass = min(probs_mu[i], probs_nu[j])
        dist += mass * dmat[i, j]
        probs_mu[i] -= mass
        probs_nu[j] -= mass
        if probs_mu[i] == 0:
            i += 1
        if probs_nu[j] == 0:
            j += 1
    return dist

def compute_curvature(regions: Dict[int, List[Point]],
                      seeds: List[Point]) -> Dict[int, float]:
    """Mean Ollivier‑Ricci curvature of each Voronoi cell with its neighbours."""
    curvature: Dict[int, float] = {}
    # Pre‑compute measures
    measures = {i: _probability_measure(regions[i], seeds[i]) for i in regions}
    for i in regions:
        neigh_idxs = [j for j in regions if j != i and distance(seeds[i], seeds[j]) < 2.0]  # adjacency heuristic
        if not neigh_idxs:
            curvature[i] = 0.0
            continue
        curvs = []
        for j in neigh_idxs:
            d_ij = distance(seeds[i], seeds[j])
            w_dist = wasserstein_distance(measures[i], measures[j])
            # Ollivier‑Ricci curvature κ = 1 - (W1 / d)
            κ = 1.0 - (w_dist / (d_ij + 1e-9))
            curvs.append(κ)
        curvature[i] = sum(curvs) / len(curvs)
    return curvature

# ----------------------------------------------------------------------
# Entropy & probabilistic labeling (from Parent A)
# ----------------------------------------------------------------------
def shannon_entropy(probs: Iterable[float]) -> float:
    """Standard Shannon entropy (base e)."""
    ent = 0.0
    for p in probs:
        if p > 0.0:
            ent -= p * math.log(p)
    return ent

def region_entropy(regions: Dict[int, List[Point]],
                   seeds: List[Point]) -> Dict[int, float]:
    """Entropy of the distance‑based pheromone distribution inside each region."""
    ent: Dict[int, float] = {}
    for i, pts in regions.items():
        if not pts:
            ent[i] = 0.0
            continue
        dists = np.array([distance(p, seeds[i]) for p in pts])
        # Convert distances to a probability distribution (closer points get higher weight)
        weights = 1.0 / (dists + 1e-9)
        probs = weights / weights.sum()
        ent[i] = shannon_entropy(probs)
    return ent

def probabilistic_labels(regions: Dict[int, List[Point]],
                         entropy: Dict[int, float]) -> Dict[int, Dict[str, float]]:
    """Create a soft label distribution per region; lower entropy → higher confidence."""
    labels: Dict[int, Dict[str, float]] = {}
    for i, pts in regions.items():
        # For demonstration we assume two possible labels: "A" and "B"
        # Probability for "A" is proportional to exp(-entropy)
        conf = math.exp(-entropy[i])
        prob_a = conf / (conf + 1.0)  # normalize with baseline for "B"
        labels[i] = {"A": prob_a, "B": 1.0 - prob_a}
    return labels

# ----------------------------------------------------------------------
# Reconstruction risk (from Parent B)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Risk score in [0,1] based on proportion of unique quasi‑identifiers."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0,
                         unique_quasi_identifiers / total_records))

# ----------------------------------------------------------------------
# Causal effect estimation (simplified version of Parent B)
# ----------------------------------------------------------------------
def estimate_ate(treatment: List[float], outcome: List[float]) -> float | None:
    """Simple difference‑in‑means ATE."""
    if not treatment or len(treatment) != len(outcome):
        return None
    treated = [y for t, y in zip(treatment, outcome) if t >= 0.5]
    control = [y for t, y in zip(treatment, outcome) if t < 0.5]
    if not treated or not control:
        return None
    return statistics.mean(treated) - statistics.mean(control)

def region_causal_effects(regions: Dict[int, List[Point]],
                          seeds: List[Point],
                          data: dict) -> Dict[int, float | None]:
    """Estimate ATE per Voronoi region using data localized to that region."""
    effects: Dict[int, float | None] = {}
    for i, pts in regions.items():
        # Simulate localisation: pick rows whose synthetic coordinate falls into region
        # Here we assume data contains parallel lists 'treatment' and 'outcome' of same length as pts
        # For the demo we just use the whole dataset (real implementation would filter)
        treatment = data.get('treatment', [])
        outcome = data.get('outcome', [])
        effects[i] = estimate_ate(treatment, outcome)
    return effects

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_curvature_entropy(points: List[Point],
                             seed_count: int = 5) -> Tuple[Dict[int, float],
                                                          Dict[int, float],
                                                          Dict[int, Dict[str, float]]]:
    """Compute Voronoi partition, curvature, entropy and probabilistic labels."""
    # Randomly initialise seeds
    seeds = random.sample(points, min(seed_count, len(points)))
    regions = assign(points, seeds)
    curv = compute_curvature(regions, seeds)
    ent = region_entropy(regions, seeds)
    labels = probabilistic_labels(regions, ent)
    return curv, ent, labels

def weighted_average_treatment_effect(points: List[Point],
                                      data: dict,
                                      unique_qi: int,
                                      total_records: int) -> float | None:
    """Combine reconstruction risk, curvature and causal effects into a WATE."""
    # Build geometry
    seeds = random.sample(points, min(5, len(points)))
    regions = assign(points, seeds)
    curv = compute_curvature(regions, seeds)

    # Causal effects per region
    ate_per_region = region_causal_effects(regions, seeds, data)

    # Risk score (global, but could be per region; we use global for simplicity)
    risk = reconstruction_risk_score(unique_qi, total_records)

    # Aggregate with weights w_i = risk * (1 + κ_i)
    numer = 0.0
    denom = 0.0
    for i, ate in ate_per_region.items():
        if ate is None:
            continue
        weight = risk * (1.0 + curv.get(i, 0.0))
        numer += weight * ate
        denom += weight
    return (numer / denom) if denom != 0.0 else None

def hybrid_labeling_with_causal_adjustment(points: List[Point],
                                            data: dict,
                                            unique_qi: int,
                                            total_records: int) -> Dict[int, Dict[str, float]]:
    """Produce label probabilities that are nudged by causal effect magnitude."""
    curv, ent, labels = hybrid_curvature_entropy(points)
    # Compute a simple global causal effect magnitude
    global_ate = estimate_ate(data.get('treatment', []), data.get('outcome', []))
    if global_ate is None:
        return labels

    # Adjust label "A" probability proportionally to sign of ATE and curvature
    adjusted: Dict[int, Dict[str, float]] = {}
    for i, prob_dict in labels.items():
        adj_factor = 1.0 + curv.get(i, 0.0) * math.tanh(global_ate)
        prob_a = prob_dict["A"] * adj_factor
        prob_a = max(0.0, min(1.0, prob_a))
        adjusted[i] = {"A": prob_a, "B": 1.0 - prob_a}
    return adjusted

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic 2‑D points
    random.seed(42)
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(200)]

    # Synthetic causal dataset
    n = 200
    treatment = [random.choice([0.0, 1.0]) for _ in range(n)]
    # Outcome depends weakly on treatment
    outcome = [t + random.gauss(0, 0.5) for t in treatment]

    data = {"treatment": treatment, "outcome": outcome}

    # Run hybrid functions
    curv, ent, labs = hybrid_curvature_entropy(points)
    print("Curvature sample:", {k: round(v, 3) for k, v in list(curv.items())[:3]})
    print("Entropy sample:", {k: round(v, 3) for k, v in list(ent.items())[:3]})
    print("Labels sample:", {k: v for k, v in list(labs.items())[:3]})

    wate = weighted_average_treatment_effect(points, data,
                                              unique_qi=30,
                                              total_records=200)
    print("Weighted Average Treatment Effect (WATE):", wate)

    adj_labels = hybrid_labeling_with_causal_adjustment(points, data,
                                                        unique_qi=30,
                                                        total_records=200)
    print("Adjusted labels sample:", {k: v for k, v in list(adj_labels.items())[:3]})