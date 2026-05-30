# DARWIN HAMMER — match 1209, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m614_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_counterfactual_effec_m316_s0.py (gen4)
# born: 2026-05-29T23:34:28Z

"""Hybrid algorithm combining geometric‑algebra inspired Voronoi/Ollivier‑Ricci curvature
with reconstruction‑risk‑adjusted causal effect estimation.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m614_s0.py (geometric product,
  Voronoi partitioning, Ollivier‑Ricci curvature, Shannon entropy, probabilistic
  labeling)
- hybrid_hybrid_hybrid_hybrid_counterfactual_effec_m316_s0.py (reconstruction
  risk scoring, causal effect estimation, weighted average treatment effect)

Mathematical bridge:
The Voronoi regions give a discrete probability distribution over space.
Ollivier‑Ricci curvature quantifies how “close” two such distributions are
relative to the distance between their seeds.  The reconstruction‑risk score
provides a scalar confidence weight for causal effect estimates.  By weighting
the Average Treatment Effect (ATE) with the risk score we obtain a
Weighted‑Average‑Treatment‑Effect (WATE).  The hybrid metric multiplies the
curvature (a geometric connectivity measure) with the WATE (a statistical
effect measure), yielding a single scalar that reflects both spatial
connectivity and confidence‑adjusted causality.

The module implements the full pipeline:
1. Voronoi partitioning of points.
2. Approximate Ollivier‑Ricci curvature between seed pairs.
3. Shannon entropy of pheromone (region‑wise) intensities.
4. Reconstruction‑risk‑adjusted causal effect estimation.
5. Hybrid metric combining curvature and weighted causal effect.
"""

import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Iterable, Any

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the closest seed (ties broken by index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to its nearest seed, returning a Voronoi region map."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Entropy utilities (from Parent A)
# ----------------------------------------------------------------------
def shannon_entropy(probs: List[float]) -> float:
    """Standard Shannon entropy H = -Σ p·log(p). Zero‑probability entries are ignored."""
    eps = 1e-12
    return -sum(p * math.log(p + eps) for p in probs if p > eps)

def pheromone_distribution(regions: Dict[int, List[Point]]) -> List[float]:
    """Treat region sizes as a proxy for pheromone intensity and normalize."""
    total = sum(len(v) for v in regions.values())
    if total == 0:
        return [0.0 for _ in regions]
    return [len(v) / total for v in regions.values()]

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature approximation (derived from Parent A)
# ----------------------------------------------------------------------
def region_centroid(region: List[Point]) -> Point:
    """Mean position of points in a region; if empty returns (0,0)."""
    if not region:
        return (0.0, 0.0)
    xs, ys = zip(*region)
    return (float(np.mean(xs)), float(np.mean(ys)))

def ollivier_ricci_curvature(
    seed_i: Point,
    seed_j: Point,
    region_i: List[Point],
    region_j: List[Point]
) -> float:
    """
    Approximate Ollivier‑Ricci curvature κ(i,j) = 1 - W1(μ_i, μ_j) / d(seed_i, seed_j)
    where μ are uniform distributions over the two Voronoi regions.
    For uniform distributions the 1‑Wasserstein distance reduces to the Euclidean
    distance between centroids.
    """
    d_ij = distance(seed_i, seed_j)
    if d_ij == 0:
        return 0.0
    c_i = region_centroid(region_i)
    c_j = region_centroid(region_j)
    w1 = distance(c_i, c_j)  # uniform‑mass transport cost
    return 1.0 - w1 / d_ij

def curvature_matrix(seeds: List[Point], regions: Dict[int, List[Point]]) -> np.ndarray:
    """Full pairwise curvature matrix κ_{ab} for all seed pairs."""
    n = len(seeds)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            κ = ollivier_ricci_curvature(seeds[i], seeds[j], regions[i], regions[j])
            mat[i, j] = mat[j, i] = κ
    return mat

# ----------------------------------------------------------------------
# Reconstruction risk (from Parent B)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk score in [0,1] proportional to the fraction of unique quasi‑identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

# ----------------------------------------------------------------------
# Causal effect estimation (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CausalEffect:
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]

def estimate_causal_effect(
    effect_id: str,
    treatment: str,
    outcome: str,
    confounders: List[str],
    data: Dict[str, List[float]],
) -> CausalEffect:
    """
    Very simple ATE estimator: difference of means between treated (t>=0.5)
    and control (t<0.5). Confidence interval is approximated with ±1.96*std/√n.
    """
    t_vals = np.array(data.get(treatment, []), dtype=float)
    y_vals = np.array(data.get(outcome, []), dtype=float)

    if t_vals.size == 0 or t_vals.shape != y_vals.shape:
        ate = None
        ci = None
    else:
        treated = y_vals[t_vals >= 0.5]
        control = y_vals[t_vals < 0.5]
        if treated.size == 0 or control.size == 0:
            ate = None
            ci = None
        else:
            ate = float(treated.mean() - control.mean())
            # Pooled standard deviation
            pooled_var = ((treated - treated.mean()) ** 2).sum() + ((control - control.mean()) ** 2).sum()
            pooled_sd = math.sqrt(pooled_var / (treated.size + control.size - 2)) if (treated.size + control.size) > 2 else 0.0
            se = pooled_sd * math.sqrt(1 / treated.size + 1 / control.size)
            ci = (ate - 1.96 * se, ate + 1.96 * se)

    return CausalEffect(
        effect_id=effect_id,
        treatment=treatment,
        outcome=outcome,
        confounders=tuple(confounders),
        ate_estimate=ate,
        ate_confidence_interval=ci,
        refutation_passed=False,
        refutation_methods=(),
        heterogeneous_effects={},
    )

def weighted_average_treatment_effect(causal: CausalEffect, risk_score: float) -> float | None:
    """
    Multiply the ATE by the reconstruction risk score to obtain a
    risk‑adjusted effect (WATE). Returns None if ATE is unavailable.
    """
    if causal.ate_estimate is None:
        return None
    return causal.ate_estimate * risk_score

# ----------------------------------------------------------------------
# Hybrid operations (new)
# ----------------------------------------------------------------------
def hybrid_metric_matrix(
    seeds: List[Point],
    regions: Dict[int, List[Point]],
    causal: CausalEffect,
    risk_score: float,
) -> np.ndarray:
    """
    Combine geometry and causality:
      M_{ij} = κ_{ij} * WATE
    where κ_{ij} is Ollivier‑Ricci curvature and WATE is the risk‑adjusted ATE.
    The resulting matrix reflects both spatial connectivity and statistical effect.
    """
    κ = curvature_matrix(seeds, regions)
    wate = weighted_average_treatment_effect(causal, risk_score)
    if wate is None:
        # If causal estimate missing, fall back to curvature only
        return κ
    return κ * wate

def region_entropy_and_curvature(
    seeds: List[Point],
    regions: Dict[int, List[Point]],
) -> Tuple[List[float], np.ndarray]:
    """
    Returns a list of Shannon entropies for each region's pheromone proxy
    and the pairwise curvature matrix.
    """
    probs = pheromone_distribution(regions)
    entropies = [shannon_entropy([p]) for p in probs]  # single‑bin entropy = 0, kept for interface
    κ = curvature_matrix(seeds, regions)
    return entropies, κ

def full_hybrid_pipeline(
    points: List[Point],
    seeds: List[Point],
    treatment: str,
    outcome: str,
    confounders: List[str],
    data: Dict[str, List[float]],
    unique_qi: int,
    total_records: int,
) -> Dict[str, Any]:
    """
    Execute the complete hybrid workflow:
      1. Voronoi partitioning
      2. Curvature computation
      3. Entropy of region sizes
      4. Reconstruction risk scoring
      5. Causal effect estimation
      6. Hybrid metric matrix
    Returns a dictionary with all intermediate results.
    """
    regions = assign(points, seeds)
    entropies, curvature = region_entropy_and_curvature(seeds, regions)

    risk = reconstruction_risk_score(unique_qi, total_records)
    causal = estimate_causal_effect(
        effect_id="hybrid_001",
        treatment=treatment,
        outcome=outcome,
        confounders=confounders,
        data=data,
    )
    hybrid_mat = hybrid_metric_matrix(seeds, regions, causal, risk)

    return {
        "regions": regions,
        "entropy_per_region": entropies,
        "curvature_matrix": curvature,
        "reconstruction_risk": risk,
        "causal_effect": asdict(causal),
        "hybrid_metric_matrix": hybrid_mat,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic spatial data
    random.seed(42)
    num_points = 200
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_points)]
    seeds = [(2.0, 2.0), (8.0, 2.0), (5.0, 8.0)]

    # Synthetic treatment/outcome data
    N = 150
    data = {
        "treatment": [random.choice([0.0, 1.0]) for _ in range(N)],
        "outcome": [random.gauss(5, 2) + 1.5 * random.choice([0, 1]) for _ in range(N)],
        "age": [random.randint(20, 70) for _ in range(N)],  # example confounder
    }

    # Run pipeline
    result = full_hybrid_pipeline(
        points=points,
        seeds=seeds,
        treatment="treatment",
        outcome="outcome",
        confounders=["age"],
        data=data,
        unique_qi=30,
        total_records=150,
    )

    # Simple sanity prints (no external libraries)
    print("Entropy per region:", result["entropy_per_region"])
    print("Curvature matrix:\n", result["curvature_matrix"])
    print("Reconstruction risk score:", result["reconstruction_risk"])
    print("Causal ATE estimate:", result["causal_effect"]["ate_estimate"])
    print("Hybrid metric matrix:\n", result["hybrid_metric_matrix"])