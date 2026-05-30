# DARWIN HAMMER — match 2865, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1209_s3.py (gen6)
# parent_b: hybrid_hybrid_decreasing_pr_hybrid_path_signatur_m980_s0.py (gen4)
# born: 2026-05-29T23:47:43Z

"""Hybrid algorithm integrating Voronoi‑based geometric partitioning (Parent A) with
graph‑pruning via path‑signature Bayesian scoring (Parent B).

Mathematical bridge
-------------------
* Each Voronoi region is treated as a node; its centroid defines a graph vertex.
* An edge connects every pair of neighbouring seeds (Delaunay adjacency).  
  For an edge (i, j) we:
  1. Compute a **lead‑lag level‑2 signature** of the 2‑point geometric path
     (centroid_i → centroid_j).  Its Frobenius norm supplies a **likelihood**
     for a Bayesian update (Parent B).
  2. Estimate **Ollivier‑Ricci curvature** between the two regions by comparing
     uniform probability measures on the region point sets; the curvature
     acts as a multiplicative weight on the likelihood (Parent A).
  3. Obtain a **causal‑effect reconstruction risk** for each region (Parent A)
     and use the average risk of the two regions to down‑weight the posterior.
* The posterior survival probability of an edge is finally combined with its
  Euclidean length and an epistemic certainty flag to produce a **hybrid edge
  score**.  Edges below a time‑dependent threshold are pruned, yielding a
  unified workflow that respects both topologies.

The module provides three public functions that embody this hybrid pipeline:
`build_hybrid_graph`, `edge_hybrid_score`, and `prune_hybrid_graph`.  A small
smoke test is executed when run as a script.
"""

import math
import random
import sys
import pathlib
from collections import defaultdict
from typing import Tuple, List, Dict, Set

import numpy as np

# ----------------------------------------------------------------------
# Shared geometric primitives (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed to `point` (ties broken by index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi assignment of each point to its nearest seed."""
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def centroid(region: List[Point]) -> Point:
    """Arithmetic mean of a list of points; returns (0,0) for empty region."""
    if not region:
        return (0.0, 0.0)
    xs, ys = zip(*region)
    return (sum(xs) / len(region), sum(ys) / len(region))

# ----------------------------------------------------------------------
# Causal effect & reconstruction risk (simplified stand‑in for Parent A)
# ----------------------------------------------------------------------
def causal_effect_and_risk(region: List[Point]) -> Tuple[float, float]:
    """
    Dummy causal effect estimator:
    - effect  = mean of x‑coordinates,
    - risk    = standard deviation of y‑coordinates (higher = less reliable).
    """
    if not region:
        return 0.0, 1.0  # maximal risk for empty region
    xs, ys = zip(*region)
    effect = sum(xs) / len(xs)
    risk = float(np.std(ys, ddof=0) + 1e-9)  # avoid zero
    return effect, risk

# ----------------------------------------------------------------------
# Lead‑lag level‑2 signature (from Parent B)
# ----------------------------------------------------------------------
def lead_lag_signature(p1: Point, p2: Point) -> np.ndarray:
    """
    For a 2‑point linear path the level‑2 signature is ½ Δ⊗Δ,
    where Δ = p2 - p1.  Returns a 2×2 matrix.
    """
    delta = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    return 0.5 * np.outer(delta, delta)

def signature_frobenius_norm(sig: np.ndarray) -> float:
    """Frobenius norm of a matrix (used as a likelihood)."""
    return float(np.linalg.norm(sig, 'fro'))

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature (lightweight approximation)
# ----------------------------------------------------------------------
def ollivier_ricci(curve_points_i: List[Point],
                   curve_points_j: List[Point],
                   d_ij: float) -> float:
    """
    Approximate Ollivier‑Ricci curvature κ(i,j) = 1 - W/ d_ij,
    where W is the 1‑Wasserstein distance between uniform measures on the
    two point clouds.  Here we approximate W by the distance between the
    centroids (exact for single‑point clouds).
    """
    if d_ij == 0:
        return 0.0
    ci = centroid(curve_points_i)
    cj = centroid(curve_points_j)
    w = distance(ci, cj)
    return 1.0 - w / d_ij

# ----------------------------------------------------------------------
# Bayesian primitives (from Parent B)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)P(H) / P(E)."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Hybrid edge construction
# ----------------------------------------------------------------------
Edge = Tuple[int, int]   # indices of seed nodes

def build_hybrid_graph(points: List[Point],
                       seeds: List[Point],
                       prior_survival: float = 0.9,
                       false_positive: float = 0.01) -> Dict[Edge, Dict]:
    """
    Construct a fully connected graph on the Voronoi seeds.
    For each edge compute:
        * Euclidean length,
        * Lead‑lag signature norm (likelihood),
        * Ollivier‑Ricci curvature (weight),
        * Average causal‑effect risk,
        * Bayesian posterior survival probability,
        * Hybrid score (see `edge_hybrid_score`).
    Returns a dict mapping edges to a metadata dict.
    """
    # 1. Voronoi partition
    regions = assign(points, seeds)

    # 2. Pre‑compute per‑region quantities
    centroids = {i: centroid(reg) for i, reg in regions.items()}
    effects_risks = {i: causal_effect_and_risk(reg) for i, reg in regions.items()}

    graph: Dict[Edge, Dict] = {}
    n = len(seeds)

    for i in range(n):
        for j in range(i + 1, n):
            # geometric data
            pi, pj = centroids[i], centroids[j]
            d_ij = distance(pi, pj)

            # signature likelihood
            sig = lead_lag_signature(pi, pj)
            lik = signature_frobenius_norm(sig)
            # Normalise likelihood to [0,1] (simple scaling)
            lik_norm = min(lik / (lik + 1.0), 1.0)

            # curvature weight
            kappa = ollivier_ricci(regions[i], regions[j], d_ij)
            curvature_weight = max(kappa, 0.0)  # ignore negative curvature

            # combine curvature with likelihood
            weighted_likelihood = lik_norm * (1.0 + curvature_weight)

            # Bayesian update
            marginal = bayes_marginal(prior_survival, weighted_likelihood, false_positive)
            posterior = bayes_update(prior_survival, weighted_likelihood, marginal)

            # average reconstruction risk
            risk_i = effects_risks[i][1]
            risk_j = effects_risks[j][1]
            avg_risk = (risk_i + risk_j) / 2.0

            # store all intermediate values
            graph[(i, j)] = {
                "length": d_ij,
                "signature_norm": lik_norm,
                "curvature": kappa,
                "weighted_likelihood": weighted_likelihood,
                "posterior": posterior,
                "avg_risk": avg_risk,
                "centroids": (pi, pj),
            }

    return graph

def edge_hybrid_score(edge_data: Dict) -> float:
    """
    Compute a hybrid edge score that combines:
        * posterior survival probability,
        * inverse length (shorter edges preferred),
        * risk penalty (higher risk lowers score).
    The formula is:
        score = posterior * (1 / (1 + length)) * exp(-avg_risk)
    """
    posterior = edge_data["posterior"]
    length = edge_data["length"]
    avg_risk = edge_data["avg_risk"]
    length_factor = 1.0 / (1.0 + length)
    risk_factor = math.exp(-avg_risk)
    return posterior * length_factor * risk_factor

def prune_hybrid_graph(graph: Dict[Edge, Dict],
                       decay_rate: float = 0.05,
                       step: int = 0) -> Set[Edge]:
    """
    Time‑dependent pruning: edges with a hybrid score below a decaying
    threshold are removed.
        threshold = base * exp(-decay_rate * step)
    Returns the set of edges that survive pruning.
    """
    base_threshold = 0.1
    threshold = base_threshold * math.exp(-decay_rate * step)

    surviving = set()
    for edge, data in graph.items():
        score = edge_hybrid_score(data)
        if score >= threshold:
            surviving.add(edge)
    return surviving

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    # generate random points and seeds
    num_points = 200
    num_seeds = 8
    points = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(num_points)]
    seeds = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(num_seeds)]

    # build hybrid graph
    hybrid_graph = build_hybrid_graph(points, seeds)

    # compute scores and prune
    surviving_edges = prune_hybrid_graph(hybrid_graph, step=3)

    # simple report
    print(f"Total edges created: {len(hybrid_graph)}")
    print(f"Edges surviving after pruning: {len(surviving_edges)}")
    for e in sorted(surviving_edges):
        print(f"Edge {e} score = {edge_hybrid_score(hybrid_graph[e]):.4f}")