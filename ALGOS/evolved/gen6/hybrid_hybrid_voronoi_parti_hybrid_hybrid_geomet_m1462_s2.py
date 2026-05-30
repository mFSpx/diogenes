# DARWIN HAMMER — match 1462, survivor 2
# gen: 6
# parent_a: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s2.py (gen2)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m461_s2.py (gen5)
# born: 2026-05-29T23:36:29Z

"""Hybrid Voronoi-Circuit-Capybara Optimizer

This module fuses the two parent algorithms:

* **Parent A – Hybrid Voronoi-Circuit-Morphology Engine** 
  (`hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s2.py`): 
  supplies a Euclidean distance metric, “nearest-seed” assignment logic, 
  reliability counter, recovery priority, and shape descriptors.
* **Parent B – Hybrid Geometric-Decision-Capybara Optimizer** 
  (`hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m461_s2.py`): 
  provides a scalar modulation of a bilinear learning step via Shannon 
  entropy and a radial-basis-function (RBF) surrogate model.

The mathematical bridge is established by modulating the 
`hybrid health-distance score` S of Parent A with the effective 
learning rates of Parent B. Specifically, we redefine the score as:

S'(e, p) = (R(e)) ^ w_r   ·   (1 – d(p, c_e)/D_max) ^ w_d
          · (P(e)) ^ w_p   ·   (σ(e)) ^ w_s   ·   (1/φ(e)) ^ w_f
          · (1 + H) ^ w_h   ·   σ(ŝ) ^ w_g

where  
- `R(e)`, `d(p, c_e)`, `D_max`, `P(e)`, `σ(e)`, and `φ(e)` are as in Parent A,
- `H` is the Shannon entropy of decision-feature counts,
- `ŝ` is the RBF surrogate prediction,
- `w_*` are configurable weights that sum to 1.

The module provides three public functions demonstrating the hybrid operation:

* `hybrid_score(endpoint, point, max_dist, counts, rbf_model)` 
  – compute S' for a single pair.
* `hybrid_assign(points, pool, counts, rbf_model)` 
  – assign a list of points to the best endpoints.
* `hybrid_score_matrix(endpoints, points, counts, rbf_model)` 
  – return a NumPy matrix of all scores.
"""

import math
import random
import sys
import numpy as np
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

def shannon_entropy(counts: np.ndarray) -> float:
    """Return the Shannon entropy of a non-negative integer count vector."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log2(probs[mask])))

def rbf_surrogate(x: float) -> float:
    """Simple radial-basis-function surrogate model."""
    return np.exp(-x**2)

def sigmoid(x: float) -> float:
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def hybrid_score(endpoint, point, max_dist, counts, rbf_model):
    R = endpoint['R']
    d = np.linalg.norm(np.array(point) - np.array(endpoint['c']))
    D_max = max_dist
    P = endpoint['P']
    sigma = endpoint['sigma']
    phi = endpoint['phi']
    H = shannon_entropy(counts)
    s = rbf_surrogate(rbf_model)
    w_r, w_d, w_p, w_s, w_f, w_h, w_g = 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1

    S = (R ** w_r) * ((1 - d/D_max) ** w_d) * (P ** w_p) * (sigma ** w_s) * ((1/phi) ** w_f) * ((1 + H) ** w_h) * (sigmoid(s) ** w_g)
    return S

def hybrid_assign(points, pool, counts, rbf_model):
    assignments = {}
    for point in points:
        best_endpoint = max(pool, key=lambda endpoint: hybrid_score(endpoint, point, max([np.linalg.norm(np.array(point) - np.array(endpoint['c'])) for point in points]), counts, rbf_model))
        assignments[point] = best_endpoint
    return assignments

def hybrid_score_matrix(endpoints, points, counts, rbf_model):
    scores = np.zeros((len(endpoints), len(points)))
    for i, endpoint in enumerate(endpoints):
        for j, point in enumerate(points):
            scores[i, j] = hybrid_score(endpoint, point, max([np.linalg.norm(np.array(point) - np.array(endpoint['c'])) for point in points]), counts, rbf_model)
    return scores

if __name__ == "__main__":
    # Example usage
    endpoints = [{'R': 1, 'c': [0, 0], 'P': 1, 'sigma': 1, 'phi': 1}, {'R': 1, 'c': [1, 1], 'P': 1, 'sigma': 1, 'phi': 1}]
    points = [[0, 0], [1, 1]]
    counts = np.array([1, 1])
    rbf_model = 1.0

    print(hybrid_score(endpoints[0], points[0], 2, counts, rbf_model))
    print(hybrid_assign(points, endpoints, counts, rbf_model))
    print(hybrid_score_matrix(endpoints, points, counts, rbf_model))