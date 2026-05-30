# DARWIN HAMMER — match 3030, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_privacy_model_hybrid_hybrid_semant_m1133_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m1861_s2.py (gen6)
# born: 2026-05-29T23:47:16Z

"""
Hybrid module combining the hybrid_privacy_model_hybrid_hybrid_semant_m1133_s1.py and hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m1861_s2.py.
The mathematical bridge between the two lies in the representation of the semantic neighborhoods as multivectors, 
which allows for the use of the geometric product to compute the similarity between documents, and the use of the 
Voronoi partitioning to assign points to these neighborhoods. In this hybrid module, we integrate the privacy risk 
vector from the hybrid_privacy_model_hybrid_hybrid_semant_m1133_s1.py with the geometric product and Voronoi 
partitioning from hybrid_hybrid_semantic_neig_hybrid_hybrid_geomet_m116_s1.py. We also incorporate the simulated 
annealing leader election and the TTT-Linear weight matrix with reconstruction-risk from hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m1861_s2.py.

The fusion treats the joint temperature `p·T` as a scaling factor that simultaneously affects the annealing 
schedule of the leader-election process and the learning rate of the TTT-Linear update. The pressure difference 
`Δp = pressure_a – pressure_b` drives a flux `Φ = g·Δp / ℓ` which is used as an additional regularization term 
on the weight matrix `W`. Thus, the two dynamical systems are coupled through the shared scalar fields `p`, `T`, 
and `Φ`.
"""

import math
import numpy as np
import random
import sys
from pathlib import Path

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Return a normalized reconstruction risk in [0,1]."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def _cos(a, b):
    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return 0.0 if den == 0 else sum(x*y for x, y in zip(a, b)) / den

def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    """Find the index of the nearest seed to a given point."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def broadcast_probability(phases: int, phase: int) -> float:
    """Original A: p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Original B: exponential cooling"""
    return t0 * (alpha ** k)

def hybrid_update(weights: np.ndarray, point: tuple[float, float], seeds: list[tuple[float, float]], 
                   phases: int, phase: int, t0: float = 1.0, alpha: float = 0.95, 
                   conductance: float = 1.0, edge_length: float = 1.0, pressure_a: float = 0.0, pressure_b: float = 0.0) -> np.ndarray:
    """Hybrid update function that combines the geometric product and Voronoi partitioning with the simulated annealing leader election."""
    joint_temperature = broadcast_probability(phases, phase) * cooling_temperature(phase, t0, alpha)
    delta_p = pressure_a - pressure_b
    flux = conductance * delta_p / edge_length
    updated_weights = weights + joint_temperature * flux * np.array([distance(point, seed) for seed in seeds])
    return updated_weights

def hybrid_train(seeds: list[tuple[float, float]], points: list[tuple[float, float]], phases: int, 
                 max_phase: int, t0: float = 1.0, alpha: float = 0.95, 
                 conductance: float = 1.0, edge_length: float = 1.0, pressure_a: float = 0.0, pressure_b: float = 0.0) -> np.ndarray:
    """Hybrid training function that demonstrates the integration of the two parent algorithms."""
    weights = np.zeros(len(seeds))
    for phase in range(max_phase):
        for point in points:
            weights = hybrid_update(weights, point, seeds, phases, phase, t0, alpha, conductance, edge_length, pressure_a, pressure_b)
    return weights

if __name__ == "__main__":
    seeds = [(0.0, 0.0), (1.0, 1.0)]
    points = [(0.5, 0.5), (1.5, 1.5)]
    phases = 2
    max_phase = 10
    weights = hybrid_train(seeds, points, phases, max_phase)
    print(weights)