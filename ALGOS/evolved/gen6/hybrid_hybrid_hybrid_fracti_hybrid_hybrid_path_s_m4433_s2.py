# DARWIN HAMMER — match 4433, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_fractional_hd_pheromone_m2184_s0.py (gen3)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s2.py (gen5)
# born: 2026-05-29T23:55:39Z

"""
This module fuses the mathematical algorithms from hybrid_hybrid_fractional_hd_pheromone_m2184_s0.py and hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s2.py.

The mathematical bridge is a pheromone-guided regret minimization strategy that integrates the fractional power binding from Hybrid Power Binding with the path signature analysis from Hybrid Path Signature Analysis. The resulting regret is computed as a weighted sum of the expected values of the actions, where the weights are given by the pheromone-guided geometric indices vector.

"""

import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import random

# ---------------------------------------------------------------------------
# MathAction and MathCounterfactual classes
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def random_hv(d=10000, kind="complex", seed=None):
    """Generate a random hypervector of dimension d.

    Parameters
    ----------
    d:
        Dimension of the hypervector.
    kind:
        "complex"  — unit-magnitude complex vector (each component e^{i*theta},
                     theta ~ Uniform[0, 2pi]).  These are the natural carriers
                     for phase-based fractional binding.
        "bipolar"  — real vector with each component in {+1, -1}.
        "real"     — Gaussian sample normalized to unit L2 norm.
    seed:
        Integer seed for reproducibility; None for random.

    Returns
    -------
    np.ndarray
        Shape (d,).  dtype=complex128 for kind="complex", float64 otherwise.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)

def pheromone_guided_health_score(vector1, vector2, pheromone_signal):
    """Compute the pheromone-guided health score.

    Parameters
    ----------
    vector1:
        Fractional power bound vector.
    vector2:
        Pheromone-guided geometric indices vector.
    pheromone_signal:
        Pheromone signal value.

    Returns
    -------
    float
        Pheromone-guided health score.
    """
    return np.dot(vector1, vector2) * pheromone_signal

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          
    running    = path[:-1] - path[0]            
    return running.T @ increments               

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2      
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    return B

def calculate_regret_weighted_probabilities(actions: list, pheromone_signal: float) -> np.ndarray:
    probabilities = np.array([action.expected_value for action in actions])
    regret_weights = np.array([pheromone_guided_health_score(vector1, vector2, pheromone_signal) for vector1, vector2 in zip([action.expected_value for action in actions], [action.risk for action in actions])])
    return regret_weights

def hybrid_health_score(actions: list, path: np.ndarray, pheromone_signal: float) -> float:
    """Compute the hybrid health score.

    Parameters
    ----------
    actions:
        List of MathAction objects.
    path:
        Path signature.
    pheromone_signal:
        Pheromone signal value.

    Returns
    -------
    float
        Hybrid health score.
    """
    lead_lag_path = lead_lag_transform(path)
    signature = signature_level2(lead_lag_path)
    bspline_basis_matrix = bspline_basis(lead_lag_path[:, 0], np.linspace(0, 1, len(lead_lag_path)), k=3)
    health_score = np.dot(signature, bspline_basis_matrix)
    regret_weights = calculate_regret_weighted_probabilities(actions, pheromone_signal)
    return np.dot(health_score, regret_weights)

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    math_action = MathAction("action1", 1.0, 0.5)
    math_action2 = MathAction("action2", 2.0, 0.5)
    path = np.array([[1.0, 2.0], [2.0, 3.0]])
    pheromone_signal = 1.0
    print(hybrid_health_score([math_action, math_action2], path, pheromone_signal))

if __name__ == "__main__":
    smoke_test()