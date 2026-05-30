# DARWIN HAMMER — match 605, survivor 1
# gen: 3
# parent_a: dense_associative_memory.py (gen0)
# parent_b: hybrid_hybrid_pheromone_inf_privacy_m54_s2.py (gen2)
# born: 2026-05-29T23:29:53Z

"""
Hybrid Dense Associative Memory - Pheromone Infotaxis Privacy System
================================================================
This module fuses the *Dense Associative Memory* (parent algorithm A) with the
*Pheromone Infotaxis Privacy System* (parent algorithm B).  

Mathematical bridge
-------------------
- The Dense Associative Memory computes an **energy** `E(xi) = -beta^{-1} log( sum_i exp(beta * M_i . xi) )`.
- The Pheromone Infotaxis Privacy System supplies a **reconstruction-risk score** `R ∈ [0,1]`.

We treat the risk score as a *scalar weight* that modulates the energy function:
   w = 1 – R                     # privacy‑preserving weight
   Ê = w · E(xi)                 # weighted energy value

The hybrid system therefore:
1. Computes raw pheromone signals and associated risk scores.
2. Forms weighted energy values `Ê = (1‑R)·E(xi)`.
3. Uses the weighted energy values in decision making and differentially‑private aggregation.

The implementation below provides three exemplar hybrid functions:
`hybrid_energy`, `hybrid_update_rule`, and `best_privacy_action`.  
"""

import numpy as np
from pathlib import Path
import math
import random
import sys
from datetime import datetime, timezone
from typing import Any, Dict

def _softmax(z):
    """Numerically stable softmax over 1-D array z."""
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def _lse(z):
    """log-sum-exp of 1-D array z (numerically stable)."""
    m = z.max()
    return m + np.log(np.exp(z - m).sum())

def energy(xi, M, beta=1.0):
    """Compute the Dense AM energy E(xi).

    Parameters
    ----------
    xi : array shape (d,)
        Query / current state vector.
    M : array shape (N, d)
        Memory matrix — each row is one stored pattern.
    beta : float
        Inverse temperature. Higher -> sharper energy wells.

    Returns
    -------
    float
        Scalar energy value. Fixed-point attractors are local minima.
    """
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * xi @ xi
    return -lse_term + quadratic_term

class PheromoneSystem:
    def __init__(self) -> None:
        self.pheromones: Dict[str, Dict[str, Any]] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {}
        if signal_kind not in self.pheromones[surface_key]:
            self.pheromones[surface_key][signal_kind] = (signal_value, now)
        else:
            prev_value, prev_time = self.pheromones[surface_key][signal_kind]
            time_diff = (now - prev_time).total_seconds()
            decayed_value = prev_value * math.exp(-math.log(2) * time_diff / half_life_seconds)
            self.pheromones[surface_key][signal_kind] = (decayed_value + signal_value, now)
        return self.pheromones[surface_key][signal_kind][0]

def hybrid_energy(xi, M, beta=1.0, R=0.0):
    """Compute the hybrid energy Ê(xi).

    Parameters
    ----------
    xi : array shape (d,)
        Query / current state vector.
    M : array shape (N, d)
        Memory matrix — each row is one stored pattern.
    beta : float
        Inverse temperature. Higher -> sharper energy wells.
    R : float
        Reconstruction-risk score.

    Returns
    -------
    float
        Scalar hybrid energy value.
    """
    w = 1 - R
    return w * energy(xi, M, beta)

def hybrid_update_rule(xi, M, beta=1.0, R=0.0):
    """Compute the hybrid update rule xi_new.

    Parameters
    ----------
    xi : array shape (d,)
        Query / current state vector.
    M : array shape (N, d)
        Memory matrix — each row is one stored pattern.
    beta : float
        Inverse temperature. Higher -> sharper energy wells.
    R : float
        Reconstruction-risk score.

    Returns
    -------
    array shape (d,)
        Updated state vector xi_new.
    """
    w = 1 - R
    scores = beta * (M @ xi)
    softmax_scores = _softmax(scores)
    return M.T @ (w * softmax_scores)

def best_privacy_action(xi, M, beta=1.0, R_values=[0.0, 0.5, 1.0]):
    """Find the best privacy action.

    Parameters
    ----------
    xi : array shape (d,)
        Query / current state vector.
    M : array shape (N, d)
        Memory matrix — each row is one stored pattern.
    beta : float
        Inverse temperature. Higher -> sharper energy wells.
    R_values : list of float
        List of reconstruction-risk scores.

    Returns
    -------
    float
        Optimal reconstruction-risk score R.
    """
    energies = [hybrid_energy(xi, M, beta, R) for R in R_values]
    return R_values[np.argmin(energies)]

if __name__ == "__main__":
    np.random.seed(0)
    M = np.random.rand(10, 5)
    xi = np.random.rand(5)
    beta = 1.0
    R = 0.5
    print(hybrid_energy(xi, M, beta, R))
    print(hybrid_update_rule(xi, M, beta, R))
    R_values = [0.0, 0.5, 1.0]
    print(best_privacy_action(xi, M, beta, R_values))