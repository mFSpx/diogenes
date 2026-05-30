# DARWIN HAMMER — match 5791, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1274_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m678_s1.py (gen5)
# born: 2026-05-30T00:04:48Z

"""
Hybrid Algorithm: Pheromone‑Physarum‑Geometric Fusion
====================================================

Parents
-------
- **Parent A** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1274_s2.py``  
  Provides pheromone‑based surface usage tracking, Fisher information on
  Gaussian‑like beams and entropy‑style uncertainty measures.

- **Parent B** – ``hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m678_s1.py``  
  Supplies a broadcast‑probability schedule, an exponential cooling schedule,
  and a Clifford‑algebra‑style geometric product used to evolve a Physarum‑
  inspired conductance matrix.

Mathematical Bridge
-------------------
Both parents expose *scalar temperature‑like quantities* that modulate updates:

* ``broadcast_probability`` (Parent B) – a decreasing weight that can be
  interpreted as a confidence in the current pheromone distribution.
* ``cooling_temperature`` (Parent B) – an exponential schedule that controls
  the aggressiveness of the Physarum conductance update.
* ``fisher_score`` (Parent A) – a local information‑theoretic curvature that
  measures how sharply the pheromone probability changes with respect to a
  virtual angular coordinate.

The hybrid algorithm treats the product  


τ = cooling_temperature(k) * broadcast_probability(phases, phase) * mean_fisher``  

as a *joint temperature* ``τ``.  This temperature simultaneously scales

1. the **Physarum conductance update** (via a geometric product) and
2. the **entropy‑regularised pheromone redistribution** (via a softmax‑like
   Fisher weighting).

Thus the two previously disjoint topologies are fused through a single
scalar field that drives both the graph‑flow dynamics and the probabilistic
surface‑usage dynamics.

The module below implements this fusion with three core functions and a
smoke‑test that runs without external data.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants (mirroring the original parents)
# ----------------------------------------------------------------------
BASE_TAU = 1.0          # baseline liquid time constant (Parent A)
ALPHA = 5.0             # gating steepness (Parent A)
LAMBDA = 0.7            # VFE weighting factor (Parent A)
MINHASH_K = 64          # number of hash functions for MinHash (Parent A)
MAX64 = (1 << 64) - 1   # mask for 64‑bit hashing (Parent A)
SEED_BASE = 123456789   # deterministic base seed for all RNGs (Parent A)

# ----------------------------------------------------------------------
# Parent‑A derived utilities
# ----------------------------------------------------------------------
def calculate_pheromone_probabilities(surface_key: str, limit: int) -> List[float]:
    """
    Simulate pheromone probabilities for ``limit`` virtual cells.
    The random seed is derived from ``surface_key`` to obtain reproducibility.
    """
    rng = random.Random(hash(surface_key) ^ SEED_BASE)
    pheromones = [rng.random() for _ in range(limit)]
    total = sum(pheromones) + 1e-12
    return [p / total for p in pheromones]


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single Gaussian‑shaped beam.
    """
    intensity = max(math.exp(-0.5 * ((theta - center) / width) ** 2), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative ** 2) / max(intensity, eps)


def fisher_information_vector(thetas: np.ndarray,
                              center: float = 0.0,
                              width: float = 1.0) -> np.ndarray:
    """
    Vectorised Fisher information for an array of angular coordinates.
    """
    intensity = np.exp(-0.5 * ((thetas - center) / width) ** 2)
    derivative = intensity * (-(thetas - center) / (width * width))
    # Guard against division by zero
    intensity = np.clip(intensity, 1e-12, None)
    return (derivative ** 2) / intensity


# ----------------------------------------------------------------------
# Parent‑B derived utilities
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    """
    Original A: p = 1 / 2^{phases‑phase}, clamped to [0,1].
    """
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """
    Original A: exponential cooling schedule.
    """
    if k < 0 or t0 < 0 or not (0.0 <= alpha < 1.0):
        raise ValueError("k, t0, and alpha must be valid")
    return t0 * (alpha ** k)


def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    A lightweight Clifford‑algebra inspired product.
    For 3‑D vectors it returns ``a·b + a∧b`` where the wedge is realised
    by the cross product.  For higher dimensions we fall back to the
    element‑wise product (which still respects associativity for the
    purposes of this hybrid).
    """
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape")
    if a.size == 3:
        dot = np.dot(a, b)
        cross = np.cross(a, b)
        # Return a vector whose first component stores the scalar part
        # and the remaining two store the bivector (cross) components.
        return np.array([dot, cross[0], cross[1]])  # simple embedding
    else:
        # Element‑wise product as a safe fallback
        return a * b


def update_conductance_matrix(conductance: np.ndarray,
                              temperature: float) -> np.ndarray:
    """
    Physarum‑style conductance update scaled by a temperature.
    The update uses the geometric product between each row and column,
    summed over the matrix to produce a symmetric perturbation.
    """
    if conductance.ndim != 2 or conductance.shape[0] != conductance.shape[1]:
        raise ValueError("Conductance must be a square matrix")
    n = conductance.shape[0]
    delta = np.zeros_like(conductance)
    for i in range(n):
        for j in range(i, n):
            prod = geometric_product(conductance[i, :], conductance[:, j])
            # Reduce the product to a scalar by taking its mean
            scalar = np.mean(prod)
            delta[i, j] = delta[j, i] = temperature * scalar
    # Simple Euler‑style update with a small damping factor
    damping = 0.1
    new_conductance = conductance + damping * delta
    # Ensure positivity (Physarum conductances are non‑negative)
    return np.clip(new_conductance, 0.0, None)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def joint_temperature(k: int,
                      phases: int,
                      phase: int,
                      fisher_vals: np.ndarray) -> float:
    """
    Combine the cooling schedule, broadcast probability and the mean
    Fisher information into a single scalar that drives both subsystems.
    """
    temp = cooling_temperature(k)
    broadcast = broadcast_probability(phases, phase)
    mean_fisher = float(np.mean(fisher_vals))
    return temp * broadcast * mean_fisher


def hybrid_pheromone_update(surface_key: str,
                            limit: int,
                            joint_temp: float) -> List[float]:
    """
    Update pheromone probabilities using a softmax that is tempered by the
    joint temperature.  Higher ``joint_temp`` sharpens the distribution,
    mimicking a lower entropy state.
    """
    raw = calculate_pheromone_probabilities(surface_key, limit)
    # Apply temperature‑scaled softmax
    scaled = np.array(raw) ** (1.0 / (joint_temp + 1e-12))
    total = np.sum(scaled) + 1e-12
    return (scaled / total).tolist()


def hybrid_iteration(surface_key: str,
                     limit: int,
                     conductance: np.ndarray,
                     k: int,
                     phases: int,
                     phase: int) -> Tuple[np.ndarray, List[float], np.ndarray]:
    """
    Perform one hybrid iteration:
      1. Compute pheromone probabilities.
      2. Derive Fisher information from the angular representation of the
         probability indices.
      3. Build the joint temperature ``τ``.
      4. Update the conductance matrix with ``τ``.
      5. Re‑weight the pheromone distribution with ``τ``.
    Returns the updated conductance matrix, the new pheromone probabilities,
    and the Fisher information vector.
    """
    # 1. Pheromone distribution
    probs = calculate_pheromone_probabilities(surface_key, limit)

    # 2. Fisher information (treat indices as angular coordinates)
    thetas = np.linspace(-math.pi, math.pi, limit, endpoint=False)
    fisher_vec = fisher_information_vector(thetas, center=0.0, width=1.0)

    # 3. Joint temperature
    tau = joint_temperature(k, phases, phase, fisher_vec)

    # 4. Conductance update
    new_conductance = update_conductance_matrix(conductance, tau)

    # 5. Temperature‑tempered pheromone update
    new_probs = hybrid_pheromone_update(surface_key, limit, tau)

    return new_conductance, new_probs, fisher_vec


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic seed for reproducibility
    random.seed(SEED_BASE)
    np.random.seed(SEED_BASE)

    # Dummy parameters
    surface = "test_surface"
    cell_limit = 8
    node_count = 5

    # Initialise a random symmetric conductance matrix
    base = np.random.rand(node_count, node_count)
    conductance_matrix = (base + base.T) / 2.0  # make symmetric

    # Hybrid iteration parameters
    iteration = 3
    total_phases = 5
    current_phase = 2

    # Run a single hybrid iteration
    updated_cond, updated_probs, fisher = hybrid_iteration(
        surface_key=surface,
        limit=cell_limit,
        conductance=conductance_matrix,
        k=iteration,
        phases=total_phases,
        phase=current_phase,
    )

    # Simple sanity prints (they are not required but harmless)
    print("Updated Conductance Matrix:\n", updated_cond)
    print("\nUpdated Pheromone Probabilities:\n", updated_probs)
    print("\nFisher Information Vector:\n", fisher)