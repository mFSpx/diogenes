# DARWIN HAMMER — match 2959, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_fisher_m2596_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2494_s2.py (gen6)
# born: 2026-05-29T23:46:52Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_distri_hybrid_hybrid_fisher_m2596_s1.py
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2494_s2.py

Mathematical Bridge
-------------------
Parent A supplies information‑theoretic primitives (Fisher information,
Hoeffding concentration bounds, simulated annealing acceptance) while
Parent B supplies a Clifford (geometric) algebra representation of
high‑dimensional objects together with a variational free‑energy
criterion based on reconstruction error.

The fusion rests on interpreting the coefficients of a multivector as
statistical parameters.  Their Fisher information matrix quantifies
sensitivity, which can be bounded by Hoeffding’s inequality when the
coefficients are estimated from a finite sample set.  The geometric
product of multivectors supplies a binding operation; weighting this
product by the annealed acceptance probability yields a stochastic
update rule.  The resulting hybrid free‑energy combines a reconstruction
term, an entropy term derived from the Hoeffding bound, and a regulariser
proportional to the Fisher information trace.

The module implements three core hybrid functions:
1. `hybrid_fisher_hoeffding_bound` – couples acceptance probability,
   Fisher information, and Hoeffding concentration.
2. `geometric_bind` – geometric (Clifford) product of two multivectors.
3. `hybrid_variational_free_energy` – free‑energy estimate that merges
   the reconstruction error of Parent B with the information‑theoretic
   regularisation of Parent A.
"""

import math
import random
import sys
import pathlib
import numpy as np
from typing import List, Tuple, FrozenSet, Dict, Set

# ----------------------------------------------------------------------
# Parent A – probabilistic primitives
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis‑Hastings acceptance for a simulated‑annealing step."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hoeffding_bound(n: int, delta: float = 0.05) -> float:
    """
    Two‑sided Hoeffding bound for the mean of bounded i.i.d. variables
    in [0,1].  Returns epsilon such that P(|mean - μ| > ε) ≤ δ.
    """
    if n <= 0:
        raise ValueError("sample size must be positive")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def fisher_information(params: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Approximate Fisher information matrix for a Gaussian likelihood
    with unit variance:  I(θ) = Σ (∂/∂θ log p(x|θ)) (∂/∂θ log p(x|θ))ᵀ.
    For N(θ,1) the gradient is (x-θ), so I = Σ (x-θ)(x-θ)ᵀ.
    """
    if params.ndim != 1 or data.ndim != 2:
        raise ValueError("params must be 1‑D, data must be 2‑D (samples × dim)")
    diffs = data - params  # shape (samples, dim)
    return diffs.T @ diffs  # (dim, dim)


# ----------------------------------------------------------------------
# Parent B – Geometric (Clifford) Algebra core
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate index cancels (e_i * e_i = 1)
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: FrozenSet[int], blade_b: FrozenSet[int]
) -> Tuple[FrozenSet[int], int]:
    """
    Geometric product of two basis blades.
    Returns (result_blade, sign) where sign ∈ {+1, -1}.
    """
    if not blade_a:
        return blade_b, 1
    if not blade_b:
        return blade_a, 1
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


Multivector = Dict[FrozenSet[int], float]  # blade → coefficient


def geometric_product(mv_a: Multivector, mv_b: Multivector) -> Multivector:
    """Full geometric (Clifford) product of two multivectors."""
    result: Multivector = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
    # prune near‑zero coefficients
    return {b: c for b, c in result.items() if abs(c) > 1e-12}


def extract_vector_part(mv: Multivector) -> np.ndarray:
    """
    Return the grade‑1 (vector) part of a multivector as a dense numpy array.
    The dimension is inferred from the highest index appearing in any blade.
    """
    max_idx = 0
    for blade in mv.keys():
        if blade:
            max_idx = max(max_idx, max(blade))
    vec = np.zeros(max_idx + 1)
    for blade, coeff in mv.items():
        if len(blade) == 1:
            idx = next(iter(blade))
            vec[idx] = coeff
    return vec


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def hybrid_fisher_hoeffding_bound(
    delta_e: float,
    temperature: float,
    params: np.ndarray,
    data: np.ndarray,
    n_samples: int,
    delta: float = 0.05,
) -> float:
    """
    Combines simulated‑annealing acceptance with a confidence factor
    derived from Hoeffding’s inequality and the trace of the Fisher
    information matrix.

    Returns a scalar in [0,1] that can be used as a stochastic weight.
    """
    # base acceptance from Parent A
    base = acceptance_probability(delta_e, temperature)

    # Hoeffding concentration (smaller epsilon → higher confidence)
    epsilon = hoeffding_bound(n_samples, delta)
    hoeffding_factor = max(0.0, 1.0 - epsilon)  # map to [0,1]

    # Fisher regularisation (trace gives overall sensitivity)
    fisher_mat = fisher_information(params, data)
    fisher_trace = np.trace(fisher_mat)
    # Normalise trace to [0,1] using a simple logistic squash
    fisher_factor = 1.0 / (1.0 + math.exp(fisher_trace - n_samples))

    # Hybrid weight – the three ingredients act multiplicatively
    return base * hoeffding_factor * fisher_factor


def geometric_bind(mv1: Multivector, mv2: Multivector, weight: float) -> Multivector:
    """
    Bind two multivectors using the geometric product and then scale the
    resulting coefficients by a scalar `weight` (e.g., the hybrid acceptance
    weight from `hybrid_fisher_hoeffding_bound`).
    """
    bound = geometric_product(mv1, mv2)
    return {blade: coeff * weight for blade, coeff in bound.items()}


def hybrid_variational_free_energy(
    belief_mv: Multivector,
    observation_mv: Multivector,
    temperature: float,
    params: np.ndarray,
    data: np.ndarray,
    n_samples: int,
) -> float:
    """
    Variational free energy that merges Parent B's reconstruction error
    with Parent A's information‑theoretic regularisation.

    - Reconstruction error: squared Euclidean distance between the vector
      parts of the belief and the observation.
    - Entropy term: -log( hybrid_fisher_hoeffding_bound ).
    - Energy scaling by temperature (simulated annealing).

    Returns the scalar free‑energy value.
    """
    # 1) Reconstruction error (L2)
    belief_vec = extract_vector_part(belief_mv)
    obs_vec = extract_vector_part(observation_mv)
    recon_error = np.sum((belief_vec - obs_vec) ** 2)

    # 2) Hybrid information weight
    delta_e = recon_error  # treat error as energy change
    info_weight = hybrid_fisher_hoeffding_bound(
        delta_e, temperature, params, data, n_samples
    )
    # Guard against log(0)
    entropy = -math.log(max(info_weight, 1e-12))

    # 3) Total free energy (energy - temperature * entropy)
    free_energy = recon_error - temperature * entropy
    return free_energy


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic data
    dim = 4
    np.random.seed(0)
    true_params = np.random.randn(dim)
    samples = np.random.randn(100, dim) + true_params  # Gaussian with unit variance

    # Construct two tiny multivectors (grade‑0 and grade‑1 components)
    mv_a: Multivector = {
        frozenset(): 0.7,                     # scalar part
        frozenset({0}): 1.2,                  # e0 component
        frozenset({2}): -0.5,                 # e2 component
    }
    mv_b: Multivector = {
        frozenset(): 0.4,
        frozenset({1}): 0.9,
        frozenset({3}): 0.3,
    }

    # Simulated annealing schedule
    k = 5
    temp = cooling_temperature(k, t0=2.0, alpha=0.9)

    # Hybrid weight
    delta_energy = 0.3
    weight = hybrid_fisher_hoeffding_bound(
        delta_energy, temp, true_params, samples, n_samples=100
    )
    print(f"Hybrid weight (0‑1): {weight:.4f}")

    # Binding operation
    bound_mv = geometric_bind(mv_a, mv_b, weight)
    print("Bound multivector coefficients:")
    for blade, coeff in bound_mv.items():
        print(f"  {sorted(list(blade))}: {coeff:.4f}")

    # Free‑energy evaluation
    free_energy = hybrid_variational_free_energy(
        belief_mv=bound_mv,
        observation_mv=mv_a,  # treat mv_a as noisy observation
        temperature=temp,
        params=true_params,
        data=samples,
        n_samples=100,
    )
    print(f"Hybrid variational free energy: {free_energy:.4f}")

    sys.exit(0)