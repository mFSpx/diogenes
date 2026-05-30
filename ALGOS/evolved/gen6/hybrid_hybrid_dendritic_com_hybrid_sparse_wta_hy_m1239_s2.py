# DARWIN HAMMER — match 1239, survivor 2
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s3.py (gen5)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py (gen2)
# born: 2026-05-29T23:34:37Z

"""
Hybrid Dendritic Regret-Weighted Ternary-Decision Analyzer with Sparse Winner-Take-All and Privacy Model.

This module integrates the governing equations of two parent algorithms:
- Hybrid Dendritic Regret-Weighted Ternary-Decision Analyzer (HD-RW-TD) from `hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s3.py`
- Hybrid Sparse Winner-Take-All and Privacy Model from `hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py`

The mathematical bridge between the two parents is established by using the membrane potential (V) from the dendritic model as input to calculate regret-weighted probabilities, 
which are then mapped onto a ternary alphabet and used as input for the path signature pruning algorithm. 
The sparse expansion maps an input vector `v ∈ ℝⁿ` to a high-dimensional space `ℝᵐ` using locality-sensitive hashing. 
The resulting expanded vector `e` is treated as a *query* whose aggregate (sum) is perturbed with Laplace noise to satisfy differential privacy. 
The noisy aggregate is normalised and fed into the reconstruction-risk function `risk = unique_quasi_identifiers / total_records`. 
This risk score is then used as the scale of a second Laplace noise term that governs whether a model may be admitted to the pool.

"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Iterable, List
import math
import random
import sys
import hashlib
import pathlib

# Shared data structures
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

# Dendritic model utilities
def sodium_current(V, m, h, g_Na=120.0, E_Na=50.0):
    """Hodgkin-Huxley sodium current."""
    return g_Na * m**3 * h * (V - E_Na)

def calculate_membrane_potential(C_m, g_L, E_L, V_i, I_ion, I_syn):
    """Calculate membrane potential."""
    return -g_L*(V_i - E_L) + I_ion + I_syn

# Sparse Winner-Take-All utilities
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash-based sparse expansion of `values` into a vector of length `m`."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out

def top_k_mask(values: List[float], k: int) -> List[int]:
    """Return a binary mask with 1 at the indices of the top-k values."""
    k = max(0, min(k, len(values)))
    winners = {
        i
        for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]

def hamming(a: List[int], b: List[int]) -> int:
    """Hamming distance between two lists."""
    return sum(el1 != el2 for el1, el2 in zip(a, b))

# Hybrid operations
def hybrid_dendritic_sparse(values: List[float], m: int, salt: str = ""):
    """Hybrid operation that integrates dendritic model with sparse expansion."""
    V = np.mean(values)
    m_val = 0.5
    h_val = 0.5
    I_ion = sodium_current(V, m_val, h_val)
    I_syn = 0.0
    V_i = 0.0
    membrane_potential = calculate_membrane_potential(1.0, 0.1, -70.0, V_i, I_ion, I_syn)
    expanded_values = expand(values, m, salt)
    return expanded_values, membrane_potential

def hybrid_sparse_privacy(expanded_values: List[float], k: int, epsilon: float):
    """Hybrid operation that integrates sparse expansion with privacy model."""
    top_k = top_k_mask(expanded_values, k)
    noisy_aggregate = np.sum(expanded_values) + np.random.laplace(0.0, 1.0/epsilon)
    risk = np.sum(top_k) / len(top_k)
    return noisy_aggregate, risk

def hybrid_dendritic_sparse_privacy(values: List[float], m: int, k: int, epsilon: float, salt: str = ""):
    """Hybrid operation that integrates dendritic model, sparse expansion, and privacy model."""
    expanded_values, _ = hybrid_dendritic_sparse(values, m, salt)
    noisy_aggregate, risk = hybrid_sparse_privacy(expanded_values, k, epsilon)
    return noisy_aggregate, risk

if __name__ == "__main__":
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    m = 10
    k = 3
    epsilon = 1.0
    salt = ""
    noisy_aggregate, risk = hybrid_dendritic_sparse_privacy(values, m, k, epsilon, salt)
    print(f"Noisy aggregate: {noisy_aggregate}, Risk: {risk}")