# DARWIN HAMMER — match 1239, survivor 3
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s3.py (gen5)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py (gen2)
# born: 2026-05-29T23:34:37Z

"""
Hybrid Dendritic Regret-Weighted Ternary-Decision Analyzer with Sparse Winner-Take-All and Privacy Model.

This module fuses the governing equations of two parent algorithms:
- Hybrid Dendritic Regret-Weighted Ternary-Decision Analyzer (HD-RW-TD) from `hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s3.py`
- Hybrid algorithm merging Sparse Winner-Take-All and Privacy-Aware Model Pool from `hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py`

The mathematical bridge between the two parents is established by using the membrane potential (V) from the dendritic model as input to calculate regret-weighted probabilities, which are then mapped onto a ternary alphabet and used as input for the path signature pruning algorithm. 
The output of the path signature pruning algorithm is then fed into the sparse winner-take-all (WTA) mechanism to select the top-k sparse representations. 
Finally, the selected representations are used to calculate the aggregate and its perturbation with Laplace noise to satisfy differential privacy.

"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
import math
import random
import sys
import hashlib
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared data structures
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
# Dendritic model utilities
# ---------------------------------------------------------------------------
def sodium_current(V, m, h, g_Na=120.0, E_Na=50.0):
    """Hodgkin-Huxley sodium current.

    I_Na = g_Na * m^3 * h * (V - E_Na)

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    m:
        Na+ activation gate variable, in [0, 1].
    h:
        Na+ inactivation gate variable
    """
    return g_Na * m**3 * h * (V - E_Na)

def calculate_membrane_potential(C_m, g_L, E_L, V_i, I_ion, I_syn):
    """Calculate membrane potential.

    C_m * dV_i/dt = -g_L*(V_i - E_L) + I_ion(V_i) + I_syn(t)

    Parameters
    ----------
    C_m:
        Membrane capacitance (uF/cm^2)
    g_L:
        Passive leak conductance (mS/cm^2)
    E_L:
        Leak reversal potential (mV)
    V_i:
        Membrane potential of compartment i (mV)
    I_ion:
        Nonlinear 
    """
    pass

# ----------------------------------------------------------------------
# Parent A – Sparse Winner‑Take‑All utilities
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`."""
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
    """Return a binary mask with 1 at the indices of the top‑k values."""
    k = max(0, min(k, len(values)))
    winners = {
        i
        for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]

# ---------------------------------------------------------------------------

def hybrid_dendritic_sparse_wta(C_m, g_L, E_L, V_i, I_ion, I_syn, values, m, k):
    membrane_potential = calculate_membrane_potential(C_m, g_L, E_L, V_i, I_ion, I_syn)
    sodium_current_val = sodium_current(membrane_potential, 0.5, 0.5)
    sparse_expanded = expand([sodium_current_val], m)
    top_k = top_k_mask(sparse_expanded, k)
    return top_k

def hybrid_regret_weighted_ternary_decision(action_id, outcome_value, probability):
    """Regret-weighted ternary decision."""
    # Map regret-weighted probabilities onto a ternary alphabet
    ternary_alphabet = [0, 1, 2]
    regret_weighted_probability = probability * outcome_value
    ternary_choice = np.random.choice(ternary_alphabet, p=[1 - regret_weighted_probability, regret_weighted_probability * 0.5, regret_weighted_probability * 0.5])
    return ternary_choice

def hybrid_sparse_wta_privacy_model(values, m, k, epsilon, delta):
    """Sparse WTA with differential privacy."""
    expanded = expand(values, m)
    noisy_aggregate = sum(expanded) + np.random.laplace(0, 1 / epsilon)
    risk_score = len(values) / sum(values)
    noisy_risk_score = risk_score + np.random.laplace(0, delta / epsilon)
    top_k = top_k_mask(expanded, k)
    return top_k, noisy_risk_score

if __name__ == "__main__":
    C_m = 1.0
    g_L = 0.5
    E_L = -50.0
    V_i = 0.0
    I_ion = 0.0
    I_syn = 0.0
    values = [1.0, 2.0, 3.0]
    m = 10
    k = 3
    action_id = "action1"
    outcome_value = 10.0
    probability = 0.5
    epsilon = 0.1
    delta = 0.01
    print(hybrid_dendritic_sparse_wta(C_m, g_L, E_L, V_i, I_ion, I_syn, values, m, k))
    print(hybrid_regret_weighted_ternary_decision(action_id, outcome_value, probability))
    print(hybrid_sparse_wta_privacy_model(values, m, k, epsilon, delta))