# DARWIN HAMMER — match 1239, survivor 4
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s3.py (gen5)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py (gen2)
# born: 2026-05-29T23:34:37Z

"""
Hybrid Dendritic Regret-Weighted Ternary-Decision Analyzer with Sparse Winner-Take-All Privacy Model.

This module fuses the governing equations of two parent algorithms:
- Hybrid Dendritic Regret-Weighted Ternary-Decision Analyzer (HD-RW-TD) from `hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s3.py`
- Hybrid Sparse Winner-Take-All with Privacy Model (HS-WTA-PM) from `hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py`

The mathematical bridge between the two parents is established by using the membrane potential (V) from the dendritic model as input to calculate regret-weighted probabilities, which are then mapped onto a ternary alphabet and used as input for the path signature pruning algorithm.
The resulting ternary decision is then fed into the sparse winner-take-all (WTA) model, where it undergoes a hash-based sparse expansion, followed by a differentially private aggregation and a risk-scale based model pool admission decision.

"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
import math
import random
import sys
import hashlib
import pathlib

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
    return -g_L*(V_i - E_L) + I_ion + I_syn

# ---------------------------------------------------------------------------
# Sparse Winner-Take-All utilities
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------
def hybrid_decision(V, m, h, g_Na=120.0, E_Na=50.0, expansion_size=100, k=10):
    """Hybrid decision-making function that combines dendritic model and sparse WTA."""
    I_ion = sodium_current(V, m, h, g_Na, E_Na)
    I_syn = 0.0  # assuming no synaptic input for simplicity
    V_i = calculate_membrane_potential(1.0, 0.1, -70.0, V, I_ion, I_syn)
    ternary_decision = [1 if V_i > 0 else -1 if V_i < 0 else 0]
    expanded_decision = expand(ternary_decision, expansion_size)
    top_k_expanded = top_k_mask(expanded_decision, k)
    return top_k_expanded

def hybrid_risk_scale(V, m, h, g_Na=120.0, E_Na=50.0, expansion_size=100, k=10, noise_scale=0.1):
    """Hybrid risk-scale function that combines dendritic model and sparse WTA with differential privacy."""
    I_ion = sodium_current(V, m, h, g_Na, E_Na)
    I_syn = 0.0  # assuming no synaptic input for simplicity
    V_i = calculate_membrane_potential(1.0, 0.1, -70.0, V, I_ion, I_syn)
    ternary_decision = [1 if V_i > 0 else -1 if V_i < 0 else 0]
    expanded_decision = expand(ternary_decision, expansion_size)
    noisy_expanded = [x + random.gauss(0, noise_scale) for x in expanded_decision]
    top_k_noisy = top_k_mask(noisy_expanded, k)
    return sum(top_k_noisy) / len(top_k_noisy)

def hybrid_model_pool_admission(V, m, h, g_Na=120.0, E_Na=50.0, expansion_size=100, k=10, noise_scale=0.1, threshold=0.5):
    """Hybrid model pool admission function that combines dendritic model and sparse WTA with differential privacy."""
    risk_scale = hybrid_risk_scale(V, m, h, g_Na, E_Na, expansion_size, k, noise_scale)
    if risk_scale > threshold:
        return True
    else:
        return False

if __name__ == "__main__":
    V = 0.0
    m = 0.5
    h = 0.5
    print(hybrid_decision(V, m, h))
    print(hybrid_risk_scale(V, m, h))
    print(hybrid_model_pool_admission(V, m, h))