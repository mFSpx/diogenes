# DARWIN HAMMER — match 3488, survivor 1
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_semant_hybrid_sparse_wta_hy_m1305_s1.py (gen4)
# born: 2026-05-29T23:50:23Z

"""
HYBRID DENDRITIC-COMPARTMENT-BASED RECOVERY PRIORITY ANALYZER (DCRPA)

This module fuses the governing equations of two parent algorithms:
- Dendritic Compartment Model (DCM) from `hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s0.py`
- Hybrid Recovery Priority Analyzer (HRPA) from `hybrid_hybrid_hybrid_semant_hybrid_sparse_wta_hy_m1305_s1.py`

The mathematical bridge between the two parents is established by mapping the dendritic compartment model's membrane potential onto a morphology-based recovery priority index, which is then used to modulate the model's tier allocation.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass

__all__ = [
    "sodium_current",
    "potassium_current",
    "leak_current",
    "alpha_beta_gates",
    "compartment_step",
    "calculate_recovery_priority",
    "map_membrane_potential_to_recovery_priority",
    "hybrid_dcrpa_step",
]

# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

# Ion channel currents (from Parent A)
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

def potassium_current(V, n, g_K=36.0, E_K=-77.0):
    """Hodgkin-Huxley potassium current.

    I_K = g_K * n^4 * (V - E_K)

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    n:
        K+ activation gate variable, in [0, 1].
    """
    return g_K * n**4 * (V - E_K)

def leak_current(V, g_L=0.3, E_L=-54.4):
    """Passive leak current.

    I_L = g_L * (V - E_L)

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    """
    return g_L * (V - E_L)

def alpha_beta_gates(V, alpha, beta):
    """Gate dynamics: dx/dt = alpha_x(V)*(1-x) - beta_x(V)*x

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    alpha:
        Gate activation rate constant.
    beta:
        Gate deactivation rate constant.
    """
    pass  # stub

def compartment_step(V, m, h, n, g_Na, g_K, g_L, E_Na, E_K, E_L):
    """Single compartment step.

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    m:
        Na+ activation gate variable, in [0, 1].
    h:
        Na+ inactivation gate variable
    n:
        K+ activation gate variable, in [0, 1].
    g_Na:
        Sodium conductance.
    g_K:
        Potassium conductance.
    g_L:
        Leak conductance.
    E_Na:
        Sodium reversal potential.
    E_K:
        Potassium reversal potential.
    E_L:
        Leak reversal potential.

    Returns
    -------
    dVdt:
        Membrane potential derivative.
    """
    INa = sodium_current(V, m, h, g_Na, E_Na)
    IK = potassium_current(V, n, g_K, E_K)
    IL = leak_current(V, g_L, E_L)
    dVdt = (INa + IK + IL) / 10.0  # arbitrary capacitance
    return dVdt

def map_membrane_potential_to_recovery_priority(V, morphology):
    """Maps membrane potential to recovery priority.

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    morphology:
        Morphology dataclass.

    Returns
    -------
    rp:
        Recovery priority.
    """
    rp = recovery_priority(morphology)
    scaled_rp = rp * (V / 100.0)  # arbitrary scaling
    return scaled_rp

def calculate_recovery_priority(morphology, max_index=10.0):
    return recovery_priority(morphology, max_index)

def hybrid_dcrpa_step(V, m, h, n, g_Na, g_K, g_L, E_Na, E_K, E_L, morphology):
    """Hybrid DCRPA step.

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    m:
        Na+ activation gate variable, in [0, 1].
    h:
        Na+ inactivation gate variable
    n:
        K+ activation gate variable, in [0, 1].
    g_Na:
        Sodium conductance.
    g_K:
        Potassium conductance.
    g_L:
        Leak conductance.
    E_Na:
        Sodium reversal potential.
    E_K:
        Potassium reversal potential.
    E_L:
        Leak reversal potential.
    morphology:
        Morphology dataclass.

    Returns
    -------
    dVdt:
        Membrane potential derivative.
    rp:
        Recovery priority.
    """
    dVdt = compartment_step(V, m, h, n, g_Na, g_K, g_L, E_Na, E_K, E_L)
    rp = map_membrane_potential_to_recovery_priority(V, morphology)
    return dVdt, rp

if __name__ == "__main__":
    morphology = Morphology(10.0, 5.0, 2.0, 1.0)
    V = -70.0
    m = 0.5
    h = 0.6
    n = 0.7
    g_Na = 120.0
    g_K = 36.0
    g_L = 0.3
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.4

    dVdt, rp = hybrid_dcrpa_step(V, m, h, n, g_Na, g_K, g_L, E_Na, E_K, E_L, morphology)
    print(f"dVdt: {dVdt}, rp: {rp}")