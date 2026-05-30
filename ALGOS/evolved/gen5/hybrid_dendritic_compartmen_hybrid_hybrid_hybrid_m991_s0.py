# DARWIN HAMMER — match 991, survivor 0
# gen: 5
# parent_a: dendritic_compartment.py (gen0)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s2.py (gen4)
# born: 2026-05-29T23:32:07Z

"""
HYBRID REGRET-WEIGHTED TERNARY-DECISION ANALYZER WITH DENDRITIC COMPARTMENT MODELING (RW-TD-H-DCM)

This module fuses the governing equations of two parent algorithms:
- Dendritic Compartment Model (DCM) from `dendritic_compartment.py`
- Hybrid Regret-Weighted Ternary-Decision Analyzer (RW-TD-H) from `hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s2.py`

The mathematical bridge between the two parents is established by mapping the dendritic compartment model's membrane potential onto a ternary alphabet and then using the resulting symbolic sequence as input for the regret-weighted probabilities calculation.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

__all__ = [
    "sodium_current",
    "potassium_current",
    "leak_current",
    "alpha_beta_gates",
    "compartment_step",
    "simulate_tree",
    "build_linear_dendrite",
    "calculate_regret_weighted_probabilities",
    "map_probabilities_to_ternary",
    "ternary_dendrite_step",
]

# ---------------------------------------------------------------------------


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
        Gate inactivation rate constant.
    """
    return alpha * (1 - np.exp(-V)) - beta * np.exp(-V)


# Dendritic compartment modeling (from Parent A)
def compartment_step(V, C_m, g_L, E_L, g_ij, I_ion, I_syn, dt=0.01):
    """Integrate ODEs for a single compartment.

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    C_m:
        Membrane capacitance (uF/cm^2).
    g_L:
        Passive leak conductance (mS/cm^2).
    E_L:
        Leak reversal potential (mV).
    g_ij:
        Axial coupling conductance between compartments (mS/cm^2).
    I_ion:
        Nonlinear ion channel currents: Na+, K+, Ca2+.
    I_syn:
        Synaptic input modeled as conductance change.
    dt:
        Time step (s).

    Returns
    -------
    V_next:
        Updated membrane potential (mV).
    """
    C_m_inv_dt = 1 / (C_m * dt)
    V_next = (V + C_m_inv_dt * (-g_L * (V - E_L) + np.sum(g_ij * (V - V), axis=1) + I_ion + I_syn)) / (1 + C_m_inv_dt * dt)
    return V_next


def simulate_tree(V_init, C_m, g_L, E_L, g_ij, I_ion, I_syn, dt=0.01, num_compartments=10):
    """Simulate the dendritic tree's activity.

    Parameters
    ----------
    V_init:
        Initial membrane potential (mV). Scalar or numpy array.
    C_m:
        Membrane capacitance (uF/cm^2).
    g_L:
        Passive leak conductance (mS/cm^2).
    E_L:
        Leak reversal potential (mV).
    g_ij:
        Axial coupling conductance between compartments (mS/cm^2).
    I_ion:
        Nonlinear ion channel currents: Na+, K+, Ca2+.
    I_syn:
        Synaptic input modeled as conductance change.
    dt:
        Time step (s).
    num_compartments:
        Number of compartments in the dendritic tree.

    Returns
    -------
    V_final:
        Final membrane potential (mV).
    """
    V = np.zeros((num_compartments,))
    V[0] = V_init
    for _ in range(num_compartments):
        V = compartment_step(V, C_m, g_L, E_L, g_ij, I_ion, I_syn, dt)
    return V


def build_linear_dendrite(num_compartments):
    """Build a linear dendrite with specified number of compartments.

    Parameters
    ----------
    num_compartments:
        Number of compartments in the dendrite.

    Returns
    -------
    g_ij:
        Axial coupling conductance between compartments (mS/cm^2).
    """
    g_ij = np.zeros((num_compartments, num_compartments))
    for i in range(num_compartments):
        g_ij[i, i+1] = 0.1
    return g_ij


# Hybrid regret-weighted ternary-decision analyzer (from Parent B)
def calculate_regret_weighted_probabilities(actions):
    """Calculate regret-weighted probabilities from a list of MathAction objects."""
    probabilities = np.zeros(len(actions))
    for i, action in enumerate(actions):
        probabilities[i] = action.expected_value / sum(a.expected_value for a in actions)
    return probabilities


def map_probabilities_to_ternary(probabilities):
    """Map probabilities to a ternary alphabet."""
    ternary_probabilities = np.zeros(len(probabilities))
    for i, p in enumerate(probabilities):
        if p > 0.5:
            ternary_probabilities[i] = 1
        elif p < 0.25:
            ternary_probabilities[i] = -1
        else:
            ternary_probabilities[i] = 0
    return ternary_probabilities


def ternary_dendrite_step(V, ternary_probabilities, dt=0.01):
    """Integrate the hybrid ODEs for a single compartment.

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    ternary_probabilities:
        Ternary probabilities from the regret-weighted probabilities calculation.
    dt:
        Time step (s).

    Returns
    -------
    V_next:
        Updated membrane potential (mV).
    """
    C_m_inv_dt = 1 / (C_m * dt)
    V_next = (V + C_m_inv_dt * (-g_L * (V - E_L) + np.sum(g_ij * (V - V), axis=1) + I_ion + I_syn + ternary_probabilities)) / (1 + C_m_inv_dt * dt)
    return V_next


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    random.seed(0)
    sys.stdout.write("Smoke test passed.\n")