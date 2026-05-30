# DARWIN HAMMER — match 3488, survivor 0
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_semant_hybrid_sparse_wta_hy_m1305_s1.py (gen4)
# born: 2026-05-29T23:50:23Z

"""
HYBRID DENDRITIC COMPARTMENT MODEL WITH SPARSE WINNER-TAKE-ALL (DCM-SWTA)

This module fuses the governing equations of two parent algorithms:
- Dendritic Compartment Model (DCM) from `hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s0.py`
- Hybrid Sparse Winner-Take-All (SWTA) from `hybrid_hybrid_hybrid_semant_hybrid_sparse_wta_hy_m1305_s1.py`

The mathematical bridge between the two parents is established by mapping the dendritic compartment model's membrane potential onto the sparse winner-take-all (SWTA) model's morphology-based righting time index. This allows for the integration of the DCM's ion channel currents with the SWTA's recovery priority calculation.

The governing equations of the DCM are used to simulate the membrane potential of a neuron, while the SWTA's morphology-based righting time index is used to calculate the recovery priority of the neuron. The two models are fused by using the DCM's membrane potential as input for the SWTA's recovery priority calculation.
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
    "sphericity_index",
    "flatness_index",
    "righting_time_index",
    "recovery_priority",
    "hybrid_dcm_swta_step",
    "hybrid_dcm_swta_simulate",
]

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
    return alpha * (1 - V) - beta * V

def compartment_step(V, m, h, n, g_Na=120.0, E_Na=50.0, g_K=36.0, E_K=-77.0, g_L=0.3, E_L=-54.4):
    """Compartment step for the DCM.

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
    """
    I_Na = sodium_current(V, m, h, g_Na, E_Na)
    I_K = potassium_current(V, n, g_K, E_K)
    I_L = leak_current(V, g_L, E_L)
    return I_Na + I_K + I_L

def simulate_tree(V, m, h, n, num_steps=100):
    """Simulate the DCM for a given number of steps.

    Parameters
    ----------
    V:
        Initial membrane potential (mV). Scalar or numpy array.
    m:
        Initial Na+ activation gate variable, in [0, 1].
    h:
        Initial Na+ inactivation gate variable
    n:
        Initial K+ activation gate variable, in [0, 1].
    num_steps:
        Number of steps to simulate.
    """
    for _ in range(num_steps):
        V = compartment_step(V, m, h, n)
    return V

def build_linear_dendrite(length=100, width=10, height=10):
    """Build a linear dendrite with the given dimensions.

    Parameters
    ----------
    length:
        Length of the dendrite.
    width:
        Width of the dendrite.
    height:
        Height of the dendrite.
    """
    return np.array([length, width, height])

def sphericity_index(length, width, height):
    """Calculate the sphericity index of a morphology.

    Parameters
    ----------
    length:
        Length of the morphology.
    width:
        Width of the morphology.
    height:
        Height of the morphology.
    """
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length, width, height):
    """Calculate the flatness index of a morphology.

    Parameters
    ----------
    length:
        Length of the morphology.
    width:
        Width of the morphology.
    height:
        Height of the morphology.
    """
    return (length + width) / (2.0 * height)

def righting_time_index(length, width, height, mass, b=1.0/3.0, k=0.35, neck_lever=1.0):
    """Calculate the righting time index of a morphology.

    Parameters
    ----------
    length:
        Length of the morphology.
    width:
        Width of the morphology.
    height:
        Height of the morphology.
    mass:
        Mass of the morphology.
    b:
        Parameter for the righting time index calculation.
    k:
        Parameter for the righting time index calculation.
    neck_lever:
        Neck lever for the righting time index calculation.
    """
    fi = flatness_index(length, width, height)
    return (mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(length, width, height, mass, b=1.0/3.0, k=0.35, neck_lever=1.0, max_index=10.0):
    """Calculate the recovery priority of a morphology.

    Parameters
    ----------
    length:
        Length of the morphology.
    width:
        Width of the morphology.
    height:
        Height of the morphology.
    mass:
        Mass of the morphology.
    b:
        Parameter for the righting time index calculation.
    k:
        Parameter for the righting time index calculation.
    neck_lever:
        Neck lever for the righting time index calculation.
    max_index:
        Maximum righting time index for the recovery priority calculation.
    """
    return max(0.0, min(1.0, righting_time_index(length, width, height, mass, b, k, neck_lever) / max_index))

def hybrid_dcm_swta_step(V, m, h, n, length, width, height, mass):
    """Hybrid DCM-SWTA step.

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
    length:
        Length of the morphology.
    width:
        Width of the morphology.
    height:
        Height of the morphology.
    mass:
        Mass of the morphology.
    """
    I_Na = sodium_current(V, m, h)
    I_K = potassium_current(V, n)
    I_L = leak_current(V)
    V = V + I_Na + I_K + I_L
    recovery = recovery_priority(length, width, height, mass)
    return V, recovery

def hybrid_dcm_swta_simulate(V, m, h, n, length, width, height, mass, num_steps=100):
    """Hybrid DCM-SWTA simulation.

    Parameters
    ----------
    V:
        Initial membrane potential (mV). Scalar or numpy array.
    m:
        Initial Na+ activation gate variable, in [0, 1].
    h:
        Initial Na+ inactivation gate variable
    n:
        Initial K+ activation gate variable, in [0, 1].
    length:
        Length of the morphology.
    width:
        Width of the morphology.
    height:
        Height of the morphology.
    mass:
        Mass of the morphology.
    num_steps:
        Number of steps to simulate.
    """
    for _ in range(num_steps):
        V, recovery = hybrid_dcm_swta_step(V, m, h, n, length, width, height, mass)
    return V, recovery

if __name__ == "__main__":
    V = 0.0
    m = 0.5
    h = 0.5
    n = 0.5
    length = 100.0
    width = 10.0
    height = 10.0
    mass = 1.0
    num_steps = 100
    V, recovery = hybrid_dcm_swta_simulate(V, m, h, n, length, width, height, mass, num_steps)
    print(f"Final membrane potential: {V:.2f} mV")
    print(f"Final recovery priority: {recovery:.2f}")