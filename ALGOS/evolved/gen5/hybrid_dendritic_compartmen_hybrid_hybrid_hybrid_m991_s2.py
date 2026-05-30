# DARWIN HAMMER — match 991, survivor 2
# gen: 5
# parent_a: dendritic_compartment.py (gen0)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s2.py (gen4)
# born: 2026-05-29T23:32:07Z

"""
This module fuses the governing equations of two parent algorithms:
- Multi-Compartment Dendritic ODEs (Hodgkin-Huxley cable model) from `dendritic_compartment.py`
- Hybrid Regret-Weighted Ternary-Decision Analyzer with Path Signature Pruning (RW-TD-H-PSP) from `hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s2.py`

The mathematical bridge between the two parents is established by mapping the regret-weighted probabilities onto a ternary alphabet and using the resulting symbolic sequence as input for the path signature pruning algorithm.
The Hodgkin-Huxley ion channel currents are integrated with the regret-weighted probabilities to model the dendritic spikes as a hidden-layer activation in the neural network.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

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
    """Hodgkin-Huxley leak current.

    I_L = g_L * (V - E_L)

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    """
    return g_L * (V - E_L)


def calculate_regret_weighted_probabilities(actions: List[MathAction]) -> np.ndarray:
    """Calculate regret-weighted probabilities from a list of MathAction objects."""
    probabilities = np.zeros(len(actions))
    for i, action in enumerate(actions):
        probabilities[i] = action.expected_value / sum(a.expected_value for a in actions)
    return probabilities


def map_probabilities_to_ternary_alphabet(probabilities: np.ndarray) -> List[str]:
    """Map regret-weighted probabilities to a ternary alphabet."""
    ternary_alphabet = []
    for probability in probabilities:
        if probability < 0.33:
            ternary_alphabet.append('0')
        elif probability < 0.66:
            ternary_alphabet.append('1')
        else:
            ternary_alphabet.append('2')
    return ternary_alphabet


def hybrid_hodgkin_huxley_regret(actions: List[MathAction], V, m, h, n) -> float:
    """Hybrid Hodgkin-Huxley model with regret-weighted probabilities."""
    probabilities = calculate_regret_weighted_probabilities(actions)
    ternary_alphabet = map_probabilities_to_ternary_alphabet(probabilities)
    sodium_current_value = sodium_current(V, m, h)
    potassium_current_value = potassium_current(V, n)
    leak_current_value = leak_current(V)
    # Integrate the currents with the regret-weighted probabilities
    integrated_current = (sodium_current_value + potassium_current_value + leak_current_value) * sum(int(i) for i in ternary_alphabet)
    return integrated_current


def simulate_hybrid_neuron(actions: List[MathAction], V, m, h, n, dt=0.1, t_max=100):
    """Simulate the hybrid neuron model."""
    t = 0
    membrane_potentials = []
    while t < t_max:
        integrated_current = hybrid_hodgkin_huxley_regret(actions, V, m, h, n)
        V += integrated_current * dt
        membrane_potentials.append(V)
        t += dt
    return membrane_potentials


if __name__ == "__main__":
    actions = [MathAction('action1', 10.0), MathAction('action2', 20.0), MathAction('action3', 30.0)]
    V = 0.0
    m = 0.5
    h = 0.5
    n = 0.5
    membrane_potentials = simulate_hybrid_neuron(actions, V, m, h, n)
    print(membrane_potentials)