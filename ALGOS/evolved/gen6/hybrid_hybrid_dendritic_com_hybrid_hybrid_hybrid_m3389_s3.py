# DARWIN HAMMER — match 3389, survivor 3
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2698_s0.py (gen5)
# born: 2026-05-29T23:49:43Z

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


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


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
        probabilities[i] = action.expected_value / (action.expected_value + action.cost)
    return probabilities


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * exp(k * fi)


def hybrid_dendritic_compartment_morphology(actions: List[MathAction], morphology: Morphology) -> np.ndarray:
    """Hybrid algorithm that combines regret-weighted probabilities and morphology.

    Parameters
    ----------
    actions:
        List of MathAction objects.
    morphology:
        Morphology object.
    """
    probabilities = calculate_regret_weighted_probabilities(actions)
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    return probabilities * sphericity


def hybrid_regret_weighted_morphology(current: float, morphology: Morphology, b: float = 1.0 / 3.0, k: float = 0.35) -> float:
    """Hybrid algorithm that combines regret-weighted probabilities and morphology.

    Parameters
    ----------
    current:
        Current value.
    morphology:
        Morphology object.
    b:
        Parameter for righting time index.
    k:
        Parameter for righting time index.
    """
    righting_time = righting_time_index(morphology, b, k)
    return (current + righting_time) / 2.0


def hybrid_free_energy_asymptotic(morphology: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Hybrid algorithm that combines morphology and free-energy asymptotic.

    Parameters
    ----------
    morphology:
        Morphology object.
    b:
        Parameter for righting time index.
    k:
        Parameter for righting time index.
    neck_lever:
        Parameter for righting time index.
    """
    righting_time = righting_time_index(morphology, b, k, neck_lever)
    return righting_time * exp(-k * righting_time)


def hybrid_rlct_regression(morphology: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Hybrid algorithm that combines morphology and RLCT regression.

    Parameters
    ----------
    morphology:
        Morphology object.
    b:
        Parameter for righting time index.
    k:
        Parameter for righting time index.
    neck_lever:
        Parameter for righting time index.
    """
    righting_time = righting_time_index(morphology, b, k, neck_lever)
    return righting_time * log2(righting_time) / (2.0 * k)


if __name__ == "__main__":
    actions = [MathAction(id="action1", expected_value=10.0), MathAction(id="action2", expected_value=20.0)]
    morphology = Morphology(length=10.0, width=20.0, height=30.0, mass=40.0)
    print(hybrid_dendritic_compartment_morphology(actions, morphology))
    print(hybrid_regret_weighted_morphology(10.0, morphology))
    print(hybrid_free_energy_asymptotic(morphology))
    print(hybrid_rlct_regression(morphology))