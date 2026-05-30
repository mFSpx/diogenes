# DARWIN HAMMER — match 2488, survivor 0
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s0.py (gen5)
# parent_b: hybrid_workshare_allocator_doomsday_calendar_m14_s1.py (gen1)
# born: 2026-05-29T23:42:36Z

"""
HYBRID DENDRITIC-COMPARTMENT-BASED WORKSHARE ALLOCATOR WITH REGRET-WEIGHTED TERNARY DECISION ANALYZER (DC-WSRA)

This module fuses the governing equations of two parent algorithms:
- Dendritic Compartment Model (DCM) from `hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s0.py`
- Hybrid Workshare Allocator with Doomsday Calendar (HWAD) from `hybrid_workshare_allocator_doomsday_calendar_m14_s1.py`

The mathematical bridge between the two parents is established by using the membrane potential from the DCM to influence the workshare allocation process in HWAD.
Specifically, the ternary decision analyzer from RW-TD-H is used to map the membrane potential onto a probability distribution,
which is then used to dynamically adjust the deterministic target percentage in the workshare allocation process.
"""

import numpy as np
from datetime import date
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
    "calculate_regret_weighted_probabilities",
    "map_probabilities_to_ternary",
    "allocate_workshare_with_dendritic_influence",
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
        Gate deactivation rate constant.
    """
    return alpha * (1 - V) - beta * V


def compartment_step(V, m, h, n, g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.4):
    """Update membrane potential and gate variables.

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
    dVdt = - (I_Na + I_K + I_L)
    dm_dt = alpha_beta_gates(m, 0.128 * np.exp((V + 25) / 4.8), 4.25 * np.exp((V + 5) / 5))
    dh_dt = alpha_beta_gates(h, 0.128 * np.exp((V + 25) / 4.8), 1 / (1 + np.exp((V + 30) / 5)))
    dn_dt = alpha_beta_gates(n, 0.032 * np.exp((V + 52) / 5), 0.5 * np.exp((V + 57) / 40))
    return dVdt, dm_dt, dh_dt, dn_dt


def calculate_regret_weighted_probabilities(probabilities, regret_weights):
    """Calculate regret-weighted probabilities.

    Parameters
    ----------
    probabilities:
        List of probabilities.
    regret_weights:
        List of regret weights.

    Returns
    -------
    Regret-weighted probabilities.
    """
    return [p * (1 - r) for p, r in zip(probabilities, regret_weights)]


def map_probabilities_to_ternary(probabilities):
    """Map probabilities to ternary alphabet.

    Parameters
    ----------
    probabilities:
        List of probabilities.

    Returns
    -------
    Ternary alphabet.
    """
    return ["1" if p > 0.7 else "0" if p < 0.3 else "-" for p in probabilities]


# ---------------------------------------------------------------------------

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7


def allocate_workshare_with_dendritic_influence(*, total_units: float, membrane_potential: float, deterministic_target_pct: float = 90.0):
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")

    # Map membrane potential to probabilities
    probabilities = [1 / (1 + np.exp(-membrane_potential))]
    ternary_alphabet = map_probabilities_to_ternary(probabilities)
    regret_weights = [0.2 if t == "-" else 0.5 if t == "1" else 0.8 for t in ternary_alphabet]
    regret_weighted_probabilities = calculate_regret_weighted_probabilities(probabilities, regret_weights)

    # Adjust deterministic target percentage based on regret-weighted probabilities
    adjusted_deterministic_target_pct = deterministic_target_pct * regret_weighted_probabilities[0]

    doomsday_value = doomsday(date.today().year, date.today().month, date.today().day)
    deterministic_units = total_units * adjusted_deterministic_target_pct / 100.0 * (1 + doomsday_value / 7)
    llm_units = total_units - deterministic_units
    groups = ("codex", "groq", "cohere", "local_models")
    per_group = llm_units / len(groups)
    lanes = [
        {
            "group": group,
            "llm_units": round(per_group, 6),
            "llm_share_pct": round(100.0 / len(groups), 6),
            "proof_required": True,
        }
        for group in groups
    ]
    return lanes


if __name__ == "__main__":
    membrane_potential = -50.0  # Example membrane potential
    total_units = 100.0
    result = allocate_workshare_with_dendritic_influence(total_units=total_units, membrane_potential=membrane_potential)
    print(result)