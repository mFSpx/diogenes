# DARWIN HAMMER — match 3389, survivor 5
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2698_s0.py (gen5)
# born: 2026-05-29T23:49:43Z

"""hybrid_dendritic_bandit_hybrid_hybrid_mixed.py
Hybrid algorithm merging:

- Parent A: Multi‑Compartment Dendritic Hodgkin‑Huxley ODEs (dendritic_compartment.py) which provides the ionic currents
  I_Na, I_K, I_L as functions of membrane potential V and gating variables (m, h, n).

- Parent B: Morphology‑driven bandit / log‑count statistics (hybrid_hybrid_hybrid_bandit_m2698_s0.py) which supplies
  a geometric descriptor (Morphology) and uses probability distributions together with empirical log‑likelihood
  (log‑count) to drive action selection.

Mathematical bridge:
The regret‑weighted action probabilities from Parent A are interpreted as a *selection weight* that
modulates the effective conductances (g_Na, g_K, g_L).  Conversely, the morphology‑derived indices
(sphericity, flatness, righting‑time) are used as *scaling factors* for the bandit’s log‑count update.
Thus the two subsystems exchange scalar multipliers at each integration step, yielding a single
dynamical system that evolves both the membrane potential and the bandit’s belief state simultaneously.
"""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action description used by the regret‑weighted selector."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome used for regret computation."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class Morphology:
    """Geometric proxy from Parent B."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Parent A – Hodgkin‑Huxley currents
# ----------------------------------------------------------------------
def sodium_current(V: float, m: float, h: float, g_Na: float = 120.0, E_Na: float = 50.0) -> float:
    """I_Na = g_Na * m³ * h * (V - E_Na)"""
    return g_Na * (m ** 3) * h * (V - E_Na)


def potassium_current(V: float, n: float, g_K: float = 36.0, E_K: float = -77.0) -> float:
    """I_K = g_K * n⁴ * (V - E_K)"""
    return g_K * (n ** 4) * (V - E_K)


def leak_current(V: float, g_L: float = 0.3, E_L: float = -54.4) -> float:
    """I_L = g_L * (V - E_L)"""
    return g_L * (V - E_L)


# ----------------------------------------------------------------------
# Parent B – Morphology indices and log‑count bandit utilities
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity = (L·W·H)^{1/3} / L  (L = longest side)"""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    longest = max(length, width, height)
    return (length * width * height) ** (1.0 / 3.0) / longest


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (L + W) / (2·H)  (H = smallest side)"""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    smallest = min(length, width, height)
    return (length + width) / (2.0 * smallest)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Empirical righting‑time proxy used as a scaling factor for log‑count updates."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


# ----------------------------------------------------------------------
# Hybrid core utilities
# ----------------------------------------------------------------------
def regret_weighted_probabilities(actions: List[MathAction],
                                  regret_factor: float = 1.0) -> np.ndarray:
    """
    Compute a soft‑max over (expected_value - cost) scaled by a regret factor.
    The result is a probability vector that sums to one.
    """
    if not actions:
        raise ValueError("action list must not be empty")
    scores = np.array([a.expected_value - a.cost for a in actions], dtype=float)
    # Regret factor acts like an inverse temperature
    scaled = scores * regret_factor
    # Numerical stability
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)
    probs = exp_vals / exp_vals.sum()
    return probs


def update_log_counts(log_counts: dict, selected_action: str) -> None:
    """
    Increment the log‑count for the selected action.
    The log‑count is stored as the natural logarithm of (1 + count) to avoid overflow.
    """
    count = log_counts.get(selected_action, 0.0)
    # Inverse of log(1+count) → count = exp(log_count) - 1
    real_count = math.exp(count) - 1.0
    real_count += 1.0
    log_counts[selected_action] = math.log(real_count)


def hybrid_conductance_scaling(morph: Morphology,
                               prob_vector: np.ndarray,
                               base_g_Na: float = 120.0,
                               base_g_K: float = 36.0,
                               base_g_L: float = 0.3) -> Tuple[float, float, float]:
    """
    Produce effective conductances by mixing morphology‑derived scalars with
    the regret‑weighted probability mass.

    - The sphericity index scales g_Na (larger, more spherical dendrites → more Na⁺ influx).
    - The flatness index scales g_K (flatter structures favor K⁺ repolarisation).
    - The righting‑time index scales g_L (longer righting → larger leak).

    The probability vector influences the scaling through its mean magnitude,
    ensuring that a higher confidence in actions amplifies the conductances.
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flat = flatness_index(morph.length, morph.width, morph.height)
    right = righting_time_index(morph)

    prob_factor = prob_vector.mean()  # between 0 and 1

    g_Na_eff = base_g_Na * (1.0 + sph) * prob_factor
    g_K_eff = base_g_K * (1.0 + flat) * prob_factor
    g_L_eff = base_g_L * (1.0 + right) * prob_factor
    return g_Na_eff, g_K_eff, g_L_eff


# ----------------------------------------------------------------------
# Hybrid dynamical step
# ----------------------------------------------------------------------
def hybrid_step(V: float,
                m: float, h: float, n: float,
                actions: List[MathAction],
                morph: Morphology,
                log_counts: dict,
                dt: float = 0.01,
                C_m: float = 1.0) -> Tuple[float, float, float, float]:
    """
    Perform a single integration step of the coupled system.

    1. Compute regret‑weighted probabilities from the action list.
    2. Scale Hodgkin‑Huxley conductances using morphology and the probability mass.
    3. Evaluate ionic currents with the scaled conductances.
    4. Update the membrane potential using Euler integration.
    5. Sample an action according to the probabilities and update its log‑count.

    Returns the new state (V, m, h, n).  Gating variables are updated with the classic
    HH first‑order kinetics (α/β formulas are approximated for brevity).
    """
    # ---- 1. Regret‑weighted probabilities ----
    probs = regret_weighted_probabilities(actions, regret_factor=1.0)

    # ---- 2. Conductance scaling via morphology ----
    g_Na_eff, g_K_eff, g_L_eff = hybrid_conductance_scaling(morph, probs)

    # ---- 3. Currents with effective conductances ----
    I_Na = g_Na_eff * (m ** 3) * h * (V - 50.0)          # E_Na fixed at +50 mV
    I_K = g_K_eff * (n ** 4) * (V + 77.0)                # E_K fixed at –77 mV
    I_L = g_L_eff * (V + 54.4)                           # E_L fixed at –54.4 mV
    I_total = I_Na + I_K + I_L

    # ---- 4. Membrane potential update (Euler) ----
    dV = -I_total / C_m
    V_new = V + dt * dV

    # ---- 5. Simple HH gating updates (using standard α/β approximations) ----
    # Alpha/Beta functions (from classic HH model)
    alpha_m = (0.1 * (25.0 - V)) / (math.exp((25.0 - V) / 10.0) - 1.0) if V != 25.0 else 1.0
    beta_m = 4.0 * math.exp(-V / 18.0)
    alpha_h = 0.07 * math.exp(-V / 20.0)
    beta_h = 1.0 / (math.exp((30.0 - V) / 10.0) + 1.0)
    alpha_n = (0.01 * (10.0 - V)) / (math.exp((10.0 - V) / 10.0) - 1.0) if V != 10.0 else 0.1
    beta_n = 0.125 * math.exp(-V / 80.0)

    dm = alpha_m * (1.0 - m) - beta_m * m
    dh = alpha_h * (1.0 - h) - beta_h * h
    dn = alpha_n * (1.0 - n) - beta_n * n

    m_new = m + dt * dm
    h_new = h + dt * dh
    n_new = n + dt * dn

    # ---- 6. Action selection and log‑count update ----
    selected_idx = np.random.choice(len(actions), p=probs)
    selected_action_id = actions[selected_idx].id
    update_log_counts(log_counts, selected_action_id)

    return V_new, m_new, h_new, n_new


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def run_hybrid_simulation(steps: int = 200) -> None:
    """
    Run a short deterministic simulation that showcases the hybrid dynamics.
    Prints the final membrane potential and a summary of log‑counts.
    """
    # Initial HH state
    V = -65.0          # mV
    m = 0.05
    h = 0.6
    n = 0.32

    # Example action set
    actions = [
        MathAction(id="A", expected_value=1.2, cost=0.1),
        MathAction(id="B", expected_value=0.8, cost=0.05),
        MathAction(id="C", expected_value=0.5, cost=0.2),
    ]

    # Example morphology (arbitrary but plausible)
    morph = Morphology(length=1.2, width=0.8, height=0.5, mass=0.03)

    # Log‑count container
    log_counts = {}

    for _ in range(steps):
        V, m, h, n = hybrid_step(V, m, h, n, actions, morph, log_counts)

    # Display results
    print(f"Final membrane potential: {V:.3f} mV")
    print("Log‑counts (natural log of 1+visits) per action:")
    for act in actions:
        cnt = math.exp(log_counts.get(act.id, 0.0)) - 1.0
        print(f"  {act.id}: {cnt:.2f}")


if __name__ == "__main__":
    try:
        run_hybrid_simulation()
    except Exception as exc:
        sys.stderr.write(f"Hybrid simulation failed: {exc}\\n")
        sys.exit(1)