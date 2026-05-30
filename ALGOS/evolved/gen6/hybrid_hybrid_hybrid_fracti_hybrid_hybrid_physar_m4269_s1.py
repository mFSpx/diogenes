# DARWIN HAMMER — match 4269, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_fractional_hd_hybrid_hybrid_hybrid_m401_s0.py (gen5)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m373_s1.py (gen4)
# born: 2026-05-29T23:55:56Z

"""Hybrid Algorithm: fractional_hdc_physarum_bandit_fusion
Parents:
- hybrid_hybrid_fractional_hd_hybrid_hybrid_hybrid_m401_s0.py (fractional power binding,
  hypervector generation, Gini coefficient)
- hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m373_s1.py (physarum flux,
  conductance update driven by bandit action)

Mathematical Bridge:
The bandit action provides an inflow/outflow quantity `q = propensity -
confidence_bound`.  A hypervector representation of the action (derived from its
attributes) is bound with a reward hypervector using fractional‑power binding.
The resulting complex magnitudes are interpreted as a distribution whose Gini
coefficient `G` measures inequality of the bound signal.  This inequality is
used as an information‑theoretic gain factor that scales the physarum
conductance update:

    Δσ = dt * ( G * |q| - decay * σ )

The physarum flux `Φ = σ / L * (p_a - p_b)` then uses the updated conductance.
Thus the HDC‑based statistical signal directly modulates the physarum dynamics
in a unified system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (fractional HDC)
# ----------------------------------------------------------------------


def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Generate a random hypervector.

    Parameters
    ----------
    d: dimension of the hypervector.
    kind: "complex", "bipolar", or "real".
    seed: optional integer seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (d,). dtype is complex128 for "complex", float64 otherwise.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice([-1, 1], size=d).astype(np.float64)
    if kind == "real":
        x = rng.normal(size=d)
        return x / np.linalg.norm(x)
    raise ValueError(f"Unsupported kind '{kind}'")


def fractional_power_binding(hv1: np.ndarray, hv2: np.ndarray, power: float) -> np.ndarray:
    """Fractional power binding between two hypervectors.

    The operation is defined as hv1**power * conj(hv2) element‑wise.
    """
    return np.power(hv1, power) * np.conjugate(hv2)


def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1‑D array of non‑negative numbers."""
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    if np.any(values < 0):
        raise ValueError("values must be non‑negative for Gini")
    sorted_vals = np.sort(values)
    n = len(values)
    if n == 0:
        return 0.0
    cumulative = np.cumsum(sorted_vals, dtype=float)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    # Gini formula: (2*sum_i i*x_i)/(n*sum_x) - (n+1)/n
    i = np.arange(1, n + 1)
    gini = (2.0 * np.sum(i * sorted_vals)) / (n * sum_vals) - (n + 1) / n
    return float(gini)


# ----------------------------------------------------------------------
# Parent B components (physarum + bandit)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float            # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float      # interpreted as outflow rate
    algorithm: str = "bandit"


def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on an edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       gain: float = 1.0, dt: float = 1.0,
                       decay: float = 0.05) -> float:
    """Physarum conductance update with an external gain factor."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


# ----------------------------------------------------------------------
# Fusion layer: linking HDC statistics to physarum dynamics
# ----------------------------------------------------------------------


def action_to_hypervector(action: BanditAction,
                          dim: int = 4096,
                          seed_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Create two hypervectors representing the action and its reward.

    The first hypervector encodes the action identifier (deterministic seed).
    The second hypervector encodes the expected reward magnitude.
    """
    # Deterministic seed from action_id hash ensures reproducibility
    seed_a = (hash(action.action_id) + seed_offset) & 0xffffffff
    hv_action = random_hv(d=dim, kind="complex", seed=seed_a)

    # Reward hypervector uses a seed derived from the reward value
    seed_b = (int(action.expected_reward * 1e6) + seed_offset) & 0xffffffff
    hv_reward = random_hv(d=dim, kind="complex", seed=seed_b)

    return hv_action, hv_reward


def binding_gini_gain(action: BanditAction,
                      power: float = 0.5,
                      dim: int = 4096) -> float:
    """Compute the gain factor G from fractional binding and Gini coefficient.

    Steps:
    1. Encode action and reward as hypervectors.
    2. Perform fractional power binding.
    3. Take magnitudes of the complex result as a distribution.
    4. Return 1 + Gini (so that G ≥ 1).
    """
    hv_a, hv_r = action_to_hypervector(action, dim=dim)
    bound = fractional_power_binding(hv_a, hv_r, power)
    magnitudes = np.abs(bound)
    gini = gini_coefficient(magnitudes)
    return 1.0 + gini


def physarum_step(conductance: float,
                  edge_length: float,
                  pressure_a: float,
                  pressure_b: float,
                  action: BanditAction,
                  dt: float = 1.0,
                  decay: float = 0.05,
                  power: float = 0.5,
                  dim: int = 4096) -> Tuple[float, float]:
    """Perform a single physarum update using the HDC‑derived gain.

    Returns
    -------
    new_conductance, flux_value
    """
    # Bandit‑driven inflow/outflow quantity
    q = action.propensity - action.confidence_bound

    # Gain from HDC statistics
    G = binding_gini_gain(action, power=power, dim=dim)

    # Conductance update
    new_sigma = update_conductance(conductance, q, gain=G, dt=dt, decay=decay)

    # Flux using updated conductance
    phi = flux(new_sigma, edge_length, pressure_a, pressure_b)

    return new_sigma, phi


# ----------------------------------------------------------------------
# Example utilities (three distinct functions as required)
# ----------------------------------------------------------------------


def simulate_network_step(state: Dict[str, float],
                          actions: List[BanditAction],
                          edge_lengths: Dict[Tuple[str, str], float],
                          pressures: Dict[str, float],
                          dt: float = 1.0) -> Dict[str, float]:
    """Update conductances for a set of edges given bandit actions.

    `state` maps edge identifiers (e.g. "A-B") to current conductance.
    `actions` must be in the same order as `state.keys()`.
    `edge_lengths` and `pressures` provide physical parameters.
    Returns a new state dictionary with updated conductances.
    """
    new_state = {}
    for (edge_id, sigma), action in zip(state.items(), actions):
        node_a, node_b = edge_id.split("-")
        L = edge_lengths.get((node_a, node_b), edge_lengths.get((node_b, node_a), 1.0))
        p_a = pressures.get(node_a, 0.0)
        p_b = pressures.get(node_b, 0.0)
        new_sigma, _ = physarum_step(sigma, L, p_a, p_b, action, dt=dt)
        new_state[edge_id] = new_sigma
    return new_state


def evaluate_network_flux(state: Dict[str, float],
                          edge_lengths: Dict[Tuple[str, str], float],
                          pressures: Dict[str, float]) -> Dict[str, float]:
    """Compute flux on every edge using the current conductances."""
    fluxes = {}
    for edge_id, sigma in state.items():
        node_a, node_b = edge_id.split("-")
        L = edge_lengths.get((node_a, node_b), edge_lengths.get((node_b, node_a), 1.0))
        p_a = pressures.get(node_a, 0.0)
        p_b = pressures.get(node_b, 0.0)
        fluxes[edge_id] = flux(sigma, L, p_a, p_b)
    return fluxes


def summarize_gain(actions: List[BanditAction],
                   power: float = 0.5,
                   dim: int = 4096) -> float:
    """Return the average gain factor G across a list of actions."""
    if not actions:
        return 1.0
    gains = [binding_gini_gain(a, power=power, dim=dim) for a in actions]
    return float(np.mean(gains))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a tiny network with two edges
    initial_state = {"A-B": 0.5, "B-C": 0.8}
    edge_lengths = {("A", "B"): 2.0, ("B", "C"): 1.5}
    pressures = {"A": 1.0, "B": 0.3, "C": 0.0}

    # Generate dummy bandit actions
    actions = [
        BanditAction(action_id="a1", propensity=0.6, expected_reward=1.2,
                     confidence_bound=0.2, algorithm="UCB"),
        BanditAction(action_id="a2", propensity=0.4, expected_reward=0.8,
                     confidence_bound=0.1, algorithm="Thompson")
    ]

    # One simulation step
    new_state = simulate_network_step(initial_state, actions, edge_lengths, pressures, dt=0.5)

    # Compute fluxes after the update
    fluxes = evaluate_network_flux(new_state, edge_lengths, pressures)

    # Summarize gain
    avg_gain = summarize_gain(actions)

    print("Updated conductances:", new_state)
    print("Fluxes:", fluxes)
    print("Average gain factor G:", avg_gain)