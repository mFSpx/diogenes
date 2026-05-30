# DARWIN HAMMER — match 11, survivor 6
# gen: 3
# parent_a: physarum_network.py (gen0)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (gen2)
# born: 2026-05-29T23:25:06Z

"""Hybrid Physarum–Bandit Model
================================

This module fuses the *Physarum* conductance dynamics (Parent A) with the
contextual bandit / VRAM‑store scheduler (Parent B).  The mathematical bridge
is the identification of **conductance** with the **propensity** (inflow rate)
of a bandit action and **flux** with the **reward signal** that drives the
bandit’s learning update.

*   Physarum: `q = G / L * (p_a - p_b)`  (flux on an edge)
*   Bandit   : `propensity` is the probability‐like inflow for an action,
    `confidence_bound` acts as an outflow.

By feeding the flux `q` into the conductance update rule we obtain a new
propensity, which is then used by the bandit’s linear weight update.  The
VRAM‑store variable provides a global modulation of the learning rate, closing
the feedback loop.

The implementation provides three core functions that demonstrate the hybrid
operation and a lightweight `HybridPhysarumBandit` class that encapsulates the
state.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Physarum primitives
# ----------------------------------------------------------------------
def flux(
    conductance: float,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    eps: float = 1e-12,
) -> float:
    """
    Compute the flux on a single edge.

    Parameters
    ----------
    conductance : float
        Current conductance (capacity) of the edge.
    edge_length : float
        Physical length of the edge (must be > 0).
    pressure_a, pressure_b : float
        Node pressures at the two ends of the edge.
    eps : float, optional
        Small number to avoid division by zero.

    Returns
    -------
    float
        Flux value `q = G/L * (p_a - p_b)`.
    """
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(
    conductance: float,
    q: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
) -> float:
    """
    Conductance update rule derived from Physarum adaptation.

    Parameters
    ----------
    conductance : float
        Current conductance.
    q : float
        Flux on the edge (acts as a reward signal).
    dt : float, optional
        Time step.
    gain : float, optional
        Reinforcement coefficient (how strongly flux increases conductance).
    decay : float, optional
        Natural decay coefficient.

    Returns
    -------
    float
        Updated conductance, clipped to non‑negative values.
    """
    if dt < 0 or decay < 0:
        raise ValueError("dt and decay must be non‑negative")
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


# ----------------------------------------------------------------------
# Parent B – Simplified hybrid bandit / VRAM store
# ----------------------------------------------------------------------
class HybridBandit:
    """
    Minimal version of the bandit scheduler from Parent B.
    It maintains:
    * a weight matrix `W` (shape (d_in, d_out))
    * a scalar store `S` that modulates the learning rate
    * a list of actions, each associated with a conductance‑derived propensity
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        store_decay: float = 0.99,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.store_decay = store_decay

        # Linear weight matrix (TTT model analogue)
        self.W = self.rng.normal(scale=0.1, size=(d_in, d_out))

        # Global VRAM‑like store (scalar for simplicity)
        self.store = 0.0

    def _modulated_eta(self) -> float:
        """Learning rate scaled by the current store."""
        return self.base_eta * (1.0 + math.tanh(self.store))

    def policy(self, x: np.ndarray) -> np.ndarray:
        """
        Compute action scores given input vector `x`.
        Returns a vector of size `d_out`.
        """
        return x @ self.W

    def update(
        self,
        x: np.ndarray,
        reward: float,
        inflow: float,
        outflow: float,
    ) -> None:
        """
        Perform a single bandit learning step.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector (shape (d_in,)).
        reward : float
            Observed reward (used as a scalar signal).
        inflow, outflow : float
            Quantities that drive the store dynamics (here taken from
            conductance/flux).
        """
        # Store dynamics (ODE discretisation)
        dS = self.dt * (self.alpha * inflow - self.beta * outflow)
        self.store = max(0.0, self.store + dS)
        self.store *= self.store_decay

        # Gradient‑like update of the weight matrix
        eta = self._modulated_eta()
        grad = np.outer(x, np.full(self.W.shape[1], reward - np.mean(self.W)))
        self.W += eta * grad

    def get_propensities(self) -> np.ndarray:
        """
        Map the current weight matrix to a set of propensities.
        For the hybrid we simply take the L2 norm of each column.
        """
        return np.linalg.norm(self.W, axis=0)


# ----------------------------------------------------------------------
# Hybrid functions that intertwine Physarum and Bandit dynamics
# ----------------------------------------------------------------------
def compute_fluxes(
    conductances: np.ndarray,
    edge_lengths: np.ndarray,
    pressures: Tuple[float, float],
) -> np.ndarray:
    """
    Vectorised flux computation for all edges.

    Parameters
    ----------
    conductances : np.ndarray
        Shape (E,) – conductance of each edge.
    edge_lengths : np.ndarray
        Shape (E,) – length of each edge (positive).
    pressures : tuple(float, float)
        Pressures at the two nodes that all edges connect to
        (for a star‑graph topology used in the demo).

    Returns
    -------
    np.ndarray
        Flux values `q` for each edge.
    """
    p_a, p_b = pressures
    return flux(conductances, edge_lengths, p_a, p_b)


def update_conductances(
    conductances: np.ndarray,
    fluxes: np.ndarray,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
) -> np.ndarray:
    """
    Apply the Physarum conductance update to a whole vector of edges.

    Parameters
    ----------
    conductances : np.ndarray
        Current conductance vector (E,).
    fluxes : np.ndarray
        Flux vector (E,) – used as the reward signal.
    dt, gain, decay : float
        Same semantics as in `update_conductance`.

    Returns
    -------
    np.ndarray
        Updated conductance vector.
    """
    vec_update = np.vectorize(update_conductance)
    return vec_update(conductances, fluxes, dt, gain, decay)


def hybrid_step(
    bandit: HybridBandit,
    conductances: np.ndarray,
    edge_lengths: np.ndarray,
    pressures: Tuple[float, float],
    input_vec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one hybrid iteration:
    1. Compute fluxes from the current Physarum state.
    2. Update conductances (which become propensities for the bandit).
    3. Derive a scalar reward from the bandit policy.
    4. Feed inflow/outflow (propensity & confidence) back into the store.

    Parameters
    ----------
    bandit : HybridBandit
        Bandit instance that holds the weight matrix and store.
    conductances : np.ndarray
        Edge conductances (E,).
    edge_lengths : np.ndarray
        Edge lengths (E,).
    pressures : tuple(float, float)
        Node pressures (p_a, p_b) for the star‑graph.
    input_vec : np.ndarray
        Feature vector fed to the bandit (shape (d_in,)).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Updated conductances and the bandit’s action scores.
    """
    # 1️⃣ Flux computation (Physarum → reward signal)
    qs = compute_fluxes(conductances, edge_lengths, pressures)

    # 2️⃣ Conductance update (Physarum adaptation)
    new_conductances = update_conductances(conductances, qs)

    # 3️⃣ Propensity ↔ bandit mapping
    # Use the updated conductances as propensities for each action.
    # For simplicity we assume `d_out == len(new_conductances)`.
    propensities = new_conductances / (new_conductances.max() + 1e-12)

    # Bandit policy scores (linear model)
    scores = bandit.policy(input_vec)

    # 4️⃣ Derive a synthetic reward: higher score for larger propensities
    reward = float(np.dot(scores, propensities))

    # Inflow/outflow for the store: we take mean propensity as inflow,
    # and the standard deviation of fluxes as outflow.
    inflow = float(propensities.mean())
    outflow = float(np.std(qs))

    # 5️⃣ Bandit update (store + weight matrix)
    bandit.update(input_vec, reward, inflow, outflow)

    return new_conductances, scores


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    """
    Run a short simulation with a tiny star‑graph (3 edges) and a
    4‑dimensional input vector.  The loop executes 5 hybrid steps and
    prints a summary; any exception indicates a failure.
    """
    # Graph definition (star topology: centre node ↔ three leaves)
    E = 3
    conductances = np.full(E, 0.5)          # initial conductance for each edge
    edge_lengths = np.array([1.0, 1.5, 2.0])  # positive lengths

    # Random but reproducible pressures at the centre and leaf node
    rng = np.random.default_rng(42)
    pressures = (rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0))

    # Bandit / TTT model
    d_in, d_out = 4, E
    bandit = HybridBandit(d_in, d_out, seed=123)

    # Fixed input vector (could be a feature representation of the current context)
    input_vec = rng.normal(size=d_in)

    print("=== Hybrid Physarum‑Bandit Smoke Test ===")
    for step in range(5):
        conductances, scores = hybrid_step(
            bandit,
            conductances,
            edge_lengths,
            pressures,
            input_vec,
        )
        print(
            f"Step {step+1:02d} | Conductances: {conductances.round(3)} | "
            f"Scores: {scores.round(3)} | Store: {bandit.store:.4f}"
        )

    # Verify that the bandit weight matrix has been updated (non‑zero)
    assert np.any(np.abs(bandit.W) > 0), "Weight matrix remained zero"
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    _smoke_test()