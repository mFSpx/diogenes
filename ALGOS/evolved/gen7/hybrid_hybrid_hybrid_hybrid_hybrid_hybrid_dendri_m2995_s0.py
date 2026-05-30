# DARWIN HAMMER — match 2995, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_model_pool_m1699_s0.py (gen6)
# parent_b: hybrid_hybrid_dendritic_com_hybrid_hybrid_hybrid_m1603_s0.py (gen6)
# born: 2026-05-29T23:47:03Z

"""
This module fuses the hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m744_s0 and hybrid_hybrid_dendritic_com_hybrid_hybrid_hybrid_m1603_s0 algorithms into a single hybrid system.
The mathematical bridge between the two structures is the application of the Hodgkin-Huxley ODEs to modulate the StoreState instance in the honeybee store, and the use of pheromone signals to scale the dendritic tree feasibility.
The ModelPool class is used to manage the loading and eviction of models based on their memory requirements, and the StoreState class is used to adaptively allocate large language model (LLM) units based on the pheromone signal values and the current state of the honeybee store.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class HybridAction:
    """Result of a hybrid action selection."""
    id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class HybridUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass
class StoreState:
    """Encapsulates the honeybee-style store and its derived control signal."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """
        Apply the store equation and recompute the dance duration.

        Returns
        -------
        new_level, delta
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ (computed lazily)."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

# ----------------------------------------------------------------------
# Hodgkin-Huxley utilities
# ----------------------------------------------------------------------
def alpha_m(V: float) -> float:
    """Alpha for Na activation."""
    return 0.1 * (V + 40.0) / (1.0 - math.exp(-(V + 40.0) / 10.0))

def beta_m(V: float) -> float:
    """Beta for Na activation."""
    return 4.0 * math.exp(-(V + 65.0) / 18.0)

def alpha_h(V: float) -> float:
    """Alpha for K activation."""
    return 0.07 * math.exp(-(V + 65.0) / 20.0)

def beta_h(V: float) -> float:
    """Beta for K activation."""
    return 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))

def tau_m(V: float) -> float:
    """Tau for membrane time constant."""
    return 1.0 / (alpha_m(V) + beta_m(V))

def tau_h(V: float) -> float:
    """Tau for membrane time constant."""
    return 1.0 / (alpha_h(V) + beta_h(V))

def membrane_potential(V: float, dt: float) -> float:
    """Compute the membrane potential at time t."""
    dV = (0.04 * (V - 10.0) - 5.0 * V + 140.0 + math.exp(-V + 18.0)) / 1000.0
    V += dt * dV
    return V

def dendritic_store(store: float, reward: float, alpha: float, beta: float, dt: float) -> float:
    """
    Evolve the membrane potential store as 
    d store / dt = α·reward - β·store.

    Parameters
    ----------
    store : float
        The current membrane potential store.
    reward : float
        The reward signal.
    alpha : float
        The alpha parameter.
    beta : float
        The beta parameter.
    dt : float
        The time step.

    Returns
    -------
    new_store : float
        The updated membrane potential store.
    """
    delta = alpha * reward - beta * store
    store += dt * delta
    return store

def fisher_information(features: List[float], angles: List[float], importance: List[float]) -> List[float]:
    """
    Compute Fisher-information weights from features, angles and per-feature importance.

    Parameters
    ----------
    features : List[float]
        The feature values.
    angles : List[float]
        The angle values.
    importance : List[float]
        The per-feature importance values.

    Returns
    -------
    weights : List[float]
        The Fisher-information weights.
    """
    weights = []
    for feature, angle, imp in zip(features, angles, importance):
        weight = (feature ** 2) * (1 - np.cos(angle)) * imp
        weights.append(weight)
    return weights

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_select_action(propensities: List[float], confidence_bounds: List[float], pheromone_signal: float, dendritic_store: float) -> HybridAction:
    """
    Perform feasibility filtering, scale bandit propensities and return a HybridAction.

    Parameters
    ----------
    propensities : List[float]
        The bandit propensities.
    confidence_bounds : List[float]
        The confidence bounds.
    pheromone_signal : float
        The pheromone signal.
    dendritic_store : float
        The dendritic store.

    Returns
    -------
    action : HybridAction
        The hybrid action selection.
    """
    scaled_propensities = [propensity * pheromone_signal for propensity in propensities]
    scaled_confidence_bounds = [confidence_bound * dendritic_store for confidence_bound in confidence_bounds]
    return HybridAction(
        id="hybrid",
        propensity=sum(scaled_propensities),
        expected_reward=sum(scaled_confidence_bounds),
        confidence_bound=max(scaled_confidence_bounds),
        algorithm="hybrid",
        expected_value=sum(scaled_propensities),
    )

def hybrid_update(context_id: str, action_id: str, reward: float, propensity: float, dendritic_store: float) -> HybridUpdate:
    """
    Update the hybrid policy.

    Parameters
    ----------
    context_id : str
        The context ID.
    action_id : str
        The action ID.
    reward : float
        The reward signal.
    propensity : float
        The propensity value.
    dendritic_store : float
        The dendritic store.

    Returns
    -------
    update : HybridUpdate
        The hybrid update.
    """
    return HybridUpdate(
        context_id=context_id,
        action_id=action_id,
        reward=reward,
        propensity=propensity,
    )

def hybrid_store(store: StoreState, inflow: List[float], outflow: List[float], pheromone_signal: float, dendritic_store: float) -> StoreState:
    """
    Update the store state.

    Parameters
    ----------
    store : StoreState
        The current store state.
    inflow : List[float]
        The inflow values.
    outflow : List[float]
        The outflow values.
    pheromone_signal : float
        The pheromone signal.
    dendritic_store : float
        The dendritic store.

    Returns
    -------
    new_store : StoreState
        The updated store state.
    """
    new_level, delta = store.update(inflow, outflow)
    store.level = max(0.0, new_level + pheromone_signal * dendritic_store)
    return store

# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Smoke test
    store = StoreState()
    inflow = [1.0, 2.0, 3.0]
    outflow = [4.0, 5.0, 6.0]
    pheromone_signal = 0.5
    dendritic_store = 10.0
    new_store = hybrid_store(store, inflow, outflow, pheromone_signal, dendritic_store)
    print(new_store)