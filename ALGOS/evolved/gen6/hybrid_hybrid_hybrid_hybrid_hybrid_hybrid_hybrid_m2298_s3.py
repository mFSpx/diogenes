# DARWIN HAMMER — match 2298, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m630_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_fisher_m432_s0.py (gen4)
# born: 2026-05-29T23:41:41Z

"""Hybrid NLMS-Pheromone-Fisher Algorithm
Combines:
- Parent A: NLMS-Omni-Chaotic-Sprint, Hybrid Bandit Router, Liquid Time Constant Diffusion Store.
- Parent B: Pheromone-based decision making with exponential decay and Fisher‑information driven ternary logic.

Mathematical Bridge
------------------
The bridge is built on two complementary information‑theoretic quantities:

1. **NLMS Velocity Vector** (`weights @ x`) – provides a fast‑changing estimate of the local signal gradient.
2. **Pheromone Decay Rate** – an exponential half‑life process that can be *scaled* by the NLMS velocity magnitude.

The hybrid algorithm:
* feeds the current pheromone signal vector into the NLMS predictor,
* uses the resulting prediction error to update the NLMS weights,
* computes a *Fisher information* estimate from the pheromone gradient,
* modulates the bandit propensities and the store diffusion coefficients with both the NLMS velocity magnitude and the Fisher information.

Thus the adaptive filter drives pheromone decay while the pheromone field supplies the adaptive filter with fresh observations, creating a closed‑loop fusion of the two parent topologies.
"""

import numpy as np
import math
import random
import sys
import pathlib
import datetime
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# ----------------------------------------------------------------------
# Data structures (merged from both parents)
# ----------------------------------------------------------------------
NodeId = str
Edge = Tuple[NodeId, NodeId, int]

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridNLMSFisher"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass
class StoreState:
    level: float = 0.0          # current store level
    alpha: float = 1.0          # inflow scaling
    beta:  float = 1.0          # outflow scaling
    dt:    float = 1.0          # time step
    base:  float = 1.0          # baseline level

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.datetime.now(datetime.timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.datetime.now(datetime.timezone.utc) -
                self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self, scaling: float = 1.0) -> None:
        """Apply exponential decay possibly scaled by an external factor."""
        factor = self.decay_factor() ** scaling
        self.signal_value *= factor
        self.last_decay = datetime.datetime.now(datetime.timezone.utc)

# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction using current NLMS weights."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalised LMS weight update.
    Returns (new_weights, prediction_error).
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    y = nlms_predict(weights, x)
    e = target - y
    norm = float(x @ x) + eps
    new_weights = weights + (mu / norm) * e * x
    return new_weights, e


def fisher_information(signal: np.ndarray, eps: float = 1e-12) -> float:
    """
    Simple scalar Fisher information estimate for a 1‑D signal.
    I = Σ ( (∂/∂θ log p(x|θ))² ) ≈ Σ ( (∇signal)² / (signal + eps) )
    Here we treat the signal itself as a proxy for the likelihood.
    """
    grad = np.gradient(signal)
    info = np.sum((grad ** 2) / (np.abs(signal) + eps))
    return float(info)


def hybrid_decay_pheromones(
    pheromones: List[PheromoneEntry],
    nlms_weights: np.ndarray,
    input_vector: np.ndarray,
    scaling_exponent: float = 0.5,
) -> None:
    """
    Scale each pheromone's decay factor by the magnitude of the NLMS velocity.
    The velocity magnitude is |w·x|; larger velocities accelerate decay.
    """
    velocity = abs(nlms_predict(nlms_weights, input_vector))
    # Map velocity to a reasonable exponent (avoid extreme scaling)
    exp_factor = scaling_exponent * (1.0 + math.tanh(velocity))
    for p in pheromones:
        p.apply_decay(scaling=exp_factor)


def hybrid_bandit_update(
    action: BanditAction,
    pheromone_signal: float,
    fisher_info: float,
    learning_rate: float = 0.1,
) -> BanditAction:
    """
    Update a BanditAction's propensity using a blend of pheromone signal
    and Fisher information. The update follows a UCB‑like rule:
        new_propensity = propensity * (1 + lr * (signal + FI))
    """
    delta = learning_rate * (pheromone_signal + fisher_info)
    new_propensity = max(0.0, action.propensity * (1.0 + delta))
    # Re‑compute confidence bound as inverse of sqrt of visits (mocked)
    new_confidence = 1.0 / math.sqrt(1.0 + delta)
    return BanditAction(
        action_id=action.action_id,
        propensity=new_propensity,
        expected_reward=action.expected_reward,  # unchanged in this step
        confidence_bound=new_confidence,
        algorithm=action.algorithm,
    )


def hybrid_store_step(
    state: StoreState,
    inflow_coeff: float,
    outflow_coeff: float,
) -> StoreState:
    """
    Liquid‑time‑constant diffusion store update.
    level_{t+1} = level_t + dt * (α * inflow - β * outflow)
    The coefficients are modulated by NLMS velocity and Fisher information
    (passed in via inflow_coeff / outflow_coeff).
    """
    delta = state.dt * (state.alpha * inflow_coeff - state.beta * outflow_coeff)
    new_level = max(0.0, state.level + delta)  # store cannot go negative
    return StoreState(
        level=new_level,
        alpha=state.alpha,
        beta=state.beta,
        dt=state.dt,
        base=state.base,
    )

# ----------------------------------------------------------------------
# High‑level hybrid routine (demonstrates the three required functions)
# ----------------------------------------------------------------------
def hybrid_iteration(
    nlms_weights: np.ndarray,
    input_vector: np.ndarray,
    target: float,
    pheromones: List[PheromoneEntry],
    bandit_actions: List[BanditAction],
    store: StoreState,
) -> Tuple[np.ndarray, List[PheromoneEntry], List[BanditAction], StoreState]:
    """
    Perform one hybrid iteration:
    1. NLMS prediction & weight update.
    2. Compute Fisher information from current pheromone values.
    3. Decay pheromones using NLMS velocity.
    4. Update bandit propensities with pheromone signal + Fisher info.
    5. Advance the diffusion store using the same blended coefficients.
    Returns updated objects.
    """
    # 1. NLMS step
    nlms_weights, error = nlms_update(nlms_weights, input_vector, target)

    # 2. Gather pheromone signal vector and Fisher information
    signal_vec = np.array([p.signal_value for p in pheromones])
    fisher = fisher_information(signal_vec)

    # 3. Decay pheromones (velocity‑scaled)
    hybrid_decay_pheromones(pheromones, nlms_weights, input_vector)

    # 4. Update each bandit action
    updated_actions = []
    for act in bandit_actions:
        # use the mean pheromone signal as a simple proxy for context
        mean_signal = float(np.mean(signal_vec)) if signal_vec.size else 0.0
        updated = hybrid_bandit_update(act, mean_signal, fisher)
        updated_actions.append(updated)

    # 5. Store dynamics – inflow/outflow coefficients derived from NLMS error & Fisher
    inflow = max(0.0, error) * (1.0 + fisher * 0.01)
    outflow = max(0.0, -error) * (1.0 + fisher * 0.01)
    new_store = hybrid_store_step(store, inflow, outflow)

    return nlms_weights, pheromones, updated_actions, new_store

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a small NLMS filter
    dim = 5
    w = np.zeros(dim)

    # Random input vector and target
    x = np.random.randn(dim)
    tgt = np.dot(np.arange(1, dim + 1), x) * 0.3  # synthetic linear target

    # Create a handful of pheromone entries
    pheros = [
        PheromoneEntry(surface_key=f"node_{i}",
                       signal_kind="typeA",
                       signal_value=random.uniform(0.5, 2.0),
                       half_life_seconds=random.randint(5, 20))
        for i in range(3)
    ]

    # Initialise bandit actions
    actions = [
        BanditAction(action_id=f"act_{i}",
                     propensity=1.0,
                     expected_reward=0.0,
                     confidence_bound=1.0)
        for i in range(2)
    ]

    # Initialise store
    store = StoreState(level=10.0, alpha=0.8, beta=0.5, dt=0.1)

    # Run a few hybrid iterations
    for step in range(4):
        w, pheros, actions, store = hybrid_iteration(
            nlms_weights=w,
            input_vector=x,
            target=tgt,
            pheromones=pheros,
            bandit_actions=actions,
            store=store,
        )
        # Simple diagnostics
        print(f"Step {step+1}")
        print(f"  NLMS weights: {w}")
        print(f"  Pheromone values: {[p.signal_value for p in pheros]}")
        print(f"  Bandit propensities: {[a.propensity for a in actions]}")
        print(f"  Store level: {store.level}\n")
    sys.exit(0)