# DARWIN HAMMER — match 2119, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_endpoi_m247_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m960_s1.py (gen4)
# born: 2026-05-29T23:40:50Z

"""Hybrid Algorithm combining DARWIN HAMMER parent A (morphology, variational free energy, health scoring)
and parent B (pheromone‑store dynamics, bandit action selection).

Mathematical Bridge
------------------
- The variational free‑energy computed in parent A is interpreted as a *belief variance*.
  It is used to scale the pheromone (store) update: a larger free‑energy (more uncertainty)
  reduces the store’s gain, i.e. `store.gain = 1 / (1 + free_energy)`.
- The health score from parent A is injected as a multiplicative factor on the
  bandit propensities, effectively biasing the action‑selection toward morphologies
  that are physically robust.
- Endpoint selection combines the health‑adjusted bandit propensities with the
  original endpoint reliability, yielding a unified score:
  
  `score = endpoint_reliability * (adjusted_propensity ** 2) / (free_energy + 1)`

The three core functions below demonstrate this fusion:
`compute_health_score`, `update_store_with_health`, and `select_hybrid_endpoint`. """

import math
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A structures and utilities
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


@dataclass
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(morphology: Morphology) -> float:
    if min(morphology.length, morphology.width, morphology.height) <= 0:
        raise ValueError("dimensions must be positive")
    return (morphology.length * morphology.width * morphology.height) ** (1.0 / 3.0) / max(
        morphology.length, morphology.width, morphology.height
    )


def flatness_index(morphology: Morphology) -> float:
    if min(morphology.length, morphology.width, morphology.height) <= 0:
        raise ValueError("dimensions must be positive")
    return (morphology.length + morphology.width) / (2.0 * morphology.height)


def righting_time_index(
    morphology: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if morphology.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(morphology)
    return (morphology.mass ** b) * math.exp(k * fi)


def variational_free_energy(
    observation: np.ndarray, belief_mean: np.ndarray, observation_noise_variance: float
) -> float:
    reconstruction_error = np.sum((observation - belief_mean) ** 2)
    free_energy = (
        0.5 * math.log(2 * math.pi * observation_noise_variance)
        + 0.5 * reconstruction_error / observation_noise_variance
    )
    return free_energy


def calculate_health_score(
    endpoint_reliability: float,
    morphology: Morphology,
    variational_free_energy_value: float,
) -> float:
    sphericity = sphericity_index(morphology)
    flatness = flatness_index(morphology)
    righting_time = righting_time_index(morphology)
    health_score = (
        endpoint_reliability
        * (sphericity ** 2)
        * (flatness ** 2)
        * (righting_time ** 2)
        / (variational_free_energy_value + 1)
    )
    return health_score


# ----------------------------------------------------------------------
# Parent B structures and utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""

    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee‑style store that produces a bounded control signal (dance)."""

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
        # bounded dance signal
        dance = min(self.limit, max(0.0, self.base + self.gain * self.level))
        self._dance = dance
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ."""
        # If update hasn't been called yet, return base value
        return getattr(self, "_dance", self.base)


# ----------------------------------------------------------------------
# Hybrid Functions (the mathematical fusion)
# ----------------------------------------------------------------------
def compute_health_score(
    endpoint: Dict[str, Any],
    morphology: Morphology,
    observation: np.ndarray,
    belief_mean: np.ndarray,
    obs_noise_var: float,
) -> float:
    """Wrapper that combines parent‑A health calculation with endpoint data."""
    vfe = variational_free_energy(observation, belief_mean, obs_noise_var)
    return calculate_health_score(endpoint["reliability"], morphology, vfe)


def update_store_with_health(
    store: StoreState, health_score: float, free_energy: float
) -> None:
    """
    Use the health score as inflow and the free‑energy‑derived gain as a scaling factor.
    The gain is reduced when uncertainty (free_energy) is high, implementing the bridge.
    """
    # Scale gain inversely with free energy (more uncertainty → weaker influence)
    store.gain = 1.0 / (1.0 + free_energy)
    inflow = [health_score]
    outflow = []  # no explicit outflow in this simple bridge
    store.update(inflow, outflow)


def select_hybrid_action(
    actions: List[BanditAction], store: StoreState, health_score: float
) -> BanditAction:
    """
    Adjust each action's propensity by the current dance signal and the health score,
    then pick the action with the highest adjusted propensity.
    """
    adjusted = []
    for a in actions:
        adj_prop = a.propensity * (store.dance + 1.0) * (health_score + 1.0)
        adjusted.append((adj_prop, a))
    _, best = max(adjusted, key=lambda pair: pair[0])
    return best


def select_hybrid_endpoint(
    endpoints: List[Dict[str, Any]],
    actions: List[BanditAction],
    morphology: Morphology,
    observation: np.ndarray,
    belief_mean: np.ndarray,
    obs_noise_var: float,
    store: StoreState,
) -> Dict[str, Any]:
    """
    Compute a unified score for each endpoint:
        score = endpoint_reliability *
                (adjusted_propensity ** 2) /
                (free_energy + 1)

    The adjusted propensity comes from the bandit action selected for that endpoint
    (we assume a 1‑to‑1 mapping for simplicity). The endpoint with the highest score
    is returned.
    """
    free_energy = variational_free_energy(observation, belief_mean, obs_noise_var)

    # For demonstration we pair endpoints with actions by index (wrapping if needed)
    best_endpoint = None
    best_score = -math.inf

    for idx, ep in enumerate(endpoints):
        health = compute_health_score(ep, morphology, observation, belief_mean, obs_noise_var)
        update_store_with_health(store, health, free_energy)

        # pick an action (cyclic pairing)
        action = actions[idx % len(actions)]
        adjusted_action = select_hybrid_action([action], store, health)

        score = (
            ep["reliability"]
            * (adjusted_action.propensity ** 2)
            / (free_energy + 1.0)
        )
        if score > best_score:
            best_score = score
            best_endpoint = ep.copy()
            best_endpoint["hybrid_score"] = score
            best_endpoint["selected_action"] = adjusted_action.action_id

    return best_endpoint


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # Dummy morphology
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Synthetic observation / belief
    obs = np.array([0.2, 0.5, 0.7])
    belief = np.array([0.1, 0.4, 0.6])
    noise_var = 0.05

    # Endpoints list
    endpoints = [
        {"id": "ep1", "reliability": 0.9},
        {"id": "ep2", "reliability": 0.6},
        {"id": "ep3", "reliability": 0.8},
    ]

    # Bandit actions
    actions = [
        BanditAction(
            action_id="a1",
            propensity=0.4,
            expected_reward=1.2,
            confidence_bound=0.1,
            algorithm="UCB",
        ),
        BanditAction(
            action_id="a2",
            propensity=0.6,
            expected_reward=0.9,
            confidence_bound=0.2,
            algorithm="Thompson",
        ),
        BanditAction(
            action_id="a3",
            propensity=0.3,
            expected_reward=1.5,
            confidence_bound=0.05,
            algorithm="EXP3",
        ),
    ]

    # Initialise store
    store = StoreState()

    # Run hybrid endpoint selection
    chosen = select_hybrid_endpoint(
        endpoints,
        actions,
        morph,
        obs,
        belief,
        noise_var,
        store,
    )

    print("Chosen endpoint (hybrid):")
    for k, v in chosen.items():
        print(f"  {k}: {v}")

    # Verify that store has been updated
    print(f"\nStore state after processing: level={store.level:.3f}, dance={store.dance:.3f}")