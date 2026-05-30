# DARWIN HAMMER — match 5028, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_minimu_m1883_s1.py (gen6)
# born: 2026-05-29T23:59:22Z

"""Hybrid Algorithm: FisherGeometricBandit

This module fuses the core mathematical components of two parent algorithms:

* **Parent A (hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s2.py)** – provides
  - a temperature‑dependent developmental rate based on the Schoolfield model,
  - a Gaussian beam intensity and its Fisher information (fisher_score).

* **Parent B (hybrid_hybrid_hybrid_geomet_hybrid_hybrid_minimu_m1883_s1.py)** – defines a
  contextual bandit framework where feature vectors are compared via a Gaussian
  radial‑basis‑function (RBF) kernel; the similarity modulates action propensities
  and the resulting reward feeds back into the policy.

**Mathematical Bridge**

For each decision step we build a *feature vector* **f** from the outputs of the
Schoolfield developmental rate and the Fisher information of a Gaussian beam:


f = [ developmental_rate(T), fisher_score(θ, μ, σ) ]


Given the feature vector of the previous step **f_prev**, the similarity
`S = exp( -||f - f_prev||² / (2·γ²) )` (Gaussian RBF) is computed.  
`S` scales the propensity of each bandit action, effectively coupling the
temperature‑driven biology model (Parent A) with the contextual bandit learning
mechanism (Parent B). The reward for an action is defined as the negative Euclidean
distance between a sampled point and a target point, mirroring the minimum‑cost
idea of Parent B.

The resulting system is a single unified hybrid that continuously updates its
policy based on biologically‑inspired features and similarity‑driven exploration."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared between parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a possible action."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridFisherGeometricBandit"


@dataclass(frozen=True)
class BanditUpdate:
    """Immutable record of a bandit update."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Parent A – biological / statistical primitives
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield temperature‑dependence model.
    Returns the developmental rate (1/time) at absolute temperature `temp_k`.
    """
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    Provides a scalar measure of sensitivity with respect to `theta`.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Parent B – contextual bandit primitives
# ----------------------------------------------------------------------


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def rbf_similarity(v1: np.ndarray, v2: np.ndarray, gamma: float = 1.0) -> float:
    """
    Gaussian radial‑basis‑function similarity.
    gamma is the length‑scale parameter (larger gamma → broader kernel).
    """
    diff = v1 - v2
    dist_sq = np.dot(diff, diff)
    return math.exp(-dist_sq / (2.0 * gamma * gamma))


# ----------------------------------------------------------------------
# Hybrid core – combines both parents
# ----------------------------------------------------------------------


class FisherGeometricBandit:
    """
    A contextual bandit whose context is built from the Schoolfield developmental
    rate and Fisher information of a Gaussian beam. The similarity between
    successive contexts modulates action propensities.
    """

    def __init__(self, actions: List[str], gamma: float = 1.0):
        # policy maps action_id -> raw propensity score (logits)
        self._policy: Dict[str, float] = {aid: 0.0 for aid in actions}
        self._prev_feature: np.ndarray = np.zeros(2)  # will be set after first step
        self.gamma = gamma
        self._action_counter = 0

    # ------------------------------------------------------------------
    # Feature construction (Parent A)
    # ------------------------------------------------------------------
    def _build_feature(self, temperature_c: float, theta: float) -> np.ndarray:
        """
        Construct a 2‑dimensional feature vector:
        [ developmental_rate(K), fisher_score(theta) ].
        """
        temp_k = c_to_k(temperature_c)
        dev_rate = developmental_rate(temp_k)
        # Use arbitrary but fixed beam parameters for the Fisher term
        center = 0.0
        width = 1.0
        fisher = fisher_score(theta, center, width)
        return np.array([dev_rate, fisher], dtype=float)

    # ------------------------------------------------------------------
    # Policy update using similarity (Parent B)
    # ------------------------------------------------------------------
    def _update_propensities(self, similarity: float) -> None:
        """
        Scale raw logits by the similarity factor, then recompute softmax
        probabilities. This couples the biological context to the bandit.
        """
        for aid in self._policy:
            self._policy[aid] *= similarity
        # Convert logits to probabilities via softmax
        logits = np.array(list(self._policy.values()))
        probs = softmax(logits)
        for aid, prob in zip(self._policy.keys(), probs):
            self._policy[aid] = prob

    # ------------------------------------------------------------------
    # Action selection and reward computation
    # ------------------------------------------------------------------
    def select_action(self) -> BanditAction:
        """Sample an action according to the current propensity distribution."""
        actions, probs = zip(*self._policy.items())
        chosen_id = random.choices(actions, weights=probs, k=1)[0]
        propensity = self._policy[chosen_id]
        # Expected reward and confidence bound are placeholders; they will be
        # updated after observing the true reward.
        return BanditAction(
            action_id=chosen_id,
            propensity=propensity,
            expected_reward=0.0,
            confidence_bound=0.0,
        )

    def compute_reward(self, point: Tuple[float, float], target: Tuple[float, float]) -> float:
        """
        Simple minimum‑cost reward: negative Euclidean distance between a sampled
        point and a target point (the lower the distance, the higher the reward).
        """
        dx = point[0] - target[0]
        dy = point[1] - target[1]
        distance = math.hypot(dx, dy)
        return -distance

    # ------------------------------------------------------------------
    # Public step interface
    # ------------------------------------------------------------------
    def step(self, context_id: str, temperature_c: float, theta: float,
             point: Tuple[float, float], target: Tuple[float, float]) -> BanditUpdate:
        """
        Perform a full hybrid iteration:
        1. Build the current feature vector.
        2. Compute similarity with the previous feature (RBF kernel).
        3. Update propensities using the similarity factor.
        4. Sample an action.
        5. Compute reward from geometric distance.
        6. Record the update and store the current feature for the next step.
        """
        feature = self._build_feature(temperature_c, theta)

        # First iteration uses similarity = 1 (no scaling)
        similarity = 1.0 if np.all(self._prev_feature == 0) else rbf_similarity(feature, self._prev_feature, self.gamma)
        self._update_propensities(similarity)

        action = self.select_action()
        reward = self.compute_reward(point, target)

        # Simple incremental learning: move the logits towards the observed reward
        lr = 0.1  # learning rate
        self._policy[action.action_id] += lr * (reward - action.expected_reward)

        # Store feature for next iteration
        self._prev_feature = feature.copy()

        return BanditUpdate(
            context_id=context_id,
            action_id=action.action_id,
            reward=reward,
            propensity=action.propensity,
        )


# ----------------------------------------------------------------------
# Helper functions demonstrating hybrid operation (requirement: ≥3)
# ----------------------------------------------------------------------


def simulate_temperature_series(start_c: float, steps: int, step_size: float = 0.5) -> List[float]:
    """Generate a synthetic temperature series (Celsius) for testing."""
    return [start_c + i * step_size for i in range(steps)]


def random_point(bounds: Tuple[float, float] = (-10.0, 10.0)) -> Tuple[float, float]:
    """Sample a random 2‑D point within square bounds."""
    low, high = bounds
    return (random.uniform(low, high), random.uniform(low, high))


def run_hybrid_demo() -> None:
    """Execute a short demo of the hybrid bandit."""
    actions = [f"action_{i}" for i in range(5)]
    bandit = FisherGeometricBandit(actions=actions, gamma=0.5)

    temps = simulate_temperature_series(start_c=20.0, steps=10)
    theta = 0.0  # fixed angle for simplicity
    target = (0.0, 0.0)

    for idx, temp in enumerate(temps):
        point = random_point()
        update = bandit.step(
            context_id=f"step_{idx}",
            temperature_c=temp,
            theta=theta,
            point=point,
            target=target,
        )
        print(f"{update.context_id}: action={update.action_id}, reward={update.reward:.3f}, propensity={update.propensity:.3f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_hybrid_demo()