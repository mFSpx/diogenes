# DARWIN HAMMER — match 1475, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m105_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s1.py (gen3)
# born: 2026-05-29T23:36:41Z

"""Hybrid NLMS‑Bandit‑PathSignature Algorithm
================================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – a Normalised Least‑Mean‑Squares (NLMS) adaptive filter that
  updates a weight vector `w` using the error `e = d – w·x` and a learning
  rate `μ`.

* **Parent B** – a contextual bandit with a store‑state dynamics (honey‑bee
  “dance” signal) and a path‑signature feature extractor that approximates
  iterated integrals with B‑spline‑like basis functions.

**Mathematical bridge**

Both parents adapt a set of parameters based on a scalar feedback signal:


NLMS   : w←w+ μ·e·x / (‖x‖²+ε)
Bandit: μ←μ·dance   (dance∈[0,1] derived from the store)


We therefore let the *dance* signal of the store modulate the NLMS learning
rate, and we use the path‑signature vector `φ(x)` as the NLMS input `x`.
The bandit selects an action whose expected reward is predicted by the NLMS
model; after observing the real reward we update **both** the store and the
NLMS weights in a single mathematically unified step.

The three public functions below demonstrate this hybrid operation:

* `nlms_predict`
* `compute_path_signature`
* `hybrid_step` – selects an action, observes a reward and performs the joint
  NLMS‑Bandit update.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# NLMS core utilities (Parent A)
# ----------------------------------------------------------------------


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction  ŷ = w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    One Normalised LMS update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Base learning rate (0 < μ < 2). It will be further scaled by the store
        ``dance`` signal.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Prediction error `e = target – w·x`.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    error = target - nlms_predict(weights, x)
    norm_sq = float(x @ x) + eps
    step = (mu * error) / norm_sq
    new_weights = weights + step * x
    return new_weights, error


# ----------------------------------------------------------------------
# Store dynamics – Bandit side (Parent B)
# ----------------------------------------------------------------------


@dataclass
class StoreState:
    """Honey‑bee‑style store whose ``dance`` signal modulates learning."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """
        Apply the store equation  Δ = α·∑inflow – β·∑outflow
        and integrate it forward.

        Returns
        -------
        new_level : float
            Updated store level (clipped at 0).
        delta : float
            The raw Δ before clipping.
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal in [0, 1] derived from the current level."""
        return min(1.0, self.level / self.limit)


# ----------------------------------------------------------------------
# Path‑signature feature extractor (approximation of Parent B)
# ----------------------------------------------------------------------


def compute_path_signature(path: List[float], order: int = 2) -> np.ndarray:
    """
    Very lightweight approximation of a path signature.

    For a 1‑D scalar path `p = [p₀, p₁, …, p_T]` the level‑1 signature is the
    cumulative sum, and level‑2 consists of pairwise products of increments.
    The function returns a concatenated vector `[ΣΔ, ΣΔ·Δ_shifted]`.

    Parameters
    ----------
    path : List[float]
        Sequence of scalar observations.
    order : int, optional
        Highest order of iterated integrals (currently only 1 or 2 are
        supported). Default is 2.

    Returns
    -------
    np.ndarray
        Feature vector φ(p) to be fed to the NLMS filter.
    """
    if not path:
        raise ValueError("path must contain at least one element")
    increments = np.diff(np.asarray(path, dtype=float))
    level1 = np.sum(increments)  # ΣΔ

    if order == 1:
        return np.array([level1], dtype=float)

    # order >= 2 : compute simple second‑order term ΣΔ_i * Δ_{i+1}
    if len(increments) < 2:
        level2 = 0.0
    else:
        level2 = np.sum(increments[:-1] * increments[1:])
    return np.array([level1, level2], dtype=float)


# ----------------------------------------------------------------------
# Bandit action representation (Parent B)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridNLMSBandit"


# ----------------------------------------------------------------------
# Hybrid system tying everything together
# ----------------------------------------------------------------------


class HybridNLMSBandit:
    """
    Combines NLMS weight adaptation with a contextual bandit whose learning
    rate is modulated by a store ``dance`` signal and whose context features are
    path‑signature vectors.
    """

    def __init__(
        self,
        action_space: List[str],
        feature_order: int = 2,
        mu_base: float = 0.5,
        eps: float = 1e-9,
    ) -> None:
        self.actions = action_space
        self.feature_order = feature_order
        self.mu_base = mu_base
        self.eps = eps

        # Initialise a weight vector sized to the signature dimension.
        dummy_feat = compute_path_signature([0.0, 1.0], order=self.feature_order)
        self.weights = np.zeros_like(dummy_feat, dtype=float)

        # Store for modulating learning rate.
        self.store = StoreState()

        # Statistics for each action: (sum_rewards, count)
        self._stats: Dict[str, Tuple[float, int]] = {
            a: (0.0, 0) for a in self.actions
        }

    # ------------------------------------------------------------------
    # Helper: compute expected reward for each action using the current NLMS model
    # ------------------------------------------------------------------
    def _expected_rewards(self, signature: np.ndarray) -> Dict[str, float]:
        base_pred = nlms_predict(self.weights, signature)
        # For illustration we treat the base prediction as a global offset and
        # add a small random bias per action to break ties.
        return {
            a: base_pred + random.uniform(-0.01, 0.01) for a in self.actions
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_action(self, path: List[float]) -> BanditAction:
        """
        Given a raw path, compute its signature, predict rewards and return a
        stochastic action (softmax sampling). The propensity is the softmax
        probability; confidence bound is a simple UCB term.
        """
        phi = compute_path_signature(path, order=self.feature_order)
        exp_rewards = self._expected_rewards(phi)

        # Softmax over expected rewards
        max_val = max(exp_rewards.values())
        exp_vals = {a: math.exp(r - max_val) for a, r in exp_rewards.items()}
        total = sum(exp_vals.values())
        propensities = {a: v / total for a, v in exp_vals.items()}

        # Choose action proportional to propensity
        rnd = random.random()
        cumulative = 0.0
        chosen_id = self.actions[0]
        for a in self.actions:
            cumulative += propensities[a]
            if rnd <= cumulative:
                chosen_id = a
                break

        # Confidence bound – simple Upper Confidence Bound (UCB1) style
        total_counts = sum(cnt for _, cnt in self._stats.values()) + 1
        _, cnt = self._stats[chosen_id]
        cb = math.sqrt(2 * math.log(total_counts) / (cnt + 1))

        return BanditAction(
            action_id=chosen_id,
            propensity=propensities[chosen_id],
            expected_reward=exp_rewards[chosen_id],
            confidence_bound=cb,
        )

    def observe(
        self,
        path: List[float],
        action_id: str,
        reward: float,
    ) -> None:
        """
        Incorporate a new observation:
        1. Update the store (reward is inflow, a dummy outflow of 0).
        2. Scale the NLMS learning rate by the current ``dance`` signal.
        3. Perform an NLMS weight update using the path signature as input and
           the received reward as the target.
        4. Update per‑action statistics.
        """
        # 1. Store dynamics
        self.store.update(inflow=[reward], outflow=[0.0])

        # 2. Modulated learning rate
        mu_mod = self.mu_base * self.store.dance  # dance∈[0,1]

        # 3. NLMS update
        phi = compute_path_signature(path, order=self.feature_order)
        self.weights, _ = nlms_update(
            self.weights,
            phi,
            target=reward,
            mu=mu_mod,
            eps=self.eps,
        )

        # 4. Action statistics
        sum_r, cnt = self._stats[action_id]
        self._stats[action_id] = (sum_r + reward, cnt + 1)

    # ------------------------------------------------------------------
    # Convenience wrapper that performs select → observe in one call
    # ------------------------------------------------------------------
    def hybrid_step(self, path: List[float], reward_fn: Callable[[str], float]) -> BanditAction:
        """
        Execute a full hybrid iteration:

        * select an action based on the current model,
        * obtain a reward via ``reward_fn(action_id)`` (user‑provided),
        * update the internal state.

        Returns the ``BanditAction`` that was taken.
        """
        action = self.select_action(path)
        reward = reward_fn(action.action_id)
        self.observe(path, action.action_id, reward)
        return action


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny synthetic environment
    ACTIONS = ["up", "down", "stay"]

    # Reward function: simple deterministic mapping for testing
    def reward_fn(aid: str) -> float:
        return {"up": 1.0, "down": -0.5, "stay": 0.0}[aid]

    # Initialise the hybrid system
    hybrid = HybridNLMSBandit(action_space=ACTIONS, feature_order=2, mu_base=0.8)

    # Simulate a few steps with random walks as paths
    for step in range(5):
        # generate a random walk path of length 4
        path = np.cumsum(np.random.randn(4)).tolist()
        act = hybrid.hybrid_step(path, reward_fn)
        print(
            f"Step {step+1}: action={act.action_id}, "
            f"propensity={act.propensity:.3f}, reward={reward_fn(act.action_id):.2f}, "
            f"dance={hybrid.store.dance:.3f}"
        )

    # Final weight vector display
    print("Final NLMS weights:", hybrid.weights)