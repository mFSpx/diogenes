# DARWIN HAMMER — match 4512, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_xgboos_m667_s1.py (gen5)
# parent_b: hybrid_nlms_omni_chaotic_sprint_m59_s0.py (gen1)
# born: 2026-05-29T23:56:14Z

"""Hybrid Bandit-NLMS Algorithm
================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – *HybridBanditXGBoost* – which treats a bandit’s propensity
  vector as a stochastic frequency vector, computes a logistic margin,
  and derives gradient/hessian via binary logistic loss.

* **Parent B** – *HybridAlgorithm* – which updates a weight vector using the
  Normalized Least‑Mean‑Squares (NLMS) rule:
  ``w ← w + μ * error * x / (‖x‖² + ε)``.

**Mathematical bridge**

Both parents rely on a gradient‑driven correction term:

* In Parent A the error is the difference between the observed reward
  (treated as a binary label) and the logistic prediction
  ``σ(margin)``; the gradient of the logistic loss w.r.t. the margin is
  ``g = σ(margin) – y``.

* In Parent B the NLMS update scales an error term by the input vector
  ``x`` normalized by its power ``‖x‖²``.

The hybrid therefore interprets the **frequency vector** produced by
``indy_learning`` as the NLMS input ``x`` and uses the logistic error
``error = reward – σ(margin)`` as the NLMS error.  The NLMS rule then
updates the internal propensity weights, closing the loop between the
bandit’s stochastic action selection and adaptive filtering.

The resulting system simultaneously:
* maintains a stochastic propensity distribution,
* predicts reward via a logistic link,
* adapts propensities with an NLMS‑style normalized gradient step.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    if isinstance(margin, np.ndarray):
        return 1.0 / (1.0 + np.exp(-margin))
    # scalar
    if margin >= 0:
        z = math.exp(-margin)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(margin)
        return z / (1.0 + z)


def binary_logistic_grad_hess(
    y_true: np.ndarray, margin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient and Hessian of binary logistic loss w.r.t. the margin.
    g = σ(margin) - y
    h = σ(margin) * (1 - σ(margin))
    """
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    error: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Normalized LMS update.
    w_{new} = w + μ * error * x / (‖x‖² + ε)
    """
    power = np.dot(x, x) + eps
    return weights + mu * error * x / power


# ----------------------------------------------------------------------
# Hybrid class
# ----------------------------------------------------------------------
class HybridBanditNLMS:
    """
    Combines a stochastic bandit propensity model with an NLMS adaptive
    filter.  The internal state ``self.propensity_weights`` lives in the
    same space as the input dimension ``d_in``.
    """

    DEFAULT_REG = 1e-6

    def __init__(
        self,
        d_in: int,
        seed: int = 0,
        mu: float = 0.5,
        learning_rate: float = 0.1,
        regularization: float = DEFAULT_REG,
    ):
        self.rng = np.random.default_rng(seed)
        self.d_in = d_in
        self.mu = mu
        self.lr = learning_rate
        self.reg = regularization

        # Initialize propensity weights (log‑space for positivity)
        self.propensity_weights = np.log(self.rng.random(d_in) + self.reg)

    # ------------------------------------------------------------------
    # Exploration: Dirichlet‑based frequency vector (Parent A)
    # ------------------------------------------------------------------
    def indy_learning(self, inflow_rate: float) -> np.ndarray:
        """
        Generate a stochastic frequency vector via a Dirichlet draw.
        The vector sums to ``inflow_rate`` and is used as the NLMS input.
        """
        freq = self.rng.dirichlet(np.ones(self.d_in))
        return freq * inflow_rate

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, action_id: str) -> BanditAction:
        """
        Produce a BanditAction whose propensity is derived from the softmax
        of the current propensity weights.
        """
        # Softmax to obtain a probability distribution
        max_w = np.max(self.propensity_weights)
        exp_w = np.exp(self.propensity_weights - max_w)
        probs = exp_w / np.sum(exp_w)

        # Sample a single propensity (for simplicity we pick the first)
        propensity = float(probs[0])

        # Expected reward is the logistic prediction of the margin
        margin = self.propensity_weights[0]
        expected_reward = float(sigmoid(margin))

        # Confidence bound (placeholder: sqrt of variance)
        confidence = math.sqrt(propensity * (1 - propensity) / (self.d_in + 1))

        return BanditAction(
            action_id=action_id,
            propensity=propensity,
            expected_reward=expected_reward,
            confidence_bound=confidence,
            algorithm="HybridBanditNLMS",
        )

    # ------------------------------------------------------------------
    # Update step (mathematical fusion)
    # ------------------------------------------------------------------
    def update(
        self,
        action: BanditAction,
        reward: float,
        inflow_rate: float = 1.0,
    ) -> BanditUpdate:
        """
        Perform a single hybrid update:
        1. Generate a frequency vector ``x`` via ``indy_learning``.
        2. Compute logistic margin from the current weight.
        3. Derive logistic error = reward - σ(margin).
        4. Apply NLMS update to ``self.propensity_weights`` using ``x`` and
           the logistic error.
        """
        # 1. NLMS input vector
        x = self.indy_learning(inflow_rate)

        # 2. Logistic margin (use the weight corresponding to the action)
        margin = self.propensity_weights[0]

        # 3. Logistic prediction and error
        pred = float(sigmoid(margin))
        error = reward - pred  # scalar error

        # 4. NLMS weight adaptation
        self.propensity_weights = nlms_update(
            self.propensity_weights, x, error, mu=self.mu, eps=self.reg
        )

        # Optional: small gradient step toward logistic loss (learning_rate)
        g, _ = binary_logistic_grad_hess(
            np.array([reward]), np.array([margin])
        )
        self.propensity_weights -= self.lr * g

        # Return a BanditUpdate record
        return BanditUpdate(
            context_id="default_context",
            action_id=action.action_id,
            reward=reward,
            propensity=action.propensity,
        )

    # ------------------------------------------------------------------
    # Utility: expose current propensity distribution
    # ------------------------------------------------------------------
    def current_propensities(self) -> np.ndarray:
        """Softmax of the internal weights."""
        max_w = np.max(self.propensity_weights)
        exp_w = np.exp(self.propensity_weights - max_w)
        return exp_w / np.sum(exp_w)


# ----------------------------------------------------------------------
# Demonstration functions (require at least three)
# ----------------------------------------------------------------------
def demo_sigmoid_and_grad():
    """Show sigmoid, gradient and Hessian on a simple margin array."""
    margins = np.array([-2.0, 0.0, 2.0])
    sig = sigmoid(margins)
    g, h = binary_logistic_grad_hess(np.array([0, 1, 1]), margins)
    print("Margins:", margins)
    print("Sigmoid:", sig)
    print("Gradient:", g)
    print("Hessian:", h)


def demo_nlms_step():
    """Run a single NLMS update on a random weight vector."""
    rng = np.random.default_rng(42)
    w = rng.random(5)
    x = rng.random(5)
    error = 0.3
    w_new = nlms_update(w, x, error, mu=0.7)
    print("Before NLMS:", w)
    print("Input x:", x)
    print("Error:", error)
    print("After NLMS:", w_new)


def demo_hybrid_cycle():
    """Execute a few cycles of the hybrid bandit/NLMS algorithm."""
    hybrid = HybridBanditNLMS(d_in=5, seed=123, mu=0.4, learning_rate=0.05)

    for step in range(3):
        action = hybrid.select_action(action_id=f"act_{step}")
        # Simulate a stochastic binary reward with probability = expected_reward
        reward = 1.0 if random.random() < action.expected_reward else 0.0
        update = hybrid.update(action, reward, inflow_rate=1.0)
        print(f"\nStep {step + 1}")
        print("Selected action:", asdict(action))
        print("Observed reward:", reward)
        print("Update record:", asdict(update))
        print("Current propensities:", hybrid.current_propensities())


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Sigmoid & Logistic Gradient ===")
    demo_sigmoid_and_grad()

    print("\n=== Demo: NLMS Update ===")
    demo_nlms_step()

    print("\n=== Demo: Hybrid Bandit‑NLMS Cycle ===")
    demo_hybrid_cycle()