# DARWIN HAMMER — match 5655, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_fisher_bandit_router_m1158_s0.py (gen3)
# parent_b: nlms.py (gen0)
# born: 2026-05-30T00:03:56Z

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class BanditAction:
    """Immutable representation of a bandit arm/action."""
    action_id: str
    propensity: float          # Probability of selection
    expected_reward: float     # Estimated reward
    confidence_bound: float    # Upper confidence bound (UCB) or similar
    algorithm: str             # Identifier of the bandit algorithm used


@dataclass
class BanditUpdate:
    """Container for a batch of bandit observations used to adapt the hybrid learner."""
    actions: List[BanditAction]
    context: Optional[np.ndarray] = None  # Optional contextual features


def gaussian_beam_model(center: float, width: float, theta: float) -> Tuple[float, float]:
    """
    Compute the intensity and derivative of a 1‑D Gaussian beam.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    intensity = math.exp(-0.5 * z * z)
    derivative = intensity * (-(theta - center) / (width * width))
    return intensity, derivative


def fisher_score(intensity: float, derivative: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a scalar Gaussian observation.
    Uses I(θ) = (∂μ/∂θ)² / σ², where intensity plays the role of the likelihood.
    """
    # Guard against division by zero when intensity is extremely small
    return (derivative * derivative) / (intensity + eps)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalized Least‑Mean‑Squares weight update.
    """
    if weights.shape != x.shape:
        raise ValueError("weights and input must have the same shape")
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must lie in (0, 2)")

    y = float(weights @ x)
    error = target - y
    power = float(x @ x) + eps
    next_weights = weights + mu * error * x / power
    return next_weights, error


def _adaptive_mu(base_mu: float, bandit_update: BanditUpdate) -> float:
    """
    Adapt the NLMS step size using bandit‑derived uncertainty.
    The adaptation factor is 1 + average confidence bound,
    capped to keep mu within (0, 2).
    """
    if not bandit_update.actions:
        return base_mu
    avg_conf = np.mean([a.confidence_bound for a in bandit_update.actions])
    adapted = base_mu * (1.0 + avg_conf)
    return max(0.0, min(adapted, 1.999))  # enforce NLMS stability limits


def hybrid_update(
    center: float,
    width: float,
    theta: float,
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    bandit_update: BanditUpdate,
    base_mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float, float]:
    """
    Perform a single hybrid learning step:
      1. Compute Gaussian beam intensity & derivative.
      2. Derive Fisher information.
      3. Adapt NLMS step size using bandit uncertainty.
      4. Update weights with NLMS.
    Returns the updated weights, Fisher information, and prediction error.
    """
    intensity, derivative = gaussian_beam_model(center, width, theta)
    fisher = fisher_score(intensity, derivative)

    mu = _adaptive_mu(base_mu, bandit_update)
    next_weights, error = nlms_update(weights, x, target, mu, eps)

    return next_weights, fisher, error


def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction using current weights."""
    return float(weights @ x)


if __name__ == "__main__":
    # Example configuration
    center = 0.5
    width = 1.0
    theta = 0.2

    # Initial NLMS weights (as a column vector)
    weights = np.array([0.1, 0.2, 0.3], dtype=float)

    # Input feature vector
    x = np.array([1.0, 2.0, 3.0], dtype=float)

    # Desired target output
    target = 1.5

    # Simulated bandit feedback (normally produced by an external bandit module)
    bandit_actions = [
        BanditAction(
            action_id="a1",
            propensity=0.3,
            expected_reward=0.6,
            confidence_bound=0.15,
            algorithm="UCB1",
        ),
        BanditAction(
            action_id="a2",
            propensity=0.7,
            expected_reward=0.4,
            confidence_bound=0.05,
            algorithm="UCB1",
        ),
    ]
    bandit_update = BanditUpdate(actions=bandit_actions)

    # Perform hybrid update
    next_weights, fisher, error = hybrid_update(
        center,
        width,
        theta,
        weights,
        x,
        target,
        bandit_update,
        base_mu=0.5,
    )

    print("Updated weights:", next_weights.tolist())
    print("Fisher information score:", fisher)
    print("Prediction error:", error)
    print("Predicted output after update:", predict(next_weights, x))