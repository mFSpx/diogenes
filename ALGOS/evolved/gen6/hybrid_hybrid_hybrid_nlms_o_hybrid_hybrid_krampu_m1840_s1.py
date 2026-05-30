# DARWIN HAMMER — match 1840, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s1.py (gen4)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m524_s3.py (gen5)
# born: 2026-05-29T23:39:07Z

"""Hybrid NLMS‑Rectified‑Flow & Krampus‑Bandit‑RBF Fusion

This module fuses the two parent algorithms:

* **Parent A** – NLMS predictor combined with a straight‑line interpolant of a
  rectified flow.  The NLMS weights are adapted to predict the flow velocity.
* **Parent B** – Krampus brain‑map bandit router that uses a Gaussian‑RBF
  surrogate to turn a context vector into a set of kernel features for action
  selection.

**Mathematical bridge**

The interpolated flow state `Z_t = t·x1 + (1‑t)·x0` is treated as a collection of
sample points.  The Krampus brain‑map supplies a *context vector* `c`.  A
Gaussian‑RBF kernel `k(c, Z_t) = exp(-‖c‑Z_t‖²/(2σ²))` converts the flow state
into a scalar feature.  This scalar is concatenated with the interpolated
state to form the NLMS input vector


φ = [ Z_t , k(c, Z_t) ] .


The NLMS predicts the flow velocity `v̂ = w·φ`.  The prediction error drives
both the NLMS weight update and the bandit policy update: a smaller error is
interpreted as a higher reward for the action selected by the router.  Thus
the two topologies are mathematically intertwined through the shared feature
vector `φ` and the common error signal.

The code below implements this hybrid system with three core functions:
`interpolant`, `gaussian_kernel`, and `hybrid_step`.  A lightweight smoke
test demonstrates end‑to‑end execution.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Parent A – NLMS utilities
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    One Normalised LMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    (weights, error) : Tuple[np.ndarray, float]
        Updated weights and the prediction error.
    """
    error = target - nlms_predict(weights, x)
    weights = weights + mu * error * x / (np.linalg.norm(x) ** 2 + eps)
    return weights, error


def interpolant(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """
    Straight‑line interpolant between two vectors.

    Z_t = t·x1 + (1‑t)·x0
    """
    return t * x1 + (1.0 - t) * x0


def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """
    Target vector field for the rectified flow:
    v_θ(Z_t, t) = (x1 - x0)  (constant velocity along the line).
    """
    return x1 - x0


# ----------------------------------------------------------------------
# Parent B – Krampus brain‑map / RBF utilities
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


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def gaussian_kernel(x: np.ndarray, c: np.ndarray, sigma: float) -> float:
    """
    Isotropic Gaussian RBF kernel.
    k(x, c) = exp( -‖x‑c‖² / (2σ²) )
    """
    diff = x - c
    return math.exp(-np.dot(diff, diff) / (2.0 * sigma * sigma))


class HybridRouter:
    """
    Simple bandit router whose policy is a mapping
    action_id → [cumulative_reward, count].
    """
    _POLICY: Dict[str, List[float]] = {}

    def __init__(self):
        self._reset_policy()

    def _reset_policy(self) -> None:
        self._POLICY.clear()

    def update_policy(self, updates: List[BanditUpdate]) -> None:
        """Accumulate rewards for each action."""
        for u in updates:
            stats = self._POLICY.setdefault(u.action_id, [0.0, 0.0])
            stats[0] += float(u.reward)      # cumulative reward
            stats[1] += 1.0                   # count

    def select_action(self, candidate_actions: List[BanditAction]) -> BanditAction:
        """
        Choose the action with the highest average reward seen so far.
        If an action is unseen, fall back to propensity‑weighted random choice.
        """
        unseen = [a for a in candidate_actions if a.action_id not in self._POLICY]
        if unseen:
            # propensity‑weighted random among unseen actions
            total = sum(a.propensity for a in unseen)
            r = random.random() * total
            cum = 0.0
            for a in unseen:
                cum += a.propensity
                if r <= cum:
                    return a
        # all seen → pick highest average reward
        best = max(
            candidate_actions,
            key=lambda a: self._POLICY[a.action_id][0] / max(self._POLICY[a.action_id][1], 1e-9)
        )
        return best


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def extract_features(
    interpolated_state: np.ndarray,
    context_vec: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Build the NLMS feature vector φ = [Z_t , k(c, Z_t)].
    """
    rbf = gaussian_kernel(interpolated_state, context_vec, sigma)
    return np.concatenate([interpolated_state, [rbf]])


def hybrid_step(
    weights: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    t: float,
    context_vec: np.ndarray,
    router: HybridRouter,
    candidate_actions: List[BanditAction],
    mu: float = 0.5,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, float, HybridRouter]:
    """
    Perform one hybrid iteration:

    1. Interpolate the flow state.
    2. Convert it to NLMS features via the Gaussian kernel.
    3. Predict the flow velocity with NLMS.
    4. Compute the true velocity from the flow target.
    5. Update NLMS weights.
    6. Use the prediction error as a reward to update the bandit policy.
    7. Return updated weights, error, and router.
    """
    # 1. Interpolation
    Z_t = interpolant(x0, x1, t)

    # 2. Feature extraction
    phi = extract_features(Z_t, context_vec, sigma)

    # 3. NLMS prediction
    pred = nlms_predict(weights, phi)

    # 4. True velocity (scalar magnitude of the constant flow vector)
    true_vec = flow_target(x0, x1)
    true_vel = float(np.linalg.norm(true_vec))

    # 5. NLMS weight update
    weights, error = nlms_update(weights, phi, true_vel, mu=mu)

    # 6. Bandit update
    # Select an action based on current policy
    action = router.select_action(candidate_actions)
    # Reward is higher when error is lower (inverse relationship)
    reward = -abs(error)
    update = BanditUpdate(
        context_id="ctx_" + str(hash(context_vec.tobytes())),
        action_id=action.action_id,
        reward=reward,
        propensity=action.propensity,
    )
    router.update_policy([update])

    return weights, error, router


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dimensionality
    dim = 4

    # Random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialise NLMS weights (dim + 1 for the RBF scalar)
    w = np.zeros(dim + 1)

    # Random flow endpoints
    x0 = np.random.randn(dim)
    x1 = np.random.randn(dim)

    # Random context vector for the brain‑map
    context = np.random.randn(dim)

    # Create a router and a few dummy actions
    router = HybridRouter()
    actions = [
        BanditAction(
            action_id=f"act_{i}",
            propensity=random.random(),
            expected_reward=0.0,
            confidence_bound=0.0,
            algorithm="Hybrid"
        )
        for i in range(3)
    ]

    # Run several hybrid steps
    for step in range(5):
        t = random.random()  # interpolation parameter in [0,1]
        w, err, router = hybrid_step(
            weights=w,
            x0=x0,
            x1=x1,
            t=t,
            context_vec=context,
            router=router,
            candidate_actions=actions,
            mu=0.4,
            sigma=1.2,
        )
        print(f"Step {step+1:02d}: error = {err:.6f}")

    # Final policy dump (for visual inspection)
    print("\nFinal bandit policy (action_id → [reward_sum, count]):")
    for aid, stats in router._POLICY.items():
        avg = stats[0] / max(stats[1], 1e-9)
        print(f"  {aid}: sum={stats[0]:.3f}, count={int(stats[1])}, avg={avg:.3f}")