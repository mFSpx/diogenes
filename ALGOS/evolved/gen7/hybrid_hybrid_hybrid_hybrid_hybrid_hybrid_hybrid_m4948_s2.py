# DARWIN HAMMER — match 4948, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_krampu_m1840_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_omni_c_hybrid_hybrid_hybrid_m1805_s1.py (gen5)
# born: 2026-05-29T23:59:04Z

"""Hybrid Algorithm combining:
- Parent A: `HybridRouter` bandit policy updater.
- Parent B: SSIM‑like similarity reward and probabilistic acceptance.

Mathematical Bridge
-------------------
Parent B provides a similarity score ρ∈[0,1] between a predictor output and a target.
We transform this score into a bandit reward `r = ρ` (higher similarity → higher reward)
and feed it to Parent A’s `HybridRouter` via a `BanditUpdate`.  

Conversely, Parent A supplies a context‑aware Gaussian kernel matrix `K`
derived from a *context vector* (here the input feature vector).  
The kernel modulates the Normalised Least‑Mean‑Squares (NLMS) weight update:

    e   = y – ŷ                         # prediction error
    K   = gaussian_kernel(x, x, σ)     # scalar kernel for the current context
    Δw = μ · K · e · x / (‖x‖² + ε)    # NLMS step weighted by the kernel

Thus the similarity reward drives the bandit policy while the kernel‑weighted
NLMS adapts the predictor that generates the similarity score, closing the
feedback loop between the two parent topologies.

The module implements this fused workflow with three core functions:
`encode_and_normalize`, `compute_similarity_reward`, and `hybrid_step`.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditUpdate:
    """Record of a single bandit learning event."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


class HybridRouter:
    """Simple bandit policy accumulator."""
    _POLICY: Dict[str, List[float]] = {}

    def __init__(self) -> None:
        self._reset_policy()

    def _reset_policy(self) -> None:
        self._POLICY.clear()

    def update_policy(self, updates: List[BanditUpdate]) -> None:
        """Accumulate reward and propensity per action."""
        for u in updates:
            stats = self._POLICY.setdefault(u.action_id, [0.0, 0.0])
            stats[0] += float(u.reward)
            stats[1] += float(u.propensity)

    def get_policy(self) -> Dict[str, Tuple[float, float]]:
        """Return a copy of the learned policy as (reward_sum, propensity_sum)."""
        return {k: (v[0], v[1]) for k, v in self._POLICY.items()}


# ----------------------------------------------------------------------
# Helper functions (from Parent B)
# ----------------------------------------------------------------------
def encode_vector(x: np.ndarray) -> np.ndarray:
    """L2‑normalize a vector (Parent B encoder)."""
    norm = np.linalg.norm(x)
    if norm == 0.0:
        return x.copy()
    return x / norm


def ssim_reward(x: np.ndarray, y: np.ndarray) -> float:
    """
    Lightweight SSIM‑like similarity in [0, 1].
    Mirrors the classic SSIM numerator/denominator structure.
    """
    C1 = 1e-4
    C2 = 9e-4

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return float(numerator / (denominator + 1e-12))


def gaussian_kernel(a: np.ndarray, b: np.ndarray, sigma: float = 1.0) -> float:
    """
    Radial‑basis Gaussian kernel between two vectors.
    Returns a scalar (since we use the same context vector for both arguments).
    """
    diff = a - b
    dist_sq = np.dot(diff, diff)
    return math.exp(-dist_sq / (2 * sigma ** 2))


# ----------------------------------------------------------------------
# NLMS predictor (core of Parent A’s NLMS component)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction ŷ = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    error: float,
    mu: float = 0.01,
    sigma: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Kernel‑weighted NLMS weight update.

    Δw = μ · K(x,x) · e · x / (‖x‖² + ε)

    where K is a Gaussian kernel encoding the context similarity.
    """
    norm_sq = np.dot(x, x) + eps
    K = gaussian_kernel(x, x, sigma)
    delta_w = (mu * K * error / norm_sq) * x
    return weights + delta_w


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def encode_and_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Wrapper that applies Parent B's encoder and ensures a float64 array.
    """
    return encode_vector(vec.astype(np.float64))


def compute_similarity_reward(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the SSIM‑like reward between prediction and target.
    The reward directly serves as the bandit reward.
    """
    return ssim_reward(pred, target)


def hybrid_step(
    router: HybridRouter,
    weights: np.ndarray,
    context_id: str,
    action_id: str,
    input_vec: np.ndarray,
    target_vec: np.ndarray,
    mu: float = 0.01,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single hybrid iteration:

    1. Encode the input (Parent B).
    2. Predict with NLMS weights (Parent A).
    3. Measure similarity reward (Parent B).
    4. Feed the reward to the bandit router (Parent A).
    5. Update NLMS weights using a kernel‑weighted NLMS rule (bridge).

    Returns the updated weight vector and the scalar reward.
    """
    # 1. Encode / normalise
    x = encode_and_normalize(input_vec)

    # 2. Linear prediction
    y_pred = nlms_predict(weights, x)

    # 3. Similarity reward (scalar)
    reward = compute_similarity_reward(np.full_like(x, y_pred), target_vec)

    # 4. Bandit update – propensity is set to 1.0 for simplicity
    update = BanditUpdate(
        context_id=context_id,
        action_id=action_id,
        reward=reward,
        propensity=1.0,
    )
    router.update_policy([update])

    # 5. NLMS weight adaptation
    error = float(target_vec.mean() - y_pred)  # scalar error proxy
    new_weights = nlms_update(weights, x, error, mu=mu, sigma=sigma)

    return new_weights, reward


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # Synthetic problem: map 5‑dim input to a scalar target
    dim = 5
    weights = rng.normal(scale=0.1, size=dim).astype(np.float64)

    router = HybridRouter()

    for step in range(10):
        inp = rng.normal(size=dim)
        # True target is a noisy linear function of the input
        true_w = np.arange(1, dim + 1, dtype=np.float64) * 0.2
        target = np.dot(true_w, inp) + rng.normal(scale=0.05)

        # Convert target to a vector of the same shape for SSIM calculation
        target_vec = np.full(dim, target, dtype=np.float64)

        weights, reward = hybrid_step(
            router=router,
            weights=weights,
            context_id=f"ctx_{step}",
            action_id="predict",
            input_vec=inp,
            target_vec=target_vec,
            mu=0.05,
            sigma=1.0,
        )
        print(f"Step {step:02d} | Reward: {reward:.4f} | Weight norm: {np.linalg.norm(weights):.4f}")

    # Display learned policy
    print("\nLearned policy (action → (reward_sum, propensity_sum)):")
    for act, (rew, prop) in router.get_policy().items():
        print(f"  {act}: reward_sum={rew:.3f}, propensity_sum={prop:.1f}")