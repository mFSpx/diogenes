# DARWIN HAMMER — match 5286, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_krampu_m2626_s0.py (gen6)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_hybrid_hybrid_m1453_s5.py (gen6)
# born: 2026-05-30T00:01:04Z

"""Hybrid Temperature‑Bandit‑VRAM‑Pheromone Scheduler

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_krampu_m2626_s0.py (A)
  Provides a temperature‑dependent developmental_rate (Schoolfield model) and
  BanditAction data structures used for reward prediction.
- hybrid_hybrid_model_vram_sc_hybrid_hybrid_hybrid_m1453_s5.py (B)
  Provides a resource‑store dynamics equation driven by bandit propensity,
  confidence bounds and pheromone decay, together with a learning update that
  mixes squared‑error loss with an SSIM‑based regulariser.

Mathematical bridge:
Both parents expose a scalar “resource” that is influenced by a bandit’s
propensity/confidence and by an external scalar (temperature or pheromone).
We therefore fuse them by letting the **store** evolve according to the
temperature‑modulated reward from the Schoolfield model while the inflow/outflow
terms are taken from the VRAM‑Pheromone dynamics:

    Δstore = α·(propensity + pheromone) – β·(confidence_bound) – γ·store
    store_{t+1} = max(0, store_t + Δstore·dt)

The reward used for learning is the developmental_rate evaluated at the current
temperature, which scales the squared‑error term.  The learning rate is also
modulated by the bandit propensity, completing the hybrid coupling.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit decision."""
    action_id: str
    propensity: float          # virtual “inflow” magnitude
    expected_reward: float
    confidence_bound: float    # virtual “outflow” magnitude


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Schoolfield temperature model."""
    rho_25: float = 1.0               # rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15                     # K
    t_high: float = 307.15                    # K
    delta_h_low: float = -45_000.0            # J mol⁻¹
    delta_h_high: float = 65_000.0            # J mol⁻¹
    r_cal: float = 1.987                      # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)


# ----------------------------------------------------------------------
# Core mathematical components
# ----------------------------------------------------------------------


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    """
    Schoolfield temperature‑dependent developmental rate.

    The denominator implements low‑ and high‑temperature deactivation terms.
    """
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")

    # activation term (Arrhenius)
    act = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * (1 / 298.15 - 1 / temp_k)
    )

    # low‑temperature deactivation
    low = 1.0 + math.exp(
        (params.delta_h_low / params.r_cal) * (1 / params.t_low - 1 / temp_k)
    )

    # high‑temperature deactivation
    high = 1.0 + math.exp(
        (params.delta_h_high / params.r_cal) * (1 / temp_k - 1 / params.t_high)
    )

    return act / (low * high)


def simple_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Very lightweight structural similarity index (SSIM) approximation.
    Works on 2‑D arrays; returns a value in [0, 1] where 1 means identical.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)

    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Hybrid dynamics
# ----------------------------------------------------------------------


class HybridStore:
    """
    Represents the unified resource pool that aggregates VRAM memory,
    pheromone concentration and temperature‑scaled reward.
    """

    def __init__(self, init_store: float = 0.0):
        self.store = float(init_store)

    def step(
        self,
        action: BanditAction,
        pheromone: float,
        dt: float,
        alpha: float = 0.6,
        beta: float = 0.4,
        gamma: float = 0.01,
    ) -> float:
        """
        Update the store according to the fused dynamics:

            Δstore = α·(propensity + pheromone) – β·(confidence_bound) – γ·store
            store_{t+1} = max(0, store_t + Δstore·dt)

        Returns the new store value.
        """
        inflow = action.propensity + pheromone
        outflow = action.confidence_bound
        delta = alpha * inflow - beta * outflow - gamma * self.store
        self.store = max(0.0, self.store + delta * dt)
        return self.store


def compute_temperature_reward(
    action: BanditAction,
    temp_c: float,
    params: SchoolfieldParams,
) -> float:
    """
    Scale the bandit's expected reward by the developmental rate at the
    supplied temperature (Celsius).  This couples the temperature model (A)
    with the bandit decision (B).
    """
    temp_k = c_to_k(temp_c)
    rate = developmental_rate(temp_k, params)
    return action.expected_reward * rate


def learning_update(
    W: np.ndarray,
    X: np.ndarray,
    estimate: np.ndarray,
    target: np.ndarray,
    ssim_target: np.ndarray,
    eta0: float,
    propensity: float,
    lam: float = 0.1,
) -> np.ndarray:
    """
    Perform a single gradient‑descent step on the weight matrix W.

    Loss:
        L = ||estimate - target||² + λ·(1 - SSIM(estimate, ssim_target))

    Effective learning rate:
        η = η₀·(1 + propensity)

    Returns the updated weight matrix.
    """
    # squared‑error gradient
    grad_err = 2.0 * (estimate - target)[:, None] * X[None, :]  # shape (out, in)

    # SSIM gradient approximation: we treat SSIM as a scalar multiplier
    ssim_val = simple_ssim(estimate, ssim_target)
    # d/dW (1 - SSIM) ≈ -(dSSIM/destimate)·destimate/dW
    # Using a simple proxy: (estimate - ssim_target)
    grad_ssim = -(estimate - ssim_target)[:, None] * X[None, :] * (1.0 / (ssim_val + 1e-8))

    grad = grad_err + lam * grad_ssim

    eta = eta0 * (1.0 + propensity)
    W_new = W - eta * grad
    return W_new


# ----------------------------------------------------------------------
# Helper utilities (Voronoi‑like region assignment)
# ----------------------------------------------------------------------


def assign_voronoi_regions(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Very lightweight Voronoi assignment: for each point, return the index of the
    nearest seed (Euclidean distance).  This mimics the Voronoi partitioning
    used in parent A.
    """
    if points.ndim != 2 or seeds.ndim != 2:
        raise ValueError("points and seeds must be 2‑D arrays (N×D, M×D)")
    # Compute squared distances efficiently
    diff = points[:, None, :] - seeds[None, :, :]          # (N, M, D)
    dists = np.einsum('nmd,nmd->nm', diff, diff)          # (N, M)
    return np.argmin(dists, axis=1)


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _smoke_test() -> None:
    # 1. Create a dummy bandit action
    action = BanditAction(
        action_id="a1",
        propensity=0.8,
        expected_reward=5.0,
        confidence_bound=0.3,
    )

    # 2. Temperature‑scaled reward
    temp_c = 30.0
    params = SchoolfieldParams()
    temp_reward = compute_temperature_reward(action, temp_c, params)
    print(f"Temperature‑scaled reward: {temp_reward:.4f}")

    # 3. Store dynamics
    store = HybridStore(init_store=10.0)
    for step in range(3):
        new_val = store.step(
            action=action,
            pheromone=0.2,   # constant pheromone inflow
            dt=1.0,
            alpha=0.6,
            beta=0.4,
            gamma=0.02,
        )
        print(f"Step {step+1} – store: {new_val:.4f}")

    # 4. Learning update
    np.random.seed(0)
    W = np.random.randn(4, 3)          # weight matrix (output×input)
    X = np.random.randn(3)             # input feature vector
    estimate = W @ X                   # model output
    target = np.array([1.0, 0.5, -0.2, 0.8])
    ssim_target = np.full_like(estimate, 0.5)  # dummy target for SSIM term

    W_new = learning_update(
        W=W,
        X=X,
        estimate=estimate,
        target=target,
        ssim_target=ssim_target,
        eta0=0.05,
        propensity=action.propensity,
        lam=0.2,
    )
    print(f"Weight matrix change norm: {np.linalg.norm(W - W_new):.6f}")

    # 5. Voronoi region assignment
    points = np.random.rand(10, 2)
    seeds = np.array([[0.2, 0.2], [0.8, 0.8], [0.5, 0.1]])
    regions = assign_voronoi_regions(points, seeds)
    print(f"Voronoi region indices: {regions}")


if __name__ == "__main__":
    _smoke_test()