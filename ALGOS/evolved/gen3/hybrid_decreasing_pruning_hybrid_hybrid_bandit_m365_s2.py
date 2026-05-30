# DARWIN HAMMER — match 365, survivor 2
# gen: 3
# parent_a: decreasing_pruning.py (gen0)
# parent_b: hybrid_hybrid_bandit_router_state_space_duality_m143_s1.py (gen2)
# born: 2026-05-29T23:28:26Z

"""Hybrid algorithm merging Decreasing-Pruning (A) and Temperature‑Dependent State‑Space (B).

Mathematical bridge:
Both parents define a *rate* that modulates a process.
- Parent A:  p(t) = λ·exp(‑α·t)  (probability of pruning an edge at time t).
- Parent B:  ρ(T) = developmental_rate(T)  (scalar factor scaling every matrix in the state‑space).

We treat the temperature‑dependent scalar ρ(T) as a *modulator* of the pruning rate, forming a combined
effective rate  

    r_eff(t, T) = λ·exp(‑α·t) · ρ(T),

clipped to the interval [0, 1].  This unified rate drives both edge‑pruning and the scaling of the
state‑space matrices, yielding a single hybrid step that respects both stochastic pruning and
thermodynamic dynamics.  The code below implements three core functions that expose this hybrid
behaviour and integrates the bandit‑policy utilities from parent B."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol‑1 K‑1


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


_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Clear the global bandit policy."""
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """Accumulate rewards per action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])  # [cumulative_reward, count]
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Temperature‑dependent developmental rate (parent B)
# ----------------------------------------------------------------------
def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Compute the Schoolfield developmental rate ρ(T)."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    # numerator
    numerator = (
        params.rho_25
        * (temp_k / 298.15)
        * math.exp(
            (params.delta_h_activation / params.r_cal)
            * ((1.0 / 298.15) - (1.0 / temp_k))
        )
    )
    # low‑temperature inhibition term
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    # high‑temperature inhibition term
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    denominator = 1.0 + low + high
    if denominator == 0:
        raise ValueError("denominator cannot be zero")
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid primitives (fusion of A and B)
# ----------------------------------------------------------------------
def hybrid_prune_rate(
    t: float,
    temp_k: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> float:
    """
    Effective pruning rate r_eff(t, T) = λ·exp(‑α·t)·ρ(T), clipped to [0, 1].

    Parameters
    ----------
    t : float
        Abstract time (must be ≥ 0).
    temp_k : float
        Temperature in Kelvin (must be > 0).
    lam, alpha : float
        Base exponential pruning parameters.
    params : SchoolfieldParams
        Parameters governing the temperature‑dependent developmental rate.

    Returns
    -------
    float
        Probability with which an edge is *removed*.
    """
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    base = lam * math.exp(-alpha * t)
    rho = developmental_rate(temp_k, params)
    rate = base * rho
    return min(1.0, max(0.0, rate))


def hybrid_prune_edges(
    edges: List[Hashable],
    t: float,
    temp_k: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> List[Hashable]:
    """
    Probabilistically prune edges using the hybrid rate r_eff(t, T).

    Returns the subset of edges that survive the pruning step.
    """
    rng = random.Random(seed)
    p = hybrid_prune_rate(t, temp_k, lam, alpha, params)
    # Keep edge if random draw ≥ p (same convention as parent A)
    return [e for e in edges if rng.random() >= p]


def temperature_scaled_matrix(
    M: np.ndarray,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> np.ndarray:
    """
    Scale an arbitrary matrix by the developmental rate ρ(T).

    This mirrors the temperature‑dependent scaling used for A, B, and C in parent B.
    """
    rate = developmental_rate(temp_k, params)
    return M * rate


def hybrid_ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    edges: List[Hashable],
    t: float,
    temp_k: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> tuple[np.ndarray, np.ndarray, List[Hashable]]:
    """
    One hybrid state‑space step:

    1. Scale A, B, C by ρ(T).
    2. Prune the supplied edge list using the hybrid rate.
    3. Propagate the hidden state:   h_next = A_scaled @ x + B_scaled @ u
    4. Produce output:               y = C_scaled @ h_next

    Returns (h_next, y, pruned_edges).
    """
    # 1. Temperature scaling
    A_s = temperature_scaled_matrix(A, temp_k, params)
    B_s = temperature_scaled_matrix(B, temp_k, params)
    C_s = temperature_scaled_matrix(C, temp_k, params)

    # 2. Edge pruning (edges are abstract identifiers; the function does not modify matrices)
    pruned_edges = hybrid_prune_edges(edges, t, temp_k, lam, alpha, seed, params)

    # 3. State propagation
    h_next = A_s @ x + B_s @ u

    # 4. Output projection
    y = C_s @ h_next

    return h_next, y, pruned_edges


# ----------------------------------------------------------------------
# Example bandit integration (demonstrating policy update after a hybrid step)
# ----------------------------------------------------------------------
def hybrid_bandit_update(
    updates: List[BanditUpdate],
    edges: List[Hashable],
    t: float,
    temp_k: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> List[Hashable]:
    """
    Apply bandit policy updates and then prune edges using the hybrid rate.

    The function returns the surviving edges after pruning.
    """
    update_policy(updates)
    return hybrid_prune_edges(edges, t, temp_k, lam, alpha, seed, params)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic test to verify that the module runs without error.
    rng = np.random.default_rng(42)

    # Dummy state‑space matrices (2‑state, 1‑input, 1‑output)
    A = rng.normal(size=(2, 2))
    B = rng.normal(size=(2, 1))
    C = rng.normal(size=(1, 2))

    # Initial hidden state, input, and output placeholders
    h0 = rng.normal(size=(2, 1))
    x0 = rng.normal(size=(2, 1))
    u0 = rng.normal(size=(1, 1))

    # Edge identifiers (could be anything hashable)
    edges = ["e1", "e2", "e3", "e4"]

    # Parameters for hybrid step
    t = 5.0               # time
    temp_k = 295.15       # ~22 °C
    lam = 0.8
    alpha = 0.15
    seed = 12345

    # Run one hybrid step
    h1, y1, remaining_edges = hybrid_ssm_step(
        h=h0,
        x=x0,
        u=u0,
        A=A,
        B=B,
        C=C,
        edges=edges,
        t=t,
        temp_k=temp_k,
        lam=lam,
        alpha=alpha,
        seed=seed,
    )

    # Print shapes to confirm successful computation
    print("h_next shape:", h1.shape)
    print("output shape:", y1.shape)
    print("edges kept:", remaining_edges)

    # Demonstrate bandit update + pruning
    dummy_updates = [
        BanditUpdate(context_id="c1", action_id="a1", reward=1.0, propensity=0.5),
        BanditUpdate(context_id="c2", action_id="a2", reward=0.2, propensity=0.7),
    ]
    kept_after_bandit = hybrid_bandit_update(
        updates=dummy_updates,
        edges=edges,
        t=t,
        temp_k=temp_k,
        lam=lam,
        alpha=alpha,
        seed=seed,
    )
    print("edges after bandit update & pruning:", kept_after_bandit)

    # Verify policy dictionary
    print("policy state:", _POLICY)