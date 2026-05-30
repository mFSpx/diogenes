# DARWIN HAMMER — match 4249, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m169_s0.py (gen3)
# parent_b: hybrid_decreasing_pruning_hybrid_hybrid_bandit_m365_s2.py (gen3)
# born: 2026-05-29T23:54:27Z

"""Hybrid Algorithm: SSIM‑Bandit‑Pruning‑Temperature Fusion

Parents:
- hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m169_s0.py (SSIM similarity, Schoolfield temperature scaling, bandit propensity)
- hybrid_decreasing_pruning_hybrid_hybrid_bandit_m365_s2.py (Exponential pruning rate, temperature‑dependent developmental rate, bandit policy)

Mathematical Bridge:
Both parents expose a temperature‑dependent scalar ρ(T) derived from the Schoolfield model.
Parent A multiplies ρ(T) with SSIM to obtain a suitability σ_i.
Parent B multiplies ρ(T) with an exponential pruning schedule λ·exp(‑α·t) to obtain an effective rate r_eff(t,T).

The fused system therefore defines:

    ρ(T)      – developmental rate from Schoolfield parameters.
    σ_i       – suitability of item i: σ_i = SSIM_i · ρ(T).
    r_eff(t,T) – combined pruning/scale rate: r_eff = λ·exp(‑α·t)·ρ(T)  (clipped to [0,1]).

These quantities drive:
1. Work‑share allocation among items (deterministic share ∝ σ_i, stochastic share via bandit propensities).
2. Edge‑pruning probability in a graph‑like structure (probability = r_eff).
3. Bandit policy updates where propensities are biased by σ_i.

The module below implements the unified mathematics and provides three public functions that showcase the hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Schoolfield temperature model (shared by both parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters for the Schoolfield developmental rate model."""
    rho_25: float = 1.0            # reference rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0   # activation enthalpy (cal mol⁻¹)
    t_low: float = 283.15          # low temperature threshold (K)
    t_high: float = 307.15         # high temperature threshold (K)
    delta_h_low: float = -45_000.0          # low‑temp deactivation enthalpy (cal mol⁻¹)
    delta_h_high: float = 65_000.0          # high‑temp deactivation enthalpy (cal mol⁻¹)
    r_cal: float = 1.987           # gas constant (cal mol⁻¹ K⁻¹)


def developmental_rate(T: float, p: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Compute the temperature‑dependent scalar ρ(T) using the Schoolfield model.
    The formulation follows the classic ectotherm developmental rate equation.
    """
    # Clamp temperature to biologically plausible range to avoid division by zero.
    T = max(p.t_low, min(p.t_high, T))
    k = p.r_cal
    num = p.rho_25 * math.exp(-p.delta_h_activation / (k * (T - 273.15)))
    den_low = 1.0 + math.exp(p.delta_h_low / (k * (p.t_low - 273.15)) - p.delta_h_low / (k * (T - 273.15)))
    den_high = 1.0 + math.exp(p.delta_h_high / (k * (p.t_high - 273.15)) - p.delta_h_high / (k * (T - 273.15)))
    return num / (den_low * den_high)


# ----------------------------------------------------------------------
# SSIM similarity (Parent A)
# ----------------------------------------------------------------------


def _pct(value: float) -> float:
    """Round to six decimal places for readability."""
    return round(float(value), 6)


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) between two 1‑D vectors.
    Returns a value in [-1, 1]; typical usage expects [0, 1].
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Bandit data structures (shared by both parents)
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


_POLICY: Dict[str, List[float]] = {}  # maps action_id -> [cumulative_reward, count]


def reset_policy() -> None:
    """Clear the global bandit policy."""
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """Accumulate rewards per action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])  # [cumulative_reward, count]
        stats[0] += float(u.reward)
        stats[1] += 1.0


def get_action_statistics(action_id: str) -> Tuple[float, float]:
    """Return (average_reward, count) for a given action."""
    cum, cnt = _POLICY.get(action_id, [0.0, 0.0])
    avg = cum / cnt if cnt > 0 else 0.0
    return avg, cnt


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------


def suitability_scores(ssim_vec: np.ndarray, T: float,
                       sf_params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    """
    Compute the hybrid suitability σ_i = SSIM_i × ρ(T) for each element i.
    The input `ssim_vec` may be a 1‑D array of pre‑computed SSIM values.
    """
    rho = developmental_rate(T, sf_params)
    return np.multiply(ssim_vec, rho)


def effective_pruning_rate(t: float, T: float, lam: float, alpha: float,
                           sf_params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Combined pruning / scaling rate:
        r_eff(t,T) = λ·exp(‑α·t)·ρ(T)
    The result is clipped to the unit interval [0,1].
    """
    base = lam * math.exp(-alpha * t)
    rho = developmental_rate(T, sf_params)
    r = base * rho
    return max(0.0, min(1.0, r))


def allocate_work_and_update_bandit(ssim_matrix: np.ndarray,
                                   T: float,
                                   actions: List[BanditAction],
                                   lam: float,
                                   alpha: float,
                                   t: float,
                                   sf_params: SchoolfieldParams = SchoolfieldParams()) -> Tuple[np.ndarray, List[BanditAction]]:
    """
    Hybrid operation that:
    1. Computes suitability σ_i for each column (action) of `ssim_matrix`.
    2. Derives a deterministic work share proportional to σ_i.
    3. Adjusts each action's propensity using the combined pruning rate r_eff(t,T).
    4. Returns the normalized work allocation vector and the updated action list.

    Parameters
    ----------
    ssim_matrix : np.ndarray
        Shape (n_items, n_actions). Each column will be reduced to a single SSIM score
        (mean across rows) before fusion with temperature.
    T : float
        Current temperature (Kelvin).
    actions : List[BanditAction]
        Existing bandit actions; their propensities are overwritten.
    lam, alpha, t : float
        Parameters of the exponential pruning schedule.
    sf_params : SchoolfieldParams
        Optional custom Schoolfield parameters.

    Returns
    -------
    work_allocation : np.ndarray
        Normalized allocation (sums to 1) for each action.
    updated_actions : List[BanditAction]
        New BanditAction objects with modified propensities and confidence bounds.
    """
    # 1️⃣ Reduce SSIM matrix to a vector of scores (mean similarity per action)
    mean_ssim = np.mean(ssim_matrix, axis=0)
    # 2️⃣ Fuse with temperature
    sigma = suitability_scores(mean_ssim, T, sf_params)
    # 3️⃣ Deterministic allocation (proportional to σ_i)
    if sigma.sum() == 0:
        deterministic = np.full_like(sigma, 1.0 / sigma.size)
    else:
        deterministic = sigma / sigma.sum()
    # 4️⃣ Effective pruning rate that modulates stochastic bandit share
    r_eff = effective_pruning_rate(t, T, lam, alpha, sf_params)
    # 5️⃣ Stochastic bandit share: propensity ∝ σ_i * r_eff
    stochastic_raw = sigma * r_eff
    if stochastic_raw.sum() == 0:
        stochastic = np.full_like(stochastic_raw, 1.0 / stochastic_raw.size)
    else:
        stochastic = stochastic_raw / stochastic_raw.sum()
    # 6️⃣ Mix deterministic and stochastic shares (simple average)
    work_allocation = 0.5 * deterministic + 0.5 * stochastic

    # 7️⃣ Update BanditAction objects
    updated_actions = []
    for idx, act in enumerate(actions):
        new_propensity = float(work_allocation[idx])
        # Confidence bound shrinks with higher propensity (heuristic)
        new_confidence = max(0.01, 1.0 - new_propensity)
        avg_reward, count = get_action_statistics(act.action_id)
        updated = BanditAction(
            action_id=act.action_id,
            propensity=new_propensity,
            expected_reward=avg_reward,
            confidence_bound=new_confidence,
            algorithm=act.algorithm
        )
        updated_actions.append(updated)

    return work_allocation, updated_actions


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Synthetic SSIM matrix: 5 items × 3 actions
    n_items, n_actions = 5, 3
    ssim_matrix = np.random.rand(n_items, n_actions)

    # Temperature (K) and time parameters
    temperature = 295.0          # ~22 °C
    current_time = 2.5           # arbitrary time units
    lam = 0.9                    # base pruning intensity
    alpha = 0.3                  # decay rate

    # Initialise dummy bandit actions
    actions = [
        BanditAction(action_id=f"a{i}", propensity=1.0 / n_actions,
                     expected_reward=0.0, confidence_bound=0.5, algorithm="hybrid")
        for i in range(n_actions)
    ]

    # Reset and seed policy with some fake updates
    reset_policy()
    fake_updates = [
        BanditUpdate(context_id="c0", action_id="a0", reward=1.2, propensity=0.3),
        BanditUpdate(context_id="c1", action_id="a1", reward=0.8, propensity=0.4),
        BanditUpdate(context_id="c2", action_id="a2", reward=1.5, propensity=0.5),
    ]
    update_policy(fake_updates)

    # Run hybrid allocation
    allocation, updated_actions = allocate_work_and_update_bandit(
        ssim_matrix=ssim_matrix,
        T=temperature,
        actions=actions,
        lam=lam,
        alpha=alpha,
        t=current_time
    )

    # Print results (rounded for readability)
    print("SSIM matrix:\n", np.round(ssim_matrix, 3))
    print("\nTemperature (K):", temperature)
    print("\nEffective pruning rate r_eff:", _pct(effective_pruning_rate(current_time, temperature, lam, alpha)))
    print("\nWork allocation per action:", np.round(allocation, 4))
    print("\nUpdated Bandit Actions:")
    for act in updated_actions:
        print(f"  {act.action_id}: propensity={_pct(act.propensity)}, expected_reward={_pct(act.expected_reward)}, confidence={_pct(act.confidence_bound)}")
    sys.exit(0)