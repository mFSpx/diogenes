# DARWIN HAMMER — match 2608, survivor 1
# gen: 3
# parent_a: hybrid_voronoi_partition_poikilotherm_schoolf_m49_s0.py (gen1)
# parent_b: hybrid_hybrid_bandit_router_variational_free_ene_m56_s0.py (gen2)
# born: 2026-05-29T23:43:06Z

"""Hybrid Voronoi‑Bandit‑Variational Free Energy Algorithm.

Parents:
- **Parent A**: Voronoi partition + Poikilotherm Schoolfield temperature model.
- **Parent B**: Multi‑armed bandit router + Variational Free Energy (VFE) inference.

Mathematical Bridge:
Each Voronoi region is interpreted as a bandit arm. The normalized
developmental rate (Schoolfield model) supplies a temperature‑driven
activity score `a_i ∈ [0,1]` that acts as the *expected reward* for arm *i*.
A variational posterior `q_i = N(μ_i, σ_i²)` is maintained for each arm.
The VFE for arm *i* is  

    F_i = ½·( (σ_i²/σ_p²) + ((μ_i-μ_p)²/σ_p²) - 1 + log(σ_p²/σ_i²) )
          - a_i·μ_i

where the first term is the KL divergence `KL(q_i‖p)` between posterior
and a fixed prior `p = N(μ_p,σ_p²)`, and the second term is the expected
negative log‑likelihood under the observation model `p(a_i|μ_i)=exp(a_i·μ_i)`.
The arm with the smallest `F_i` (i.e. highest evidence) is selected,
balancing temperature‑driven reward with epistemic uncertainty.

The code below implements:
1. Voronoi assignment of points to seeds.
2. Temperature‑driven activity per region (Schoolfield).
3. Bandit‑style selection using VFE‑augmented scores.
4. Policy update driven by observed rewards.

Only the allowed standard‑library modules and NumPy are used.
"""

from __future__ import annotations
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Voronoi & Schoolfield core
# ----------------------------------------------------------------------


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError("seed list empty")
    return min(range(len(seeds)), key=lambda i: _euclidean(point, seeds[i]))


def assign_regions(points: List[Tuple[float, float]],
                   seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """Assign each point to its nearest seed, returning a region dict."""
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[_nearest(p, seeds)].append(p)
    return regions


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15   # Kelvin
    t_high: float = 307.15  # Kelvin
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


def _c_to_k(celsius: float) -> float:
    return celsius + 273.15


def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield developmental rate (Eq. from parent A)."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("invalid temperature or rho_25")
    num = params.rho_25 * (temp_k / 298.15) * np.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = np.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = np.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    return num / (1.0 + low + high)


def normalized_activity(temp_c: float,
                        low_c: float = 5.0,
                        high_c: float = 40.0,
                        samples: int = 141) -> float:
    """Map temperature to activity ∈[0,1] using the Schoolfield curve."""
    params = SchoolfieldParams(t_low=_c_to_k(low_c), t_high=_c_to_k(high_c))
    rate = developmental_rate(_c_to_k(temp_c), params)
    # compute max rate over the sampled interval
    temps = np.linspace(low_c, high_c, samples)
    max_rate = max(
        developmental_rate(_c_to_k(t), params) for t in temps
    )
    if max_rate <= 0:
        return 0.0
    return max(0.0, min(1.0, rate / max_rate))


def region_activity(regions: Dict[int, List[Tuple[float, float]]],
                    temp_map: Dict[int, float]) -> Dict[int, float]:
    """
    Compute a temperature‑driven activity for each region.
    `temp_map` supplies an ambient temperature (°C) per region index.
    """
    activity: Dict[int, float] = {}
    for idx, pts in regions.items():
        # Use the provided temperature; fallback to 20 °C if missing.
        temp_c = temp_map.get(idx, 20.0)
        activity[idx] = normalized_activity(temp_c)
    return activity


# ----------------------------------------------------------------------
# Parent B – Bandit & Variational Free Energy core
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    region_id: int
    propensity: float          # derived from VFE (lower is better)
    expected_reward: float     # temperature activity
    confidence_bound: float    # UCB style term
    algorithm: str = "HybridVFE"


@dataclass(frozen=True)
class BanditUpdate:
    region_id: int
    reward: float
    propensity: float


# Global mutable policy state (posterior parameters per region)
_POLICY_MU: Dict[int, float] = {}
_POLICY_SIGMA2: Dict[int, float] = {}
_POLICY_COUNTS: Dict[int, int] = {}
_PRIOR_MU: float = 0.0
_PRIOR_SIGMA2: float = 1.0


def reset_policy() -> None:
    _POLICY_MU.clear()
    _POLICY_SIGMA2.clear()
    _POLICY_COUNTS.clear()


def _kl_gaussian(mu_q: float, sigma2_q: float,
                 mu_p: float = _PRIOR_MU,
                 sigma2_p: float = _PRIOR_SIGMA2) -> float:
    """Closed‑form KL divergence KL(N_q‖N_p)."""
    return 0.5 * (
        (sigma2_q / sigma2_p)
        + ((mu_q - mu_p) ** 2) / sigma2_p
        - 1.0
        + math.log(sigma2_p / sigma2_q)
    )


def _vfe(mu_q: float, sigma2_q: float, activity: float) -> float:
    """
    Variational Free Energy for a single arm.
    The likelihood term is taken as -activity·mu_q (higher activity lowers F).
    """
    return _kl_gaussian(mu_q, sigma2_q) - activity * mu_q


def hybrid_select_action(activity: Dict[int, float],
                         total_steps: int) -> BanditAction:
    """
    Choose a region (arm) by minimizing VFE penalized UCB score.

    Score_i = F_i + λ·UCB_i
    where λ balances inference vs exploration (set to 0.5).
    """
    lambda_expl = 0.5
    best_score = math.inf
    best_action: BanditAction | None = None

    for region_id, act in activity.items():
        # Retrieve posterior parameters, initialise if absent
        mu = _POLICY_MU.get(region_id, _PRIOR_MU)
        sigma2 = _POLICY_SIGMA2.get(region_id, _PRIOR_SIGMA2)
        n = _POLICY_COUNTS.get(region_id, 0)

        # VFE component
        f = _vfe(mu, sigma2, act)

        # UCB component (using empirical mean = mu as proxy for reward)
        if n == 0:
            ucb = math.sqrt(2 * math.log(max(1, total_steps + 1)))
        else:
            ucb = math.sqrt(2 * math.log(total_steps + 1) / n)

        score = f + lambda_expl * ucb

        if score < best_score:
            best_score = score
            best_action = BanditAction(
                region_id=region_id,
                propensity=f,
                expected_reward=act,
                confidence_bound=ucb,
            )
    if best_action is None:
        raise RuntimeError("No action could be selected")
    return best_action


def hybrid_update_policy(updates: List[BanditUpdate]) -> None:
    """
    Simple Bayesian update: treat reward as a noisy observation of μ.
    Posterior update for Gaussian with known variance = 1.0.
    """
    obs_variance = 1.0
    for u in updates:
        n = _POLICY_COUNTS.get(u.region_id, 0)
        mu_prior = _POLICY_MU.get(u.region_id, _PRIOR_MU)
        sigma2_prior = _POLICY_SIGMA2.get(u.region_id, _PRIOR_SIGMA2)

        # Posterior precision = prior precision + obs precision
        precision_post = 1.0 / sigma2_prior + 1.0 / obs_variance
        mu_post = (mu_prior / sigma2_prior + u.reward / obs_variance) / precision_post
        sigma2_post = 1.0 / precision_post

        _POLICY_MU[u.region_id] = mu_post
        _POLICY_SIGMA2[u.region_id] = sigma2_post
        _POLICY_COUNTS[u.region_id] = n + 1


# ----------------------------------------------------------------------
# Hybrid orchestration utilities
# ----------------------------------------------------------------------


def simulate_step(points: List[Tuple[float, float]],
                  seeds: List[Tuple[float, float]],
                  temp_map: Dict[int, float],
                  step_counter: int) -> Tuple[BanditAction, List[BanditUpdate]]:
    """
    One iteration:
    1. Partition points → regions.
    2. Compute temperature activity per region.
    3. Select region via VFE‑augmented bandit.
    4. Generate a synthetic reward (e.g., activity + Gaussian noise).
    5. Return selected action and update objects.
    """
    regions = assign_regions(points, seeds)
    activity = region_activity(regions, temp_map)
    action = hybrid_select_action(activity, step_counter)

    # Synthetic reward: true activity perturbed by Gaussian noise
    true_reward = activity[action.region_id]
    noisy_reward = float(np.random.normal(loc=true_reward, scale=0.1))

    update = BanditUpdate(
        region_id=action.region_id,
        reward=noisy_reward,
        propensity=action.propensity,
    )
    return action, [update]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # generate random points and seeds in a unit square
    num_points = 200
    num_seeds = 5
    points = [(random.random(), random.random()) for _ in range(num_points)]
    seeds = [(random.random(), random.random()) for _ in range(num_seeds)]

    # assign a temperature (°C) to each seed/region (varying around 20 °C)
    temp_map = {i: 15.0 + 10.0 * math.sin(i) for i in range(num_seeds)}

    reset_policy()
    total_steps = 0
    for epoch in range(10):
        action, updates = simulate_step(points, seeds, temp_map, total_steps)
        hybrid_update_policy(updates)
        total_steps += 1
        print(
            f"Step {epoch+1:02d}: selected region {action.region_id}, "
            f"activity {action.expected_reward:.3f}, propensity {action.propensity:.3f}"
        )
    print("Final posterior means per region:", _POLICY_MU)