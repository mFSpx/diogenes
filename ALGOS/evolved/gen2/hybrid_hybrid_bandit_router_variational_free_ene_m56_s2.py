# DARWIN HAMMER — match 56, survivor 2
# gen: 2
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s4.py (gen1)
# parent_b: variational_free_energy.py (gen0)
# born: 2026-05-29T23:26:33Z

"""
Hybrid Bandit-Routing Active Inference Model

This module combines the core topologies of the BanditRouter (hybrid_bandit_router_poikilotherm_schoolf_m20_s4.py) and Active Inference (variational_free_energy.py) algorithms.
The mathematical interface is established by integrating the Gaussian distributions from Active Inference with the probability updating rules of BanditRouter.

The BanditRouter's policy updates are now conditioned on the variational free energy, which is minimized by iteratively updating the agent's beliefs and actions.
The agent's actions are represented as bandit actions, where the reward is given by the normalized activity in the Schoolfield temperature model.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Parent A – bandit router core (lightly adapted)
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

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def normalized_activity(temp_c: float, low_c: float = 5) -> float:
    params = SchoolfieldParams()
    return developmental_rate(c_to_k(temp_c), params)

# ----------------------------------------------------------------------
# Parent B – Active Inference core (lightly adapted)
# ----------------------------------------------------------------------
R_CAL = 1.987  
K25 = 298.15  

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / K25) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / K25) - (1.0 / temp_k))
    )
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    return numerator / (1.0 + low + high)

def kl_gaussian(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray,
) -> float:
    """KL divergence KL[N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)].

    Closed form for univariate Gaussians (scalar or array; arrays are summed):

        KL = ln(sigma_p/sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2) / (2 sigma_p^2) - 1/2

    Parameters
    ----------
    mu_q, sigma_q:
        Mean and standard deviation of the variational distribution q.
    mu_p, sigma_p:
        Mean and standard deviation of the prior p.

    Returns
    -------
    float — sum of KL over all dimensions if array inputs.
    """
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    mu_p = np.asarray(mu_p, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)

    if np.any(sigma_q <= 0) or np.any(sigma_p <= 0):
        raise ValueError("Standard deviations must be strictly positive.")

    kl = (
        np.log(sigma_p / sigma_q)
        + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2)
        - 0.5
    )
    return np.sum(kl) if isinstance(kl, np.ndarray) else kl

def free_energy_gaussian(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray,
    observation: float,
) -> float:
    """Variational free energy F = KL[q || p] - ln p(o)."""
    return kl_gaussian(mu_q, sigma_q, mu_p, sigma_p) - np.log(np.exp(-((observation - mu_q) / sigma_q)**2 / 2) / (sigma_q * np.sqrt(2 * np.pi)))

def belief_update(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    observation: float,
) -> Tuple[float, float]:
    """Update the variational distribution q given an observation."""
    mu_q_new = mu_q + (observation - mu_q) / (1 + 1 / (sigma_q**2))
    sigma_q_new = 1 / (1 / sigma_q**2 + 1 / (sigma_q**2))
    return mu_q_new, sigma_q_new

def precision_weighted_update(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    observation: float,
    precision: float | np.ndarray,
) -> Tuple[float, float]:
    """Update the variational distribution q given an observation and precision."""
    mu_q_new = mu_q + (observation - mu_q) / (precision + 1 / (sigma_q**2))
    sigma_q_new = 1 / np.sqrt(1 / (precision * sigma_q**2) + 1 / (sigma_q**2))
    return mu_q_new, sigma_q_new

def hybrid_active_inference(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray,
    observation: float,
    precision: float | np.ndarray,
    action_id: str,
) -> Tuple[float, float, float]:
    """Hybrid active inference algorithm."""
    kl = kl_gaussian(mu_q, sigma_q, mu_p, sigma_p)
    f = free_energy_gaussian(mu_q, sigma_q, mu_p, sigma_p, observation)
    mu_q_new, sigma_q_new = precision_weighted_update(mu_q, sigma_q, observation, precision)
    reward = _reward(action_id)
    return kl, f, mu_q_new, sigma_q_new, reward

def hybrid_bandit_router(
    context_id: str,
    action_id: str,
    reward: float,
    propensity: float,
) -> None:
    """Hybrid bandit router algorithm."""
    update_policy([BanditUpdate(context_id, action_id, reward, propensity)])

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    mu_q = 0.0
    sigma_q = 1.0
    mu_p = 0.0
    sigma_p = 1.0
    observation = 1.0
    precision = 1.0
    action_id = "action_1"
    hybrid_active_inference(mu_q, sigma_q, mu_p, sigma_p, observation, precision, action_id)
    hybrid_bandit_router("context_1", action_id, 1.0, 0.5)