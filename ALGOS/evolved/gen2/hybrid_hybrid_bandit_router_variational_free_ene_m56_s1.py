# DARWIN HAMMER — match 56, survivor 1
# gen: 2
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s4.py (gen1)
# parent_b: variational_free_energy.py (gen0)
# born: 2026-05-29T23:26:33Z

"""
Hybrid Algorithm: Fusing Bandit Router and Variational Free Energy (Active Inference)

This module mathematically fuses the core topologies of two parent algorithms:
1. hybrid_bandit_router_poikilotherm_schoolf_m20_s4.py (Bandit Router + Schoolfield temperature model)
2. variational_free_energy.py (Variational Free Energy for Active Inference)

The bridge between the two parents lies in the concept of "utility" or "reward" in the Bandit Router,
which can be interpreted as the negative of the surprise or variational free energy (F) in the Variational Free Energy framework.
By casting the Bandit Router's expected rewards as a probabilistic distribution over outcomes,
we can integrate it with the Variational Free Energy framework to create a hybrid algorithm.

The hybrid algorithm uses the Schoolfield temperature model to modulate the precision of the variational distribution
in the Variational Free Energy framework, and the Bandit Router's update policy to adapt the expected rewards
based on the outcomes.

This implementation provides a novel approach to integrating decision-making under uncertainty (Bandit Router)
with active inference (Variational Free Energy) in a thermodynamically-inspired framework.
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

# ----------------------------------------------------------------------
# Parent B – Variational Free Energy (lightly adapted)
# ----------------------------------------------------------------------
def kl_gaussian(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray,
) -> float:
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    mu_p = np.asarray(mu_p, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)

    if np.any(sigma_q <= 0) or np.any(sigma_p <= 0):
        raise ValueError("Standard deviations must be strictly positive.")

    kl = (
        np.log(sigma_p/sigma_q) + 
        (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5
    )
    return np.sum(kl)

def free_energy_gaussian(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray,
) -> float:
    return kl_gaussian(mu_q, sigma_q, mu_p, sigma_p)

def belief_update(
    mu_q: float,
    sigma_q: float,
    mu_p: float,
    sigma_p: float,
    precision: float,
) -> Tuple[float, float]:
    sigma_q_updated = 1 / (1 / sigma_q**2 + precision / sigma_p**2)
    mu_q_updated = sigma_q_updated * (mu_q / sigma_q**2 + precision * mu_p / sigma_p**2)
    return mu_q_updated, sigma_q_updated

# ----------------------------------------------------------------------
# Schoolfield temperature model (lightly adapted)
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
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
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

# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------
def hybrid_update(
    bandit_update: BanditUpdate,
    schoolfield_params: SchoolfieldParams,
    variational_mu: float,
    variational_sigma: float,
) -> Tuple[float, float]:
    temperature = c_to_k(20)  # assuming 20°C
    developmental_rate_value = developmental_rate(temperature, schoolfield_params)
    precision = developmental_rate_value

    variational_mu_updated, variational_sigma_updated = belief_update(
        variational_mu,
        variational_sigma,
        bandit_update.reward,
        1.0,
        precision,
    )

    update_policy([bandit_update])
    return variational_mu_updated, variational_sigma_updated

def hybrid_step(
    action: BanditAction,
    schoolfield_params: SchoolfieldParams,
    variational_mu: float,
    variational_sigma: float,
) -> Tuple[float, float]:
    bandit_update = BanditUpdate(
        context_id="",
        action_id=action.action_id,
        reward=action.expected_reward,
        propensity=action.propensity,
    )
    return hybrid_update(bandit_update, schoolfield_params, variational_mu, variational_sigma)

# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    schoolfield_params = SchoolfieldParams()
    variational_mu = 0.0
    variational_sigma = 1.0

    bandit_action = BanditAction(
        action_id="test_action",
        propensity=1.0,
        expected_reward=10.0,
        confidence_bound=0.1,
        algorithm="test_algorithm",
    )

    bandit_update = BanditUpdate(
        context_id="test_context",
        action_id="test_action",
        reward=10.0,
        propensity=1.0,
    )

    updated_mu, updated_sigma = hybrid_step(bandit_action, schoolfield_params, variational_mu, variational_sigma)
    print(f"Updated Variational Mu: {updated_mu}, Updated Variational Sigma: {updated_sigma}")

    update_policy([bandit_update])
    print(_reward("test_action"))