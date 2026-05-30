# DARWIN HAMMER — match 56, survivor 0
# gen: 2
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s4.py (gen1)
# parent_b: variational_free_energy.py (gen0)
# born: 2026-05-29T23:26:33Z

"""
Hybrid Algorithm: Fusing Bandit Router and Variational Free Energy

This module integrates the core topologies of the Bandit Router (hybrid_bandit_router_poikilotherm_schoolf_m20_s4.py) 
and Variational Free Energy (variational_free_energy.py) algorithms. The mathematical bridge between the two 
structures lies in the use of probabilistic updates and expectations. Specifically, we utilize the Variational Free 
Energy (VFE) framework to inform the bandit policy updates, effectively creating a hybrid algorithm that balances 
exploration-exploitation trade-offs with Bayesian inference.

The Bandit Router's multi-armed bandit problem is fused with the Variational Free Energy's active inference framework, 
enabling the algorithm to adaptively sample actions based on both their expected rewards and the uncertainty 
associated with those expectations.

Imports: numpy, standard library, math, random, sys, pathlib
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
# Parent B – Variational Free Energy core
# ---------------------------------------------------------------------------
def kl_gaussian(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray,
) -> float:
    """KL divergence KL[N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)]."""
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    mu_p = np.asarray(mu_p, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)

    if np.any(sigma_q <= 0) or np.any(sigma_p <= 0):
        raise ValueError("Standard deviations must be strictly positive.")

    kl = (
        np.log(sigma_p/sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 1/2
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
    new_mu_q = (sigma_q**-2 * mu_q + precision * mu_p) / (sigma_q**-2 + precision)
    new_sigma_q = 1 / np.sqrt(sigma_q**-2 + precision)
    return new_mu_q, new_sigma_q

# ----------------------------------------------------------------------
# Hybrid Algorithm
# ---------------------------------------------------------------------------
@dataclass
class HybridParams:
    learning_rate: float = 0.1
    precision: float = 1.0

def hybrid_bandit_vfe(
    bandit_updates: List[BanditUpdate],
    params: HybridParams = HybridParams(),
) -> None:
    for update in bandit_updates:
        action_id = update.action_id
        reward = update.reward
        propensity = update.propensity

        # Variational Free Energy update
        mu_q = _reward(action_id)
        sigma_q = 1.0
        mu_p = reward
        sigma_p = propensity
        kl_divergence = free_energy_gaussian(mu_q, sigma_q, mu_p, sigma_p)

        # Update bandit policy using VFE-informed updates
        stats = _POLICY.setdefault(action_id, [0.0, 0.0])
        stats[0] += float(reward) * params.learning_rate * kl_divergence
        stats[1] += 1.0

def demonstrate_hybrid_operation() -> None:
    # Initialize bandit policy
    reset_policy()

    # Sample bandit updates
    updates = [
        BanditUpdate("context1", "action1", 10.0, 0.5),
        BanditUpdate("context1", "action2", 20.0, 0.3),
        BanditUpdate("context2", "action1", 15.0, 0.4),
    ]

    # Run hybrid algorithm
    hybrid_bandit_vfe(updates)

    # Print updated bandit policy
    for action_id, stats in _POLICY.items():
        print(f"Action {action_id}: reward = {stats[0]}, count = {stats[1]}")

if __name__ == "__main__":
    demonstrate_hybrid_operation()