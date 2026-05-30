# DARWIN HAMMER — match 468, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hoeffding_tre_m301_s2.py (gen4)
# parent_b: hybrid_hybrid_bandit_router_hybrid_hybrid_endpoi_m134_s1.py (gen3)
# born: 2026-05-29T23:29:05Z

"""Hybrid Regret‑Weighted Hoeffding Tree + Bandit Developmental Rate Fusion

Parent A: hybrid_hybrid_hybrid_regret_hoeffding_tre_m301_s2.py  
Parent B: hybrid_hybrid_bandit_router_hybrid_hybrid_endpoi_m134_s1.py  

Mathematical Bridge
-------------------
The Gini coefficient (A) quantifies inequality among the expected values of
candidate actions.  In the bandit formulation (B) each action carries a
propensity‑adjusted confidence bound.  We fuse the two by letting the Gini
coefficient modulate the *regret‑weighted* confidence term:

    ε = base_ε * (1 + λ_g * G)

where G is the Gini coefficient of the action‑value distribution and λ_g is a
tunable scaling factor.  The developmental rate (temperature‑dependent) from
Parent B provides a contextual scaling factor ρ(T) that further adapts the
split decision of the Hoeffding tree.  The final split‑gain estimate becomes

    gain_gap = ρ(T) * (max_gain - ε)

Thus the split decision, bandit policy update and hybrid scoring all share a
common mathematical core built from Gini‑weighted regret and temperature‑driven
developmental rates.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action used in the regret‑weighted Hoeffding tree."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class BanditAction:
    """Bandit arm with propensity‑adjusted confidence bound."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


# ----------------------------------------------------------------------
# Parent‑A utilities (Gini, signature, etc.)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def compute_gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative vector."""
    arr = np.array([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Gini coefficient requires non‑negative values")
    if np.allclose(arr, 0):
        return 0.0
    sorted_arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(sorted_arr)
    sum_y = cumulative[-1]
    # Gini based on the Lorenz curve area
    gini = 1.0 - (2.0 / (n - 1)) * (np.sum(cumulative) / sum_y - (n + 1) / 2.0)
    return max(0.0, min(1.0, gini))


# ----------------------------------------------------------------------
# Parent‑B utilities (developmental rate, policy handling)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield‐type temperature dependent rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


# Simple in‑memory policy store (mirrors Parent B)
_POLICY: dict[str, Tuple[float, int]] = {}


def reset_policy() -> None:
    _POLICY.clear()


def update_policy(updates: List[Tuple[str, float]]) -> None:
    """updates: list of (action_id, reward)."""
    for action_id, reward in updates:
        total, cnt = _POLICY.get(action_id, (0.0, 0))
        _POLICY[action_id] = (total + float(reward), cnt + 1)


def average_reward(action_id: str) -> float:
    total, cnt = _POLICY.get(action_id, (0.0, 0))
    return total / cnt if cnt else 0.0


# ----------------------------------------------------------------------
# Hybrid core functions (the required three+ functions)
# ----------------------------------------------------------------------
def hybrid_regret_weighted_split(
    actions: List[MathAction],
    temperature_c: float,
    base_epsilon: float = 0.05,
    lambda_g: float = 0.5,
) -> Tuple[bool, float, float]:
    """
    Decide whether to split a Hoeffding node using a regret‑weighted term
    modulated by the Gini coefficient of the action values and the
    temperature‑dependent developmental rate.

    Returns:
        (should_split, epsilon, gain_gap)
    """
    if not actions:
        return False, base_epsilon, 0.0

    # 1. Gini of expected values (Parent A)
    gini = compute_gini_coefficient([a.expected_value for a in actions])

    # 2. Temperature scaling via developmental rate (Parent B)
    temp_k = c_to_k(temperature_c)
    rho = developmental_rate(temp_k)

    # 3. Regret‑weighted epsilon
    epsilon = base_epsilon * (1.0 + lambda_g * gini) * rho

    # 4. Simple gain estimate: max expected – min expected
    values = np.array([a.expected_value for a in actions])
    max_gain = float(values.max() - values.min())
    gain_gap = max_gain - epsilon

    should_split = gain_gap > 0.0
    return should_split, epsilon, gain_gap


def hybrid_bandit_confidence(
    bandit_actions: List[BanditAction],
    temperature_c: float,
    lambda_g: float = 0.3,
) -> List[BanditAction]:
    """
    Adjust each bandit arm's confidence bound by injecting the Gini‑derived
    regret term and the developmental rate.  The returned list contains new
    BanditAction instances with updated confidence bounds.
    """
    if not bandit_actions:
        return []

    # Gini of expected rewards across arms
    gini = compute_gini_coefficient([a.expected_reward for a in bandit_actions])

    # Temperature scaling factor
    rho = developmental_rate(c_to_k(temperature_c))

    adjusted = []
    for a in bandit_actions:
        # Original bound + regret term
        regret_term = lambda_g * gini * rho
        new_bound = a.confidence_bound + regret_term
        adjusted.append(
            BanditAction(
                action_id=a.action_id,
                propensity=a.propensity,
                expected_reward=a.expected_reward,
                confidence_bound=new_bound,
                algorithm=a.algorithm,
            )
        )
    return adjusted


def hybrid_policy_score(action_id: str, temperature_c: float, lambda_g: float = 0.4) -> float:
    """
    Compute a unified score for a given action that blends:
        • empirical average reward (Parent B)
        • Gini‑weighted regret penalty
        • developmental rate (temperature context)
    The score can be used for ranking or as a selection metric.
    """
    # Base reward from the policy store
    base = average_reward(action_id)

    # Gini of the whole policy distribution (captures inequality)
    all_rewards = [average_reward(aid) for aid in _POLICY.keys()]
    gini = compute_gini_coefficient(all_rewards) if all_rewards else 0.0

    # Temperature scaling
    rho = developmental_rate(c_to_k(temperature_c))

    # Final hybrid score
    score = base * rho * (1.0 + lambda_g * gini)
    return max(0.0, score)


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset and populate a tiny policy
    reset_policy()
    update_policy([("a1", 1.2), ("a2", 0.5), ("a3", 0.8)])

    # Define some MathActions for the split decision
    math_actions = [
        MathAction(id="m1", expected_value=0.3),
        MathAction(id="m2", expected_value=0.7),
        MathAction(id="m3", expected_value=0.4),
    ]

    # Perform hybrid split decision at 25 °C
    split_ok, eps, gain = hybrid_regret_weighted_split(math_actions, temperature_c=25.0)
    print(f"Split decision → ok:{split_ok}, epsilon:{eps:.4f}, gain_gap:{gain:.4f}")

    # Define bandit arms
    bandit_arms = [
        BanditAction(action_id="a1", propensity=0.4, expected_reward=average_reward("a1"), confidence_bound=0.1),
        BanditAction(action_id="a2", propensity=0.35, expected_reward=average_reward("a2"), confidence_bound=0.15),
        BanditAction(action_id="a3", propensity=0.25, expected_reward=average_reward("a3"), confidence_bound=0.2),
    ]

    # Adjust confidence bounds using hybrid logic at 30 °C
    adjusted_arms = hybrid_bandit_confidence(bandit_arms, temperature_c=30.0)
    for arm in adjusted_arms:
        print(f"Adjusted arm {arm.action_id}: confidence={arm.confidence_bound:.4f}")

    # Compute hybrid policy scores for each arm
    for aid in _POLICY.keys():
        sc = hybrid_policy_score(aid, temperature_c=22.0)
        print(f"Hybrid score for {aid}: {sc:.4f}")

    print("Smoke test completed without errors.")