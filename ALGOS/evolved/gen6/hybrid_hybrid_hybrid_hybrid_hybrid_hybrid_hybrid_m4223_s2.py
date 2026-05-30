# DARWIN HAMMER — match 4223, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_workshare_all_m1367_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m1353_s2.py (gen4)
# born: 2026-05-29T23:54:16Z

"""Hybrid algorithm combining temperature-dependent developmental rates (Parent A) with
document morphology sphericity (Parent B). 

Mathematical bridge:
    The expected reward for a bandit action is scaled by a *thermal factor* ρ(T) 
    (Schoolfield developmental rate) and a *shape factor* σ(L,W,H) (sphericity 
    index). Both factors are multiplicative, yielding a unified reward model:

        R̂ = ρ(T) · σ(L,W,H) · w_dow

    where w_dow is a weekday weight derived from the original weekday_weight_vector 
    (Parent A). This unified reward feeds a Thompson‑sampling update that respects 
    both thermodynamic and morphological contexts.
"""

import math
import random
import sys
from pathlib import Path
from datetime import date, datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (merged from both parents)
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15   # reference temperature in K


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float = 1.0  # default non‑zero mass


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Core physics / morphology functions (parents)
# ----------------------------------------------------------------------
def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature‑dependence (Parent A)."""
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
    return numerator * (low + high)


def sphericity_index(length: float, width: float, height: float) -> float:
    """Sphericity of a 3‑D object (Parent B)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
    """Cyclic weekday weighting (Parent A)."""
    N = len(groups)
    theta = np.linspace(0, 2 * math.pi, N, endpoint=False)
    phi = 2 * math.pi * dow / 7
    alpha = 0.2
    weights = 1 + alpha * np.sin(theta + phi)
    return weights / np.sum(weights)


# ----------------------------------------------------------------------
# Hybrid policy handling
# ----------------------------------------------------------------------
_POLICY: Dict[str, Tuple[float, float]] = {}  # action_id -> (sum_rewards, count)


def reset_policy() -> None:
    """Clear the global policy."""
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate raw bandit updates into the global policy."""
    for u in updates:
        total, cnt = _POLICY.get(u.action_id, (0.0, 0.0))
        total += float(u.reward)
        cnt += 1.0
        _POLICY[u.action_id] = (total, cnt)


def thompson_sample(action_id: str) -> float:
    """Draw a Thompson‑sampling sample for an action using a Beta posterior."""
    total, cnt = _POLICY.get(action_id, (0.0, 0.0))
    # Beta parameters: successes = reward_sum + 1, failures = (cnt - reward_sum) + 1
    # Clamp reward_sum to [0, cnt] to keep parameters valid.
    reward_sum = max(0.0, min(total, cnt))
    alpha = reward_sum + 1.0
    beta = (cnt - reward_sum) + 1.0
    return random.betavariate(alpha, beta)


# ----------------------------------------------------------------------
# Hybrid functions demonstrating the fused mathematics
# ----------------------------------------------------------------------
def combined_expected_reward(temp_c: float,
                             morph: Morphology,
                             dow: int,
                             groups: List[str]) -> float:
    """
    Compute the unified expected reward:
        R̂ = ρ(T) * σ(L,W,H) * w_dow

    - ρ(T) : developmental_rate at temperature T (K)
    - σ    : sphericity_index from morphology
    - w_dow: weight of the current weekday for the given group set
    """
    temp_k = c_to_k(temp_c)
    thermal = developmental_rate(temp_k)
    shape = sphericity_index(morph.length, morph.width, morph.height)
    # Use the first group weight as a proxy for the current context
    weights = weekday_weight_vector(groups, dow)
    weekday_weight = weights[0] if weights.size > 0 else 1.0
    return thermal * shape * weekday_weight


def hybrid_bandit_router(updates: List[BanditUpdate],
                         temp_c: float,
                         morph: Morphology,
                         year: int,
                         month: int,
                         day: int,
                         groups: List[str]) -> List[BanditAction]:
    """
    Core routing routine (hybrid of Parent A's router and Parent B's Thompson sampling).

    1. Update the global policy with raw updates.
    2. For each distinct action, compute a Thompson sample.
    3. Scale the sample by the combined expected reward (thermal × shape × weekday).
    4. Return a list of BanditAction objects ready for downstream consumption.
    """
    update_policy(updates)

    dow = (date(year, month, day).weekday() + 1) % 7  # 0=Mon … 6=Sun → 0‑6 cyclic
    base_reward = combined_expected_reward(temp_c, morph, dow, groups)

    actions = []
    for action_id in _POLICY.keys():
        sample = thompson_sample(action_id)
        expected = base_reward * sample
        confidence = math.sqrt(expected)  # simplistic confidence proxy
        actions.append(
            BanditAction(
                action_id=action_id,
                propensity=sample,
                expected_reward=expected,
                confidence_bound=confidence,
                algorithm="hybrid"
            )
        )
    return actions


def simulate_hybrid_cycle(num_actions: int = 5) -> None:
    """
    Demonstration helper that:
    * creates dummy updates,
    * runs the hybrid router,
    * prints the resulting actions.
    """
    # Dummy morphology (arbitrary but positive)
    morph = Morphology(length=2.3, width=1.7, height=0.9, mass=1.2)

    # Generate synthetic updates for a set of actions
    dummy_updates = [
        BanditUpdate(
            context_id="ctx",
            action_id=f"act_{i}",
            reward=random.random(),
            propensity=0.5
        )
        for i in range(num_actions)
    ]

    # Run the router for a fixed date
    actions = hybrid_bandit_router(
        updates=dummy_updates,
        temp_c=22.0,
        morph=morph,
        year=2026,
        month=5,
        day=29,
        groups=["A", "B", "C"]
    )

    for a in actions:
        print(f"Action {a.action_id}: expected={a.expected_reward:.4f}, "
              f"propensity={a.propensity:.4f}, confidence={a.confidence_bound:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    reset_policy()
    simulate_hybrid_cycle()
    sys.exit(0)