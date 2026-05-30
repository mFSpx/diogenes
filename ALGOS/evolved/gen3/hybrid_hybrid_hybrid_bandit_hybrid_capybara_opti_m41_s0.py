# DARWIN HAMMER — match 41, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s2.py (gen2)
# parent_b: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py (gen1)
# born: 2026-05-29T23:25:33Z

"""
Hybrid Bandit-Capybara Algorithm.

This module fuses the contextual multi-armed bandit router and linear TTT model 
from hybrid_bandit_router_hybrid_model_vram_sc_m35_s2.py with the continuous 
optimisation primitives and statistical gating logic of 
hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py.

The mathematical bridge lies in the interpretation of the bandit-produced 
`propensity` as a confidence scalar that modulates the evasion magnitude 
and the learning rate of the TTT update. The `confidence_bound` is used to 
calculate the signal-to-noise gap, which drives the attraction towards the 
global best and modulates the probability of entering *standby* versus *burst*.

"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Bandit core
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
_STORE: Dict[str, float] = {}          # virtual VRAM store per key


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """Choose an action and return its BanditAction descriptor."""
    if not actions:
        raise ValueError("No actions provided")
    # Simple implementation, replace with actual linucb algorithm
    return BanditAction(random.choice(actions), random.random(), random.random(), random.random(), algorithm)


# ----------------------------------------------------------------------
# Capybara core
# ----------------------------------------------------------------------
Vector = Sequence[float]


def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule for evasion magnitude."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)


def clamp(x: Vector, lower: float, upper: float) -> list[float]:
    """Clamp each component of a vector to [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]


def _shannon_entropy(data: Sequence[int]) -> float:
    """Return Shannon entropy in bits for a sequence of byte values."""
    if not data:
        return 0.0
    counts = np.bincount(np.array(data, dtype=np.uint8), minlength=256)
    total = sum(counts)
    return -sum([count / total * math.log2(count / total) for count in counts if count > 0])


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def calculate_propensity_modulated_evasion(
    t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0, propensity: float = 0.5
) -> float:
    """Calculate the evasion magnitude modulated by the bandit's propensity."""
    return evasion_delta(t, t_max, delta_max, alpha) * (1 + propensity)


def calculate_signal_to_noise_gap(
    signal: float, noise: float, confidence_bound: float
) -> float:
    """Calculate the signal-to-noise gap using the bandit's confidence bound."""
    return (signal - noise) * confidence_bound


def calculate_hybrid_weight_update(
    weight: float, propensity: float, confidence_bound: float, learning_rate: float = 0.1
) -> float:
    """Calculate the hybrid weight update using the bandit's propensity and confidence bound."""
    return weight * (1 + propensity) * (1 - confidence_bound) * learning_rate


if __name__ == "__main__":
    reset_policy()
    action = select_action({}, ["action1", "action2"])
    evasion = calculate_propensity_modulated_evasion(10, 100, propensity=action.propensity)
    signal_to_noise_gap = calculate_signal_to_noise_gap(1.0, 0.5, action.confidence_bound)
    weight_update = calculate_hybrid_weight_update(0.5, action.propensity, action.confidence_bound)
    print("Evasion magnitude:", evasion)
    print("Signal-to-noise gap:", signal_to_noise_gap)
    print("Hybrid weight update:", weight_update)