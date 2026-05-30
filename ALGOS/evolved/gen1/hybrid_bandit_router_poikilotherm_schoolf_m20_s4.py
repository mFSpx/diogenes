# DARWIN HAMMER — match 20, survivor 4
# gen: 1
# parent_a: bandit_router.py (gen0)
# parent_b: poikilotherm_schoolfield.py (gen0)
# born: 2026-05-29T23:23:03Z

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
# Parent B – Schoolfield temperature model (lightly adapted)
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

def normalized_activity(
    temp_c: float,
    low_c: float = 5.0,
    high_c: float = 40.0,
    samples: int = 141,
) -> float:
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)

    temps = np.linspace(low_c, high_c, max(2, samples))
    rates = np.vectorize(lambda t: developmental_rate(c_to_k(t), params))(temps)
    max_rate = np.max(rates)

    if max_rate <= 0:
        return 0.0
    return max(0.0, min(1.0, rate / max_rate))

# ----------------------------------------------------------------------
# Hybrid layer – temperature‑aware bandit selection
# ----------------------------------------------------------------------
def temperature_activity(temp_c: float) -> float:
    try:
        return normalized_activity(temp_c)
    except Exception:
        return 0.0

def _temperature_scaled_scale(context: Dict[str, float], temp_c: float) -> float:
    if not context:
        return 1.0
    vec = np.fromiter(context.values(), dtype=float)
    norm = float(np.linalg.norm(vec))
    activity = temperature_activity(temp_c)
    return activity * norm if activity > 0 else 1e-6  

def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    temperature_c: float,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    if not actions:
        raise ValueError("actions required")

    rng = random.Random(seed)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
        return BanditAction(
            action_id=chosen,
            propensity=1.0 / len(actions),
            expected_reward=_reward(chosen),
            confidence_bound=0.0,
            algorithm=algorithm,
        )

    if algorithm == "thompson":
        def thompson_score(a: str) -> float:
            mean = _reward(a)
            successes = max(0.0, mean) * temperature_activity(temperature_c) * 10.0
            failures = max(0.0, 1 - mean) * temperature_activity(temperature_c) * 10.0
            return np.random.beta(successes + 1, failures + 1)

        scores = {a: thompson_score(a) for a in actions}
        chosen = max(scores, key=scores.get)
        return BanditAction(
            action_id=chosen,
            propensity=1.0 / len(actions),
            expected_reward=scores[chosen],
            confidence_bound=0.0,
            algorithm=algorithm,
        )

    scale = _temperature_scaled_scale(context, temperature_c)
    ucbs = {}
    for a in actions:
        mean = _reward(a)
        count = _POLICY.get(a, [0.0, 0.0])[1]
        if count == 0:
            ucbs[a] = float('inf')
        else:
            ucbs[a] = mean + scale / math.sqrt(1 + count)

    chosen = max(ucbs, key=ucbs.get)
    return BanditAction(
        action_id=chosen,
        propensity=1.0 / len(actions),
        expected_reward=_reward(chosen),
        confidence_bound=ucbs[chosen] - _reward(chosen),
        algorithm=algorithm,
    )

def hybrid_update_policy(updates: List[BanditUpdate], temperature_c: float) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward) * temperature_activity(temperature_c)
        stats[1] += 1.0