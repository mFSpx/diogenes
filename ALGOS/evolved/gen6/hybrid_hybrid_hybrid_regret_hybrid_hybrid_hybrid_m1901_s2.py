# DARWIN HAMMER — match 1901, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_regret_engine_hybrid_hybrid_hdc_hy_m590_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_fold_c_state_space_duality_m407_s2.py (gen5)
# born: 2026-05-29T23:39:44Z

import math
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime as dt
from typing import List, Dict, Iterable, Tuple

@dataclass(frozen=True, slots=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True, slots=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

@dataclass(frozen=True, slots=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"

@dataclass(frozen=True, slots=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

def weekday_index(year: int, month: int, day: int) -> int:
    return dt(year, month, day).weekday()

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    n = len(xs)
    if n == 0 or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0.0:
        raise ValueError("values must be non-negative")
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))

_POLICY: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY[action]
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY[action][1]

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    return log_count_ratio * count

def _fold_change_detection(x: float, eps: float = 1e-12) -> float:
    return math.log(x / max(abs(x), eps)) if x != 0 else 0.0

def _ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    h_new = A @ h + B @ x
    y = C @ h_new
    return h_new, y

def weekday_modulated_matrix(
    base_A: np.ndarray,
    year: int,
    month: int,
    day: int,
    alpha: float = 0.1,
) -> np.ndarray:
    w = weekday_index(year, month, day)
    mu = 1.0 + alpha * math.sin(math.pi * w / 7.0)
    return base_A * mu

def gini_weighted_confidence(actions: List[BanditAction], beta: float = 0.2) -> List[BanditAction]:
    expected = [a.expected_reward for a in actions]
    G = gini_coefficient(expected)
    factor = 1.0 + beta * G
    adjusted = [
        BanditAction(
            action_id=a.action_id,
            propensity=a.propensity,
            expected_reward=a.expected_reward,
            confidence_bound=a.confidence_bound * factor,
            algorithm=a.algorithm,
        )
        for a in actions
    ]
    return adjusted

def hybrid_ssm_bandit_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    actions: List[BanditAction],
    log_count_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h_new, y = _ssm_step(h, x, A, B, C)
    actions_adj = gini_weighted_confidence(actions)
    bandit_term = 0.0
    for a in actions_adj:
        count = _count(a.action_id)
        S_i = _hybrid_store_factor(a.action_id, count, log_count_ratio)
        bandit_term += S_i * a.confidence_bound
    y_adj = y + bandit_term
    return h_new, y_adj

def hybrid_regret_ssm_router(
    base_A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    actions: List[BanditAction],
    inputs: List[np.ndarray],
    log_count_ratios: List[float],
    year: int,
    month: int,
    day: int,
    alpha: float = 0.1,
    beta: float = 0.2,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    A = weekday_modulated_matrix(base_A, year, month, day, alpha)
    outputs = []
    h = np.zeros((B.shape[0],))
    for x, log_count_ratio in zip(inputs, log_count_ratios):
        h_new, y_adj = hybrid_ssm_bandit_step(h, x, A, B, C, actions, log_count_ratio)
        outputs.append((h_new, y_adj))
        h = h_new
    return outputs