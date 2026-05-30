# DARWIN HAMMER — match 41, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s2.py (gen2)
# parent_b: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py (gen1)
# born: 2026-05-29T23:25:33Z

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
ETA0 = 0.01          # base learning rate for matrix updates
DELTA_MAX = 1.0      # max evasion magnitude
ALPHA_EVASION = 3.0  # decay rate for evasion schedule
HOEFFDING_DELTA = 0.05  # confidence level for Hoeffding bound
CLAMP_LO = -5.0
CLAMP_HI = 5.0

# ----------------------------------------------------------------------
# Bandit core (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # interpreted as inflow
    expected_reward: float
    confidence_bound: float    # interpreted as outflow
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, List[float]] = {}   # action_id -> [total_reward, count]
_STORE: Dict[str, float] = {}          # virtual VRAM store per key


def reset_policy() -> None:
    """Clear learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward_estimate(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def _confidence_bound(action_id: str) -> float:
    """Hoeffding‑type confidence bound for the given action."""
    _, n = _POLICY.get(action_id, [0.0, 0.0])
    if n == 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / HOEFFDING_DELTA) / (2.0 * n))


def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    if seed is not None:
        random.seed(seed)

    if not actions:
        raise ValueError("action list cannot be empty")

    scores: List[Tuple[str, float, float]] = []
    for a in actions:
        mu = _reward_estimate(a)
        conf = _confidence_bound(a)
        score = mu + epsilon * conf
        scores.append((a, score, conf))

    best_action, best_score, best_conf = max(scores, key=lambda x: x[1])

    exp_vals = [math.exp(s[1]) for s in scores]
    total_exp = sum(exp_vals)
    propensity = math.exp(best_score) / total_exp if total_exp > 0 else 0.0

    return BanditAction(
        action_id=best_action,
        propensity=propensity,
        expected_reward=_reward_estimate(best_action),
        confidence_bound=best_conf,
        algorithm=algorithm,
    )


def update_policy(update: BanditUpdate) -> None:
    total, n = _POLICY.get(update.action_id, [0.0, 0.0])
    _POLICY[update.action_id] = [total + update.reward, n + 1]


# ----------------------------------------------------------------------
# Capybara optimisation primitives (Parent B)
# ----------------------------------------------------------------------
Vector = Sequence[float]


def evasion_delta(t: int, t_max: int, delta_max: float = DELTA_MAX, alpha: float = ALPHA_EVASION) -> float:
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule parameters")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)


def clamp(x: Vector, lower: float, upper: float) -> list[float]:
    return [min(upper, max(lower, xi)) for xi in x]


def _shannon_entropy(data: Sequence[int]) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.array(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return -np.sum(probs * np.log2(probs))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def _update_store(action: BanditAction, key: str = "global") -> float:
    inflow = action.propensity
    outflow = action.confidence_bound
    delta_store = ALPHA * inflow - BETA * outflow
    prev = _STORE.get(key, 0.0)
    new_store = max(0.0, prev + delta_store * DT)
    _STORE[key] = new_store
    return new_store


def _gradient_from_positions(W: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if positions.size == 0:
        return np.zeros_like(W)

    g_best = positions.mean(axis=0)               
    preds = positions @ W.T                        
    error = preds - g_best                         
    grad = (error.T @ positions) / positions.shape[0]  
    return grad


def hybrid_step(
    W: np.ndarray,
    positions: np.ndarray,
    t: int,
    t_max: int,
    context: Dict[str, float],
    actions: List[str],
) -> Tuple[np.ndarray, np.ndarray, float]:
    ba = select_action(context, actions)

    store_value = _update_store(ba)

    eta = ETA0 * (1 + ba.propensity)
    grad = _gradient_from_positions(W, positions)
    W_new = W - eta * grad

    epsilon = math.sqrt(math.log(2.0 / HOEFFDING_DELTA) / (2.0 * t))
    delta_h = evasion_delta(t, t_max) * (1 + epsilon)
    directions = np.random.uniform(-1, 1, size=positions.shape)
    positions_new = clamp(positions + delta_h * directions, CLAMP_LO, CLAMP_HI)

    return W_new, positions_new, store_value