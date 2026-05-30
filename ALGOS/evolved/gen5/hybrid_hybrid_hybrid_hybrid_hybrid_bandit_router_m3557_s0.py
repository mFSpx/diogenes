# DARWIN HAMMER — match 3557, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m7_s2.py (gen4)
# parent_b: hybrid_bandit_router_honeybee_store_m9_s3.py (gen1)
# born: 2026-05-29T23:50:47Z

import math
import random
import sys
import pathlib
import numpy as np
from typing import List, Tuple, Sequence, Dict

class BanditAction:
    def __init__(self, action_id: str, propensity: float, expected_reward: float, confidence_bound: float, algorithm: str):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

class BanditUpdate:
    def __init__(self, context_id: str, action_id: str, reward: float, propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    global _POLICY
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

class Honeybee:
    def __init__(self, store: float, alpha: float = 1.0, beta: float = 1.0, dt: float = 1.0):
        self.store = store
        self.alpha = alpha
        self.beta = beta
        self.dt = dt

    def update_store(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        new_store = max(0.0, self.store + self.dt * delta)
        return new_store, delta

    def dance_duration(self, delta_store: float, base: float = 1.0, gain: float = 1.0, limit: float = 10.0) -> float:
        return max(0.0, min(limit, base + gain * delta_store))

def hybrid_honeybee_bandit(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    store_factor = 1.0 + store / (store + 1.0)

    store_honeybee = Honeybee(store=store, alpha=alpha, beta=beta, dt=dt)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        def sample(a: str) -> float:
            r = _reward(a)
            n = _count(a)
            a_param = 1.0 + max(0.0, r) * store_factor
            b_param = 1.0 + max(0.0, 1.0 - r) * store_factor
            return rng.betavariate(a_param, b_param)

        chosen = max(actions, key=sample)
    else:
        scale = np.linalg.norm(list(context.values()))
        def ucb_score(a: str) -> float:
            # UCB calculation
            propensity = (1 / _count(a)) * (1 + _reward(a) * store_factor)
            return propensity

        chosen = max(actions, key=ucb_score)

        # Update honeybee store and dance duration
        new_store, delta = store_honeybee.update_store(inflow=[1.0], outflow=[_count(chosen)])
        dance_dur = store_honeybee.dance_duration(delta_store=delta)

        return BanditAction(action_id=chosen, propensity=ucb_score(chosen), expected_reward=_reward(chosen), confidence_bound=dance_dur, algorithm=algorithm)

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [x / div for x in m[col]]

    return [m[col][-1] for col in range(n)]

if __name__ == "__main__":
    # Smoke test
    context = {"a": 1.0, "b": 2.0}
    actions = ["action1", "action2"]
    store = 10.0
    hybrid_honeybee_bandit(context, actions, store)