# DARWIN HAMMER — match 9, survivor 3
# gen: 1
# parent_a: bandit_router.py (gen0)
# parent_b: honeybee_store.py (gen0)
# born: 2026-05-29T23:16:48Z

from __future__ import annotations

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

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

def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

def dance_duration(
    delta_store: float,
    base: float = 1.0,
    gain: float = 1.0,
    limit: float = 10.0,
) -> float:
    return max(0.0, min(limit, base + gain * delta_store))

def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    store_factor = 1.0 + store / (store + 1.0)

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
            mean = _reward(a)
            cnt = _count(a)
            conf = store_factor / math.sqrt(1.0 + cnt)
            return mean + eta * scale * conf

        chosen = max(actions, key=ucb_score)

    prop = 1.0 / len(actions)
    exp_reward = _reward(chosen)
    conf_bound = store_factor / math.sqrt(1.0 + _count(chosen))

    return BanditAction(
        action_id=chosen,
        propensity=prop,
        expected_reward=exp_reward,
        confidence_bound=conf_bound,
        algorithm=algorithm,
    )

def hybrid_step(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    true_reward_fn,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
    seed: int | str | None = 7,
    cost: float = 0.0
) -> Tuple[BanditAction, float, float]:
    action = hybrid_select_action(context, actions, store, algorithm, epsilon, eta, seed)
    reward = true_reward_fn(action.action_id)
    update_policy([BanditUpdate("", action.action_id, reward, 1.0)])
    new_store, _ = update_store(store, [reward], [cost], alpha, beta, dt)
    return action, reward, new_store

def main():
    reset_policy()
    context = {"feature1": 1.0, "feature2": 2.0}
    actions = ["action1", "action2"]
    store = 1.0
    def true_reward_fn(action_id: str) -> float:
        return random.random()
    action, reward, new_store = hybrid_step(context, actions, store, true_reward_fn)
    print(action, reward, new_store)

if __name__ == "__main__":
    main()