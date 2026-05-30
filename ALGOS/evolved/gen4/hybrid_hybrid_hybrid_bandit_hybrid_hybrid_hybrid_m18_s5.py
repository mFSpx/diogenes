# DARWIN HAMMER — match 18, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s4.py (gen2)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_capybara_opti_m52_s2.py (gen3)
# born: 2026-05-29T23:26:28Z

import math
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Callable, Sequence

Vector = Sequence[float]

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
_STORE: Dict[str, float] = {}                 
_SURROGATE = None                             

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()
    global _SURROGATE
    _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=1.0)

def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
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
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def _encode_action(action_id: str, action_space: List[str]) -> List[float]:
    encoding = [0.0] * len(action_space)
    try:
        idx = action_space.index(action_id)
    except ValueError as exc:
        raise ValueError(f"unknown action {action_id}") from exc
    encoding[idx] = 1.0
    return encoding

def _build_feature_vector(context: Dict[str, float],
                          action_id: str,
                          action_space: List[str]) -> List[float]:
    ctx_vals = [context[k] for k in sorted(context.keys())]
    return ctx_vals + _encode_action(action_id, action_space)

def update_surrogate(context: Dict[str, float],
                     action_id: str,
                     reward: float,
                     action_space: List[str],
                     epsilon: float = 1.0) -> None:
    global _SURROGATE
    if _SURROGATE is None:
        _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=epsilon)

    new_center = tuple(_build_feature_vector(context, action_id, action_space))
    new_centers = _SURROGATE.centers + [new_center]

    size = len(new_centers)
    K = np.array([[gaussian(euclidean(new_centers[i], new_centers[j]), epsilon) for j in range(size)] for i in range(size)])
    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        raise ValueError("singular surrogate system")

    y = np.array((_SURROGATE.weights + [reward]) if _SURROGATE.weights else [reward])
    new_weights = np.dot(K_inv, y)

    _SURROGATE = RBFSurrogate(centers=new_centers,
                              weights=new_weights.tolist(),
                              epsilon=epsilon)

def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    else:
        surrogate_rewards = {}
        for a in actions:
            if _SURROGATE and _SURROGATE.centers:
                feat = _build_feature_vector(context, a, actions)
                surrogate_rewards[a] = _SURROGATE.predict(feat)
            else:
                surrogate_rewards[a] = 0.0

        if algorithm == "thompson":
            chosen = max(
                actions,
                key=lambda a: rng.betavariate(
                    1 + max(0.0, surrogate_rewards[a]),
                    1 + max(0.0, 1 - surrogate_rewards[a]),
                ),
            )
        else:  
            scale = math.sqrt(
                sum(float(v) * float(v) for v in context.values())
            ) if context else 1.0
            chosen = max(
                actions,
                key=lambda a: surrogate_rewards[a]
                + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
            )

    propensity = 1.0 / len(actions)
    confidence = 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1])

    _POLICY.setdefault(chosen, [0.0, 0.0])
    _POLICY[chosen][0] += surrogate_rewards[chosen]
    _POLICY[chosen][1] += 1

    return BanditAction(action_id=chosen,
                        propensity=propensity,
                        expected_reward=surrogate_rewards[chosen],
                        confidence_bound=confidence,
                        algorithm=algorithm)

def update_policy(update: BanditUpdate) -> None:
    _POLICY.setdefault(update.action_id, [0.0, 0.0])
    _POLICY[update.action_id][0] += update.reward * update.propensity
    _POLICY[update.action_id][1] += 1
    update_surrogate({k: v for k, v in update.context_id.items()},
                     update.action_id,
                     update.reward,
                     list(_POLICY.keys()))