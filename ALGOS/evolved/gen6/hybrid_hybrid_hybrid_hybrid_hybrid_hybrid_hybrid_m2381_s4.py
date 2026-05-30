# DARWIN HAMMER — match 2381, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_fold_c_hybrid_hybrid_hybrid_m653_s0.py (gen5)
# born: 2026-05-29T23:42:12Z

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Clifford algebra utilities
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  
                n -= 2
                i = -1  
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}
        self.n = int(n)

    def __add__(self, other: "Multivector") -> "Multivector":
        comp = self.components.copy()
        for k, v in other.components.items():
            comp[k] = comp.get(k, 0.0) + v
        return Multivector(comp, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({k: v * scalar for k, v in self.components.items()}, self.n)

    __rmul__ = __mul__

    def copy(self) -> "Multivector":
        return Multivector(self.components.copy(), self.n)

    def __repr__(self) -> str:
        terms = [f"{v:.3g}*e{sorted(list(k))}" if k else f"{v:.3g}" for k, v in self.components.items()]
        return " + ".join(terms) if terms else "0"


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    result: Dict[FrozenSet[int], float] = defaultdict(float)
    for blade_a, coeff_a in a.components.items():
        for blade_b, coeff_b in b.components.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            result[blade_res] += coeff_a * coeff_b * sign
    return Multivector(dict(result), a.n)


# ----------------------------------------------------------------------
# Bandit utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridLTCBandit"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, List[float]] = {}  


def reset_policy() -> None:
    _POLICY.clear()


def _record(action: str, reward: float) -> None:
    total, cnt = _POLICY.get(action, [0.0, 0.0])
    _POLICY[action] = [total + reward, cnt + 1.0]


def _reward(action: str) -> float:
    total, cnt = _POLICY.get(action, [0.0, 0.0])
    return total / cnt if cnt else 0.0


def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]


def _log_count_ratio(action_a: str, action_b: str) -> float:
    cnt_a = _count(action_a) + 1e-9
    cnt_b = _count(action_b) + 1e-9
    return math.log(cnt_a / cnt_b)


def _fold_change_detection(x: float, eps: float = 1e-9) -> float:
    return math.log(max(abs(x), eps) / (eps + 1))


def _hybrid_store_factor(action_id: str, reference_action: str) -> float:
    lc_ratio = _log_count_ratio(action_id, reference_action)
    fc = _fold_change_detection(_reward(action_id))
    return lc_ratio * fc


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_update(
    W: Multivector,
    X: Multivector,
    action_id: str,
    reference_action: str = "baseline",
) -> Multivector:
    P = geometric_product(W, X)
    λ = _hybrid_store_factor(action_id, reference_action)
    W_next = P * λ
    return W_next


def select_action(context_id: str, candidate_ids: Iterable[str]) -> BanditAction:
    for aid in candidate_ids:
        if aid not in _POLICY:
            _POLICY[aid] = [0.0, 0.0]

    total_counts = sum(_count(a) for a in candidate_ids) + 1e-9
    scores: List[Tuple[float, str]] = []
    for aid in candidate_ids:
        avg = _reward(aid)
        cnt = _count(aid)
        explore = math.sqrt(math.log(total_counts) / (cnt + 1e-9))
        ucb = avg + explore
        scores.append((ucb, aid))

    best_score, best_aid = max(scores)
    best_cnt = _count(best_aid)
    propensity = 1 / (1 + best_cnt)
    return BanditAction(best_aid, propensity, _reward(best_aid), best_score)


def update_policy(action: str, reward: float) -> None:
    _record(action, reward)


# Usage
if __name__ == "__main__":
    reset_policy()
    W = Multivector({frozenset(): 1.0}, 2)
    X = Multivector({frozenset([0]): 1.0}, 2)

    action_id = "test_action"
    reference_action = "baseline"

    for _ in range(10):
        W_next = hybrid_update(W, X, action_id, reference_action)
        print(W_next)

        update_policy(action_id, 1.0)
        selected_action = select_action("context", ["test_action", "other_action"])
        print(selected_action)

        W = W_next