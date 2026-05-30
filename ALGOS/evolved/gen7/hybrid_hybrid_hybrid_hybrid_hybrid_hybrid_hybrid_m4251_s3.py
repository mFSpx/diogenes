# DARWIN HAMMER — match 4251, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1218_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1669_s0.py (gen6)
# born: 2026-05-29T23:54:39Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, FrozenSet
import numpy as np

# ----------------------------------------------------------------------
# Multivector utilities
# ----------------------------------------------------------------------
class Multivector:
    """
    Sparse multivector for a Euclidean Clifford algebra 𝔾(n).
    """

    def __init__(self, components: Dict[FrozenSet[int], float] | None = None, n: int = 3):
        self.n = int(n)
        # filter near‑zero entries for sparsity
        self.components: Dict[FrozenSet[int], float] = {
            k: float(v) for k, v in (components or {}).items() if abs(v) > 1e-15
        }

    # ------------------------------------------------------------------
    # Algebraic helpers
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
            if abs(result[blade]) < 1e-15:
                del result[blade]
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()}, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

    __rmul__ = __mul__

    # ------------------------------------------------------------------
    # Norm and distance
    # ------------------------------------------------------------------
    def norm(self) -> float:
        """Euclidean norm of the coefficient vector."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    def distance(self, other: "Multivector") -> float:
        """‖self − other‖."""
        return (self - other).norm()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def to_array(self) -> np.ndarray:
        max_grade = self.n
        array = np.zeros(int(2**max_grade))
        for blade, coef in self.components.items():
            index = sum(2**i for i in blade)
            array[index] = coef
        return array

    @classmethod
    def from_array(cls, array: np.ndarray, n: int) -> "Multivector":
        max_grade = n
        components = {}
        for i, coef in enumerate(array):
            blade = frozenset(j for j in range(max_grade) if (i >> j) & 1)
            if abs(coef) > 1e-15:
                components[blade] = coef
        return cls(components, n)


# ----------------------------------------------------------------------
# Bandit data structures
# ----------------------------------------------------------------------
@dataclass
class ActionStats:
    count: float = 0.0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0

    def mean(self) -> float:
        return self.reward_sum / self.count if self.count > 0 else 0.0

    def variance(self) -> float:
        if self.count == 0:
            return 0.0
        mean = self.mean()
        return max(self.reward_sq_sum / self.count - mean * mean, 0.0)


# ----------------------------------------------------------------------
# Core mathematical bridge utilities
# ----------------------------------------------------------------------
def gaussian_kernel(mv1: Multivector, mv2: Multivector, epsilon: float) -> float:
    array1 = mv1.to_array()
    array2 = mv2.to_array()
    d = np.linalg.norm(array1 - array2)
    return math.exp(- (epsilon ** 2) * (d ** 2))


def fractional_learning_rate(base_lr: float, t: int, alpha: float) -> float:
    if t <= 0:
        return base_lr
    return base_lr * (t ** (alpha - 1.0))


def update_conductance(
    conductance: Multivector,
    context: Multivector,
    reward: float,
    t: int,
    alpha: float,
    base_lr: float,
) -> Multivector:
    lr = fractional_learning_rate(base_lr, t, alpha)
    delta = context * (lr * reward)
    return conductance + delta


def select_action_ucb(
    context: Multivector,
    history: List[Tuple[Multivector, Dict[str, ActionStats]]],
    actions: List[str],
    epsilon: float,
    c: float,
) -> Tuple[str, float]:
    weights = [gaussian_kernel(context, past_ctx, epsilon) for past_ctx, _ in history]
    total_weight = sum(weights) + 1e-12

    agg_stats: Dict[str, ActionStats] = {a: ActionStats() for a in actions}
    for (past_ctx, stats_dict), w in zip(history, weights):
        for a in actions:
            s = stats_dict.get(a)
            if s is None:
                continue
            agg = agg_stats[a]
            agg.count += w * s.count
            agg.reward_sum += w * s.reward_sum
            agg.reward_sq_sum += w * s.reward_sq_sum

    global_count = sum(st.count for st in agg_stats.values()) + 1e-12

    ucb_values: Dict[str, float] = {}
    for a in actions:
        st = agg_stats[a]
        if st.count == 0:
            ucb = float("inf")
        else:
            mean = st.mean()
            var = st.variance()
            bonus = c * math.sqrt(var * math.log(global_count) / (st.count + 1e-9))
            ucb = mean + bonus
        ucb_values[a] = ucb

    chosen_action = max(ucb_values, key=ucb_values.get)

    propensity = agg_stats[chosen_action].count / total_weight

    return chosen_action, propensity


# ----------------------------------------------------------------------
# High‑level hybrid step
# ----------------------------------------------------------------------
def hybrid_step(
    conductance: Multivector,
    context: Multivector,
    history: List[Tuple[Multivector, Dict[str, ActionStats]]],
    actions: List[str],
    epsilon: float,
    c: float,
    t: int,
    alpha: float,
    base_lr: float,
) -> Tuple[Multivector, str, float]:
    chosen_action, propensity = select_action_ucb(context, history, actions, epsilon, c)
    reward = np.random.normal()  # Replace with actual reward
    conductance = update_conductance(conductance, context, reward, t, alpha, base_lr)
    return conductance, chosen_action, propensity