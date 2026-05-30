# DARWIN HAMMER — match 979, survivor 0
# gen: 5
# parent_a: hybrid_hoeffding_tree_gini_coefficient_m13_s7.py (gen1)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hoeffding_tre_m301_s3.py (gen4)
# born: 2026-05-29T23:31:57Z

"""
Hybrid Hoeffding-Gini Regret Engine

Parents:
- hybrid_hoeffding_tree_gini_coefficient_m13_s7.py (Hoeffding bound + Gini coefficient)
- hybrid_hybrid_hybrid_regret_hybrid_hoeffding_tree_m301_s3.py (Regret-Weighted Hoeffding-Gini Engine)

Mathematical bridge:
The regret of each action forms a non-negative distribution.  The Gini coefficient of this
distribution quantifies inequality among regrets.  By feeding the Gini-scaled regret
vector into the Hoeffding bound we obtain a statistically sound split criterion that
operates on the regret space used by the Regret-Weighted component.  Thus the regret
similarity guides the construction of candidate splits while the Gini-weighted regret
supplies the confidence term for Hoeffding-based decisions.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Hoeffding bound utilities (from Parent A)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Return the Hoeffding bound ε for range ``r``, confidence ``1‑δ`` and
    sample size ``n``.
    
    Raises:
        ValueError: if arguments are out of the admissible domain.
    """
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Gini utilities (from Parent B)
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Compute the Gini inequality coefficient for a non-negative iterable."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def gini_impurity(labels: Iterable[int]) -> float:
    """Gini impurity of a categorical label distribution.

    ``labels`` can be any iterable of hashable class identifiers.
    """
    total = 0
    counts: Counter = Counter()
    for lbl in labels:
        counts[lbl] += 1
        total += 1
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    return 1.0 - np.sum(probs ** 2)


# ----------------------------------------------------------------------
# Shared data structures (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Regret utilities (from Parent B)
# ----------------------------------------------------------------------
def regret(values: Iterable[float]) -> float:
    """Compute the regret (maximiser's loss) of a non-negative iterable."""
    return max(values)


def regret_impurity(labels: Iterable[int]) -> float:
    """Regret impurity of a categorical label distribution.

    ``labels`` can be any iterable of hashable class identifiers.
    """
    total = 0
    counts: Counter = Counter()
    for lbl in labels:
        counts[lbl] += 1
        total += 1
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    return max(probs)


# ----------------------------------------------------------------------
# Hybrid Regret-Hoeffding Gini Engine
# ----------------------------------------------------------------------
def hybrid_regret_hoeffding_gini(values: Iterable[float], delta: float, n: int) -> Tuple[float, float]:
    """Hybrid regret-Hoeffding Gini engine.

    Compute the hybrid regret-Hoeffding Gini bound for a non-negative iterable.

    Args:
        values: iterable of non-negative values
        delta: confidence level
        n: sample size

    Returns:
        hybrid regret-Hoeffding Gini bound
    """
    regret_val = regret(values)
    gini_val = gini_coefficient(values)
    hoeffding_bound_val = hoeffding_bound(regret_val, delta, n)
    return gini_val * hoeffding_bound_val


def hybrid_regret_hoeffding_gini_split(parent_labels: Iterable[int],
                                      left_labels: Iterable[int],
                                      right_labels: Iterable[int],
                                      delta: float,
                                      n: int) -> float:
    """Hybrid regret-Hoeffding Gini split.

    Compute the hybrid regret-Hoeffding Gini split for a categorical label distribution.

    Args:
        parent_labels: iterable of parent labels
        left_labels: iterable of left child labels
        right_labels: iterable of right child labels
        delta: confidence level
        n: sample size

    Returns:
        hybrid regret-Hoeffding Gini split
    """
    parent_imp = gini_impurity(parent_labels)
    left_imp = gini_impurity(left_labels)
    right_imp = gini_impurity(right_labels)
    regret_left = regret_impurity(left_labels)
    regret_right = regret_impurity(right_labels)
    hoeffding_bound_left = hoeffding_bound(regret_left, delta, n)
    hoeffding_bound_right = hoeffding_bound(regret_right, delta, n)
    return (parent_imp - left_imp) * hoeffding_bound_left + (parent_imp - right_imp) * hoeffding_bound_right


if __name__ == "__main__":
    # Smoke test
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    delta = 0.1
    n = 10
    print(hybrid_regret_hoeffding_gini(values, delta, n))
    print(hybrid_regret_hoeffding_gini_split([1, 2, 3], [1, 2], [3, 4], delta, n))