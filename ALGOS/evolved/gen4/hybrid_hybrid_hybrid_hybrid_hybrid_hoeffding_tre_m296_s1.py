# DARWIN HAMMER — match 296, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s2.py (gen3)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s7.py (gen1)
# born: 2026-05-29T23:28:13Z

import math
import random
import sys
from datetime import date
from pathlib import Path
import numpy as np
from collections import Counter
from dataclasses import dataclass
from functools import reduce

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k):
        return Multivector(
            {bl: self.components[bl] for bl in self.components if len(bl) == k}, self.n
        )

    def __mul__(self, other):
        if isinstance(other, Multivector):
            result = {}
            for k, v in self.components.items():
                for k2, v2 in other.components.items():
                    new_k, sign = _multiply_blades(k, k2)
                    result[new_k] = result.get(new_k, 0) + sign * v * v2
            return Multivector(result, self.n)
        elif isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        else:
            raise TypeError("Unsupported operand type")

    def __rmul__(self, other):
        return self * other

def hoeffding_bound(r: float, delta: float, n: int, multivector: Multivector) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    scaled_r = r * multivector.components.get(frozenset(), 1.0)
    return math.sqrt((scaled_r * scaled_r * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def gini_coefficient(values: list[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def gini_impurity(labels: list[int]) -> float:
    total = 0
    counts: Counter = Counter()
    for lbl in labels:
        counts[lbl] += 1
        total += 1
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    return 1.0 - np.sum(probs ** 2)

def gini_gain(parent_labels: list[int],
              left_labels: list[int],
              right_labels: list[int], 
              multivector: Multivector) -> float:
    parent_imp = gini_impurity(parent_labels)
    n_parent = len(parent_labels)
    scaled_n_parent = n_parent * multivector.components.get(frozenset(), 1.0)
    left_imp = gini_impurity(left_labels)
    right_imp = gini_impurity(right_labels)
    left_weight = len(left_labels) / scaled_n_parent
    right_weight = len(right_labels) / scaled_n_parent
    return parent_imp - left_weight * left_imp - right_weight * right_imp

def hybrid_hoeffding_gini_multivector_test():
    multivector = Multivector({frozenset(): 0.5, frozenset({1}): 0.3}, 2)
    r = 1.0
    delta = 0.01
    n = 100
    epsilon = hoeffding_bound(r, delta, n, multivector)
    print(f"Hoeffding bound ε: {epsilon}")
    
    parent_labels = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    left_labels = [1, 1, 2]
    right_labels = [2, 2, 3, 3]
    gini_red = gini_gain(parent_labels, left_labels, right_labels, multivector)
    print(f"Gini gain: {gini_red}")
    print(f"Multivector components: {multivector.components}")
    print(f"Multivector grade 0: {multivector.grade(0).components}")
    print(f"Multivector grade 1: {multivector.grade(1).components}")

if __name__ == "__main__":
    hybrid_hoeffding_gini_multivector_test()