# DARWIN HAMMER — match 685, survivor 0
# gen: 5
# parent_a: hoeffding_tree.py (gen0)
# parent_b: hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s1.py (gen4)
# born: 2026-05-29T23:30:28Z

"""
This module integrates the governing equations of 'hoeffding_tree.py' and 'hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s1.py'. 
The mathematical bridge lies in the use of the Gini coefficient to inform the Hoeffding bound in the decision to split in the Hoeffding tree. 
By evaluating the Gini coefficient of the features at each node, we can leverage the Hoeffding bound to guide the splitting process in a way that minimizes the impact of noise in the data stream.
The hybrid algorithm fuses the core topologies of both parents by using the Gini coefficient to inform the Hoeffding bound, creating a more robust and adaptive decision-making process.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from collections.abc import Iterable

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0: raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def should_split(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

def hybrid_split_decision(values: Iterable[float], best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    gini = gini_coefficient(values)
    adjusted_r = r * (1 - gini)
    return should_split(best_gain, second_best_gain, adjusted_r, delta, n, tie_threshold)

def similarity_matrix(features: Dict[Hashable, Sequence[float]]) -> Tuple[np.ndarray, List[Hashable]]:
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i == j:
                S[i, j] = 1.0
            else:
                S[i, j] = 1 - (gini_coefficient(features[ni]) + gini_coefficient(features[nj])) / 2
    return S, nodes

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

if __name__ == "__main__":
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    best_gain = 0.6
    second_best_gain = 0.4
    r = 0.5
    delta = 0.01
    n = 100
    tie_threshold = 0.05
    split_decision = hybrid_split_decision(values, best_gain, second_best_gain, r, delta, n, tie_threshold)
    print(split_decision)
    features = {"node1": [0.1, 0.2, 0.3], "node2": [0.4, 0.5, 0.6]}
    S, nodes = similarity_matrix(features)
    print(S)
    print(nodes)