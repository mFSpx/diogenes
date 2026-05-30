# DARWIN HAMMER — match 685, survivor 1
# gen: 5
# parent_a: hoeffding_tree.py (gen0)
# parent_b: hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s1.py (gen4)
# born: 2026-05-29T23:30:28Z

"""
This module integrates the governing equations of 'hoeffding_tree.py' and 
'hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s1.py'. The mathematical bridge lies in the use 
of the Gini coefficient to inform the Hoeffding bound in the decision to split in the Hoeffding tree. 
By evaluating the Gini coefficient of the feature values at each node, we can leverage the Hoeffding bound 
to guide the splitting process in a way that minimizes the impact of noise in the data stream.

The mathematical interface between the two parents is established by using the Gini coefficient to adjust 
the Hoeffding bound. The Gini coefficient is used to quantify the inequality of the feature values at each 
node, which in turn affects the Hoeffding bound. This fusion enables a more robust and adaptive decision-making 
process by combining the benefits of both parents.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from collections.abc import Iterable

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    gini = gini_coefficient([best_gain, second_best_gain])
    split = gap > eps or gini > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("gini_exceeds_bound" if gini > eps else ("tie_threshold" if eps < tie_threshold else "wait"))
    return SplitDecision(split, eps, gap, reason)


def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0: raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def similarity_matrix(features: Dict[Hashable, Sequence[float]]) -> Tuple[np.ndarray, List[Hashable]]:
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(list(features[ni]))
        for j, nj in enumerate(nodes):
            hj = compute_phash(list(features[nj]))
            S[i, j] = 1 - hamming_distance(hi, hj) / 64
    return S, nodes


def hybrid_feature_selection(features: Dict[Hashable, Sequence[float]], delta: float, n: int, tie_threshold: float = 0.05) -> List[Hashable]:
    similarity_matrix_values, nodes = similarity_matrix(features)
    selected_features = []
    for node in nodes:
        best_gain = 0.0
        second_best_gain = 0.0
        for neighbor in nodes:
            if neighbor != node:
                similarity = similarity_matrix_values[nodes.index(node), nodes.index(neighbor)]
                gain = gaussian(similarity)
                if gain > best_gain:
                    second_best_gain = best_gain
                    best_gain = gain
                elif gain > second_best_gain:
                    second_best_gain = gain
        split = should_split(best_gain, second_best_gain, 0.1, delta, n, tie_threshold).should_split
        if split:
            selected_features.append(node)
    return selected_features


def hybrid_node_split(features: Dict[Hashable, Sequence[float]], delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    best_gain = 0.0
    second_best_gain = 0.0
    for feature in features.keys():
        values = features[feature]
        if isinstance(values, list):
            gini = gini_coefficient(values)
            if gini > 0.5:
                gain = gaussian(gini)
                if gain > best_gain:
                    second_best_gain = best_gain
                    best_gain = gain
                elif gain > second_best_gain:
                    second_best_gain = gain
    return should_split(best_gain, second_best_gain, 0.1, delta, n, tie_threshold)


if __name__ == "__main__":
    features = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "feature3": [3.0, 6.0, 9.0, 12.0, 15.0]
    }
    delta = 0.1
    n = 10
    tie_threshold = 0.05
    print(hybrid_feature_selection(features, delta, n, tie_threshold))
    print(hybrid_node_split(features, delta, n, tie_threshold))