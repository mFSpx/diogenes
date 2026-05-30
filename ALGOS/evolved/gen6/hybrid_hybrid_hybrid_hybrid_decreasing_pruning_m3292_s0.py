# DARWIN HAMMER — match 3292, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2447_s0.py (gen5)
# parent_b: decreasing_pruning.py (gen0)
# born: 2026-05-29T23:48:59Z

"""
Hybrid algorithm fusing 'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2447_s0' and 'decreasing_pruning'.
The mathematical bridge between the two parents is based on integrating the contextual multi-armed bandit with the Shapley value calculation and the decreasing-rate pruning schedule.
The bandit's expected reward is modified to incorporate the Shapley value of each feature in the context, while the Shapley value calculation is used to explain the bandit's decisions.
The pruning schedule is used to adaptively prune the edges of the bandit's context graph, reducing the dimensionality of the feature space and improving the efficiency of the Shapley value calculation.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# Shared Types
Vector = list[float]

# Bandit core
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

# Shapley value calculation
class Morphology:
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        return not self.open

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    return math.factorial(subset_size) * math.factorial(feature_count - subset_size - 1) / math.factorial(feature_count)

def exact_shapley_value(
    value_fn: callable[[frozenset[int]], float],
    feature_index: int,
    feature_count: int,
) -> float:
    shapley_value = 0.0
    for subset_size in range(feature_count + 1):
        weight = shapley_kernel_weight(subset_size, feature_count)
        for subset in combinations(range(feature_count), subset_size):
            subset = frozenset(subset)
            if feature_index in subset:
                value_with_feature = value_fn(subset)
            else:
                value_with_feature = value_fn(subset | {feature_index})
            shapley_value += weight * (value_with_feature - value_fn(subset))
    return shapley_value

from math import comb

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non-negative')
    return min(1.0, lam * math.exp(-alpha * t))

def prune_edges(edges: list, t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> list:
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]

def hybrid_bandit_pruning(
    value_fn: callable[[frozenset[int]], float],
    feature_count: int,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> float:
    edges = list(range(feature_count))
    pruned_edges = prune_edges(edges, t, lam, alpha, seed)
    shapley_values = []
    for feature_index in pruned_edges:
        shapley_value = exact_shapley_value(value_fn, feature_index, feature_count)
        shapley_values.append(shapley_value)
    return sum(shapley_values) / len(shapley_values)

def hybrid_bandit_pruning_with_context(
    value_fn: callable[[frozenset[int]], float],
    feature_count: int,
    context: list,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> float:
    edges = list(range(feature_count))
    pruned_edges = prune_edges(edges, t, lam, alpha, seed)
    shapley_values = []
    for feature_index in pruned_edges:
        context_value = value_fn(frozenset(context))
        shapley_value = exact_shapley_value(value_fn, feature_index, feature_count)
        shapley_values.append(shapley_value)
    return sum(shapley_values) / len(shapley_values)

def hybrid_bandit_pruning_with_multiple_contexts(
    value_fn: callable[[frozenset[int]], float],
    feature_count: int,
    contexts: list[list],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> float:
    edges = list(range(feature_count))
    pruned_edges = prune_edges(edges, t, lam, alpha, seed)
    shapley_values = []
    for context in contexts:
        for feature_index in pruned_edges:
            context_value = value_fn(frozenset(context))
            shapley_value = exact_shapley_value(value_fn, feature_index, feature_count)
            shapley_values.append(shapley_value)
    return sum(shapley_values) / len(shapley_values)

if __name__ == "__main__":
    def value_fn(subset: frozenset[int]) -> float:
        return sum(subset)

    feature_count = 10
    t = 1.0
    lam = 1.0
    alpha = 0.2
    seed = 42

    result = hybrid_bandit_pruning(value_fn, feature_count, t, lam, alpha, seed)
    print(result)

    context = [1, 2, 3]
    result = hybrid_bandit_pruning_with_context(value_fn, feature_count, context, t, lam, alpha, seed)
    print(result)

    contexts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = hybrid_bandit_pruning_with_multiple_contexts(value_fn, feature_count, contexts, t, lam, alpha, seed)
    print(result)