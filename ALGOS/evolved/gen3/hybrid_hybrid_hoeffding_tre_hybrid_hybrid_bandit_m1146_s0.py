# DARWIN HAMMER — match 1146, survivor 0
# gen: 3
# parent_a: hybrid_hoeffding_tree_gini_coefficient_m13_s6.py (gen1)
# parent_b: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s2.py (gen2)
# born: 2026-05-29T23:32:59Z

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Hoeffding helpers
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Return the Hoeffding bound ε for range *r*, confidence *δ* and sample count *n*."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑based split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """Standard Hoeffding split test."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    if gap >= eps:
        return SplitDecision(should_split=True, epsilon=eps, gain_gap=gap, reason="Gap significant")
    else:
        return SplitDecision(should_split=False, epsilon=eps, gain_gap=gap, reason="Gap not significant")


# ----------------------------------------------------------------------
# Parent B – Sketch primitives (adapted from Parent B)
# ----------------------------------------------------------------------
def estimate_log_likelihood(reward_stream: Iterable[float]) -> float:
    """Approximate the log likelihood contribution of the reward stream."""
    return np.log(np.sum(reward_stream)) - np.sum(np.log(reward_stream))


def estimate_cardinality(reward_stream: Iterable[float]) -> int:
    """Estimate the number of distinct contexts using a HyperLogLog sketch."""
    sketch = HyperLogLog()
    for reward in reward_stream:
        sketch.add(reward)
    return sketch.estimate_cardinality()


class HyperLogLog:
    def __init__(self, m: int = 256):
        self.size = m
        self.buckets = [None] * m

    def add(self, x: float):
        digest = hashlib.sha1(str(x).encode()).digest()
        index = int.from_bytes(digest[:4], byteorder='big')
        bucket = index % self.size
        self.buckets[bucket] = x if self.buckets[bucket] is None else self.buckets[bucket]

    def estimate_cardinality(self) -> int:
        buckets_with_elements = sum(1 for bucket in self.buckets if bucket is not None)
        return 2 ** (self.size - math.log(2.0 / buckets_with_elements, 2))


# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------
class HybridBandit:
    def __init__(self, confidence_delta: float, n: int):
        self.confidence_delta = confidence_delta
        self.n = n
        self.store = 0.0
        self.distinct_contexts = 0

    def update_store(self, reward: float, distinct_contexts: int):
        self.store = self.store + reward
        self.distinct_contexts = distinct_contexts

    def estimate_mean_reward(self) -> float:
        return self.store / self.distinct_contexts

    def estimate_var_reward(self) -> float:
        return self.estimate_log_likelihood() / self.distinct_contexts

    def estimate_log_likelihood(self) -> float:
        return estimate_log_likelihood(self.store)

    def estimate_cardinality(self) -> int:
        return estimate_cardinality(self.store)

    def rlct_estimate(self) -> float:
        return math.log(self.distinct_contexts) / self.estimate_cardinality()

    def should_split(self, best_gain: float, second_best_gain: float):
        r = 1 - 1 / self.distinct_contexts
        eps = hoeffding_bound(r, self.confidence_delta, self.n)
        gap = best_gain - second_best_gain
        if gap >= eps:
            return SplitDecision(should_split=True, epsilon=eps, gain_gap=gap, reason="Gap significant")
        else:
            return SplitDecision(should_split=False, epsilon=eps, gain_gap=gap, reason="Gap not significant")


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    hybrid_bandit = HybridBandit(confidence_delta=0.01, n=100)
    # Simulate rewards and distinct contexts
    rewards = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    distinct_contexts = 3
    for i in range(len(rewards)):
        reward = rewards[i]
        distinct_context = distinct_contexts
        hybrid_bandit.update_store(reward, distinct_context)
        split_decision = hybrid_bandit.should_split(1.0, 0.5)
        print(f"Split decision: {split_decision.should_split}, epsilon: {split_decision.epsilon}, gain gap: {split_decision.gain_gap}, reason: {split_decision.reason}")