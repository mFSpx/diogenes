# DARWIN HAMMER — match 5653, survivor 0
# gen: 6
# parent_a: hybrid_hoeffding_tree_gini_coefficient_m13_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m2606_s3.py (gen5)
# born: 2026-05-30T00:03:58Z

"""
This module integrates the Hoeffding tree splits from hoeffding_tree.py and the Gini inequality coefficient from gini_coefficient.py with the pheromone signal system and entropy search algorithms from hybrid_hybrid_pheromone_inf_hybrid_pheromone_inf_m894_s0.py and hybrid_hybrid_hybrid_path_s_path_signature_m501_s1.py.
The mathematical bridge between these structures lies in the concept of pheromone signals as a form of entropy optimization, which can be combined with the Gini coefficient as a measure of inequality to determine the optimal split points in a Hoeffding tree.
"""

import math
from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np
import random
import sys
import pathlib

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

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0: raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def should_split_gini(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, values: Iterable[float], tie_threshold: float = 0.05) -> SplitDecision:
    gini = gini_coefficient(values)
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    gini_weighted_gap = gap * (1 - gini)
    split = gini_weighted_gap > eps or eps < tie_threshold
    reason = "gini_weighted_gap_exceeds_bound" if gini_weighted_gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gini_weighted_gap, reason)

def calculate_gini_weighted_split_point(values: Iterable[float], r: float, delta: float, n: int) -> float:
    gini = gini_coefficient(values)
    eps = hoeffding_bound(r, delta, n)
    return gini * eps

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(random.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = pathlib.Path(__file__).modified_time()
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (pathlib.Path(__file__).modified_time() - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Return the multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = pathlib.Path(__file__).modified_time()


class PheromoneStore:
    """Singleton-like in-memory store for demo purposes."""
    _entries: dict = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> list:
        rows = []
        for entry in cls.get_by_surface(surface_key):
            entry.apply_decay()
            rows.append(entry.signal_value)
        return rows

def pheromone_gini_weighted_gap(surface_key: str, r: float, delta: float, n: int, values: Iterable[float], tie_threshold: float = 0.05) -> SplitDecision:
    pheromone_values = PheromoneStore.decay_surface(surface_key)
    gini = gini_coefficient(values)
    best_gain = np.random.uniform(0, 1)
    second_best_gain = np.random.uniform(0, 1)
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    gini_weighted_gap = gap * (1 - gini)
    pheromone_weighted_gap = gini_weighted_gap * np.mean(pheromone_values)
    split = pheromone_weighted_gap > eps or eps < tie_threshold
    reason = "pheromone_weighted_gap_exceeds_bound" if pheromone_weighted_gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, pheromone_weighted_gap, reason)

def calculate_pheromone_gini_weighted_split_point(surface_key: str, r: float, delta: float, n: int, values: Iterable[float]) -> float:
    pheromone_values = PheromoneStore.decay_surface(surface_key)
    gini = gini_coefficient(values)
    eps = hoeffding_bound(r, delta, n)
    return gini * eps * np.mean(pheromone_values)

def hybrid_split_tree(values: Iterable[float], surface_key: str, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    best_gain = np.random.uniform(0, 1)
    second_best_gain = np.random.uniform(0, 1)
    pheromone_split = pheromone_gini_weighted_gap(surface_key, r, delta, n, values, tie_threshold)
    gini_split = should_split_gini(best_gain, second_best_gain, r, delta, n, values, tie_threshold)
    split = pheromone_split.should_split or gini_split.should_split
    reason = "pheromone" if pheromone_split.should_split else "gini"
    return SplitDecision(split, pheromone_split.epsilon, pheromone_split.gain_gap, reason)

if __name__ == "__main__":
    surface_key = "surface1"
    r = 0.1
    delta = 0.01
    n = 100
    values = [1, 2, 3, 4, 5]
    print(hybrid_split_tree(values, surface_key, r, delta, n))