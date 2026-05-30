# DARWIN HAMMER — match 5653, survivor 4
# gen: 6
# parent_a: hybrid_hoeffding_tree_gini_coefficient_m13_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m2606_s3.py (gen5)
# born: 2026-05-30T00:03:58Z

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterable

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    _uid_counter = 0

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        PheromoneEntry._uid_counter += 1
        self.uuid = f"ph-{PheromoneEntry._uid_counter}"
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = int(half_life_seconds)
        now = pathlib.Path(__file__).stat().st_mtime
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (pathlib.Path(__file__).stat().st_mtime - self.last_decay)

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = pathlib.Path(__file__).stat().st_mtime


class PheromoneStore:
    _entries: dict = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> None:
        for entry in cls.get_by_surface(surface_key):
            entry.apply_decay()

    @classmethod
    def clear(cls) -> None:
        cls._entries.clear()


def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0.0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def hoeffding_bound_pheromone(r: float, base_delta: float,
                              n: int, pheromone: PheromoneEntry) -> float:
    if r <= 0 or not (0 < base_delta < 1) or n <= 0:
        raise ValueError("Invalid parameters for Hoeffding bound")
    pheromone.apply_decay()
    scale = min(1.0, pheromone.signal_value / (pheromone.signal_value + 1.0))
    effective_delta = base_delta * (1.0 - scale) + 1e-9  
    return math.sqrt((r * r * math.log(1.0 / effective_delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    weighted_gap: float
    reason: str


def should_split_hybrid(best_gain: float, second_best_gain: float,
                        r: float, base_delta: float, n: int,
                        values: Iterable[float],
                        pheromone: PheromoneEntry,
                        tie_threshold: float = 0.05) -> SplitDecision:
    gini = gini_coefficient(values)
    eps = hoeffding_bound_pheromone(r, base_delta, n, pheromone)
    gap = best_gain - second_best_gain
    weighted_gap = gap * (1.0 - gini)

    split = weighted_gap > eps or eps < tie_threshold
    if weighted_gap > eps:
        reason = "weighted_gap_exceeds_bound"
    elif eps < tie_threshold:
        reason = "epsilon_below_tie_threshold"
    else:
        reason = "await_more_data"

    return SplitDecision(split, eps, weighted_gap, reason)


def update_pheromone_from_split(surface_key: str,
                                succeeded: bool,
                                reward: float = 0.1,
                                decay_half_life: int = 30) -> None:
    entries = PheromoneStore.get_by_surface(surface_key)
    if succeeded:
        if entries:
            entry = max(entries, key=lambda e: e.signal_value)
            entry.signal_value += reward
            entry.last_decay = pathlib.Path(__file__).stat().st_mtime
        else:
            entry = PheromoneEntry(surface_key=surface_key,
                                   signal_kind="split_success",
                                   signal_value=reward,
                                   half_life_seconds=decay_half_life)
            PheromoneStore.add(entry)
    else:
        PheromoneStore.decay_surface(surface_key)


def reset_pheromone_store() -> None:
    PheromoneStore.clear()


def example_utility() -> None:
    # Create a pheromone entry
    entry = PheromoneEntry(surface_key="example_surface",
                           signal_kind="example_signal",
                           signal_value=1.0,
                           half_life_seconds=30)
    PheromoneStore.add(entry)

    # Test should_split_hybrid
    best_gain = 0.5
    second_best_gain = 0.3
    r = 1.0
    base_delta = 0.1
    n = 100
    values = [0.2, 0.3, 0.5]
    decision = should_split_hybrid(best_gain, second_best_gain,
                                   r, base_delta, n, values, entry)
    print(decision)

    # Test update_pheromone_from_split
    update_pheromone_from_split("example_surface", True)
    print(PheromoneStore.get_by_surface("example_surface")[0].signal_value)

    reset_pheromone_store()


if __name__ == "__main__":
    example_utility()