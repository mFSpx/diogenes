# DARWIN HAMMER — match 894, survivor 0
# gen: 2
# parent_a: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# parent_b: hybrid_pheromone_infotaxis_m3_s1.py (gen1)
# born: 2026-05-29T23:31:40Z

"""
This module integrates the Darwinian surface pheromone worker (hybrid_pheromone_infotaxis_m3_s4) with the gradient-free entropy search helpers (hybrid_pheromone_infotaxis_m3_s1).
The mathematical bridge between these two structures is the concept of pheromone signals and their decay rates, which can be seen as a form of entropy optimization.
By combining the pheromone signal system with the entropy search algorithms, we can create a novel hybrid algorithm that adapts to changing environments and optimizes the search process.
"""

import argparse
import json
import math
import numpy as np
import random
import sys
from datetime import datetime, timezone

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
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Return the multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


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
            before = entry.signal_value
            entry.apply_decay()
            rows.append({
                "pheromone_uuid": entry.uuid,
                "surface_key": entry.surface_key,
                "signal_kind": entry.signal_kind,
                "signal_value_before": before,
                "signal_value_after": entry.signal_value
            })
        return rows


def calculate_pheromone_signal(surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int) -> float:
    """
    Calculates the pheromone signal strength based on the surface key, signal kind, signal value, and half-life seconds.
    """
    return signal_value * math.pow(0.5, (datetime.now(timezone.utc) - datetime.now(timezone.utc)).total_seconds() / half_life_seconds)


def calculate_entropy(probabilities: list, eps: float = 1e-12) -> float:
    """
    Calculates the entropy of a given probability distribution.
    """
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)


def expected_entropy(p_hit: float, hit_state: list, miss_state: list) -> float:
    """
    Calculates the expected entropy of a given probability distribution and hit/miss states.
    """
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)


def best_action(actions: dict) -> str:
    """
    Determines the best action based on the expected entropy of each action.
    """
    if not actions:
        raise ValueError('actions required')
    return min(actions, key=lambda a: (expected_entropy(*actions[a]), a))


def hybrid_pheromone_infotaxis(surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int, p_hit: float, hit_state: list, miss_state: list) -> tuple:
    """
    Combines pheromone signal calculation with entropy-based decision making.
    """
    pheromone_signal = calculate_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds)
    entropy = expected_entropy(p_hit, hit_state, miss_state)
    return pheromone_signal, entropy


def hybrid_pheromone_infotaxis_action(surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int, p_hit: float, hit_state: list, miss_state: list, actions: dict) -> str:
    """
    Combines pheromone signal calculation with entropy-based decision making and selects the best action.
    """
    pheromone_signal, entropy = hybrid_pheromone_infotaxis(surface_key, signal_kind, signal_value, half_life_seconds, p_hit, hit_state, miss_state)
    best_action_id = best_action(actions)
    return best_action_id


if __name__ == "__main__":
    pheromone_entry = PheromoneEntry("surface_key", "signal_kind", 1.0, 3600)
    PheromoneStore.add(pheromone_entry)
    PheromoneStore.decay_surface("surface_key")
    pheromone_signal = calculate_pheromone_signal("surface_key", "signal_kind", 1.0, 3600)
    entropy = calculate_entropy([0.5, 0.5])
    expected_entropy_val = expected_entropy(0.5, [0.5, 0.5], [0.5, 0.5])
    best_action_id = best_action({"action1": (0.5, [0.5, 0.5], [0.5, 0.5])})
    hybrid_pheromone_infotaxis_result = hybrid_pheromone_infotaxis("surface_key", "signal_kind", 1.0, 3600, 0.5, [0.5, 0.5], [0.5, 0.5])
    hybrid_pheromone_infotaxis_action_result = hybrid_pheromone_infotaxis_action("surface_key", "signal_kind", 1.0, 3600, 0.5, [0.5, 0.5], [0.5, 0.5], {"action1": (0.5, [0.5, 0.5], [0.5, 0.5])})