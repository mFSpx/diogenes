# DARWIN HAMMER — match 894, survivor 2
# gen: 2
# parent_a: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# parent_b: hybrid_pheromone_infotaxis_m3_s1.py (gen1)
# born: 2026-05-29T23:31:40Z

import math
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Pheromone infrastructure
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
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
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory singleton store for pheromone entries."""
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> None:
        for e in cls.get_by_surface(surface_key):
            e.apply_decay()

    @classmethod
    def total_signal(cls, surface_key: str, kind: str = None) -> float:
        """Sum of (decayed) signal values for a surface, optionally filtered by kind."""
        cls.decay_surface(surface_key)
        entries = cls.get_by_surface(surface_key)
        if kind is not None:
            entries = [e for e in entries if e.signal_kind == kind]
        return sum(e.signal_value for e in entries)


# ----------------------------------------------------------------------
# Entropy helpers
# ----------------------------------------------------------------------
def calculate_entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    norm = [p / total for p in probabilities if p > eps]
    return -sum(p * math.log(max(p, eps)) for p in norm)


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Weighted average entropy after a binary observation."""
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)


def best_action(actions: Dict[str, Tuple[float, List[float], List[float]]]) -> str:
    """
    Choose the action with minimal expected entropy.
    actions: mapping action_id → (p_hit, hit_state, miss_state)
    """
    if not actions:
        raise ValueError("no actions to evaluate")
    return min(actions, key=lambda a: expected_entropy(*actions[a]))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def add_pheromone_signal(
    surface_key: str,
    signal_kind: str,
    delta: float,
    half_life_seconds: int = 60,
) -> None:
    PheromoneStore.decay_surface(surface_key)

    candidates = [
        e for e in PheromoneStore.get_by_surface(surface_key) if e.signal_kind == signal_kind
    ]
    if candidates:
        entry = max(candidates, key=lambda e: e.last_decay)
        entry.signal_value += delta
        entry.last_decay = datetime.now(timezone.utc)
    else:
        entry = PheromoneEntry(surface_key, signal_kind, delta, half_life_seconds)
        PheromoneStore.add(entry)


def pheromone_distribution(surface_key: str) -> List[float]:
    PheromoneStore.decay_surface(surface_key)
    entries = PheromoneStore.get_by_surface(surface_key)
    values = np.array([e.signal_value for e in entries])
    values = np.maximum(values, 1e-12)  # Ensure numerical stability
    total = np.sum(values)
    if total <= 0:
        return [1.0 / len(entries) if entries else 0.0]
    return values / total


def select_best_move(
    current_surface: str,
    neighbor_surfaces: List[str],
    hit_probability: float = 0.5,
) -> str:
    current_dist = pheromone_distribution(current_surface)

    actions: Dict[str, Tuple[float, List[float], List[float]]] = {}

    for nb in neighbor_surfaces:
        hit_dist = np.array(current_dist)
        hit_dist = np.append(hit_dist, 1.0)
        hit_dist = hit_dist / np.sum(hit_dist)

        miss_dist = current_dist

        actions[nb] = (hit_probability, hit_dist.tolist(), miss_dist)

    return best_action(actions)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    surfaces = ["A", "B", "C"]
    random.seed(0)

    add_pheromone_signal("A", "explore", 5.0, half_life_seconds=120)
    add_pheromone_signal("A", "explore", 3.0, half_life_seconds=120)

    best_move = select_best_move("A", ["B", "C"])
    print(best_move)