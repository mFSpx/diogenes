# DARWIN HAMMER — match 1097, survivor 0
# gen: 4
# parent_a: hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s1.py (gen2)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_worksh_m146_s0.py (gen3)
# born: 2026-05-29T23:32:45Z

"""
This module implements a novel HYBRID algorithm that mathematically fuses the core topologies of two parent algorithms: 
hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s1 and hybrid_hybrid_fisher_locali_hybrid_hybrid_worksh_m146_s0. 
The mathematical bridge between these two algorithms is found in the concept of information density and entropy, 
where the Fisher information scoring and sinusoidal rotation from the Fisher localization algorithm are used to 
determine the best angle for off-axis sensing, and the entropy and information gain from the pheromone infotaxis 
algorithm are used to make decisions based on pheromone signals. The hybrid algorithm combines these two concepts 
by using the vector representation from krampus_brainmap as the input to the infotaxis decision-making process 
in hybrid_pheromone_infotaxis_m3_s4, and the Fisher information scoring to determine the most informative date 
candidates.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import datetime, timezone

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
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
    _entries: dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def parse_loose_datetime(raw: str) -> datetime | None:
    text = raw.strip().strip("'\"`[]()")
    if not text:
        return None
    try:
        val = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    except ValueError:
        return None


def entropy(p: np.ndarray) -> float:
    return -np.sum(p * np.log2(p))


def hybrid_fisher_pheromone(p: np.ndarray, theta: float, center: float, width: float) -> float:
    fisher = fisher_score(theta, center, width)
    entr = entropy(p)
    return fisher * entr


def hybrid_krampus_pheromone(p: np.ndarray, surface_key: str) -> float:
    pheromones = PheromoneStore.get_by_surface(surface_key)
    signal_values = [pheromone.signal_value for pheromone in pheromones]
    if not signal_values:
        return 0.0
    return np.sum(signal_values) * entropy(p)


if __name__ == "__main__":
    import uuid
    surface_key = "test_surface"
    pheromone = PheromoneEntry(surface_key, "test_kind", 1.0, 3600)
    PheromoneStore.add(pheromone)
    p = np.array([0.5, 0.5])
    theta = 0.5
    center = 0.0
    width = 1.0
    print(hybrid_fisher_pheromone(p, theta, center, width))
    print(hybrid_krampus_pheromone(p, surface_key))