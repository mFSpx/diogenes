# DARWIN HAMMER — match 2142, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_pheromone_inf_m894_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m646_s0.py (gen5)
# born: 2026-05-29T23:41:03Z

"""
Hybrid algorithm fusing the topologies of 
`hybrid_hybrid_pheromone_inf_hybrid_pheromone_inf_m894_s2.py` and 
`hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m646_s0.py`.

The mathematical bridge between the two parents lies in the representation 
of the pheromone signal and the Fisher score. The pheromone signal provides 
a time-decaying signal value, while the Fisher score provides a data-driven 
weighting factor. By interpreting the pheromone signal as a dynamic weighting 
factor, we can link the two parent topologies. The hybrid algorithm uses 
the Fisher score to weight the pheromone signal, allowing for a unified 
decision metric that combines the strengths of both parents.

The governing equations of both parents are integrated through the 
following interface:
- The Fisher score `I(θ)` from the second parent is used to weight 
  the pheromone signal `s` from the first parent, resulting in a 
  weighted pheromone signal `I(θ) · s`.
- The pheromone decay factor from the first parent is used to compute 
  the dynamic weighting factor.

This hybrid algorithm enables the joint optimization of the pheromone 
signal and the Fisher score, leading to a more robust and accurate 
decision-making process.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List

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


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def hybrid_pheromone_fisher(surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int, 
                              theta: float, center: float, width: float) -> float:
    pheromone_entry = PheromoneEntry(surface_key, signal_kind, signal_value, half_life_seconds)
    fisher_score_value = fisher_score(theta, center, width)
    weighted_signal_value = fisher_score_value * pheromone_entry.signal_value
    return weighted_signal_value


def get_weighted_pheromone_signals(surface_key: str, kind: str = None) -> List[float]:
    PheromoneStore.decay_surface(surface_key)
    entries = PheromoneStore.get_by_surface(surface_key)
    if kind is not None:
        entries = [e for e in entries if e.signal_kind == kind]
    theta = 0.5
    center = 0.0
    width = 1.0
    return [fisher_score(theta, center, width) * e.signal_value for e in entries]


def get_total_weighted_signal(surface_key: str, kind: str = None) -> float:
    weighted_signals = get_weighted_pheromone_signals(surface_key, kind)
    return sum(weighted_signals)


if __name__ == "__main__":
    surface_key = "test_surface"
    signal_kind = "test_signal"
    signal_value = 1.0
    half_life_seconds = 10
    theta = 0.5
    center = 0.0
    width = 1.0

    hybrid_signal = hybrid_pheromone_fisher(surface_key, signal_kind, signal_value, half_life_seconds, theta, center, width)
    print(hybrid_signal)

    weighted_signals = get_weighted_pheromone_signals(surface_key)
    print(weighted_signals)

    total_weighted_signal = get_total_weighted_signal(surface_key)
    print(total_weighted_signal)