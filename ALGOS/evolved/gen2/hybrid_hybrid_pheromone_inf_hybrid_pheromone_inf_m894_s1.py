# DARWIN HAMMER — match 894, survivor 1
# gen: 2
# parent_a: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# parent_b: hybrid_pheromone_infotaxis_m3_s1.py (gen1)
# born: 2026-05-29T23:31:40Z

"""Hybrid Pheromone‑Infotaxis Algorithm
Parents:
- hybrid_pheromone_infotaxis_m3_s4.py (pheromone handling)
- hybrid_pheromone_infotaxis_m3_s1.py (entropy‑based infotaxis)

Mathematical bridge:
Pheromone values on a surface are interpreted as a non‑negative measure μ_i.
Normalising μ_i yields a probability distribution p_i = μ_i / Σ μ_i.
Entropy H(p) = −Σ p_i log p_i quantifies uncertainty of the agent’s belief.
The decay of each μ_i (half‑life exponential) continuously reshapes p_i,
while infotaxis selects the action that minimises the *expected* entropy after a
hypothetical observation (hit / miss).  Thus the hybrid core consists of:
1. Time‑dependent pheromone update → dynamic probabilities.
2. Entropy computation on those probabilities.
3. Expected‑entropy minimisation to choose the next move.

The implementation below fuses these steps into a single coherent system.
"""

import math
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Pheromone infrastructure (from parent A)
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
# Entropy helpers (from parent B)
# ----------------------------------------------------------------------
def calculate_entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    norm = [p / total for p in probabilities if p > 0]
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
    """
    Record a new pheromone signal (or reinforce an existing one) on a surface.
    The signal value is added to the most recent entry of the same kind;
    otherwise a fresh entry is created.
    """
    # Decay existing entries first
    PheromoneStore.decay_surface(surface_key)

    # Look for a recent entry of the same kind
    candidates = [
        e for e in PheromoneStore.get_by_surface(surface_key) if e.signal_kind == signal_kind
    ]
    if candidates:
        # Reinforce the newest entry
        entry = max(candidates, key=lambda e: e.last_decay)
        entry.signal_value += delta
        entry.last_decay = datetime.now(timezone.utc)
    else:
        # Create a brand‑new entry
        entry = PheromoneEntry(surface_key, signal_kind, delta, half_life_seconds)
        PheromoneStore.add(entry)


def pheromone_distribution(surface_key: str) -> List[float]:
    """
    Return the normalised pheromone values for all entries on a surface.
    This distribution is the probability vector used by infotaxis.
    """
    PheromoneStore.decay_surface(surface_key)
    entries = PheromoneStore.get_by_surface(surface_key)
    values = [e.signal_value for e in entries if e.signal_value > 0]
    if not values:
        # Avoid division by zero – return a uniform tiny distribution
        return [1.0]
    total = sum(values)
    return [v / total for v in values]


def select_best_move(
    current_surface: str,
    neighbor_surfaces: List[str],
    hit_probability: float = 0.5,
) -> str:
    """
    Evaluate each neighboring surface as a candidate action.
    For each candidate we imagine two possible outcomes:
        * hit   – the agent finds a cue, increasing pheromone on the neighbour
        * miss  – no cue, leaving the distribution unchanged
    The action with the lowest expected entropy after the imagined update is chosen.
    """
    # Current belief distribution
    current_dist = pheromone_distribution(current_surface)

    actions: Dict[str, Tuple[float, List[float], List[float]]] = {}

    for nb in neighbor_surfaces:
        # Simulate a "hit": we would add a unit pheromone of kind "hit" on the neighbour.
        # This changes the joint distribution of current+neighbour surfaces.
        # For simplicity we treat the neighbour as an independent bucket.
        # Build hit_state distribution:
        hit_dist = current_dist + [1.0]  # extra mass for the neighbour
        hit_dist = [p / sum(hit_dist) for p in hit_dist]

        # Simulate a "miss": distribution stays as current_dist
        miss_dist = current_dist

        actions[nb] = (hit_probability, hit_dist, miss_dist)

    return best_action(actions)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny environment with three surfaces
    surfaces = ["A", "B", "C"]
    # Seed deterministic behaviour
    random.seed(0)

    # Initialise pheromones
    add_pheromone_signal("A", "explore", 5.0, half_life_seconds=120)
    add_pheromone_signal("A", "explore", 3.0, half_life_seconds=120)
    add_pheromone_signal("B", "explore", 2.0, half_life_seconds=120)
    add_pheromone_signal("C", "explore", 1.0, half_life_seconds=120)

    # Compute entropy on surface A
    dist_A = pheromone_distribution("A")
    print(f"Distribution on A: {dist_A}")
    print(f"Entropy on A: {calculate_entropy(dist_A):.4f}")

    # Choose best move from A to either B or C
    best = select_best_move("A", ["B", "C"], hit_probability=0.6)
    print(f"Best next surface from A: {best}")

    # Apply the chosen move (simulate a hit)
    add_pheromone_signal(best, "hit", delta=1.0, half_life_seconds=90)

    # Verify that entropy on the new surface decreased (or at least changed)
    new_dist = pheromone_distribution(best)
    print(f"New distribution on {best}: {new_dist}")
    print(f"New entropy on {best}: {calculate_entropy(new_dist):.4f}")

    sys.exit(0)