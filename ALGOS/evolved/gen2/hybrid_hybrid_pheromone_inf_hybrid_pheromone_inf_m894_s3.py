# DARWIN HAMMER — match 894, survivor 3
# gen: 2
# parent_a: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# parent_b: hybrid_pheromone_infotaxis_m3_s1.py (gen1)
# born: 2026-05-29T23:31:40Z

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Pheromone infrastructure (parent A)
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

    @classmethod
    def all_surfaces(cls) -> List[str]:
        return list({e.surface_key for e in cls._entries.values()})


# ----------------------------------------------------------------------
# Entropy helpers (parent B) – now vectorised with numpy
# ----------------------------------------------------------------------
def calculate_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector (base e)."""
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))


def expected_entropy(p_hit: float, hit_state: np.ndarray, miss_state: np.ndarray) -> float:
    """Weighted average entropy after a binary observation."""
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)


def best_action(actions: Dict[str, Tuple[float, np.ndarray, np.ndarray]]) -> str:
    """Return the key with minimal expected entropy."""
    if not actions:
        raise ValueError("no actions to evaluate")
    return min(actions, key=lambda a: expected_entropy(*actions[a]))


# ----------------------------------------------------------------------
# Deepened hybrid operations
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
    # Decay all entries of this surface first
    PheromoneStore.decay_surface(surface_key)

    # Look for a recent entry of the same kind
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


def global_pheromone_distribution(surface_keys: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute a probability distribution over the supplied surface keys.
    Decay is applied lazily; if total mass is zero a uniform Dirichlet prior is used.
    Returns (prob_vector, ordered_surface_keys).
    """
    masses = []
    for key in surface_keys:
        mass = PheromoneStore.total_signal(key)
        masses.append(mass)

    total = sum(masses)
    if total <= 0.0:
        # Uniform prior over the known surfaces
        probs = np.full(len(surface_keys), 1.0 / len(surface_keys))
    else:
        probs = np.array(masses) / total
    return probs, surface_keys


def simulate_distribution_update(
    probs: np.ndarray,
    surfaces: List[str],
    target_surface: str,
    delta: float,
) -> np.ndarray:
    """
    Return a new probability vector that results from adding `delta` mass to
    `target_surface` without mutating the underlying store.
    """
    if target_surface not in surfaces:
        # Extend the vector with a new surface
        probs = np.append(probs, 0.0)
        surfaces = surfaces + [target_surface]

    idx = surfaces.index(target_surface)
    new_mass = probs[idx] + delta
    total = probs.sum() + delta
    new_probs = probs.copy()
    new_probs[idx] = new_mass / total
    # Renormalise the rest
    if total > 0:
        new_probs = new_probs / total
    return new_probs


def select_best_move(
    current_surface: str,
    neighbor_surfaces: List[str],
    hit_increment: float = 1.0,
) -> str:
    """
    Choose the neighbour that minimises the expected entropy of the belief
    distribution after a hypothetical observation.
    The hit probability is taken as the current belief that the source lies on
    the neighbour (i.e. the current probability mass of that neighbour).
    """
    # Build the full set of surfaces the agent is aware of
    all_surfaces = list({current_surface, *neighbor_surfaces})
    probs, ordered = global_pheromone_distribution(all_surfaces)

    # Map surface → index for quick lookup
    surface_to_idx = {s: i for i, s in enumerate(ordered)}

    actions: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}

    for nb in neighbor_surfaces:
        # Current belief that the source is on the neighbour
        p_hit = probs[surface_to_idx[nb]] if nb in surface_to_idx else 0.0

        # Simulate hit: add `hit_increment` to neighbour
        hit_state = simulate_distribution_update(probs, ordered, nb, hit_increment)

        # Simulate miss: distribution unchanged
        miss_state = probs

        actions[nb] = (p_hit, hit_state, miss_state)

    return best_action(actions)


# ----------------------------------------------------------------------
# Smoke test (unchanged interface)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic scenario
    import random

    random.seed(0)

    # Initialise pheromones on three surfaces
    add_pheromone_signal("A", "explore", 5.0, half_life_seconds=120)
    add_pheromone_signal("A", "explore", 3.0, half_life_seconds=120)
    add_pheromone_signal("B", "explore", 2.0, half_life_seconds=120)
    add_pheromone_signal("C", "explore", 1.0, half_life_seconds=120)

    # Agent at A, neighbours B and C
    best = select_best_move("A", ["B", "C"])
    print(f"Best move from A: {best}")