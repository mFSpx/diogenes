# DARWIN HAMMER — match 3868, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_krampus_stick_hybrid_rbf_surrogate_m556_s0.py (gen4)
# parent_b: hybrid_rete_bandit_gate_hybrid_workshare_all_m43_s0.py (gen2)
# born: 2026-05-29T23:52:17Z

"""Hybrid Algorithm combining Pheromone Decay (Krampus Stickers) and Regret‑Weighted Allocation (Rete Bandit)

Parents:
- hybrid_hybrid_krampus_stick_hybrid_rbf_surrogate_m556_s0.py
- hybrid_rete_bandit_gate_hybrid_workshare_all_m43_s0.py

Mathematical Bridge:
The bridge is a Gaussian radial‑basis kernel that turns the scalar pheromone
levels (decayed signal values) associated with each work‑group into a
similarity matrix.  This matrix is then used to modulate the regret‑weighted
strategy of the bandit allocator.  In effect the allocation weights become
a blend of a pure regret‑minimization vector and a diffusion of pheromone
information through the Gaussian kernel, yielding a unified hybrid system.
"""

import math
import random
import sys
import pathlib
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Pheromone infrastructure (from Parent A)
# ----------------------------------------------------------------------
MAX_COMPONENT_TOKENS = 500

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(random.getrandbits(128))
        self.surface_key = surface_key          # e.g. work‑group name
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        self.created_at = datetime.utcnow()
        self.last_decay = self.created_at

    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.utcnow()


class PheromoneStore:
    """Simple in‑memory singleton store."""
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def all_entries(cls) -> List[PheromoneEntry]:
        return list(cls._entries.values())

    @classmethod
    def decay_all(cls) -> None:
        for e in cls._entries.values():
            e.apply_decay()


# ----------------------------------------------------------------------
# Gaussian radial‑basis function (core of Parent A)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF: exp( -r² / (2·ε²) )."""
    return math.exp(-(r ** 2) / (2.0 * epsilon ** 2))


# ----------------------------------------------------------------------
# Work‑group definitions (from Parent B)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


# ----------------------------------------------------------------------
# Helper: aggregate pheromone signal per group
# ----------------------------------------------------------------------
def aggregated_signal(group: str) -> float:
    """Sum of decayed signal values for a given group."""
    entries = PheromoneStore.get_by_surface(group)
    return sum(e.signal_value for e in entries)


# ----------------------------------------------------------------------
# Build Gaussian similarity matrix from pheromone levels
# ----------------------------------------------------------------------
def pheromone_similarity_matrix(groups: List[str],
                                epsilon: float = 1.0) -> np.ndarray:
    """
    Returns an N×N matrix S where
        S_ij = gaussian( |agg_i - agg_j| , epsilon )
    agg_k = aggregated pheromone signal for groups[k].
    """
    agg = np.array([aggregated_signal(g) for g in groups], dtype=float)
    diff = np.abs(agg[:, None] - agg[None, :])
    return np.vectorize(lambda r: gaussian(r, epsilon))(diff)


# ----------------------------------------------------------------------
# Regret‑weighted strategy blended with pheromone diffusion
# ----------------------------------------------------------------------
def compute_hybrid_weights(groups: List[str],
                           epsilon: float = 1.0,
                           alpha: float = 0.5) -> np.ndarray:
    """
    Hybrid weight computation:
      1. Start from a uniform regret‑minimization vector u.
      2. Diffuse u through the pheromone similarity matrix S.
      3. Blend: w = (1‑α)·u + α·(S·u)
      4. Normalize to sum to 1.
    Parameters
    ----------
    groups : list of group names
    epsilon : kernel width for the Gaussian RBF
    alpha : mixing coefficient (0 → pure regret, 1 → pure pheromone diffusion)
    """
    n = len(groups)
    if n == 0:
        return np.array([])

    # Uniform regret‑minimization vector
    u = np.full(n, 1.0 / n, dtype=float)

    # Pheromone similarity diffusion
    S = pheromone_similarity_matrix(groups, epsilon=epsilon)

    # Blended weights
    w = (1.0 - alpha) * u + alpha * (S @ u)

    # Normalisation
    w_sum = w.sum()
    if w_sum > 0:
        w /= w_sum
    else:
        w = u  # fallback to uniform

    return w


# ----------------------------------------------------------------------
# Allocation routine (core of Parent B) now using hybrid weights
# ----------------------------------------------------------------------
def compute_allocation(*,
                       total_units: float,
                       year: int,
                       month: int,
                       day: int,
                       epsilon: float = 1.0,
                       alpha: float = 0.5,
                       deterministic_target_pct: float = 90.0) -> dict[str, Any]:
    """
    Compute work‑unit allocation for the given date.
    The allocation weights are obtained from the hybrid
    regret‑plus‑pheromone mechanism.
    """
    # 1️⃣  Apply decay to all stored pheromones so they reflect current time.
    PheromoneStore.decay_all()

    # 2️⃣  Day of week (doomsday calendar)
    day_of_week = (date(year, month, day).weekday() + 1) % 7

    # 3️⃣  Hybrid weight vector
    groups = list(GROUPS)
    weights = compute_hybrid_weights(groups, epsilon=epsilon, alpha=alpha)

    # 4️⃣  Allocate units
    raw_units = total_units * weights
    total_allocated = raw_units.sum()
    # Guard against division by zero
    if total_allocated == 0:
        allocated = np.zeros_like(raw_units)
    else:
        allocated = raw_units / total_allocated * total_units

    # 5️⃣  Round to 6 decimal places for readability
    allocated = np.round(allocated.astype(float), 6)

    allocation = {
        "total_units": round(total_units, 6),
        "deterministic_target_pct": round(deterministic_target_pct, 6),
        "day_of_week": day_of_week,
        "lanes": [
            {"group": grp, "units": float(allocated[i])}
            for i, grp in enumerate(groups)
        ],
    }
    return allocation


def summarize_allocation(allocation: dict[str, Any]) -> None:
    """Pretty‑print the allocation dictionary."""
    print(f"Allocation Summary (Day {allocation['day_of_week']}):")
    print(f"  Total Units: {allocation['total_units']}")
    print(f"  Target %   : {allocation['deterministic_target_pct']}")
    for lane in allocation["lanes"]:
        print(f"  - {lane['group']}: {lane['units']} units")


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
def _populate_demo_pheromones() -> None:
    """Create a few random pheromone entries for each group."""
    random.seed(42)
    for group in GROUPS:
        # Create 2‑3 entries per group with varying half‑life
        for _ in range(random.randint(2, 3)):
            value = random.uniform(0.5, 5.0)
            half_life = random.choice([300, 600, 1800])  # seconds
            entry = PheromoneEntry(
                surface_key=group,
                signal_kind="demo",
                signal_value=value,
                half_life_seconds=half_life,
            )
            # Pretend the entry was created some seconds ago
            entry.created_at -= timedelta(seconds=random.randint(0, 1200))
            entry.last_decay = entry.created_at
            PheromoneStore.add(entry)


if __name__ == "__main__":
    # Populate demo pheromones
    _populate_demo_pheromones()

    # Compute allocation for today
    today = datetime.utcnow().date()
    alloc = compute_allocation(
        total_units=1000.0,
        year=today.year,
        month=today.month,
        day=today.day,
        epsilon=2.0,
        alpha=0.7,
        deterministic_target_pct=85.0,
    )
    summarize_allocation(alloc)