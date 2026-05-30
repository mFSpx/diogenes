# DARWIN HAMMER — match 2821, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m2093_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m2075_s0.py (gen6)
# born: 2026-05-29T23:46:05Z

"""Hybrid Pheromone‑Voronoi Engine
Parent A: hybrid_hybrid_krampus_brain_hybrid_pheromone_inf_m37_s0.py
Parent B: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m2075_s0.py

Mathematical bridge
-------------------
The pheromone model (Parent A) provides a scalar signal that decays exponentially
with a half‑life.  The Voronoi partition (Parent B) yields a binary region matrix
R∈{0,1}^{k×n} that assigns each of n data points to one of k seeds.  We fuse the
two by using the region weights to scale the initial pheromone value of a new
entry:

    σ_i = G(‖q‑s_i‖)·R_{i,q}

where G is the Gaussian radial‑basis function and R_{i,q} is the assignment of
the query point q to seed i (0 or 1).  The collection of σ_i values is then
interpreted as a probability distribution; its Shannon entropy H modulates the
global decay factor applied to every stored pheromone entry:

    decay = 0.5^{ (Δt / τ)·(1+H) }

Thus Voronoi geometry controls signal magnitude, while the entropy of the
resulting signal distribution controls the speed of pheromone ageing.
"""

import uuid
import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Pheromone core (Parent A)
# ----------------------------------------------------------------------
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

    def decay_factor(self, entropy: float = 0.0) -> float:
        """Decay factor that incorporates entropy (0 ≤ entropy ≤ log2(k))."""
        if self.half_life_seconds <= 0:
            return 0.0
        # Entropy amplifies the effective decay rate
        effective_half_life = self.half_life_seconds / (1.0 + entropy)
        return 0.5 ** (self.age_seconds() / effective_half_life)

    def apply_decay(self, entropy: float = 0.0) -> None:
        factor = self.decay_factor(entropy)
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory singleton store for pheromone entries."""
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def all(cls) -> List[PheromoneEntry]:
        return list(cls._entries.values())

    @classmethod
    def decay_all(cls, entropy: float) -> None:
        for e in cls._entries.values():
            e.apply_decay(entropy)


# ----------------------------------------------------------------------
# Voronoi core (Parent B)
# ----------------------------------------------------------------------
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def nearest_seed(point: np.ndarray, seeds: np.ndarray) -> int:
    """Index of the nearest seed to *point*."""
    if seeds.size == 0:
        raise ValueError("seed array is empty")
    return int(np.argmin(np.linalg.norm(seeds - point, axis=1)))


def voronoi_assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Return a binary region matrix R of shape (k, n) where k = len(seeds)
    and n = len(points).  R[i, j] == 1 iff point j belongs to seed i.
    """
    k = seeds.shape[0]
    n = points.shape[0]
    R = np.zeros((k, n), dtype=int)
    for j, p in enumerate(points):
        i = nearest_seed(p, seeds)
        R[i, j] = 1
    return R


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Radial‑basis function G(r) = exp(-(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Entropy utilities
# ----------------------------------------------------------------------
def shannon_entropy(values: np.ndarray) -> float:
    """
    Compute Shannon entropy of a non‑negative vector.
    The vector is first normalised to a probability distribution.
    """
    if values.size == 0:
        return 0.0
    total = np.sum(values)
    if total == 0:
        return 0.0
    p = values / total
    # avoid log(0) by masking zeros
    mask = p > 0
    return -np.sum(p[mask] * np.log2(p[mask]))


# ----------------------------------------------------------------------
# Hybrid operations (the new fused algorithm)
# ----------------------------------------------------------------------
def hybrid_initialize_pheromones(
    points: np.ndarray,
    seeds: np.ndarray,
    query_point: np.ndarray,
    half_life: int = 60,
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Create pheromone entries whose initial signal values are the Gaussian‑weighted
    Voronoi assignments of *query_point* to each seed.

    Returns
    -------
    signals : np.ndarray
        Array of the raw (pre‑decay) signal values σ_i.
    entropy : float
        Shannon entropy of the signal distribution (used later for decay scaling).
    """
    # 1. Voronoi region matrix for the whole dataset
    region_matrix = voronoi_assign(points, seeds)          # shape (k, n)

    # 2. Determine which seed owns the *query_point*
    q_seed_idx = nearest_seed(query_point, seeds)

    # 3. Compute Gaussian weight for each seed relative to the query point
    distances = np.linalg.norm(seeds - query_point, axis=1)  # shape (k,)
    gauss_weights = np.vectorize(gaussian_rbf)(distances, epsilon)  # shape (k,)

    # 4. Combine: region assignment of the query point (one‑hot) multiplies the weight
    assignment_one_hot = np.zeros(seeds.shape[0])
    assignment_one_hot[q_seed_idx] = 1.0
    raw_signals = gauss_weights * assignment_one_hot  # only the owning seed gets a signal

    # 5. Create a pheromone entry for each non‑zero signal
    for i, sig in enumerate(raw_signals):
        if sig > 0:
            entry = PheromoneEntry(
                surface_key=f"seed_{i}",
                signal_kind="voronoi_gauss",
                signal_value=sig,
                half_life_seconds=half_life,
            )
            PheromoneStore.add(entry)

    # 6. Compute entropy of the signal vector (used as a global factor)
    entropy = shannon_entropy(raw_signals)
    return raw_signals, entropy


def hybrid_decay_cycle(entropy: float) -> None:
    """
    Apply a decay step to **all** stored pheromone entries, using the entropy‑augmented
    decay factor defined in :class:`PheromoneEntry`.
    """
    PheromoneStore.decay_all(entropy)


def hybrid_aggregate_signal(query_point: np.ndarray, seeds: np.ndarray) -> float:
    """
    Produce a single aggregated signal for *query_point* by:
    1. Selecting the pheromone entries whose surface_key matches the nearest seed.
    2. Summing their (already decayed) signal values.
    3. Scaling by a Gaussian of the distance to the seed (adds a smooth spatial term).

    This demonstrates the closed‑loop interaction between Voronoi geometry,
    pheromone decay, and signal aggregation.
    """
    nearest_idx = nearest_seed(query_point, seeds)
    key = f"seed_{nearest_idx}"
    relevant = [
        e for e in PheromoneStore.all()
        if e.surface_key == key
    ]
    if not relevant:
        return 0.0
    # Sum the decayed signals
    total_signal = sum(e.signal_value for e in relevant)

    # Additional spatial scaling (optional but illustrates the bridge)
    dist = euclidean_distance(query_point, seeds[nearest_idx])
    spatial_factor = gaussian_rbf(dist)
    return total_signal * spatial_factor


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Synthetic dataset
    np.random.seed(42)
    points = np.random.rand(100, 2) * 10.0          # 100 points in 2‑D
    seeds = np.random.rand(5, 2) * 10.0            # 5 Voronoi seeds
    query = np.array([5.0, 5.0])

    # 2. Initialise pheromones based on the query point
    signals, entropy = hybrid_initialize_pheromones(
        points=points,
        seeds=seeds,
        query_point=query,
        half_life=30,
        epsilon=0.5,
    )
    print(f"Initial raw signals: {signals}")
    print(f"Entropy of signal distribution: {entropy:.4f}")

    # 3. Perform a decay cycle (simulating passage of time)
    hybrid_decay_cycle(entropy)

    # 4. Aggregate a signal for the same query (could be a different point)
    agg = hybrid_aggregate_signal(query_point=query, seeds=seeds)
    print(f"Aggregated signal after decay: {agg:.6f}")

    # 5. Verify that the store still holds entries
    print(f"Number of pheromone entries stored: {len(PheromoneStore.all())}")