# DARWIN HAMMER — match 4148, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_krampu_m973_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m2272_s1.py (gen5)
# born: 2026-05-29T23:53:45Z

"""Hybrid Voronoi‑Curvature‑Pheromone Engine
Parents:
- hybrid_hybrid_geometric_pro_hybrid_hybrid_krampu_m973_s0.py (geometric product, Voronoi, pheromone handling)
- hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m2272_s1.py (feature‑curvature matrix, deterministic RNG, pheromone dynamics)

Mathematical bridge:
The curvature matrix **C** derived from a deterministic feature vector (Parent B) is used as a *metric tensor* inside the Clifford‑algebra‑style geometric product of Parent A.  
The squared geometric‑product distance between two points **x** and **y** becomes  

    d²_C(x, y) = (x‑y)ᵀ C (x‑y)  

instead of the Euclidean norm.  
Thus the Voronoi partitioning (Parent A) operates on a curved space defined by the feature‑curvature matrix (Parent B).  
Pheromone updates then weight the reward by the same curvature matrix, closing the loop between geometry and dynamics.

The module provides three high‑level hybrid functions that illustrate this integration.
"""

import math
import random
import sys
import pathlib
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Dict, List, Tuple, Any, Iterable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Blade arithmetic (from Parent A) – kept for completeness, not used directly
# ----------------------------------------------------------------------
def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Very small stub – full Clifford algebra not required for the hybrid."""
    def __init__(self, components: Dict[frozenset, float]):
        self.components = components

# ----------------------------------------------------------------------
# Pheromone entry (shared by both parents)
# ----------------------------------------------------------------------
@dataclass
class PheromoneEntry:
    uuid: uuid.UUID
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: int
    created_at: datetime
    last_decay: datetime

    @staticmethod
    def create(surface_key: str,
               signal_kind: str,
               signal_value: float,
               half_life_seconds: int) -> "PheromoneEntry":
        now = datetime.utcnow()
        return PheromoneEntry(
            uuid=uuid.uuid4(),
            surface_key=surface_key,
            signal_kind=signal_kind,
            signal_value=signal_value,
            half_life_seconds=half_life_seconds,
            created_at=now,
            last_decay=now,
        )

    def decay(self, now: datetime = None) -> None:
        """Exponential decay based on half‑life."""
        now = now or datetime.utcnow()
        elapsed = (now - self.last_decay).total_seconds()
        if elapsed <= 0:
            return
        decay_factor = 0.5 ** (elapsed / self.half_life_seconds)
        self.signal_value *= decay_factor
        self.last_decay = now


# ----------------------------------------------------------------------
# Feature extraction & curvature matrix (Parent B)
# ----------------------------------------------------------------------
def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from a stable SHA‑256 hash of *text*."""
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def extract_features(text: str, d: int = 24) -> np.ndarray:
    """Deterministic d‑dimensional feature vector from *text*."""
    rng = _rng_from_text(text)
    return np.array([rng.random() for _ in range(d)], dtype=np.float64)


def compute_curvature_matrix(feature_vector: np.ndarray) -> np.ndarray:
    """
    Normalise the feature vector and build a rank‑1 curvature matrix
    C = v vᵀ, where v = f/‖f‖.
    """
    norm = np.linalg.norm(feature_vector)
    if norm == 0:
        raise ValueError("Zero‑norm feature vector")
    v = feature_vector / norm
    return np.outer(v, v)


# ----------------------------------------------------------------------
# Geometric‑product distance with curvature metric (Hybrid core)
# ----------------------------------------------------------------------
def curved_geometric_distance_sq(p: np.ndarray,
                                 q: np.ndarray,
                                 metric: np.ndarray) -> float:
    """
    Squared distance using the curvature matrix as a metric tensor:
        d² = (p‑q)ᵀ C (p‑q)
    """
    diff = p - q
    return float(diff.T @ metric @ diff)


def assign_to_voronoi(seeds: np.ndarray,
                      points: np.ndarray,
                      metric: np.ndarray) -> List[int]:
    """
    Assign each point to the index of the nearest seed using the
    curved geometric distance. Returns a list of seed indices.
    """
    assignments = []
    for pt in points:
        dists = [curved_geometric_distance_sq(pt, seed, metric) for seed in seeds]
        assignments.append(int(np.argmin(dists)))
    return assignments


# ----------------------------------------------------------------------
# Pheromone dynamics weighted by curvature (Hybrid core)
# ----------------------------------------------------------------------
def _weekday_index(year: int, month: int, day: int) -> int:
    """Monday→1 … Sunday→0 (mod 7)."""
    return (date(year, month, day).weekday() + 1) % 7


def update_pheromone(entry: PheromoneEntry,
                     reward: float,
                     curvature: np.ndarray,
                     weekday: int,
                     now: datetime = None) -> None:
    """
    Decay the entry, then add a curvature‑weighted reward.
    The weight is w = trace(C) / 𝑑, i.e. the average eigenvalue,
    providing a scalar that reflects overall curvature magnitude.
    """
    entry.decay(now)
    avg_curvature = float(np.trace(curvature) / curvature.shape[0])
    # weekday modulates the reward slightly (adds a deterministic seasonal factor)
    weekday_factor = 1.0 + (weekday - 3) * 0.02  # centre at Wednesday (3)
    weighted_reward = reward * avg_curvature * weekday_factor
    entry.signal_value += weighted_reward


def pheromone_entropy(entries: List[PheromoneEntry]) -> float:
    """
    Shannon entropy of the normalized pheromone signal values.
    """
    values = np.array([e.signal_value for e in entries], dtype=np.float64)
    total = values.sum()
    if total == 0:
        return 0.0
    probs = values / total
    # Avoid log(0) by masking zero probabilities
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log2(probs[mask])))


# ----------------------------------------------------------------------
# Public hybrid API (three demonstrative functions)
# ----------------------------------------------------------------------
def hybrid_feature_curvature(text: str) -> np.ndarray:
    """
    Combine feature extraction and curvature matrix computation.
    Returns the curvature matrix C for the given *text*.
    """
    fv = extract_features(text)
    return compute_curvature_matrix(fv)


def hybrid_voronoi_allocation(seeds: np.ndarray,
                              points: np.ndarray,
                              curvature: np.ndarray) -> List[int]:
    """
    Perform Voronoi allocation on *points* using *seeds* and the curvature‑
    based metric. Returns a list of seed indices (one per point).
    """
    return assign_to_voronoi(seeds, points, curvature)


def hybrid_pheromone_step(entries: List[PheromoneEntry],
                          rewards: Sequence[float],
                          curvature: np.ndarray,
                          ref_date: Tuple[int, int, int]) -> None:
    """
    Update a collection of pheromone entries in place.
    *rewards* must be the same length as *entries*.
    *ref_date* is a (year, month, day) tuple used to compute the weekday index.
    """
    if len(entries) != len(rewards):
        raise ValueError("entries and rewards must have the same length")
    weekday = _weekday_index(*ref_date)
    now = datetime.utcnow()
    for entry, rew in zip(entries, rewards):
        update_pheromone(entry, rew, curvature, weekday, now=now)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    rng = np.random.default_rng(42)

    # 1) Build curvature matrix from a sample text
    sample_text = "The quick brown fox jumps over the lazy dog."
    C = hybrid_feature_curvature(sample_text)

    # 2) Generate random seeds (3 seeds) and random points (15 points) in 24‑D space
    dim = 24
    seeds = rng.random((3, dim))
    points = rng.random((15, dim))

    # 3) Allocate points to Voronoi cells using the curved metric
    allocations = hybrid_voronoi_allocation(seeds, points, C)
    print("Voronoi allocations:", allocations)

    # 4) Create pheromone entries, one per seed
    entries = [
        PheromoneEntry.create(
            surface_key=f"seed_{i}",
            signal_kind="allocation",
            signal_value=1.0,
            half_life_seconds=300
        )
        for i in range(3)
    ]

    # 5) Produce synthetic rewards proportional to how many points each seed received
    reward_counts = np.bincount(allocations, minlength=3)
    rewards = reward_counts.astype(float) * 0.5  # scale factor

    # 6) Update pheromones using the curvature matrix and today's date
    today = date.today()
    hybrid_pheromone_step(entries, rewards, C, (today.year, today.month, today.day))

    # 7) Report pheromone values and entropy
    for e in entries:
        print(f"Pheromone {e.surface_key}: {e.signal_value:.4f}")

    print("Pheromone entropy:", pheromone_entropy(entries))