# DARWIN HAMMER — match 3537, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2529_s0.py (gen6)
# parent_b: hybrid_hybrid_jepa_energy_h_hybrid_hybrid_infota_m141_s1.py (gen4)
# born: 2026-05-29T23:50:34Z

"""
Hybrid Algorithm: Fusion of
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2529_s0.py (Count‑Min sketch,
  Fisher score, Gaussian beam)
- hybrid_hybrid_jepa_energy_h_hybrid_hybrid_infota_m141_s1.py (MinHash‑based
  representative selection, drag‑equation cost, ModelPool energy accounting)

Mathematical Bridge
------------------
The bridge is the *uncertainty estimate* derived from a Count‑Min sketch.
This scalar `U ∈ [0, ∞)` is used to *modulate* two distinct cost
components:

1. **Fisher‑adjusted beam score** `F(θ) = fisher_score(θ, μ, σ)`.
   The hybrid score becomes `F̂ = F * (1 + U)`, i.e. sketch uncertainty
   inflates the Fisher information penalty.

2. **Drag‑based selection cost** `D = ½·ρ·C_d·A·v²`.
   When selecting a representative element from a cluster via a MinHash
   signature, the effective cost is `C = D * (1 + U)`.  The element with the
   minimal `C` is chosen, thus the sketch‑derived uncertainty directly
   influences the cluster‑wise admission model.

Both modulated quantities are accumulated in a `ModelPool` instance, whose
internal energy counter records rewards/penalties for loading/evicting
models.  The final hybrid metric for a candidate model `m` is


HybridMetric(m) = (F̂(m) + C(m)) - pool.energy


where `F̂` uses the model’s hyper‑parameters as a Gaussian beam and `C`
uses the drag equation with parameters supplied by the caller.

The code below implements the three core functions that embody this
fusion, plus a small smoke test.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility functions from Parent A
# ----------------------------------------------------------------------
def _hash(item: str, seed: int) -> int:
    """Deterministic integer hash for a given seed."""
    h = hashlib.blake2b(digest_size=8)
    h.update(item.encode("utf-8"))
    h.update(seed.to_bytes(2, "little"))
    return int.from_bytes(h.digest(), "little")


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def count_min_sketch(items: Iterable[str], width: int = 128, depth: int = 5) -> List[List[int]]:
    """Construct a Count‑Min sketch from an iterable of hashable strings."""
    sketch = [[0] * width for _ in range(depth)]
    for item in items:
        for i in range(depth):
            idx = _hash(item, i) % width
            sketch[i][idx] += 1
    return sketch


def uncertainty_estimate(sketch: List[List[int]]) -> float:
    """
    Simple uncertainty proxy: mean of the per‑row minima.
    Lower counts indicate higher uncertainty; we invert and shift.
    """
    mins = [min(row) for row in sketch if row]
    if not mins:
        return 0.0
    mean_min = sum(mins) / len(mins)
    # Transform to a positive scaling factor; add 1 to avoid zero.
    return 1.0 / (mean_min + 1.0)


# ----------------------------------------------------------------------
# MinHash & drag‑equation utilities (Parent B)
# ----------------------------------------------------------------------
def minhash_signature(item: str, num_perm: int = 64) -> Tuple[int, ...]:
    """Very lightweight MinHash: hash the item with a sequence of seeds."""
    return tuple(_hash(item, seed) for seed in range(num_perm))


def drag_cost(velocity: float, area: float, density: float = 1.225, drag_coeff: float = 0.47) -> float:
    """
    Classical drag equation: ½·ρ·C_d·A·v²
    Returns kinetic energy dissipated by drag.
    """
    return 0.5 * density * drag_coeff * area * velocity ** 2


# ----------------------------------------------------------------------
# ModelPool from Parent B (energy accounting)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str


class ModelPool:
    """
    Simple RAM‑bounded pool with energy accounting.
    Energy is a scalar that can be added to or subtracted from.
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0

    @property
    def energy(self) -> float:
        return self._energy

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def add_model(self, model: ModelTier) -> None:
        """Add a model, applying penalties for tier conflicts or RAM overflow."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10  # conflict penalty
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy += 1e6  # RAM penalty
        self.loaded[model.name] = model

    def load(self, model: ModelTier) -> None:
        """Load a model with a reward."""
        self._energy -= 1e4
        self.add_model(model)

    def load_with_eviction(self, model: ModelTier) -> None:
        """Load a model, evicting the largest‑RAM model(s) if needed."""
        self._energy -= 1e3
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            evict_name = max(self.loaded, key=lambda k: self.loaded[k].ram_mb)
            evicted = self.loaded.pop(evict_name)
            self._energy += 1e2  # mild eviction penalty
        self.add_model(model)


# ----------------------------------------------------------------------
# Hybrid core functions (the required three+)
# ----------------------------------------------------------------------
def hybrid_fisher_with_uncertainty(
    theta: float, center: float, width: float, sketch: List[List[int]]
) -> float:
    """
    Compute Fisher score scaled by sketch‑derived uncertainty.
    Returns F̂ = fisher_score * (1 + U).
    """
    base = fisher_score(theta, center, width)
    U = uncertainty_estimate(sketch)
    return base * (1.0 + U)


def select_cluster_representative(
    cluster_items: List[str],
    sketch: List[List[int]],
    velocity: float,
    area: float,
    density: float = 1.225,
    drag_coeff: float = 0.47,
) -> str:
    """
    Choose a representative element from `cluster_items` by:
    1. Computing a MinHash signature for each item.
    2. Evaluating drag cost D.
    3. Scaling D by sketch uncertainty U → C = D * (1 + U).
    4. Picking the item with minimal C; ties broken by smallest MinHash value.
    """
    if not cluster_items:
        raise ValueError("cluster_items must be non‑empty")

    U = uncertainty_estimate(sketch)
    base_drag = drag_cost(velocity, area, density, drag_coeff)
    scaled_drag = base_drag * (1.0 + U)

    best_item = None
    best_score = math.inf

    for item in cluster_items:
        # MinHash as a deterministic tie‑breaker
        signature = minhash_signature(item, num_perm=4)  # small for speed
        tie_breaker = sum(signature)  # deterministic scalar
        total_cost = scaled_drag + tie_breaker * 1e-9  # negligible perturbation
        if total_cost < best_score:
            best_score = total_cost
            best_item = item

    return best_item  # type: ignore


def hybrid_metric_for_model(
    model: ModelTier,
    theta: float,
    center: float,
    width: float,
    sketch: List[List[int]],
    velocity: float,
    area: float,
    pool: ModelPool,
) -> float:
    """
    Compute the unified hybrid metric for a given `model`.

    HybridMetric(m) = (F̂ + C) - pool.energy
      where
        F̂ = fisher_score(θ) scaled by sketch uncertainty,
        C  = drag_cost scaled by the same uncertainty,
        pool.energy accumulates rewards/penalties from model loading.
    """
    fisher = hybrid_fisher_with_uncertainty(theta, center, width, sketch)
    U = uncertainty_estimate(sketch)
    drag = drag_cost(velocity, area) * (1.0 + U)
    return (fisher + drag) - pool.energy


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Build a toy dataset and its sketch
    items = [f"item_{i}_{random.choice('abcde')}" for i in range(200)]
    sketch = count_min_sketch(items)

    # 2. Define simple clusters by first character after the last underscore
    clusters: Dict[str, List[str]] = {}
    for it in items:
        key = it.split("_")[-1]  # character a‑e
        clusters.setdefault(key, []).append(it)

    # 3. Choose representative per cluster using the hybrid selector
    reps = {}
    for key, grp in clusters.items():
        reps[key] = select_cluster_representative(
            grp,
            sketch,
            velocity=12.0,
            area=0.05,
        )
    print("Representatives per cluster:", reps)

    # 4. Initialise a ModelPool and load a few models
    pool = ModelPool(ram_ceiling_mb=4000)
    models = [
        ModelTier(name="M1", ram_mb=500, tier="T1"),
        ModelTier(name="M2", ram_mb=1500, tier="T2"),
        ModelTier(name="M3", ram_mb=2500, tier="T3"),
    ]
    for m in models:
        pool.load_with_eviction(m)

    # 5. Compute hybrid metric for each model
    theta = 0.7
    center = 0.5
    width = 0.1
    velocity = 15.0
    area = 0.07

    for m in models:
        metric = hybrid_metric_for_model(
            m,
            theta,
            center,
            width,
            sketch,
            velocity,
            area,
            pool,
        )
        print(f"Hybrid metric for {m.name}: {metric:.6f}")

    # 6. Final energy check (should be a finite number)
    print("Final ModelPool energy:", pool.energy)