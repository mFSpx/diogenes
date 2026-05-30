# DARWIN HAMMER — match 228, survivor 2
# gen: 3
# parent_a: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s0.py (gen2)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m62_s0.py (gen2)
# born: 2026-05-29T23:27:41Z

"""
Hybrid Voronoi‑WTA Privacy‑Model Pool

This module fuses the core topologies of two parent algorithms:

* **Parent A** – Voronoi partitioning combined with a morphological
  description of hybrid endpoint circuit breakers.
* **Parent B** – Sparse Winner‑Take‑All (WTA) tagging together with a
  privacy‑aware model‑pool manager.

The mathematical bridge is the **region‑importance vector** produced by
Voronoi distances.  The vector is expanded into a high‑dimensional
space (the “expansion” step of the sparse‑WTA parent) and a sparse
top‑k mask selects the most important regions.  Those selected regions
drive the loading/eviction decisions of the privacy‑model pool, while
the morphology of each model supplies a recovery‑priority factor that
modulates the final loading score.

The result is a single unified system that:
1. Partitions a point cloud into Voronoi cells.
2. Scores each cell by distance‑based importance and morphological
   priority.
3. Expands the scores, applies sparse WTA, and loads the corresponding
   models into a RAM‑bounded pool respecting privacy constraints.
"""

import math
import random
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Iterable
import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Parent A – Voronoi & Morphology
# ----------------------------------------------------------------------
def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest_seed(point: Point, seeds: List[Point]) -> int:
    """Return index of the nearest seed (ties broken by lower index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean(point, seeds[i]), i))

def assign_voronoi(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the index of its nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        idx = nearest_seed(p, seeds)
        regions[idx].append(p)
    return regions

@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width:  float
    height: float
    mass:   float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def density(self) -> float:
        """Mass per unit volume – used as a recovery‑priority factor."""
        vol = self.volume
        return self.mass / vol if vol != 0 else 0.0

# ----------------------------------------------------------------------
# Parent B – Sparse WTA & Privacy Model Pool
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model that can be loaded into the pool."""
    name:   str
    ram_mb: int
    tier:   str               # e.g. "T1", "T2", "T3"
    morphology: Morphology    # attach a morphology for priority calculations

class ModelPool:
    """RAM‑bounded pool with simple privacy‑aware eviction."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model if constraints allow; raise otherwise."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with any T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """Evict oldest entries until the new model fits."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # FIFO eviction: pop the first inserted key
            oldest_key = next(iter(self.loaded))
            self.loaded.pop(oldest_key)
        self.load(model)

    def __repr__(self) -> str:
        return f"<ModelPool used={self._used()} / {self.ram_ceiling_mb} MB, models={list(self.loaded)}>"


def expand(values: List[float], m: int, salt: str = '') -> List[float]:
    """
    Sparse random projection used in the original sparse_wta parent.
    Each input value contributes to three random positions with random sign.
    """
    if m <= 0:
        raise ValueError("target dimension m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f'{salt}:{i}:{r}'.encode()).digest()
            j = int.from_bytes(h[:8], 'big') % m
            sign = 1.0 if (h[8] & 1) else -1.0
            out[j] += sign * v
    return out

def top_k_mask(values: List[float], k: int) -> List[int]:
    """
    Return a binary mask (list of 0/1) where exactly k positions with the
    largest values are set to 1 (ties resolved by index order).
    """
    n = len(values)
    k = max(0, min(k, n))
    # indices of the k largest values
    winners = {i for i, _ in sorted(enumerate(values), key=lambda iv: (-iv[1], iv[0]))[:k]}
    return [1 if i in winners else 0 for i in range(n)]

# ----------------------------------------------------------------------
# Hybrid Functions (the required three+ functions)
# ----------------------------------------------------------------------
def region_importance(seeds: List[Point], points: List[Point]) -> List[float]:
    """
    Compute an importance score for each seed based on the average distance
    of points assigned to its Voronoi cell.  Larger average distance → higher
    importance (the cell is more “stretched”).
    """
    regions = assign_voronoi(points, seeds)
    scores: List[float] = []
    for idx in range(len(seeds)):
        pts = regions[idx]
        if not pts:
            scores.append(0.0)
            continue
        avg_dist = sum(euclidean(p, seeds[idx]) for p in pts) / len(pts)
        scores.append(avg_dist)
    return scores

def compute_model_scores(models: List[ModelTier],
                         seed_importance: List[float],
                         morphology_weight: float = 0.5) -> List[float]:
    """
    Fuse Voronoi importance with morphological priority.
    For each model we pick a seed (by modulo) and combine:
        score = (1‑w) * seed_importance + w * density
    """
    n = len(models)
    scores = []
    for i, model in enumerate(models):
        seed_idx = i % len(seed_importance)
        imp = seed_importance[seed_idx]
        dens = model.morphology.density
        score = (1 - morphology_weight) * imp + morphology_weight * dens
        scores.append(score)
    return scores

def allocate_models(points: List[Point],
                    seeds: List[Point],
                    models: List[ModelTier],
                    pool: ModelPool,
                    expansion_dim: int = 64,
                    wta_k: int = 3) -> None:
    """
    End‑to‑end hybrid allocation:

    1. Compute Voronoi‑based importance per seed.
    2. Fuse with model morphologies → raw scores.
    3. Expand the score vector to a higher dimension.
    4. Apply sparse WTA (top‑k mask) to obtain a binary selection.
    5. Load the selected models into the pool, evicting if necessary.
    """
    # Step 1
    seed_imp = region_importance(seeds, points)

    # Step 2
    raw_scores = compute_model_scores(models, seed_imp)

    # Step 3 – random projection
    expanded = expand(raw_scores, expansion_dim, salt='hybrid')

    # Step 4 – sparse WTA
    mask = top_k_mask(expanded, wta_k)

    # Step 5 – map mask back to original models (simple modulo)
    selected_indices = [i for i, flag in enumerate(mask) if flag]
    # Reduce to at most len(models) unique indices
    selected_models = {models[i % len(models)] for i in selected_indices}

    for model in selected_models:
        try:
            pool.load(model)
        except Exception:
            # If loading fails (e.g., RAM limit), try with eviction
            pool.load_with_eviction(model)

def sample_morphology() -> Morphology:
    """Utility to generate a random morphology for demo purposes."""
    return Morphology(
        length=random.uniform(0.5, 2.0),
        width =random.uniform(0.5, 2.0),
        height=random.uniform(0.5, 2.0),
        mass  =random.uniform(1.0, 10.0)
    )

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate random seeds and points
    random.seed(42)
    seeds = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(5)]
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(200)]

    # 2. Create a list of dummy models with attached morphologies
    models = [
        ModelTier(name=f"model_{i}", ram_mb=random.randint(200, 800),
                  tier=random.choice(["T1", "T2", "T3"]),
                  morphology=sample_morphology())
        for i in range(12)
    ]

    # 3. Initialise a model pool with a modest RAM ceiling
    pool = ModelPool(ram_ceiling_mb=3000)

    # 4. Run the hybrid allocation
    allocate_models(points, seeds, models, pool,
                    expansion_dim=128, wta_k=5)

    # 5. Print resulting pool state
    print(pool)
    for name, mdl in pool.loaded.items():
        print(f"{name}: RAM={mdl.ram_mb}MB, tier={mdl.tier}, density={mdl.morphology.density:.3f}")