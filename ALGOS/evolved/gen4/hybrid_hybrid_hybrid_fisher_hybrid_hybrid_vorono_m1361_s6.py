# DARWIN HAMMER — match 1361, survivor 6
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s0.py (gen3)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s0.py (gen3)
# born: 2026-05-29T23:37:12Z

import math
import random
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Core from Parent A
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator

def hygiene_score(text: str, reference_text: str,
                  center: float, width: float) -> float:
    """Weighted similarity using Fisher score."""
    x = np.array([ord(c) for c in text], dtype=np.float64)
    y = np.array([ord(c) for c in reference_text], dtype=np.float64)
    similarity = ssim(x, y)
    fisher = fisher_score(similarity, center, width)
    return fisher * similarity

# ----------------------------------------------------------------------
# Core from Parent B
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)),
               key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi assignment of a list of points to the nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str   # e.g., "T1", "T2", "T3"

class ModelPool:
    """Memory‑constrained pool with simple eviction."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.model_usage_count: Dict[str, int] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model if constraints allow; otherwise raise."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model
        self.model_usage_count[model.name] = self.model_usage_count.get(model.name, 0) + 1

    def evict(self, name: str) -> None:
        """Remove a model from the pool."""
        self.loaded.pop(name, None)
        self.model_usage_count.pop(name, None)

    def load_with_eviction(self, model: ModelTier) -> None:
        """Load model, evicting least‑recently‑used models until enough space."""
        while model.ram_mb + self._used() > self.ram_ceiling_mb:
            if not self.loaded:
                raise Exception("Cannot free memory for model")
            # Evict the least recently used model
            lru_model_name = min(self.loaded, key=lambda m: self.model_usage_count.get(m, float('inf')))
            self.evict(lru_model_name)
        self.load(model)

# ----------------------------------------------------------------------
# Sparse Winner‑Take‑All (WTA)
# ----------------------------------------------------------------------
def sparse_wta(scores: Dict[str, float], winners: int = 1) -> Dict[str, float]:
    """
    Keep the top `winners` scores, zero out the rest.
    Returns a new dict with the same keys.
    """
    if winners <= 0:
        return {k: 0.0 for k in scores}
    # Sort by score descending
    sorted_items = sorted(scores.items(),
                          key=lambda item: item[1],
                          reverse=True)
    top_keys = {k for k, _ in sorted_items[:winners]}
    return {k: (v if k in top_keys else 0.0) for k, v in scores.items()}

# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def hybrid_feature_point(packet: Dict[str, Any],
                         reference_text: str,
                         center: float,
                         width: float) -> Point:
    """
    Compute a 2‑D feature point (similarity, fisher_weight) for a packet.
    """
    text = str(packet.get("text_surface") or
               packet.get("raw_command") or
               packet.get("text") or "")
    similarity = hygiene_score(text, reference_text, center, width) / max(1e-12, fisher_score(1.0, center, width))
    # The division rescales similarity back to raw SSIM (0‑1) while preserving the weighting.
    fisher = fisher_score(similarity, center, width)
    return (similarity, fisher)

def route_packet_voronoi(packet: Dict[str, Any],
                         reference_text: str,
                         seeds: List[Point],
                         center: float,
                         width: float,
                         pool: ModelPool,
                         model_catalog: List[ModelTier]) -> Tuple[int, ModelTier]:
    """
    Route a packet to a Voronoi region, select (or load) the region's model
    using sparse WTA on the reconstruction‑risk scores.
    Returns the region index and the chosen ModelTier.
    """
    point = hybrid_feature_point(packet, reference_text, center, width)
    region_idx = nearest(point, seeds)

    # Build a score map for all candidate models (one per region)
    scores: Dict[str, float] = {}
    for model in model_catalog:
        # Use the hybrid score as a proxy for reconstruction risk
        scores[model.name] = point[0] * point[1]  # similarity * fisher

    # Apply sparse WTA to keep only the best model for this region
    wta_scores = sparse_wta(scores, winners=1)

    # The winning model name should correspond to the region index
    winning_name = f"model_{region_idx}"
    winning_model = next(m for m in model_catalog if m.name == winning_name)

    # Ensure the winning model is resident, loading with eviction if necessary
    if not pool.is_loaded(winning_model.name):
        pool.load_with_eviction(winning_model)

    return region_idx, winning_model