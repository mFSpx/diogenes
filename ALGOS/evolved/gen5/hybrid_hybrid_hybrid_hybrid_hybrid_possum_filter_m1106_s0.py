# DARWIN HAMMER — match 1106, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hdc_se_m172_s0.py (gen4)
# parent_b: hybrid_possum_filter_hybrid_privacy_model_m53_s1.py (gen2)
# born: 2026-05-29T23:32:45Z

"""
Hybrid algorithm fusing 
- **hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hdc_se_m172_s0.py** 
  (JEPA-HDC with Fisher score) and 
- **hybrid_possum_filter_hybrid_privacy_model_m53_s1.py** 
  (Possum filter with Hybrid privacy model).

The mathematical bridge lies in using the Fisher score from JEPA-HDC 
as a distance metric to filter models based on their resource usage and 
privacy risk in the Hybrid privacy model. The Fisher score quantifies 
the informativeness of a timestamp candidate, which is used as a 
distance threshold to limit the selection of models.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Dict, Set, Tuple

# ---------------------------------------------------------------------------
# Algorithm A – Fisher-based date extraction (from JEPA-HDC)
# ---------------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Hyperdimensional primitives and Possum filter (from Hybrid privacy model)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g., "T1", "T2", "T3"

def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

def keep_candidate(candidate: Entity, selected: list[Entity], delta_m: float) -> bool:
    for existing in selected:
        same_kind = signature(candidate) == signature(existing) or candidate.category.strip().lower() == existing.category.strip().lower()
        if same_kind and haversine_m((candidate.lat, candidate.lon), (existing.lat, existing.lon)) <= delta_m:
            return False
    return True

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def filter_models(entities: Iterable[Entity], fisher_theta: float, delta_m: float = 75.0, sort_by_score: bool = True) -> list[Entity]:
    fisher_dist = fisher_score(fisher_theta)
    ordered = list(entities)
    if sort_by_score:
        ordered.sort(key=lambda e: (-e.score, e.id))
    selected: list[Entity] = []
    for entity in ordered:
        if keep_candidate(entity, selected, delta_m) and haversine_m((entity.lat, entity.lon), (0.0, 0.0)) <= fisher_dist:
            selected.append(entity)
    return selected

def evaluate_informativeness(entity: Entity, fisher_theta: float) -> float:
    return fisher_score(fisher_theta) * entity.score

def hybrid_fusion(entities: Iterable[Entity], fisher_theta: float) -> Tuple[List[Entity], float]:
    filtered_entities = filter_models(entities, fisher_theta)
    informativeness = sum(evaluate_informativeness(e, fisher_theta) for e in filtered_entities)
    return filtered_entities, informativeness

if __name__ == "__main__":
    entities = [
        Entity("id1", 37.7749, -122.4194, "category1", score=0.8),
        Entity("id2", 34.0522, -118.2437, "category2", score=0.9),
        Entity("id3", 40.7128, -74.0060, "category1", score=0.7),
    ]
    fisher_theta = 0.5
    filtered_entities, informativeness = hybrid_fusion(entities, fisher_theta)
    print(f"Filtered Entities: {[e.id for e in filtered_entities]}")
    print(f"Informativeness: {informativeness}")