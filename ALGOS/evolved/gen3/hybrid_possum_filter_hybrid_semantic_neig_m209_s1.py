# DARWIN HAMMER — match 209, survivor 1
# gen: 3
# parent_a: possum_filter.py (gen0)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s6.py (gen2)
# born: 2026-05-29T23:27:39Z

"""Hybrid Diversity Filter and Semantic-Morphology System
Parents:
- possum_filter.py (local diversity filter based on haversine distance and category/address signature)
- hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s6.py (hybrid semantic-morphology system with recovery priority and cosine similarity)

The mathematical bridge is established by replacing the ranking score in the diversity filter with the hybrid score from the semantic-morphology system.
The hybrid score `h(i,j)` is defined as a convex combination:

    h(i,j) = α * (1 - haversine_distance / max_distance) + (1-α) * recovery_priority

where `α ∈ [0,1]` balances pure spatial diversity against the physical robustness and semantic meaning of the neighbors.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    morphology: Morphology
    score: float = 0.0
    address_signature: str = ""

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def haversine_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def hybrid_score(entity: Entity, max_distance: float = 75_000.0, alpha: float = 0.5) -> float:
    distance = haversine_distance((entity.lat, entity.lon), (0.0, 0.0))  # Use a reference point (0, 0) for demonstration
    priority = recovery_priority(entity.morphology)
    return alpha * (1 - distance / max_distance) + (1 - alpha) * priority

def keep_candidate(candidate: Entity, selected: list[Entity], delta_m: float, max_distance: float) -> bool:
    for existing in selected:
        same_kind = candidate.category.strip().lower() == existing.category.strip().lower()
        distance = haversine_distance((candidate.lat, candidate.lon), (existing.lat, existing.lon))
        if same_kind and distance <= delta_m:
            return False
    return True

def filter_entities(entities: Iterable[Entity], delta_m: float = 75_000.0, max_distance: float = 75_000.0, alpha: float = 0.5) -> list[Entity]:
    ordered = sorted(entities, key=lambda e: (-hybrid_score(e, max_distance, alpha), e.id))
    selected: list[Entity] = []
    for entity in ordered:
        if keep_candidate(entity, selected, delta_m, max_distance):
            selected.append(entity)
    return selected

if __name__ == "__main__":
    entities = [
        Entity("1", 37.7749, -122.4194, "A", Morphology(1.0, 2.0, 3.0, 10.0)),
        Entity("2", 37.7859, -122.4364, "A", Morphology(4.0, 5.0, 6.0, 20.0)),
        Entity("3", 37.7963, -122.4575, "B", Morphology(7.0, 8.0, 9.0, 30.0)),
    ]
    filtered_entities = filter_entities(entities)
    for entity in filtered_entities:
        print(entity.id, entity.category, hybrid_score(entity))