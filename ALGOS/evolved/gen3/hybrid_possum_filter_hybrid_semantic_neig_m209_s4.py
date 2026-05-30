# DARWIN HAMMER — match 209, survivor 4
# gen: 3
# parent_a: possum_filter.py (gen0)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s6.py (gen2)
# born: 2026-05-29T23:27:39Z

import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Entity and Morphology Dataclasses
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    mass: float = 0.0

# ----------------------------------------------------------------------
# Morphology and Recovery Priority
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Entity, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Entity, max_index: float = 10.0) -> float:
    """Normalised priority in [0,1] derived from morphology."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

# ----------------------------------------------------------------------
# Haversine Distance and Signature
# ----------------------------------------------------------------------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

# ----------------------------------------------------------------------
# Hybrid Filter Functions
# ----------------------------------------------------------------------
def keep_candidate(candidate: Entity, selected: List[Entity], delta_m: float, alpha: float) -> bool:
    for existing in selected:
        same_kind = signature(candidate) == signature(existing) or candidate.category.strip().lower() == existing.category.strip().lower()
        if same_kind and haversine_m((candidate.lat, candidate.lon), (existing.lat, existing.lon)) <= delta_m:
            # Calculate hybrid score
            recovery_p_candidate = recovery_priority(candidate)
            recovery_p_existing = recovery_priority(existing)
            hybrid_score = alpha * (1 - (haversine_m((candidate.lat, candidate.lon), (existing.lat, existing.lon)) / delta_m)) + (1 - alpha) * (1 - abs(recovery_p_candidate - recovery_p_existing) / max(recovery_p_candidate, recovery_p_existing))
            if hybrid_score < 0.5:
                return False
    return True

def filter_entities(entities: List[Entity], delta_m: float = 75.0, alpha: float = 0.5, sort_by_score: bool = True) -> List[Entity]:
    if delta_m < 0:
        raise ValueError("delta_m must be non-negative")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")
    ordered = list(entities)
    if sort_by_score:
        ordered.sort(key=lambda e: (-e.score, e.id))
    selected: List[Entity] = []
    for entity in ordered:
        if keep_candidate(entity, selected, delta_m, alpha):
            selected.append(entity)
    return selected

def calculate_hybrid_score(entity1: Entity, entity2: Entity, delta_m: float, alpha: float) -> float:
    same_kind = signature(entity1) == signature(entity2) or entity1.category.strip().lower() == entity2.category.strip().lower()
    if same_kind:
        recovery_p_entity1 = recovery_priority(entity1)
        recovery_p_entity2 = recovery_priority(entity2)
        hybrid_score = alpha * (1 - (haversine_m((entity1.lat, entity1.lon), (entity2.lat, entity2.lon)) / delta_m)) + (1 - alpha) * (1 - abs(recovery_p_entity1 - recovery_p_entity2) / max(recovery_p_entity1, recovery_p_entity2))
        return hybrid_score
    else:
        return 0.0

if __name__ == "__main__":
    entity1 = Entity("1", 37.7749, -122.4194, "restaurant", 0.5, "", 10.0, 5.0, 2.0, 100.0)
    entity2 = Entity("2", 37.7858, -122.4364, "restaurant", 0.3, "", 8.0, 4.0, 1.5, 80.0)
    entity3 = Entity("3", 37.7963, -122.4575, "cafe", 0.2, "", 6.0, 3.0, 1.0, 60.0)
    entities = [entity1, entity2, entity3]
    filtered_entities = filter_entities(entities, delta_m=50.0, alpha=0.7)
    for entity in filtered_entities:
        print(entity.id, entity.category)
    hybrid_score = calculate_hybrid_score(entity1, entity2, delta_m=50.0, alpha=0.7)
    print("Hybrid score:", hybrid_score)