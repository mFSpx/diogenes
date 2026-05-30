# DARWIN HAMMER — match 209, survivor 2
# gen: 3
# parent_a: possum_filter.py (gen0)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s6.py (gen2)
# born: 2026-05-29T23:27:39Z

#!/usr/bin/env python3
"""
HYBRID Possum-Semantic System
Parents:
- possum_filter.py (possum-style local diversity filter)
- hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s6.py (hybrid semantic-morphology system)

Mathematical Bridge:
The bridge is the combined use of the haversine distance metric from the Possum filter and the cosine similarity measure from the Semantic Neighbors system.
A unified hybrid score `h(i,j)` is defined as a convex combination:

    h(i,j) = α * c(v_i, v_j) + (1‑α) * (1 - e^(-d_m / (2 * δ_m)))

where `α ∈ [0,1]` balances pure semantic closeness against the physical robustness of the neighbor.
This single scalar drives both neighbor ranking and dynamic adjustment of the circuit‑breaker threshold.
"""
from __future__ import annotations
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Entity & Morphology (frozen dataclasses)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""
    morphology: Morphology = Morphology(length=0.0, width=0.0, height=0.0, mass=0.0)


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Mathematical Bridge
# ----------------------------------------------------------------------
def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Haversine distance metric."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def cosine_similarity(v_i: np.ndarray, v_j: np.ndarray) -> float:
    """Cosine similarity measure."""
    dot_product = np.dot(v_i, v_j)
    magnitude_i = np.linalg.norm(v_i)
    magnitude_j = np.linalg.norm(v_j)
    return dot_product / (magnitude_i * magnitude_j)


def hybrid_score(alpha: float, entity_i: Entity, entity_j: Entity) -> float:
    """Unified hybrid score."""
    d_m = haversine_m((entity_i.lat, entity_i.lon), (entity_j.lat, entity_j.lon))
    delta_m = 75.0
    return alpha * cosine_similarity(entity_i.morphology, entity_j.morphology) + (1 - alpha) * (1 - math.exp(-d_m / (2 * delta_m)))


# ----------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------
def filter_entities(entities: List[Entity], alpha: float = 0.5, delta_m: float = 75.0) -> List[Entity]:
    """Hybrid Possum-Semantic filter."""
    ordered = entities
    ordered.sort(key=lambda e: (-e.score, e.id))
    selected: List[Entity] = []
    for entity in ordered:
        for existing in selected:
            same_kind = (signature(entity) == signature(existing) or
                         entity.category.strip().lower() == existing.category.strip().lower())
            if same_kind and haversine_m((entity.lat, entity.lon), (existing.lat, existing.lon)) <= delta_m:
                break
        else:
            selected.append(entity)
    return selected


def signature(e: Entity) -> str:
    """Address signature."""
    return (e.address_signature or e.category).strip().lower()


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] derived from morphology."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Righting time index."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness index."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    morphology = Morphology(length=10.0, width=5.0, height=2.0, mass=1.0)
    entity = Entity(id="E1", lat=37.7749, lon=-122.4194, category="Restaurant", morphology=morphology)
    print(hybrid_score(0.5, entity, entity))
    print(filter_entities([entity]))