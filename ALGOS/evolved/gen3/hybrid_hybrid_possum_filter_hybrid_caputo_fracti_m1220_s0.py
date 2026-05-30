# DARWIN HAMMER — match 1220, survivor 0
# gen: 3
# parent_a: hybrid_possum_filter_hybrid_privacy_model_m53_s0.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py (gen1)
# born: 2026-05-29T23:34:28Z

"""Hybrid Spatial-Aware Fractional Minimum Cost Tree Module.

This module fuses two parent algorithms:

* **Hybrid Possum Filter Hybrid Privacy Model (hybrid_possum_filter_hybrid_privacy_model_m53_s0.py)** – provides a method for filtering entities based on their spatial distance and category similarity, while managing model resources under RAM ceiling and tier exclusivity constraints.
* **Hybrid Caputo Fractional Minimum Cost Tree (hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py)** – computes a material length of a tree plus a linear path-weight term based on distances from a root, using a Caputo fractional derivative kernel.

The mathematical bridge is the *integration of spatial-aware privacy risk into the fractional memory term*, allowing for a more accurate representation of the entity relationships in the tree cost calculation.

The implementation below provides:

* `haversine_m` – computes the distance between two entities using the Haversine formula.
* `reconstruction_risk_score` – calculates the reconstruction risk score for an entity based on its unique quasi-identifiers and total records.
* `spatial_aware_caputo_weights` – computes the Caputo kernel weights for a history, taking into account the spatial-aware privacy risk.
* `incremental_fractional_tree_cost` – builds the tree edge-by-edge, updates distances, and evaluates the hybrid cost using the fractional memory term and spatial-aware privacy risk.
* `fractional_ssm_step` – a generic state-space update that also uses the same Caputo weighting, illustrating the deeper algebraic connection.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from pathlib import Path
import random
import sys

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def spatial_aware_caputo_weights(alpha: float, history: List[float], entities: List[Entity]) -> np.ndarray:
    weights = []
    for i, entity in enumerate(entities):
        similar_entities = [e for j, e in enumerate(entities) if i != j and signature(entity) == signature(e) and haversine_m((entity.lat, entity.lon), (e.lat, e.lon)) <= 1.0]
        unique_quasi_identifiers = len(set(e.id for e in similar_entities))
        risk = reconstruction_risk_score(unique_quasi_identifiers, len(entities))
        weight = (1.0 - risk) * (history[i] ** (alpha - 1))
        weights.append(weight)
    return np.array(weights, dtype=float) / sum(weights)

def incremental_fractional_tree_cost(entities: List[Entity], alpha: float, path_weight: float) -> float:
    tree_cost = 0.0
    for i, entity in enumerate(entities):
        distances = [haversine_m((entity.lat, entity.lon), (e.lat, e.lon)) for e in entities]
        weights = spatial_aware_caputo_weights(alpha, distances, entities)
        tree_cost += weights[i] * distances[i] + path_weight * sum(distances)
    return tree_cost

def fractional_ssm_step(state: np.ndarray, alpha: float, entities: List[Entity]) -> np.ndarray:
    weights = spatial_aware_caputo_weights(alpha, state, entities)
    return weights * state

if __name__ == "__main__":
    entities = [
        Entity("1", 37.7749, -122.4194, "Restaurant", 4.5),
        Entity("2", 37.7859, -122.4364, "Hotel", 4.2),
        Entity("3", 37.7959, -122.4574, "Restaurant", 4.8),
        Entity("4", 37.8069, -122.4784, "Hotel", 4.0),
    ]
    alpha = 0.5
    path_weight = 1.0
    tree_cost = incremental_fractional_tree_cost(entities, alpha, path_weight)
    print("Tree cost:", tree_cost)
    state = np.array([0.5, 0.3, 0.2, 0.1])
    new_state = fractional_ssm_step(state, alpha, entities)
    print("New state:", new_state)