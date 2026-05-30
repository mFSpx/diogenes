# DARWIN HAMMER — match 1220, survivor 3
# gen: 3
# parent_a: hybrid_possum_filter_hybrid_privacy_model_m53_s0.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py (gen1)
# born: 2026-05-29T23:34:28Z

"""
This module fuses the mathematical structures of hybrid_possum_filter_hybrid_privacy_model_m53_s0.py and hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py.
The mathematical bridge between these two structures is established by introducing a fractional-memory tree cost that incorporates spatial-aware privacy risk.
The reconstruction risk for each entity is weighted by its distance to other entities in the dataset, and the Caputo-weighted sum is used to compute the hybrid cost.

The hybrid algorithm, named **SpatialAwareFractionalTree**, combines the possum filter's spatial-aware privacy risk model with the Caputo fractional derivative's power-law memory kernel.
The **SpatialAwareFractionalTree** computes a material length of a tree plus a linear path-weight term based on distances from a root, where the distances are weighted by the spatial-aware privacy risk.

The implementation below provides:
* `spatial_aware_privacy_risk_vector` – compute spatial-aware privacy risk vector for a list of entities.
* `caputo_weights` – compute normalized Caputo kernel weights for a history.
* `incremental_fractional_tree_cost` – builds the tree edge-by-edge, updates distances, and evaluates the hybrid cost using the fractional memory term.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Spatial-aware privacy risk model
# ----------------------------------------------------------------------
class Entity:
    def __init__(self, id: str, lat: float, lon: float, category: str, score: float = 0.0, address_signature: str = ""):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.category = category
        self.score = score
        self.address_signature = address_signature

def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
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

def spatial_aware_privacy_risk_vector(entities: List[Entity], delta_m: float) -> np.ndarray:
    risks = []
    for i, entity in enumerate(entities):
        similar_entities = [e for j, e in enumerate(entities) if i != j and signature(entity) == signature(e) and haversine_m((entity.lat, entity.lon), (e.lat, e.lon)) <= delta_m]
        unique_quasi_identifiers = len(set(e.id for e in similar_entities))
        risk = reconstruction_risk_score(unique_quasi_identifiers, len(entities))
        risks.append(risk)
    return np.array(risks, dtype=float)

# ----------------------------------------------------------------------
# Parent B – Caputo fractional derivative and minimum-cost tree
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857
])

def gamma_lanczos(z: float) -> float:
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        x = 1.0 / (z * z)
        p = _LANCZOS_C[0] + x * (_LANCZOS_C[1] + x * (_LANCZOS_C[2] + x * (_LANCZOS_C[3] + x * (_LANCZOS_C[4] + x * (_LANCZOS_C[5] + x * _LANCZOS_C[6])))))
        return math.sqrt(2 * math.pi / z) * math.pow(z / math.e, z) * p

def caputo_weights(alpha: float, T: int) -> np.ndarray:
    w = np.zeros(T)
    for k in range(T):
        w[k] = math.pow(k + 1, alpha - 1) / gamma_lanczos(alpha)
    w /= np.sum(w)
    return w

def fractional_weighted_sum(history: np.ndarray, weights: np.ndarray) -> float:
    return np.sum(history * weights)

# ----------------------------------------------------------------------
# Hybrid SpatialAwareFractionalTree
# ----------------------------------------------------------------------
def incremental_fractional_tree_cost(entities: List[Entity], delta_m: float, alpha: float, T: int) -> float:
    risks = spatial_aware_privacy_risk_vector(entities, delta_m)
    weights = caputo_weights(alpha, T)
    material_length = 0.0
    path_weight = 0.0
    for i in range(T):
        material_length += haversine_m((entities[i].lat, entities[i].lon), (entities[(i + 1) % T].lat, entities[(i + 1) % T].lon))
        path_weight += risks[i] * weights[i]
    return material_length + path_weight

if __name__ == "__main__":
    entities = [Entity("id1", 37.7749, -122.4194, "category1"), Entity("id2", 37.7859, -122.4364, "category1"), Entity("id3", 37.7963, -122.4574, "category2")]
    delta_m = 1000.0
    alpha = 0.5
    T = 3
    cost = incremental_fractional_tree_cost(entities, delta_m, alpha, T)
    print(f"Hybrid cost: {cost}")