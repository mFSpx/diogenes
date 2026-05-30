# DARWIN HAMMER — match 1220, survivor 2
# gen: 3
# parent_a: hybrid_possum_filter_hybrid_privacy_model_m53_s0.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py (gen1)
# born: 2026-05-29T23:34:28Z

"""
This module fuses the mathematical structures of hybrid_possum_filter_hybrid_privacy_model_m53_s0.py and 
hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py. The hybrid_possum_filter_hybrid_privacy_model_m53_s0.py 
provides a method for filtering entities based on their spatial distance and category similarity, while 
hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py presents a framework for managing model resources under 
RAM ceiling and tier exclusivity constraints using a fractional-memory tree cost. The mathematical bridge 
between these two structures is established by introducing a spatial-aware fractional-memory tree cost model. 
In this model, the reconstruction risk for each entity is weighted by its distance to other entities in the 
dataset and the fractional-memory term, resulting in a modified risk vector that incorporates both spatial, 
categorical, and fractional-memory information.
"""

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple
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

def gamma_lanczos(z: float) -> float:
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
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1
    x = _LANCZOS_P / (z + _LANCZOS_G)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

_LANCZOS_P = np.array([0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857])

def caputo_weights(alpha: float, t_max: int) -> np.ndarray:
    weights = np.zeros(t_max)
    for t in range(1, t_max + 1):
        weights[t - 1] = ((t - 1) ** (alpha - 1)) / gamma_lanczos(alpha)
    return weights / np.sum(weights)

def fractional_weighted_sum(history: np.ndarray, alpha: float) -> float:
    weights = caputo_weights(alpha, len(history))
    return np.dot(weights, history)

def length(entity1: Entity, entity2: Entity) -> float:
    return haversine_m((entity1.lat, entity1.lon), (entity2.lat, entity2.lon))

def incremental_fractional_tree_cost(entities: List[Entity], alpha: float, delta_m: float) -> float:
    tree_cost = 0.0
    for i in range(len(entities)):
        entity1 = entities[i]
        similar_entities = [e for j, e in enumerate(entities) if i != j and signature(entity1) == signature(e) and haversine_m((entity1.lat, entity1.lon), (e.lat, e.lon)) <= delta_m]
        distances = [length(entity1, e) for e in similar_entities]
        tree_cost += fractional_weighted_sum(np.array(distances), alpha)
    return tree_cost

def fractional_ssm_step(entity: Entity, alpha: float, delta_m: float, entities: List[Entity]) -> float:
    similar_entities = [e for e in entities if signature(entity) == signature(e) and haversine_m((entity.lat, entity.lon), (e.lat, e.lon)) <= delta_m]
    distances = [length(entity, e) for e in similar_entities]
    return fractional_weighted_sum(np.array(distances), alpha)

if __name__ == "__main__":
    entities = [
        Entity("1", 37.7749, -122.4194, "category1"),
        Entity("2", 37.7859, -122.4364, "category1"),
        Entity("3", 37.7963, -122.4574, "category2"),
        Entity("4", 37.8067, -122.4784, "category2"),
        Entity("5", 37.8171, -122.4994, "category1")
    ]
    alpha = 0.5
    delta_m = 10000.0
    print(incremental_fractional_tree_cost(entities, alpha, delta_m))
    print(fractional_ssm_step(entities[0], alpha, delta_m, entities))