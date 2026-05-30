# DARWIN HAMMER — match 1220, survivor 4
# gen: 3
# parent_a: hybrid_possum_filter_hybrid_privacy_model_m53_s0.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py (gen1)
# born: 2026-05-29T23:34:28Z

"""
This module fuses the mathematical structures of hybrid_possum_filter_hybrid_privacy_model_m53_s0.py and hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py.
The mathematical bridge between these two structures is established by introducing a fractional-memory tree cost that incorporates both spatial-aware privacy risk and material length.
The reconstruction risk score is used to weight the Caputo kernel values, resulting in a modified fractional-memory tree cost that considers both spatial and categorical information.

Parent A: hybrid_possum_filter_hybrid_privacy_model_m53_s0.py
Parent B: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Tuple
from pathlib import Path

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
    t = 1.0 / (z * z)
    return math.sqrt(2 * math.pi / z) * math.exp(-z) * math.pow(1.0 + t * _LANCZOS_C[0] + t * t * _LANCZOS_C[1] + t * t * t * _LANCZOS_C[2] + t * t * t * t * _LANCZOS_C[3] + t * t * t * t * t * _LANCZOS_C[4] + t * t * t * t * t * t * _LANCZOS_C[5] + t * t * t * t * t * t * t * _LANCZOS_C[6], _LANCZOS_G)

def caputo_weights(alpha: float, T: int) -> np.ndarray:
    w = np.zeros(T)
    for k in range(T):
        w[k] = math.pow(k + 1, alpha - 1) / gamma_lanczos(alpha)
    return w / np.sum(w)

def fractional_weighted_sum(history: np.ndarray, weights: np.ndarray) -> float:
    return np.sum(history * weights)

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return haversine_m(a, b)

def incremental_fractional_tree_cost(entities: List[Entity], alpha: float, delta_m: float, path_weight: float) -> float:
    risks = spatial_aware_privacy_risk_vector(entities, delta_m)
    material = 0.0
    history = np.zeros(len(entities))
    for i in range(1, len(entities)):
        d = length((entities[i-1].lat, entities[i-1].lon), (entities[i].lat, entities[i].lon))
        material += d
        history[i] = d
    w = caputo_weights(alpha, len(entities))
    fractional_memory_term = path_weight * fractional_weighted_sum(risks * history, w)
    return material + fractional_memory_term

def hybrid_privacy_risk(entities: List[Entity], alpha: float, delta_m: float) -> float:
    return incremental_fractional_tree_cost(entities, alpha, delta_m, 1.0)

if __name__ == "__main__":
    entities = [
        Entity("1", 37.7749, -122.4194, "A"),
        Entity("2", 37.7859, -122.4364, "A"),
        Entity("3", 37.7963, -122.4575, "B"),
        Entity("4", 37.8067, -122.4786, "B"),
    ]
    alpha = 0.5
    delta_m = 1000.0
    print(hybrid_privacy_risk(entities, alpha, delta_m))