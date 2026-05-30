# DARWIN HAMMER — match 3578, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m528_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_semantic_neig_m1539_s1.py (gen6)
# born: 2026-05-29T23:50:52Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple
import numpy as np

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

@dataclass(frozen=True)
class PheromoneEntry:
    uuid: str
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: int
    created_at: float
    last_decay: float

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        self.uuid = f"{random.getrandbits(128):032x}"
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = random.random()
        self.created_at = now
        self.last_decay = now

def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def spatial_aware_privacy_risk_vector(entities: List[Entity], delta_m: float) -> np.ndarray:
    n = len(entities)
    risks = np.zeros(n, dtype=float)
    for i, ent in enumerate(entities):
        similar = [e for j, e in enumerate(entities) if i != j and signature(ent) == signature(e) and haversine_m((ent.lat, ent.lon), (e.lat, e.lon)) <= delta_m]
        uq = len(similar)
        total = len(entities) - 1
        risks[i] = reconstruction_risk_score(uq, total)
    return risks

def curvature_adjusted_distance_matrix(entities: List[Entity], curvature: np.ndarray) -> np.ndarray:
    n = len(entities)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_m((entities[i].lat, entities[i].lon), (entities[j].lat, entities[j].lon))
            D[i, j] = D[j, i] = d
    if curvature.shape != (n, n):
        raise ValueError("Curvature matrix must be of shape (n, n)")
    C = np.clip(curvature, -1.0, 1.0)
    D_prime = D * (1.0 - C)
    return D_prime

def pheromone_similarity_matrix(D_prime: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    S_raw = 1.0 / (D_prime + epsilon)
    row_sums = S_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = S_raw / row_sums
    return P

def shannon_entropy(prob_matrix: np.ndarray) -> float:
    flat = prob_matrix.ravel()
    flat = flat[flat > 0]
    return -np.sum(flat * np.log2(flat))

def hybrid_neighbor_scores(entities: List[Entity], curvature: np.ndarray, delta_m: float, epsilon: float = 1e-6) -> np.ndarray:
    n = len(entities)
    if n == 0:
        return np.empty((0, 0), dtype=float)
    r = spatial_aware_privacy_risk_vector(entities, delta_m)
    D_prime = curvature_adjusted_distance_matrix(entities, curvature)
    S_raw = 1.0 / (D_prime + epsilon)
    row_sums = S_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = S_raw / row_sums
    risk_factor = (1.0 - r).reshape(-1, 1)
    hybrid = S_raw * P * risk_factor
    hybrid = np.clip(hybrid, 0.0, 1.0)
    return hybrid

def improved_hybrid_neighbor_scores(entities: List[Entity], curvature: np.ndarray, delta_m: float, epsilon: float = 1e-6) -> np.ndarray:
    n = len(entities)
    if n == 0:
        return np.empty((0, 0), dtype=float)
    r = spatial_aware_privacy_risk_vector(entities, delta_m)
    D_prime = curvature_adjusted_distance_matrix(entities, curvature)
    S_raw = 1.0 / (D_prime + epsilon)
    row_sums = S_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = S_raw / row_sums
    risk_factor = (1.0 - r).reshape(-1, 1)
    hybrid = S_raw * P * risk_factor
    hybrid = np.clip(hybrid, 0.0, 1.0)
    # Apply a normalization to the hybrid scores
    hybrid = hybrid / hybrid.sum(axis=1, keepdims=True)
    return hybrid

# Usage
entities = [Entity("1", 37.7749, -122.4194, "A"), Entity("2", 37.7859, -122.4364, "B")]
curvature = np.array([[0.5, 0.3], [0.3, 0.5]])
delta_m = 1000.0
hybrid_scores = improved_hybrid_neighbor_scores(entities, curvature, delta_m)
print(hybrid_scores)