# DARWIN HAMMER — match 53, survivor 0
# gen: 2
# parent_a: possum_filter.py (gen0)
# parent_b: hybrid_privacy_model_pool_m7_s2.py (gen1)
# born: 2026-05-29T23:23:51Z

"""
This module fuses the mathematical structures of possum_filter.py and hybrid_privacy_model_pool_m7_s2.py.
The possum_filter.py provides a method for filtering entities based on their spatial distance and category similarity,
while hybrid_privacy_model_pool_m7_s2.py presents a framework for managing model resources under RAM ceiling and tier exclusivity constraints.
The mathematical bridge between these two structures is established by introducing a spatial-aware privacy risk model.
In this model, the reconstruction risk for each entity is weighted by its distance to other entities in the dataset,
resulting in a modified risk vector that incorporates both spatial and categorical information.
This allows us to build a combined resource matrix that considers both RAM consumption and spatial-aware privacy load.
"""

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np
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

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

def model_resource_matrix(models: List[ModelTier], privacy_risk: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if privacy_risk.size == 0:
        mean_risk = 0.0
    else:
        mean_risk = float(privacy_risk.mean())

    tier_factor = {"T1": 1.0, "T2": 2.0, "T3": 3.0}
    rows = []
    for m in models:
        ram = float(m.ram_mb)
        pf = tier_factor.get(m.tier, 1.0)
        privacy_load = alpha * pf * mean_risk
        rows.append([ram, privacy_load])
    return np.array(rows, dtype=float)

def select_models_hybrid(models: List[ModelTier], entities: List[Entity], delta_m: float, ram_ceiling_mb: int = 6000, privacy_budget: float = 1.0, alpha: float = 0.5) -> List[ModelTier]:
    privacy_risk = spatial_aware_privacy_risk_vector(entities, delta_m)
    resource_matrix = model_resource_matrix(models, privacy_risk, alpha)
    selected_models = []
    total_ram = 0
    total_privacy_load = 0
    for i, (ram, privacy_load) in enumerate(resource_matrix):
        if total_ram + ram <= ram_ceiling_mb and total_privacy_load + privacy_load <= privacy_budget:
            selected_models.append(models[i])
            total_ram += ram
            total_privacy_load += privacy_load
    return selected_models

if __name__ == "__main__":
    entities = [
        Entity("1", 37.7749, -122.4194, "category1"),
        Entity("2", 37.7751, -122.4196, "category1"),
        Entity("3", 37.7743, -122.4192, "category2"),
    ]
    models = [
        ModelTier("model1", 1024, "T1"),
        ModelTier("model2", 2048, "T2"),
        ModelTier("model3", 4096, "T3"),
    ]
    selected_models = select_models_hybrid(models, entities, 100.0, 6000, 1.0)
    print(selected_models)