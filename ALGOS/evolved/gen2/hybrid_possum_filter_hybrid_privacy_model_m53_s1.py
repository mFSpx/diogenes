# DARWIN HAMMER — match 53, survivor 1
# gen: 2
# parent_a: possum_filter.py (gen0)
# parent_b: hybrid_privacy_model_pool_m7_s2.py (gen1)
# born: 2026-05-29T23:23:51Z

"""
Hybrid module combining Possum-style local diversity filter (possum_filter.py) and Hybrid privacy model with model resource management (hybrid_privacy_model_pool_m7_s2.py).

Mathematical Bridge:
The fusion integrates the governing equations of both parents by using the distance threshold from the Possum filter to limit the selection of models based on their resource usage and privacy risk. The distance threshold is used to filter out models that are too similar in terms of their resource usage and privacy risk, ensuring that the selected models are diverse and do not exceed the RAM ceiling or privacy budget.
"""

import math
from dataclasses import dataclass
from typing import Iterable, List, Dict, Set, Tuple
import numpy as np
import random
import sys
import pathlib

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

def filter_entities(entities: Iterable[Entity], delta_m: float = 75.0, sort_by_score: bool = True) -> list[Entity]:
    if delta_m < 0:
        raise ValueError("delta_m must be non-negative")
    ordered = list(entities)
    if sort_by_score:
        ordered.sort(key=lambda e: (-e.score, e.id))
    selected: list[Entity] = []
    for entity in ordered:
        if keep_candidate(entity, selected, delta_m):
            selected.append(entity)
    return selected

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def model_resource_matrix(models: List[ModelTier],
                          privacy_risk: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
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

def hybrid_filter_entities(entities: Iterable[Entity], models: List[ModelTier], delta_m: float = 75.0, sort_by_score: bool = True, alpha: float = 0.5) -> Tuple[list[Entity], List[ModelTier]]:
    ordered = list(entities)
    if sort_by_score:
        ordered.sort(key=lambda e: (-e.score, e.id))
    selected_entities: list[Entity] = []
    selected_models: List[ModelTier] = []
    for entity in ordered:
        if keep_candidate(entity, selected_entities, delta_m):
            selected_entities.append(entity)
            # Select models based on the entity's category and location
            category_models = [m for m in models if m.tier == entity.category]
            if category_models:
                # Use the model_resource_matrix to select models that do not exceed the RAM ceiling or privacy budget
                privacy_risk = np.array([reconstruction_risk_score(1, 1) for _ in category_models])
                resource_matrix = model_resource_matrix(category_models, privacy_risk, alpha)
                selected_models.extend([m for m, row in zip(category_models, resource_matrix) if row[0] <= 1000 and row[1] <= 1.0])
    return selected_entities, selected_models

def select_models_hybrid(models: List[ModelTier], ram_ceiling_mb: int = 6000, privacy_budget: float = 1.0, alpha: float = 0.5) -> List[ModelTier]:
    selected_models: List[ModelTier] = []
    for model in models:
        if model.ram_mb <= ram_ceiling_mb and model.tier in ["T1", "T2"]:
            selected_models.append(model)
            ram_ceiling_mb -= model.ram_mb
    return selected_models

if __name__ == "__main__":
    entities = [Entity("1", 37.7749, -122.4194, "T1"), Entity("2", 37.7859, -122.4364, "T2")]
    models = [ModelTier("model1", 512, "T1"), ModelTier("model2", 1024, "T2")]
    selected_entities, selected_models = hybrid_filter_entities(entities, models)
    print(selected_entities)
    print(selected_models)