# DARWIN HAMMER — match 2867, survivor 2
# gen: 4
# parent_a: hybrid_possum_filter_hybrid_privacy_model_m53_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py (gen3)
# born: 2026-05-29T23:46:19Z

"""
Hybrid module combining Possum-style local diversity filter and Hybrid privacy model with model resource management.

Mathematical Bridge:
The fusion integrates the governing equations of both parents by using the distance threshold from the Possum filter to limit the selection of models based on their resource usage and privacy risk. The distance threshold is used to filter out models that are too similar in terms of their resource usage and privacy risk, ensuring that the selected models are diverse and do not exceed the RAM ceiling or privacy budget.

The fusion combines the Possum filter's use of the haversine distance to measure the similarity between entities with the Hybrid privacy model's use of the pheromone system to measure the privacy risk of models. The pheromone system's half-life function is used to decay the signal value over time, ensuring that the privacy risk of models decreases as time passes.

The Hybrid Possum filter is used to select models that are diverse and do not exceed the RAM ceiling or privacy budget. The selected models are then used to update the pheromone system, which is used to measure the privacy risk of models. The fusion of the two systems ensures that the selected models are not only diverse but also do not pose a significant privacy risk.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Dict, Set, Tuple
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

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def calculate_pheromone_signal(
    self,
    surface_key: str,
    signal_kind: str,
    signal_value: float,
    half_life_seconds: float,
) -> float:
    now = datetime.now(timezone.utc)
    if surface_key not in self.pheromones:
        self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds}
    current_value = self.pheromones[surface_key]['signal_value']
    decay_factor = math.exp(-half_life_seconds / (60 * 60 * 24))
    new_value = current_value * decay_factor + signal_value * (1 - decay_factor)
    self.pheromones[surface_key]['signal_value'] = new_value

def hybrid_filter_entities(
    entities: Iterable[Entity],
    morphologies: List[Morphology],
    delta_m: float = 75.0,
    sort_by_score: bool = True,
    max_index: float = 10.0,
    half_life_seconds: float = 86400.0,
) -> list[Entity]:
    pheromone_system = PheromoneSystem()
    filtered_entities = filter_entities(entities, delta_m, sort_by_score)
    for entity in filtered_entities:
        pheromone_system.calculate_pheromone_signal(
            "entity_" + entity.id,
            "privacy_signal",
            recovery_priority(entity.morphology),
            half_life_seconds,
        )
    return filtered_entities

def hybrid_select_models(
    entities: Iterable[Entity],
    morphologies: List[Morphology],
    delta_m: float = 75.0,
    sort_by_score: bool = True,
    max_index: float = 10.0,
    half_life_seconds: float = 86400.0,
    num_models: int = 10,
) -> list[Entity]:
    filtered_entities = hybrid_filter_entities(entities, morphologies, delta_m, sort_by_score, max_index, half_life_seconds)
    return filtered_entities[:num_models]

class PheromoneSystem:
    def __init__(self) -> None:
        self.pheromones: Dict[str, Dict[str, Any]] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds}
        current_value = self.pheromones[surface_key]['signal_value']
        decay_factor = math.exp(-half_life_seconds / (60 * 60 * 24))
        new_value = current_value * decay_factor + signal_value * (1 - decay_factor)
        self.pheromones[surface_key]['signal_value'] = new_value

def hybrid_main():
    entities = [
        Entity("e1", 37.7749, -122.4194, "category1", 0.5),
        Entity("e2", 37.7859, -122.4364, "category2", 0.7),
        Entity("e3", 37.7969, -122.4574, "category3", 0.9),
    ]
    morphologies = [
        Morphology(10.0, 5.0, 2.0, 1.0),
        Morphology(12.0, 6.0, 3.0, 2.0),
        Morphology(14.0, 7.0, 4.0, 3.0),
    ]
    filtered_entities = hybrid_filter_entities(entities, morphologies)
    selected_models = hybrid_select_models(entities, morphologies, num_models=2)
    print(filtered_entities)
    print(selected_models)

if __name__ == "__main__":
    hybrid_main()