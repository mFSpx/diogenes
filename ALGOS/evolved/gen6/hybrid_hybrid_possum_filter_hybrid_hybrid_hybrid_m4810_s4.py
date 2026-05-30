# DARWIN HAMMER — match 4810, survivor 4
# gen: 6
# parent_a: hybrid_possum_filter_hybrid_hybrid_ternar_m1381_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_xgboos_m1946_s2.py (gen5)
# born: 2026-05-29T23:58:15Z

import math
import random
import sys
import hashlib
from pathlib import Path
from datetime import date, datetime
from typing import Iterable, Tuple, List, Dict
import numpy as np

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

def keep_candidate(candidate: Entity, selected: List[Entity], delta_m: float, pheromone_signal: float, alpha: float = 0.5) -> bool:
    for existing in selected:
        same_kind = signature(candidate) == signature(existing) or candidate.category.strip().lower() == existing.category.strip().lower()
        distance = haversine_m((candidate.lat, candidate.lon), (existing.lat, existing.lon))
        similarity = math.exp(-distance / delta_m)
        weighted_similarity = similarity * pheromone_signal
        if same_kind and weighted_similarity >= alpha:
            return False
    return True

def _rng_from_text(text: str) -> random.Random:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)

def extract_full_features(text: str) -> Dict[str, float]:
    rnd = _rng_from_text(text)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "psyche_wrath_velocity",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_fusion_entropy"
    ]
    return {k: rnd.random() for k in keys}

def pheromone_signal(text: str, day: int) -> float:
    features = extract_full_features(text)
    return features["psyche_poetic_entropy"] * (1 + day / 7)

def filter_entities(entities: Iterable[Entity], delta_m: float = 75.0, sort_by_score: bool = True, text: str = "", day: int = 0, alpha: float = 0.5) -> List[Entity]:
    pheromone_sig = pheromone_signal(text, day)
    ordered = list(entities)
    if sort_by_score:
        ordered.sort(key=lambda e: (-e.score, e.id))
    selected: List[Entity] = []
    for entity in ordered:
        if keep_candidate(entity, selected, delta_m, pheromone_sig, alpha):
            selected.append(entity)
    return selected

def evaluate_entity(entity: Entity, text: str = "", day: int = 0) -> float:
    pheromone_sig = pheromone_signal(text, day)
    return entity.score * pheromone_sig

def hybrid_operation(entities: Iterable[Entity], text: str = "", day: int = 0, alpha: float = 0.5) -> Tuple[List[Entity], float]:
    filtered_entities = filter_entities(entities, text=text, day=day, alpha=alpha)
    total_score = sum(evaluate_entity(e, text=text, day=day) for e in filtered_entities)
    return filtered_entities, total_score

if __name__ == "__main__":
    entities = [Entity("1", 37.7749, -122.4194, "A"), Entity("2", 34.0522, -118.2437, "B")]
    text = "example text"
    day = 3
    alpha = 0.5
    filtered_entities, total_score = hybrid_operation(entities, text=text, day=day, alpha=alpha)
    print(filtered_entities)
    print(total_score)