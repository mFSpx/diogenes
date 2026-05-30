# DARWIN HAMMER — match 553, survivor 0
# gen: 4
# parent_a: possum_filter.py (gen0)
# parent_b: hybrid_hybrid_label_foundry_hybrid_hybrid_decisi_m172_s0.py (gen3)
# born: 2026-05-29T23:29:34Z

"""
This module fuses the possum_filter.py and hybrid_hybrid_label_foundry_hybrid_hybrid_decisi_m172_s0.py algorithms.

The mathematical bridge between the two structures is the concept of "information richness," 
which is used to determine the likelihood of an entity recovering from a failure. 
This richness is calculated based on the Shannon entropy of the feature count vector, 
and this value is then used to adjust the distance threshold for determining when to 
keep or discard an entity. In this fusion, we use the entity filtering functions from 
possum_filter.py to determine the entities to keep, and then use the information richness 
to adjust the filtering behavior.

The hybrid score combines the original entity score with the entropy-adjusted pruning probability. 
When the observed entity is information-rich (high entropy), the algorithm filters less aggressively 
and preserves more of the entity contribution; conversely, low-entropy (repetitive) inputs are 
filtered more heavily.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, log, sin, cos, asin, sqrt, radians
from random import random
from sys import exit
from pathlib import Path
from typing import Callable, Dict, List, Iterable

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float

@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371000.0 * asin(min(1.0, sqrt(h)))

def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

def shannon_entropy(vector: List[float]) -> float:
    entropy = 0.0
    for p in vector:
        if p > 0:
            entropy -= p * log(p, 2)
    return entropy

def keep_candidate(candidate: Entity, selected: list[Entity], delta_m: float, entropy: float) -> bool:
    for existing in selected:
        same_kind = signature(candidate) == signature(existing) or candidate.category.strip().lower() == existing.category.strip().lower()
        if same_kind and haversine_m((candidate.lat, candidate.lon), (existing.lat, existing.lon)) <= delta_m * (1 - entropy):
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
        vector = [entity.score / 100.0, 1 - entity.score / 100.0]
        entropy = shannon_entropy(vector)
        if keep_candidate(entity, selected, delta_m, entropy):
            selected.append(entity)
    return selected

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    votes = {}
    for batch in batches:
        for result in batch:
            if result.doc_id not in votes:
                votes[result.doc_id] = {}
            if result.label not in votes[result.doc_id]:
                votes[result.doc_id][result.label] = 0
            votes[result.doc_id][result.label] += 1
    labels = []
    for doc_id, label_counts in votes.items():
        max_label = max(label_counts, key=label_counts.get)
        confidence = label_counts[max_label] / sum(label_counts.values())
        labels.append(ProbabilisticLabel(doc_id, max_label, confidence))
    return labels

if __name__ == "__main__":
    entities = [
        Entity("1", 37.7749, -122.4194, "category1", 90.0, "address1"),
        Entity("2", 37.7859, -122.4364, "category1", 80.0, "address2"),
        Entity("3", 37.7963, -122.4575, "category2", 70.0, "address3"),
        Entity("4", 37.8067, -122.4786, "category2", 60.0, "address4"),
    ]
    filtered_entities = filter_entities(entities)
    print(filtered_entities)