# DARWIN HAMMER — match 553, survivor 1
# gen: 4
# parent_a: possum_filter.py (gen0)
# parent_b: hybrid_hybrid_label_foundry_hybrid_hybrid_decisi_m172_s0.py (gen3)
# born: 2026-05-29T23:29:34Z

"""
This module fuses the possum_filter.py and hybrid_hybrid_label_foundry_hybrid_hybrid_decisi_m172_s0.py algorithms.

The mathematical bridge between the two structures is the concept of "information 
richness," which is used to determine the likelihood of an entity being 
representative of its category. This richness is calculated based on the Shannon 
entropy of the feature count vector of entities within a certain distance 
threshold, and this value is then used to adjust the filtering behavior.

The hybrid score combines the original possum filter score with the 
entropy-adjusted pruning probability. When the observed entities are 
information-rich (high entropy), the algorithm filters less aggressively and 
preserves more of the possum filter contribution; conversely, low-entropy 
(repetitive) inputs are filtered more heavily.
"""

import numpy as np
from dataclasses import dataclass
from math import exp, log, sin, cos, asin, radians
from random import random
from sys import exit
from pathlib import Path
from typing import Iterable, List, Tuple

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * asin(min(1.0, (h ** 0.5)))

def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

def shannon_entropy(feature_counts: List[int]) -> float:
    total = sum(feature_counts)
    entropy = 0.0
    for count in feature_counts:
        if count > 0:
            prob = count / total
            entropy -= prob * log(prob, 2)
    return entropy

def pruning_probability(entropy: float) -> float:
    return exp(-entropy)

def hybrid_filter(entities: Iterable[Entity], delta_m: float = 75.0) -> List[Entity]:
    if delta_m < 0:
        raise ValueError("delta_m must be non-negative")

    # Group entities by category
    categories = {}
    for entity in entities:
        category = entity.category.strip().lower()
        if category not in categories:
            categories[category] = []
        categories[category].append(entity)

    # Calculate feature counts and Shannon entropy for each category
    category_entropies = {}
    for category, entities_in_category in categories.items():
        feature_counts = [0] * len(entities_in_category)
        for i, entity in enumerate(entities_in_category):
            feature_counts[i] = 1
        entropy = shannon_entropy(feature_counts)
        category_entropies[category] = entropy

    # Filter entities based on possum filter and entropy-adjusted pruning probability
    selected = []
    for entity in sorted(entities, key=lambda e: (-e.score, e.id)):
        category = entity.category.strip().lower()
        entropy = category_entropies.get(category, 0.0)
        pruning_prob = pruning_probability(entropy)
        keep = True
        for existing in selected:
            same_kind = signature(entity) == signature(existing) or entity.category.strip().lower() == existing.category.strip().lower()
            if same_kind and haversine_m((entity.lat, entity.lon), (existing.lat, existing.lon)) <= delta_m:
                if random() < pruning_prob:
                    keep = False
                break
        if keep:
            selected.append(entity)
    return selected

def get_entities() -> List[Entity]:
    return [
        Entity("1", 37.7749, -122.4194, "A", 0.9),
        Entity("2", 37.7859, -122.4364, "A", 0.8),
        Entity("3", 37.7963, -122.4575, "B", 0.7),
        Entity("4", 37.8067, -122.4786, "B", 0.6),
        Entity("5", 37.8169, -122.4997, "A", 0.5),
    ]

if __name__ == "__main__":
    entities = get_entities()
    filtered_entities = hybrid_filter(entities)
    for entity in filtered_entities:
        print(entity)