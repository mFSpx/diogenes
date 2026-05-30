# DARWIN HAMMER — match 3911, survivor 0
# gen: 5
# parent_a: hybrid_possum_filter_hybrid_semantic_neig_m209_s0.py (gen3)
# parent_b: hybrid_hybrid_privacy_model_hybrid_hybrid_semant_m1133_s0.py (gen4)
# born: 2026-05-29T23:52:26Z

"""
Hybrid Possum-Filtered Semantic Morphology System with Hybrid Privacy Model Pool

This module integrates the Hybrid Possum-Filtered Semantic Morphology System from
hybrid_possum_filter_hybrid_semantic_neig_m209_s0.py with the Hybrid Privacy Model Pool from
hybrid_hybrid_privacy_model_hybrid_hybrid_semant_m1133_s0.py. The mathematical bridge between
the two lies in the idea of treating the semantic neighborhoods as additional soft resources
that must be allocated together with RAM and privacy-load. A unified hybrid score is defined
as a weighted sum of the semantic similarity, recovery priority, and spatial diversity constraint.
The Voronoi partitioning is used to assign points to these neighborhoods based on their proximity
to the seeds, while the geometric product is used to compute the similarity between documents.

Mathematical Interface:
The mathematical interface between the two parents is the representation of the semantic neighborhoods
as additional soft resources that must be allocated together with RAM and privacy-load. The semantic-load
for a model m is defined as s(m) = β * semantic_similarity(m.doc_vector, seed_vectors), where β is a scaling constant.
The total load for a selection vector x (binary indicator of loaded models) is L = Aᵀ · x, where A is the combined resource matrix.
"""

import math
import numpy as np
import random
import sys
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""
    morphology: 'Morphology' = None
    doc_vector: np.ndarray = None

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

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

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width + height) / (3 * max(length, width, height))

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Return a normalized reconstruction risk in [0,1]."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def semantic_similarity(doc_vector: np.ndarray, seed_vectors: List[np.ndarray]) -> float:
    """Compute the cosine similarity between a document vector and the seed vectors."""
    dot_products = [np.dot(doc_vector, seed_vector) for seed_vector in seed_vectors]
    norms = [np.linalg.norm(seed_vector) for seed_vector in seed_vectors]
    return np.sum([dot_product / (norm * len(seed_vectors)) for dot_product, norm in zip(dot_products, norms)])

def hybrid_score(e: Entity, selected: list[Entity], delta_m: float, beta: float, ram_ceiling: float, privacy_budget: float, semantic_budget: float) -> float:
    """Compute the hybrid score for an entity based on its semantic similarity, recovery priority, and spatial diversity constraint."""
    semantic_load = beta * semantic_similarity(e.doc_vector, [seed.doc_vector for seed in selected])
    recovery_priority = 1.0 - e.score
    spatial_diversity = 1.0 / len([existing for existing in selected if signature(e) == signature(existing)])
    return semantic_load * 0.4 + recovery_priority * 0.3 + spatial_diversity * 0.3

def allocate_resources(selected: list[Entity], delta_m: float, beta: float, ram_ceiling: float, privacy_budget: float, semantic_budget: float) -> np.ndarray:
    """Allocate resources to the selected entities based on their hybrid scores."""
    scores = [hybrid_score(e, selected, delta_m, beta, ram_ceiling, privacy_budget, semantic_budget) for e in selected]
    weights = np.array([score / sum(scores) for score in scores])
    resource_matrix = np.array([[1, 1, 1], [ram_ceiling, privacy_budget, semantic_budget]])
    return np.linalg.lstsq(resource_matrix, weights, rcond=None)[0]

def voronoi_partitioning(entities: List[Entity], seeds: List[Entity]) -> Dict[str, List[Entity]]:
    """Partition the entities into Voronoi cells based on their proximity to the seeds."""
    cells = {}
    for entity in entities:
        nearest_seed = min(seeds, key=lambda seed: haversine_m((entity.lat, entity.lon), (seed.lat, seed.lon)))
        if nearest_seed not in cells:
            cells[nearest_seed] = []
        cells[nearest_seed].append(entity)
    return cells

if __name__ == "__main__":
    import random
    random.seed(0)
    np.random.seed(0)
    entities = [
        Entity(id="e1", lat=37.7749, lon=-122.4194, category="cat1", score=0.5),
        Entity(id="e2", lat=37.7859, lon=-122.4364, category="cat2", score=0.7),
        Entity(id="e3", lat=37.7959, lon=-122.4574, category="cat1", score=0.3),
        Entity(id="e4", lat=37.8059, lon=-122.4784, category="cat2", score=0.9),
    ]
    seeds = [entities[0], entities[2]]
    delta_m = 0.1
    beta = 0.5
    ram_ceiling = 1024
    privacy_budget = 100
    semantic_budget = 1000
    selected = voronoi_partitioning(entities, seeds)
    resource_allocation = allocate_resources(selected["e1"], delta_m, beta, ram_ceiling, privacy_budget, semantic_budget)
    print(resource_allocation)