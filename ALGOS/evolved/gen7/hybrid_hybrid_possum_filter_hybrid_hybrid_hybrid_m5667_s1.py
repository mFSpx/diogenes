# DARWIN HAMMER — match 5667, survivor 1
# gen: 7
# parent_a: hybrid_possum_filter_hybrid_semantic_neig_m209_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_physarum_netw_m1724_s0.py (gen6)
# born: 2026-05-30T00:04:04Z

"""
Hybrid algorithm fusing the core topologies of 
hybrid_possum_filter_hybrid_semantic_neig_m209_s1 and 
hybrid_hybrid_hybrid_hybrid_hybrid_physarum_netw_m1724_s0.

The mathematical bridge is established by integrating the sphericity index 
from the Morphology-Shapley analysis with the hybrid score from the 
semantic-morphology system. This is achieved by using the sphericity index 
as a scalar "pressure" node in a Physarum-type flow network and 
interpreting the hybrid score as a measure of the physical robustness and 
semantic meaning of the neighbors.

The resulting hybrid algorithm combines the strengths of both parents, 
allowing for a more comprehensive analysis of physical entities and their 
relationships.
"""

import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import Iterable, Tuple, FrozenSet
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations, chain

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    morphology: Morphology
    score: float = 0.0
    address_signature: str = ""

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def haversine_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def hybrid_score(entity1: Entity, entity2: Entity, alpha: float = 0.5) -> float:
    distance = haversine_distance((entity1.lat, entity1.lon), (entity2.lat, entity2.lon))
    max_distance = 2 * 6_371_000.0
    sphericity1 = sphericity_index(entity1.morphology.length, entity1.morphology.width, entity1.morphology.height)
    sphericity2 = sphericity_index(entity2.morphology.length, entity2.morphology.width, entity2.morphology.height)
    return alpha * (1 - distance / max_distance) + (1 - alpha) * (sphericity1 + sphericity2) / 2

def morphology_to_hypervector(morphology: Morphology) -> np.ndarray:
    return np.array([morphology.length, morphology.width, morphology.height, morphology.mass])

def shapley_weighted_conductance(morphology: Morphology, other_morphologies: Iterable[Morphology]) -> float:
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    num_morphologies = len(list(other_morphologies))
    return sphericity / num_morphologies

def physarum_shapley_step(morphologies: Iterable[Morphology], conductances: Iterable[float]) -> Tuple[float, float]:
    pressures = [sphericity_index(m.length, m.width, m.height) for m in morphologies]
    return np.mean(pressures), np.mean(conductances)

if __name__ == "__main__":
    entity1 = Entity("1", 37.7749, -122.4194, "category1", Morphology(1.0, 1.0, 1.0, 1.0))
    entity2 = Entity("2", 37.7858, -122.4364, "category2", Morphology(2.0, 2.0, 2.0, 2.0))
    print(hybrid_score(entity1, entity2))
    print(morphology_to_hypervector(entity1.morphology))
    print(shapley_weighted_conductance(entity1.morphology, [entity2.morphology]))
    print(physarum_shapley_step([entity1.morphology, entity2.morphology], [shapley_weighted_conductance(entity1.morphology, [entity2.morphology])]))