# DARWIN HAMMER — match 1389, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_percyphon_hyb_pheromone_m337_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_tempor_hybrid_hybrid_doomsd_m676_s2.py (gen3)
# born: 2026-05-29T23:35:56Z

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import math
import random
import sys
import pathlib
import json
from dataclasses import asdict
import hashlib

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("Dimensions must be positive")
    volume = length * width * height
    surface_area = 2 * (length * width + width * height + height * length)
    return (volume * (36 * math.pi)**(1/3)) / surface_area

def haversine_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return 6371 * c  # radius of the Earth in kilometers

def gini_coefficient(values: list[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    n = len(xs)
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * xs)) / (n * np.sum(xs)))

def symbol_vector(symbol: str, dim: int = 10000) -> list[int]:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def bind(a: list[int], b: list[int]) -> list[int]:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def hybrid_operation(morphology: Morphology, entity: Entity) -> list[int]:
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    vector_id = symbol_vector(entity.id)
    vector_category = symbol_vector(entity.category)
    return bind([int(x * sphericity) for x in vector_id], vector_category)

def compute_gini_coefficient(hybrid_vectors: list[list[int]]) -> float:
    values = [sum(abs(x)) for x in hybrid_vectors]
    return gini_coefficient(values)

def compute_haversine_distance(entity1: Entity, entity2: Entity) -> float:
    return haversine_distance((entity1.lat, entity1.lon), (entity2.lat, entity2.lon))

def compute_cosine_similarity(vector1: list[int], vector2: list[int]) -> float:
    dot_product = sum(x * y for x, y in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(x**2 for x in vector1))
    magnitude2 = math.sqrt(sum(x**2 for x in vector2))
    if magnitude1 * magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

if __name__ == "__main__":
    morphology = Morphology(10.0, 5.0, 2.0, 100.0)
    entity1 = Entity("1", 40.7128, 74.0060, "A")
    entity2 = Entity("2", 34.0522, 118.2437, "B")

    hybrid_vector1 = hybrid_operation(morphology, entity1)
    hybrid_vector2 = hybrid_operation(morphology, entity2)

    gini_coeff = compute_gini_coefficient([hybrid_vector1, hybrid_vector2])
    haversine_dist = compute_haversine_distance(entity1, entity2)
    cosine_similarity = compute_cosine_similarity(hybrid_vector1, hybrid_vector2)

    print(f"Gini Coefficient: {gini_coeff}")
    print(f"Haversine Distance: {haversine_dist}")
    print(f"Cosine Similarity: {cosine_similarity}")