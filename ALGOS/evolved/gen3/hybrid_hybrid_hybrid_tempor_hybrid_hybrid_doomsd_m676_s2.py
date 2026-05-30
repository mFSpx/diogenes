# DARWIN HAMMER — match 676, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_temporal_moti_hybrid_doomsday_cale_m218_s1.py (gen2)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hdc_serpentin_m313_s3.py (gen2)
# born: 2026-05-29T23:30:28Z

import datetime as dt
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any
import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Parent A – spatial-temporal utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

def haversine_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return 6371 * c  # radius of the Earth in kilometers

def doomsday_numpy(date: dt.date) -> int:
    t = [int(date.strftime("%Y")) // 100, int(date.strftime("%Y")) % 100]
    return (t[0] // 4 - 2 * t[0] + t[1] // 4 + 26 * (t[1] % 28 + 26 + 28 * ((t[1] + 27) % 28)) // 28) % 7

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    n = len(xs)
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * xs)) / (n * np.sum(xs)))

# ----------------------------------------------------------------------
# Parent B – hyperdimensional primitives
# ----------------------------------------------------------------------
Vector = list[int]

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: Iterable[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        return []
    dim = len(vecs[0])
    return [sum([vecs[i][j] for i in range(len(vecs))]) / len(vecs) for j in range(dim)]

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_temporal_motif_encoding(motif: List[Entity], date: dt.date) -> Tuple[float, Vector]:
    weekday = doomsday_numpy(date)
    gini = gini_coefficient([entity.score for entity in motif])
    vector = bind(symbol_vector(str(weekday), dim=len(motif)), 
                  [int(x) for x in np.random.choice([-1, 1], size=len(motif))])
    return gini, vector

def hyperdimensional_weekday_analysis(vectors: Iterable[Vector], weekdays: Iterable[int]) -> Dict[int, Vector]:
    result = {}
    for vector, weekday in zip(vectors, weekdays):
        if weekday not in result:
            result[weekday] = []
        result[weekday].append(vector)
    for weekday in result:
        result[weekday] = bundle(result[weekday])
    return result

def hybrid_gini_score(motif: List[Entity], date: dt.date) -> float:
    weekday = doomsday_numpy(date)
    gini = gini_coefficient([entity.score for entity in motif])
    return gini * (1 - gini)

def weighted_hybrid_temporal_motif_encoding(motif: List[Entity], date: dt.date) -> Tuple[float, Vector]:
    weekday = doomsday_numpy(date)
    scores = [entity.score for entity in motif]
    gini = gini_coefficient(scores)
    vector_dim = len(motif)
    vectors = [symbol_vector(str(weekday), dim=vector_dim) for _ in range(len(motif))]
    weights = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]
    weighted_vectors = [bind(vectors[i], [int(x) for x in np.random.choice([-1, 1], size=vector_dim, p=[1 - w, w])]) for i, w in enumerate(weights)]
    return gini, bundle(weighted_vectors)

if __name__ == "__main__":
    entity1 = Entity("id1", 37.7749, -122.4194, "category1", 10.0)
    entity2 = Entity("id2", 37.7859, -122.4364, "category2", 20.0)
    motif = [entity1, entity2]
    date = dt.date(2022, 1, 1)
    score, vector = weighted_hybrid_temporal_motif_encoding(motif, date)
    print(score)
    print(vector)
    vectors = [random_vector() for _ in range(7)]
    weekdays = [i for i in range(7)]
    result = hyperdimensional_weekday_analysis(vectors, weekdays)
    print(result)
    gini = hybrid_gini_score(motif, date)
    print(gini)