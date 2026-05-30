# DARWIN HAMMER — match 1596, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s1.py (gen3)
# parent_b: semantic_neighbors.py (gen0)
# born: 2026-05-29T23:37:42Z

import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import sys
import hashlib

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Voronoi partitioning
# ----------------------------------------------------------------------

Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def weighted_voronoi(points: list[Point], seeds: list[Point], weights: list[float]) -> dict[int, list[Point]]:
    if len(seeds) != len(weights):
        raise ValueError("Number of seeds and weights must match")
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        dists = [(distance(p, seeds[i]), i) for i in range(len(seeds))]
        dists.sort(key=lambda x: (x[0] * (1 / weights[i]), i))
        regions[dists[0][1]].append(p)
    return regions

# ----------------------------------------------------------------------
# Semantic neighbor enclave
# ----------------------------------------------------------------------

_ENCLAVE: dict[str, list[float]] = {}

def clear_enclave() -> None:
    _ENCLAVE.clear()

def register_document(doc_id: str, vector: list[float]) -> None:
    if doc_id in _ENCLAVE:
        raise ValueError("Document ID already exists")
    _ENCLAVE[doc_id] = vector

def _cos(a, b):
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den

def semantic_neighbors(doc_id: str, k: int = 5) -> list[tuple[str, float]]:
    if doc_id not in _ENCLAVE:
        raise ValueError("Document ID does not exist")
    v = _ENCLAVE[doc_id]
    return sorted(((d, _cos(v, w)) for d, w in _ENCLAVE.items() if d != doc_id), key=lambda x: (-x[1], x[0]))[:k]

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------

def hybrid_operation(points: list[Point], seeds: list[Point], doc_id: str, vector: list[float]) -> dict[int, list[Point]]:
    register_document(doc_id, vector)
    weights = [1.0 + _cos(vector, _ENCLAVE[d]) for d in _ENCLAVE]
    weighted_regions = weighted_voronoi(points, seeds, weights)
    return weighted_regions

def generate_random_vector(length: int) -> list[float]:
    return [random.uniform(0, 1) for _ in range(length)]

def smoke_test() -> None:
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(100)]
    seeds = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(10)]
    doc_id = "test_doc"
    vector_length = 10
    vector = generate_random_vector(vector_length)
    weighted_regions = hybrid_operation(points, seeds, doc_id, vector)
    for region in weighted_regions.values():
        print(region)

if __name__ == "__main__":
    smoke_test()