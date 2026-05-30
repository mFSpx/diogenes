# DARWIN HAMMER — match 5768, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s1.py (gen3)
# parent_b: pheromone.py (gen0)
# born: 2026-05-30T00:04:40Z

import numpy as np
import math
import random
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

# Hyperdimensional primitives
Vector = List[int]  # bipolar hypervector

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: Iterable[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    dim = len(vecs[0])
    for v in vecs:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    # element-wise sum then binarize by sign
    summed = [sum(comp) for comp in zip(*vecs)]
    return [1 if s >= 0 else -1 for s in summed]

def similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity for bipolar vectors (identical to normalized dot)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)  # because |a| = |b| = sqrt(dim) for bipolar vectors

# Sparse WTA primitives
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash-based sparse projection of a short list into an m-dimensional vector."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):  # three hash collisions per entry
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            idx = int.from_bytes(h, 'big') % m
            out[idx] += v
    return out

# Pheromone signal processing
@dataclass
class PheromoneSignal:
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: float
    timestamp: float

def decay(pheromone_signal: PheromoneSignal, current_time: float) -> float:
    elapsed_time = current_time - pheromone_signal.timestamp
    return pheromone_signal.signal_value * math.exp(-elapsed_time / pheromone_signal.half_life_seconds)

# Hybrid operations
def pheromone_vector(pheromone_signal: PheromoneSignal, dim: int = 10000) -> Vector:
    values = [pheromone_signal.signal_value]
    sparse_vector = expand(values, dim, salt=pheromone_signal.surface_key)
    return [1 if x >= 0 else -1 for x in sparse_vector]

def hybrid_similarity(pheromone_signal1: PheromoneSignal, pheromone_signal2: PheromoneSignal) -> float:
    vector1 = pheromone_vector(pheromone_signal1)
    vector2 = pheromone_vector(pheromone_signal2)
    return similarity(vector1, vector2)

def hybrid_decay(pheromone_signal: PheromoneSignal, current_time: float) -> Vector:
    decayed_signal = decay(pheromone_signal, current_time)
    return pheromone_vector(PheromoneSignal(pheromone_signal.surface_key, pheromone_signal.signal_kind, decayed_signal, pheromone_signal.half_life_seconds, current_time))

def temporal_similarity(signal1: PheromoneSignal, signal2: PheromoneSignal, current_time: float) -> float:
    vector1 = hybrid_decay(signal1, current_time)
    vector2 = hybrid_decay(signal2, current_time)
    return similarity(vector1, vector2)

if __name__ == "__main__":
    pheromone_signal1 = PheromoneSignal("surface_key1", "signal_kind1", 1.0, 10.0, 0.0)
    pheromone_signal2 = PheromoneSignal("surface_key2", "signal_kind2", 0.5, 5.0, 0.0)
    current_time = 3.0
    print(hybrid_similarity(pheromone_signal1, pheromone_signal2))
    print(temporal_similarity(pheromone_signal1, pheromone_signal2, current_time))