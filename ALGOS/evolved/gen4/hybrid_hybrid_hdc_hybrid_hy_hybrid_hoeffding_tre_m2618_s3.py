# DARWIN HAMMER — match 2618, survivor 3
# gen: 4
# parent_a: hybrid_hdc_hybrid_hybrid_bandit_m146_s1.py (gen3)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s3.py (gen1)
# born: 2026-05-29T23:43:16Z

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional Computing primitives
# ----------------------------------------------------------------------
Vector = List[int]

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(hashlib.sha256(symbol.encode('utf-8')).digest()[:8], 'big')
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: List[Vector]) -> Vector:
    if not vectors:
        raise ValueError('at least one vector is required')
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError('vectors must have equal length')
    sums = [0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    return [1 if x >= 0 else -1 for x in sums]

def similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    if not a:
        raise ValueError('vectors must not be empty')
    return sum(x * y for x, y in zip(a, b)) / len(a)

# ----------------------------------------------------------------------
# Hoeffding‑Gini primitives
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gini: float, second_best_gini: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gini - second_best_gini
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

# ----------------------------------------------------------------------
# Hybrid HD‑HG core
# ----------------------------------------------------------------------
_CONFIDENCE_VECTOR: Vector = random_vector()

def hd_encode_record(record: Tuple[float, ...],
                    dim: int = 10000) -> List[Vector]:
    vectors: List[Vector] = []
    for i, val in enumerate(record):
        bits = np.float64(val).tobytes()
        seed = int.from_bytes(hashlib.sha256(bits + i.to_bytes(2, 'little')).digest()[:8], 'big')
        vectors.append(random_vector(dim, seed))
    return vectors

def hd_encode_stream(stream: Iterable[Tuple[float, ...]],
                    dim: int = 10000) -> List[List[Vector]]:
    return [hd_encode_record(rec, dim) for rec in stream]

def hd_gini_from_bundle(bundled: Vector,
                        reference: Vector | None = None) -> float:
    if reference is None:
        reference = random_vector(len(bundled))
    sims = [(1 + similarity([b], [r])) / 2 for b, r in zip(bundled, reference)]
    return gini_coefficient(sims)

def hd_hoeffding_split(node_vectors: List[List[Vector]],
                       r: float,
                       delta: float,
                       n_observations: int,
                       tie_threshold: float = 0.05) -> SplitDecision:
    if not node_vectors:
        raise ValueError("node_vectors must contain at least one record")

    num_features = len(node_vectors[0])
    feature_columns: List[List[Vector]] = [[] for _ in range(num_features)]
    for rec in node_vectors:
        for idx, vec in enumerate(rec):
            feature_columns[idx].append(vec)

    gini_per_feature: List[float] = []
    for col in feature_columns:
        bundled = bundle(col)
        gini = hd_gini_from_bundle(bundled)
        gini_per_feature.append(gini)

    sorted_ginis = sorted(gini_per_feature, reverse=True)
    best_gini = sorted_ginis[0]
    second_best_gini = sorted_ginis[1] if len(sorted_ginis) > 1 else 0.0

    eps_scalar = hoeffding_bound(r, delta, n_observations)
    eps_hd = bind(_CONFIDENCE_VECTOR, [1 if x > 0 else -1 for x in [eps_scalar]])

    mean_eps_hd = sum(eps_hd) / len(eps_hd)
    split_decision = should_split(best_gini, second_best_gini, r, delta, n_observations, tie_threshold)

    return SplitDecision(should_split=mean_eps_hd > 0, 
                         epsilon=eps_scalar, 
                         gain_gap=best_gini - second_best_gini, 
                         reason=split_decision.reason)

def improve_hd_hg_stream_encoding(stream: Iterable[Tuple[float, ...]],
                                dim: int = 10000,
                                num_hash_functions: int = 5) -> List[List[Vector]]:
    encoded_stream = hd_encode_stream(stream, dim)
    improved_encoded_stream = []

    for record in encoded_stream:
        hash_values = []
        for i in range(num_hash_functions):
            hash_value = 0
            for vec in record:
                hash_value += similarity(vec, random_vector(dim))
            hash_values.append(hash_value)

        improved_record = []
        for i, vec in enumerate(record):
            hash_value = hash_values[i % num_hash_functions]
            improved_vec = bind(vec, [1 if x > 0 else -1 for x in [hash_value]])
            improved_record.append(improved_vec)

        improved_encoded_stream.append(improved_record)

    return improved_encoded_stream