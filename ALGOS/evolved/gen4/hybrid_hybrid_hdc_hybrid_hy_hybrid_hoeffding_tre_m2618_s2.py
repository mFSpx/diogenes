# DARWIN HAMMER — match 2618, survivor 2
# gen: 4
# parent_a: hybrid_hdc_hybrid_hybrid_bandit_m146_s1.py (gen3)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s3.py (gen1)
# born: 2026-05-29T23:43:16Z

"""Hybrid Hyperdimensional Hoeffding‑Gini (HD‑HG) algorithm.

Parents:
- hybrid_hdc_hybrid_hybrid_bandit_m146_s1.py (Hyperdimensional Computing primitives)
- hybrid_hoeffding_tree_gini_coefficient_m13_s3.py (Hoeffding bound + Gini coefficient)

Mathematical bridge:
Each attribute of a streaming example is encoded as a high‑dimensional (HD) bipolar vector.
The *binding* operation is used to modulate the Hoeffding confidence term:
    ε̂ = ε ⊗ c
where ε is the scalar Hoeffding bound, c is a binary confidence hypervector
(and ⊗ denotes element‑wise multiplication, i.e. binding).  The sign of ε̂
determines whether a split should be taken.

The *bundle* operation aggregates HD statistics of a node (sum of bound
vectors for all observed examples).  From the bundled vector we extract a
real‑valued proxy (average similarity to a reference hypervector) that
serves as the Gini‑like impurity measure for that node.  This allows the
classic Hoeffding‑Gini split test to operate entirely in the HD space,
yielding a single unified decision rule.

The three public functions below demonstrate:
1. hd_encode_stream – encode a stream of numeric records into HD vectors.
2. hd_gini_from_bundle – compute a Gini‑style impurity from a bundled HD
   vector.
3. hd_hoeffding_split – decide whether to split a node using the bound
   modulated by binding and the HD‑derived Gini impurity.
"""

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional Computing primitives (Parent A)
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
# Hoeffding‑Gini primitives (Parent B)
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
# Global confidence hypervector (static, can be re‑seeded by the user)
_CONFIDENCE_VECTOR: Vector = random_vector()

def hd_encode_record(record: Tuple[float, ...],
                    dim: int = 10000) -> List[Vector]:
    """Encode each numeric field of *record* into a bipolar HD vector.

    The value is first transformed into a deterministic seed via its
    binary representation; then a random bipolar vector of length *dim*
    is generated.  The result is a list of vectors, one per field.
    """
    vectors: List[Vector] = []
    for i, val in enumerate(record):
        # deterministic seed from float bits + position
        bits = np.float64(val).tobytes()
        seed = int.from_bytes(hashlib.sha256(bits + i.to_bytes(2, 'little')).digest()[:8], 'big')
        vectors.append(random_vector(dim, seed))
    return vectors

def hd_encode_stream(stream: Iterable[Tuple[float, ...]],
                    dim: int = 10000) -> List[List[Vector]]:
    """Encode an entire stream of records; returns a list of per‑record
    vector lists."""
    return [hd_encode_record(rec, dim) for rec in stream]

def hd_gini_from_bundle(bundled: Vector,
                        reference: Vector | None = None) -> float:
    """Derive a Gini‑style impurity from a bundled HD vector.

    The impurity is the Gini coefficient of the absolute similarity scores
    between the bundled vector and a reference hypervector (default:
    a random reference).  Higher similarity variance indicates a more
    heterogeneous node, analogous to higher Gini impurity.
    """
    if reference is None:
        reference = random_vector(len(bundled))
    # similarity per dimension is either 1 or -1; map to 0/1 for Gini
    sims = [(1 + similarity([b], [r])) / 2 for b, r in zip(bundled, reference)]
    return gini_coefficient(sims)

def hd_hoeffding_split(node_vectors: List[List[Vector]],
                       r: float,
                       delta: float,
                       n_observations: int,
                       tie_threshold: float = 0.05) -> SplitDecision:
    """Hybrid split decision for a node.

    *node_vectors* – list of encoded records belonging to the node.
    The function bundles each feature column, computes an HD‑derived Gini
    impurity per feature, and applies the Hoeffding bound whose scalar
    ε is modulated (bound vector) by *binding* with the global confidence
    hypervector.  The bound’s sign (positive → split) is extracted via the
    mean of the bound vector.

    Returns a SplitDecision dataclass.
    """
    if not node_vectors:
        raise ValueError("node_vectors must contain at least one record")

    # Transpose: feature_index -> list of vectors for that feature
    num_features = len(node_vectors[0])
    feature_columns: List[List[Vector]] = [[] for _ in range(num_features)]
    for rec in node_vectors:
        for idx, vec in enumerate(rec):
            feature_columns[idx].append(vec)

    # Compute bundled vector per feature and its HD‑Gini impurity
    gini_per_feature: List[float] = []
    for col in feature_columns:
        bundled = bundle(col)
        gini = hd_gini_from_bundle(bundled)
        gini_per_feature.append(gini)

    # Identify best and second‑best Gini (higher impurity is better for split)
    sorted_ginis = sorted(gini_per_feature, reverse=True)
    best_gini = sorted_ginis[0]
    second_best_gini = sorted_ginis[1] if len(sorted_ginis) > 1 else 0.0

    # Hoeffding bound ε (scalar) and its HD modulation
    eps_scalar = hoeffding_bound(r, delta, n_observations)
    # Bind scalar ε with confidence vector: treat ε as +1 or -1 depending on magnitude
    eps_sign = 1 if eps_scalar >= 0 else -1
    eps_vector = bind(_CONFIDENCE_VECTOR, [eps_sign] * len(_CONFIDENCE_VECTOR))

    # Use the mean of eps_vector as a signed confidence modifier
    eps_mod = sum(eps_vector) / len(eps_vector)  # will be +1 or -1
    # Effective epsilon for split test
    effective_eps = eps_scalar * eps_mod

    # Apply split rule with the effective epsilon
    split = best_gini - second_best_gini > effective_eps or abs(effective_eps) < tie_threshold
    reason = ("gap_exceeds_mod_bound" if best_gini - second_best_gini > effective_eps
              else ("tie_threshold" if abs(effective_eps) < tie_threshold else "wait"))
    return SplitDecision(split, effective_eps, best_gini - second_best_gini, reason)

# ----------------------------------------------------------------------
# Simple demonstration / smoke test
# ----------------------------------------------------------------------
def _demo():
    # Create a tiny synthetic stream (two features)
    stream = [
        (0.1, 0.9),
        (0.2, 0.8),
        (0.15, 0.85),
        (0.9, 0.1),
        (0.85, 0.15),
    ]

    # Encode the stream into HD vectors
    encoded = hd_encode_stream(stream, dim=2048)

    # Simulate a node that currently holds the first three records
    node_vectors = encoded[:3]

    # Parameters for Hoeffding bound
    r = 1.0          # range of Gini (0‑1)
    delta = 0.05
    n_observations = len(node_vectors)

    decision = hd_hoeffding_split(node_vectors, r, delta, n_observations)

    print("HD‑Gini split decision:")
    print(f"  should_split : {decision.should_split}")
    print(f"  epsilon      : {decision.epsilon:.5f}")
    print(f"  gain_gap     : {decision.gain_gap:.5f}")
    print(f"  reason       : {decision.reason}")

if __name__ == "__main__":
    _demo()