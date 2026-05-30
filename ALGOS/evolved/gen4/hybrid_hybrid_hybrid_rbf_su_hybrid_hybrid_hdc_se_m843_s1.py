# DARWIN HAMMER — match 843, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s7.py (gen3)
# parent_b: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s1.py (gen3)
# born: 2026-05-29T23:31:21Z

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic Types
# ----------------------------------------------------------------------
Node = Hashable
FeatureVec = Sequence[float]
Vector = List[int]  # bipolar hypervector (‑1 / +1)

# ----------------------------------------------------------------------
# Utility Functions (Parent A)
# ----------------------------------------------------------------------
def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def compute_phash(values: List[float]) -> int:
    """
    Simple perceptual hash: 1‑bit per value relative to the median.
    Uses up to 64 bits; remaining values are ignored.
    """
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit strings."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Hyperdimensional Primitives (Parent B)
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministic hypervector for a symbolic token."""
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (XOR for bipolar vectors)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Superposition: element‑wise sum followed by sign‑binarisation."""
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    dim = len(vecs[0])
    for v in vecs:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    summed = [sum(comp) for comp in zip(*vecs)]
    return [1 if s >= 0 else -1 for s in summed]


def hd_similarity(a: Vector, b: Vector) -> float:
    """Cosine‑like similarity for bipolar vectors (dot / dim)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)  # because |a| = |b| = sqrt(dim) for bipolar vectors


# ----------------------------------------------------------------------
# Sparse WTA Expansion (Parent B – partially shown)
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """
    Hash‑based sparse projection of a short list into an m‑dimensional vector.
    Each input value contributes to three positions (collision count = 3).
    The position keeps the maximum absolute value seen (preserving sign).
    """
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):  # three hash collisions per entry
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            idx = int.from_bytes(h[:4], "big") % m
            # keep the value with larger magnitude (preserve sign)
            if abs(v) > abs(out[idx]):
                out[idx] = v
    return out


def binarize_dense(vec: List[float]) -> Vector:
    """Convert a dense real‑valued vector to a bipolar hypervector by sign."""
    return [1 if x >= 0 else -1 for x in vec]


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def hypervector_from_features(values: List[float], dim: int = 10000, salt: str = "") -> Vector:
    """
    Project a real‑valued feature list into a bipolar hypervector.
    1. Sparse‑WTA expansion to a dense float vector of length `dim`.
    2. Sign‑binarisation to obtain a bipolar vector.
    """
    dense = expand(values, dim, salt)
    return binarize_dense(dense)


def rbf_kernel_on_hypervectors(
    hv_dict: Dict[Node, Vector],
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute a dense RBF kernel matrix where each entry is
        K[i, j] = exp(-ε² * ‖h_i – h_j‖²)

    Because hypervectors are bipolar, ‖h_i – h_j‖² = 4 * Hamming(h_i, h_j),
    which yields the same result as a Hamming‑based kernel.
    """
    nodes = list(hv_dict.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    # Pre‑convert hypervectors to numpy arrays of type float64 for speed
    arr = np.vstack([np.array(hv_dict[node], dtype=np.float64) for node in nodes])

    # Calculate Hamming distance directly
    hamming_distances = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            hamming_distances[i, j] = hamming_distance(int(''.join(map(str, arr[i])), 2), int(''.join(map(str, arr[j])), 2))
            hamming_distances[j, i] = hamming_distances[i, j]

    # Compute RBF kernel
    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            dist = 4 * hamming_distances[i, j]
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes


def fused_similarity(
    node_a: Node,
    node_b: Node,
    hv_dict: Dict[Node, Vector],
    epsilon: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """
    Blend two similarity measures:
      * RBF similarity derived from Euclidean distance of hypervectors.
      * HD cosine similarity (binding‑aware) between the same hypervectors.

    The final score is `alpha * rbf + (1‑alpha) * hd`.
    """
    if node_a not in hv_dict or node_b not in hv_dict:
        raise KeyError("Both nodes must exist in the hypervector dictionary")

    ha = np.array(hv_dict[node_a], dtype=np.float64)
    hb = np.array(hv_dict[node_b], dtype=np.float64)

    # RBF part
    dist = np.linalg.norm(ha - hb)
    rbf = gaussian(dist, epsilon)

    # HD cosine part (identical to hd_similarity)
    hd = hd_similarity(hv_dict[node_a], hv_dict[node_b])

    return alpha * rbf + (1.0 - alpha) * hd


def bind_with_symbol(node: Node, hv_dict: Dict[Node, Vector], symbol: str) -> Vector:
    """
    Demonstrates a binding operation that couples a node’s hypervector
    with a symbolic token's hypervector.
    """
    symbol_hv = symbol_vector(symbol)
    node_hv = hv_dict[node]
    return bind(node_hv, symbol_hv)