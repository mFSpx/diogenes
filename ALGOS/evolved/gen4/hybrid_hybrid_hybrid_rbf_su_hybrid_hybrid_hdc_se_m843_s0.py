# DARWIN HAMMER — match 843, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s7.py (gen3)
# parent_b: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s1.py (gen3)
# born: 2026-05-29T23:31:21Z

"""
Hybrid Algorithm: RBF‑HD Fusion

This module combines the two parent algorithms:

* **Parent A** – builds a dense Radial Basis Function (RBF) kernel matrix from
  real‑valued feature vectors using Euclidean distance and a Gaussian kernel.
* **Parent B** – creates binary (bipolar) hyperdimensional vectors, provides
  binding/bundling operations and a cosine‑like similarity measure.

**Mathematical Bridge**

For bipolar hypervectors `v ∈ {‑1,+1}ⁿ` the squared Euclidean distance is
directly proportional to the Hamming distance:


‖v_i – v_j‖² = 4 * Hamming(v_i, v_j)


Consequently the Gaussian RBF kernel can be expressed in terms of the Hamming
distance of hypervectors.  The fusion therefore proceeds as:

1. Project arbitrary real‑valued features into a binary hypervector using the
   sparse‑WTA expansion (Parent B).
2. Compute the RBF kernel on the resulting hypervectors via Euclidean distance
   (which is equivalent to a Hamming‑based kernel).
3. Combine the RBF similarity with the conventional HD cosine similarity
   (binding‑aware) to obtain a unified similarity measure.

The three public functions below illustrate the hybrid pipeline.
"""

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

    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            # Euclidean distance on bipolar vectors
            dist = np.linalg.norm(arr[i] - arr[j])
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
    Demonstrates a binding operation that couples a node’s hypervector with a
    deterministic symbol hypervector.  The result can be used as a query key
    in associative memory or as part of a larger compositional representation.
    """
    if node not in hv_dict:
        raise KeyError(f"Node {node!r} not found")
    sym_vec = symbol_vector(symbol, dim=len(hv_dict[node]))
    return bind(hv_dict[node], sym_vec)


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    raw_features: Dict[Node, List[float]] = {
        "A": [0.1, 0.4, -0.2],
        "B": [0.0, 0.35, -0.15],
        "C": [0.9, -0.1, 0.3],
    }

    # Step 1 – turn each feature list into a hypervector
    dim = 2048  # modest dimensionality for the demo
    hv_dict = {
        node: hypervector_from_features(feat, dim=dim, salt="demo")
        for node, feat in raw_features.items()
    }

    # Step 2 – compute the hybrid RBF kernel matrix
    K, order = rbf_kernel_on_hypervectors(hv_dict, epsilon=0.5)
    print("RBF kernel matrix (order = {}):".format(order))
    print(K)

    # Step 3 – compute a fused similarity between two nodes
    sim_AB = fused_similarity("A", "B", hv_dict, epsilon=0.5, alpha=0.7)
    print(f"Fused similarity A‑B: {sim_AB:.4f}")

    # Step 4 – bind a node with a symbolic hypervector and show its similarity
    bound_vec = bind_with_symbol("A", hv_dict, symbol="weather")
    # Compare bound vector to original via HD similarity (should be ~0 because binding flips signs)
    bound_sim = hd_similarity(bound_vec, hv_dict["A"])
    print(f"Similarity after binding with 'weather': {bound_sim:.4f}")

    # Verify that the perceptual hash bridge works (optional sanity)
    phash_A = compute_phash([float(x) for x in hv_dict["A"][:64]])
    phash_B = compute_phash([float(x) for x in hv_dict["B"][:64]])
    ham = hamming_distance(phash_A, phash_B)
    print(f"Hamming distance between perceptual hashes of A and B: {ham}")

    sys.exit(0)