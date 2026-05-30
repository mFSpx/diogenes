# DARWIN HAMMER — match 843, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s7.py (gen3)
# parent_b: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s1.py (gen3)
# born: 2026-05-29T23:31:21Z

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic Types
# ----------------------------------------------------------------------
Node = Hashable
FeatureVec = Sequence[float]
Vector = np.ndarray  # bipolar hypervector stored as int8 array (+1 / -1)

# ----------------------------------------------------------------------
# Utility Functions (Parent A)
# ----------------------------------------------------------------------
def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two equal‑length real‑valued vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, gamma: float) -> float:
    """Gaussian‑like kernel based on distance *and* bandwidth γ."""
    return math.exp(-gamma * (r ** 2))


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


def hamming_int(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit strings."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Hyperdimensional Primitives (Parent B)
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    """Generate a random bipolar hypervector (+1 / -1) as int8 NumPy array."""
    rng = random.Random(seed)
    arr = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)),
        dtype=np.int8,
        count=dim,
    )
    return arr


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministic hypervector for a symbolic token."""
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (XOR for bipolar vectors)."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal length")
    return a * b  # int8 multiplication yields +1 / -1


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Superposition: element‑wise sum followed by sign‑binarisation."""
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    stacked = np.vstack(vecs).astype(np.int16)  # avoid overflow
    summed = stacked.sum(axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


def hd_similarity(a: Vector, b: Vector) -> float:
    """Cosine‑like similarity for bipolar vectors (dot / dim)."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal length")
    dot = int(a.dot(b))  # dot is integer in [-dim, dim]
    return dot / a.shape[0]


# ----------------------------------------------------------------------
# Sparse WTA Expansion (Parent B – refined)
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """
    Hash‑based sparse projection of a short list into an m‑dimensional vector.
    Each input value contributes to three positions (collision count = 3).
    The position keeps the value with larger absolute magnitude (preserving sign).
    """
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):  # three hash collisions per entry
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            idx = int.from_bytes(h[:4], "big") % m
            if abs(v) > abs(out[idx]):
                out[idx] = v
    return out


def binarize_dense(vec: List[float]) -> Vector:
    """Convert a dense real‑valued vector to a bipolar hypervector by sign."""
    arr = np.where(np.array(vec) >= 0, 1, -1).astype(np.int8)
    return arr


# ----------------------------------------------------------------------
# Hybrid Functions (Improved)
# ----------------------------------------------------------------------
def hypervector_from_features(
    values: List[float], dim: int = 10000, salt: str = ""
) -> Vector:
    """
    Project a real‑valued feature list into a bipolar hypervector.
    1. Sparse‑WTA expansion to a dense float vector of length `dim`.
    2. Sign‑binarisation to obtain a bipolar vector.
    """
    dense = expand(values, dim, salt)
    return binarize_dense(dense)


def _hamming_distance_vec(a: Vector, b: Vector) -> int:
    """
    Fast Hamming distance for bipolar int8 vectors using bit‑packing.
    Packs +1 → 0, -1 → 1 and counts differing bits.
    """
    if a.shape != b.shape:
        raise ValueError("vectors must have equal length")
    # Map +1 → 0, -1 → 1
    a_bits = np.packbits(((a + 1) // 2).astype(np.uint8))
    b_bits = np.packbits(((b + 1) // 2).astype(np.uint8))
    xor = np.bitwise_xor(a_bits, b_bits)
    return int(np.unpackbits(xor).sum())


def rbf_kernel_on_hypervectors(
    hv_dict: Dict[Node, Vector],
    gamma: float = 0.001,
) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute a dense RBF‑like kernel matrix directly on Hamming distance:

        K[i, j] = exp(-γ * Hamming(h_i, h_j))

    This avoids the unnecessary Euclidean conversion and respects the natural
    binary geometry of hypervectors.
    """
    nodes = list(hv_dict.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    # Pre‑pack vectors once for speed
    packed = [
        np.packbits(((hv_dict[node] + 1) // 2).astype(np.uint8))
        for node in nodes
    ]

    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            xor = np.bitwise_xor(packed[i], packed[j])
            ham = int(np.unpackbits(xor).sum())
            val = math.exp(-gamma * ham)
            K[i, j] = val
            K[j, i] = val
    return K, nodes


def fused_similarity(
    node_a: Node,
    node_b: Node,
    hv_dict: Dict[Node, Vector],
    gamma: float = 0.001,
    alpha: float = 0.5,
) -> float:
    """
    Deeply integrate the two similarity notions by forming a *geometric*
    blend of the Hamming‑based RBF kernel and the angular (cosine‑like) HD
    similarity:

        fused = (rbf ** α) * ((1 + hd) / 2) ** (1‑α)

    The `(1+hd)/2` term rescales the HD similarity from [-1, 1] to [0, 1],
    making the product well‑behaved.
    """
    if node_a not in hv_dict or node_b not in hv_dict:
        raise KeyError("Both nodes must exist in the hypervector dictionary")

    ha = hv_dict[node_a]
    hb = hv_dict[node_b]

    # Hamming‑based RBF part
    ham = _hamming_distance_vec(ha, hb)
    rbf = math.exp(-gamma * ham)

    # Rescaled HD cosine similarity
    hd = hd_similarity(ha, hb)
    hd_rescaled = (1.0 + hd) / 2.0  # now in [0, 1]

    # Geometric blend (equivalent to weighted log‑sum)
    return (rbf ** alpha) * (hd_rescaled ** (1.0 - alpha))


def bind_with_symbol(
    node: Node, hv_dict: Dict[Node, Vector], symbol: str
) -> Vector:
    """
    Bind a node’s hypervector with a deterministic symbol hypervector.
    Returns a *new* hypervector without mutating the original dictionary.
    """
    if node not in hv_dict:
        raise KeyError(f"Node {node!r} not found in hv_dict")
    symbol_hv = symbol_vector(symbol, dim=hv_dict[node].shape[0])
    return bind(hv_dict[node], symbol_hv)