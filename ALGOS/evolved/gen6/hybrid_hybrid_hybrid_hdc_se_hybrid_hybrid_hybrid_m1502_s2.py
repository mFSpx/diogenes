# DARWIN HAMMER — match 1502, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m424_s0.py (gen5)
# born: 2026-05-29T23:36:50Z

"""
DARWIN HAMMER — hybrid match 425, survivor 1
gen: 5
parent_a: hybrid_hybrid_hdc_serpentina_self_righ_m50_s2.py (gen1)
parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m424_s0.py (gen4)
born: 2026-05-29T23:30:55Z

This module integrates the hybrid_hdc_serpentina_self_righ and hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid algorithms into a single hybrid system.
The mathematical bridge between the two structures is formed by using the NLMS prediction error as a proxy for the cosine similarity between two hypervectors.
This is achieved by combining the NLMS prediction error with the cosine similarity factor to obtain an effective similarity measure,
and then using the hybrid hypervector representation to inform the probabilistic transformation of the edge contributions.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def random_vector(dim: int = 10000, seed: str | int | None = None) -> np.ndarray:
    """Generate a bipolar random hypervector."""
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)), dtype=np.int8, count=dim
    )
    return data

def symbol_vector(symbol: str, dim: int = 10000) -> np.ndarray:
    """Deterministically map a symbol to a bipolar hypervector."""
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], byteorder="big"
    )
    return random_vector(dim, seed)

def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise binding (multiplication) of two hypervectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b

def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Superposition of hypervectors followed by binarization (sign)."""
    vecs = vectors
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vecs, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot-product prediction w·x."""
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update.
    """
    prediction = nlms_predict(weights, x)
    error = target - prediction
    weights_update = mu * error * x / (x @ x + eps)
    new_weights = weights + weights_update
    return new_weights, error

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def words(text: str) -> List[str]:
    return [word for word in text.lower().split()]

def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = {word: ws.count(word) / total for word in set(ws)}
    return cnt

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """True cosine similarity handling sparse (zero) entries."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

def hybrid_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Hybrid similarity measure combining NLMS prediction error and cosine similarity."""
    error = nlms_update(np.zeros_like(a), a, nlms_predict(b, a))[1]
    return error * cosine_similarity(a, b)

def hybrid_bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Hybrid superposition of hypervectors followed by binarization (sign)."""
    vecs = vectors
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vecs, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    hybrid_sim = [hybrid_similarity(vecs[i], vecs[0]) for i in range(len(vecs))]
    hybrid_sim = np.array(hybrid_sim)
    return np.where(summed >= 0, 1, -1).astype(np.int8)

def hybrid_test():
    dim = 10000
    seed = random.getrandbits(32)
    np.random.seed(seed)
    a = random_vector(dim)
    b = random_vector(dim)
    c = random_vector(dim)
    vectors = [a, b, c]
    bundle_vec = hybrid_bundle(vectors)
    print(bundle_vec)

if __name__ == "__main__":
    hybrid_test()