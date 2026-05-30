# DARWIN HAMMER — match 1502, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m424_s0.py (gen5)
# born: 2026-05-29T23:36:50Z

"""
This module integrates the hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s2 and 
hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m424_s0 algorithms into a single hybrid system. 
The mathematical bridge between the two structures is formed by using the cosine similarity 
between hyperdimensional vectors as a proxy for the likelihood of error in the NLMS prediction 
and the epistemic certainty calculation. This is achieved by combining the cosine similarity 
with the NLMS prediction error to obtain an effective edge weight.

Parent A: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s2
Parent B: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m424_s0
"""

import numpy as np
import math
import random
import sys
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Tuple

Vector = np.ndarray  # bipolar hypervector of dtype int8

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    """Generate a bipolar random hypervector."""
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)), dtype=np.int8, count=dim
    )
    return data

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministically map a symbol to a bipolar hypervector."""
    seed = int.from_bytes(
        hashlib.md5(symbol.encode("utf-8")).digest()[:8], byteorder="big"
    )
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise binding (multiplication) of two hypervectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b

def bundle(vectors: List[Vector]) -> Vector:
    """Superposition of hypervectors followed by binarization (sign)."""
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vecs, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)

def cosine_similarity(a: Vector, b: Vector) -> float:
    """True cosine similarity handling sparse (zero) entries."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

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

def calculate_epistemic_certainty(factor: float, error: float) -> float:
    return factor * (1 - error)

def hybrid_operation(symbol1: str, symbol2: str, weights: np.ndarray, x: np.ndarray, target: float) -> Tuple[float, Vector]:
    v1 = symbol_vector(symbol1)
    v2 = symbol_vector(symbol2)
    similarity = cosine_similarity(v1, v2)
    new_weights, error = nlms_update(weights, x, target)
    epistemic_certainty = calculate_epistemic_certainty(0.5, abs(error))
    effective_weight = similarity * epistemic_certainty
    return effective_weight, bundle([v1, v2])

def another_hybrid_operation(text: str, weights: np.ndarray, x: np.ndarray, target: float) -> Tuple[float, Dict[str, float]]:
    lsm = {word: text.count(word) / len(text) for word in set(text)}
    new_weights, error = nlms_update(weights, x, target)
    epistemic_certainty = calculate_epistemic_certainty(0.5, abs(error))
    return epistemic_certainty, lsm

def yet_another_hybrid_operation(vectors: List[Vector], weights: np.ndarray, x: np.ndarray, target: float) -> Tuple[float, Vector]:
    bundled_vector = bundle(vectors)
    new_weights, error = nlms_update(weights, x, target)
    similarity = cosine_similarity(bundled_vector, random_vector())
    return similarity, bundled_vector

if __name__ == "__main__":
    symbol1 = "apple"
    symbol2 = "banana"
    weights = np.random.rand(10)
    x = np.random.rand(10)
    target = 0.5
    effective_weight, bundled_vector = hybrid_operation(symbol1, symbol2, weights, x, target)
    print(effective_weight)
    print(bundled_vector)

    text = "apple banana apple"
    epistemic_certainty, lsm = another_hybrid_operation(text, weights, x, target)
    print(epistemic_certainty)
    print(lsm)

    vectors = [random_vector() for _ in range(10)]
    similarity, bundled_vector = yet_another_hybrid_operation(vectors, weights, x, target)
    print(similarity)
    print(bundled_vector)