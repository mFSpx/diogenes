# DARWIN HAMMER — match 3415, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_model_vram_sc_m2303_s2.py (gen3)
# born: 2026-05-29T23:50:05Z

"""Hybrid NLMS‑Hyperdimensional Adaptive Module
Parents:
- hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s2.py (NLMS, graph, phash)
- hybrid_hybrid_hybrid_doomsd_hybrid_model_vram_sc_m2303_s2.py (hyperdimensional vectors, Gini scaling, Doomsday‑derived symbol)

Mathematical bridge:
The Gini coefficient of the NLMS weight distribution scales updates of a
hyper‑dimensional weight‑matrix W.  The weekday obtained from the Doomsday
algorithm (here via datetime.weekday) is turned into a symbolic hypervector.
That hypervector is bound to a hypervector derived from the current NLMS
weights; the outer product of the bound vector is added to W, weighted by the
Gini coefficient.  Thus the adaptive filter’s statistics directly modulate a
hyperdimensional associative memory, fusing both topologies into a single
system.
"""

import numpy as np
import math
import random
import sys
import pathlib
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

# ----------------------------------------------------------------------
# Parent A – NLMS and similarity utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    id: int
    weight: float

def predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(np.dot(weights, x))

def nlms_update(weights: np.ndarray,
                x: np.ndarray,
                target: float,
                mu: float = 0.5,
                eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """Standard Normalised LMS update."""
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    next_weights = weights + mu * error * x / power
    return next_weights, error

def construct_graph(weights: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    graph: Dict[int, List[Tuple[int, float]]] = {}
    n = len(weights)
    for i in range(n):
        node = Node(i, weights[i])
        graph[node.id] = []
        for j in range(n):
            if i == j:
                continue
            similarity = 1.0 - abs(node.weight - weights[j]) / (1.0 + abs(node.weight - weights[j]))
            graph[node.id].append((j, similarity))
    return graph

def compute_phash(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    avg = float(np.mean(values))
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')

def similarity_matrix(features: Dict[int, np.ndarray]) -> np.ndarray:
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(features[ni])
        for j, nj in enumerate(nodes):
            if j < i:
                hj = compute_phash(features[nj])
                S[i, j] = 1.0 - hamming_distance(hi, hj) / 64.0
                S[j, i] = S[i, j]
            elif i == j:
                S[i, j] = 1.0
    return S

# ----------------------------------------------------------------------
# Parent B – Hyperdimensional primitives, Gini, Doomsday
# ----------------------------------------------------------------------
Vector = List[int]

def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
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
    return [int(round(sum(v[i] for v in vecs) / len(vecs))) for i in range(dim)]

def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient of a list/array of non‑negative numbers."""
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Gini coefficient is undefined for negative values")
    sorted_arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(sorted_arr)
    sum_y = cumulative[-1]
    if sum_y == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_y) / n
    return float(gini)

def weekday_symbol(date: dt.date) -> str:
    """Return a deterministic weekday name used as a symbol."""
    return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][date.weekday()]

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def nlms_update_with_hypercontext(weights: np.ndarray,
                                  x: np.ndarray,
                                  target: float,
                                  date: dt.date,
                                  W: np.ndarray,
                                  mu: float = 0.5,
                                  alpha: float = 0.1) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Perform an NLMS weight update, then use the Gini of the weight vector to
    scale a hyperdimensional outer‑product update of matrix W.
    The date determines a weekday‑symbol hypervector which is bound to a
    weight‑derived hypervector before the outer‑product.
    """
    # 1) NLMS step
    new_w, err = nlms_update(weights, x, target, mu)

    # 2) Gini coefficient of the new weight distribution
    gini = gini_coefficient(new_w)

    # 3) Hypervectors
    #   a) Weekday symbol hypervector
    wd_sym = weekday_symbol(date)
    hv_date = np.array(symbol_vector(wd_sym, dim=W.shape[0]), dtype=np.int8)

    #   b) Weight‑derived hypervector – seed from mean weight
    seed_val = int(np.mean(new_w) * 1e6) if np.isfinite(np.mean(new_w)) else 0
    hv_weight = np.array(random_vector(dim=W.shape[0], seed=seed_val), dtype=np.int8)

    #   c) Bind the two hypervectors
    hv_bound = bind(hv_date.tolist(), hv_weight.tolist())
    hv_bound_np = np.array(hv_bound, dtype=np.int8)

    # 4) Update the hyperdimensional matrix W
    #    Outer product of the bound vector, scaled by Gini and alpha
    delta_W = alpha * gini * np.outer(hv_bound_np.astype(np.float64),
                                     hv_bound_np.astype(np.float64))
    new_W = W + delta_W

    return new_w, err, new_W

def compute_hypergraph_similarity(weights: np.ndarray, dim: int = 1024) -> np.ndarray:
    """
    Build a graph based on NLMS weight similarities, assign each node a
    hypervector symbol, and return the pairwise similarity matrix derived
    from perceptual hashing of those hypervectors.
    """
    graph = construct_graph(weights)
    # Create a hypervector feature for each node
    features: Dict[int, np.ndarray] = {}
    for node_id in graph.keys():
        symbol = f"node_{node_id}"
        hv = np.array(symbol_vector(symbol, dim=dim), dtype=np.int8)
        features[node_id] = hv
    # Use the existing similarity_matrix (phash‑based)
    S = similarity_matrix(features)
    return S

def gini_weighted_matrix_update(W: np.ndarray,
                                weights: np.ndarray,
                                date: dt.date,
                                alpha: float = 0.05) -> np.ndarray:
    """
    Independent helper that updates matrix W using only the Gini coefficient
    and a date‑derived hypervector (no NLMS step).  Useful for scheduler‑style
    budgeting where the weight vector is treated as a resource profile.
    """
    gini = gini_coefficient(weights)

    # Date‑derived hypervector
    hv_date = np.array(symbol_vector(weekday_symbol(date), dim=W.shape[0]), dtype=np.int8)

    # Simple outer‑product update
    delta_W = alpha * gini * np.outer(hv_date.astype(np.float64),
                                     hv_date.astype(np.float64))
    return W + delta_W

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a small NLMS problem
    dim_input = 8
    rng = np.random.default_rng(42)
    w = rng.normal(size=dim_input).astype(np.float64)
    x = rng.normal(size=dim_input).astype(np.float64)
    target = 0.7

    # Hyperdimensional matrix (square, same dim as hypervectors)
    hv_dim = 512
    W = np.zeros((hv_dim, hv_dim), dtype=np.float64)

    today = dt.date.today()

    # Perform a hybrid update
    w, err, W = nlms_update_with_hypercontext(w, x, target, today, W,
                                              mu=0.3, alpha=0.2)
    print(f"NLMS error after update: {err:.4f}")
    print(f"Weight vector norm: {np.linalg.norm(w):.4f}")
    print(f"Hyperdimensional matrix Frobenius norm: {np.linalg.norm(W, 'fro'):.4f}")

    # Compute similarity matrix of hypergraph
    S = compute_hypergraph_similarity(w, dim=hv_dim)
    print(f"Similarity matrix shape: {S.shape}")
    print(f"Mean similarity (excluding diagonal): {np.mean(S[np.triu_indices_from(S, k=1)]):.4f}")

    # Apply a pure Gini‑weighted matrix update
    W = gini_weighted_matrix_update(W, w, today, alpha=0.1)
    print(f"Updated matrix Frobenius norm after Gini step: {np.linalg.norm(W, 'fro'):.4f}")