# DARWIN HAMMER — match 4581, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_ssim_hybrid_h_hybrid_sketches_rlct_m1064_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s3.py (gen5)
# born: 2026-05-29T23:56:41Z

"""Hybrid Multivector‑NLMS‑Sketch Module
Parents:
- hybrid_hybrid_ssim_hybrid_h_hybrid_sketches_rlct_m1064_s1.py (Geometric algebra + sketch/RLCT)
- hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s3.py (NLMS adaptive filter + perceptual hash graph)

Mathematical bridge
-------------------
The *Multivector* class provides a geometric product whose scalar part is exactly the
dot‑product of two vectors (grade‑1 blades).  The NLMS update rule

w ← w + μ·e·x / (‖x‖²+ε)

modifies a weight vector by a scalar multiple of the input vector.  By interpreting the
weight vector and the input as multivectors, the NLMS step can be written as an
update of the **vector part** of a multivector while the prediction uses the **scalar
part of the geometric product**.

The sketch side reduces high‑dimensional feature vectors to a low‑dimensional
count‑min sketch.  Each sketch bucket is mapped to a basis blade, turning the sketch
into another multivector.  The scalar part of the geometric product between two
sketch‑derived multivectors yields an inexpensive similarity estimate that can be
combined with the perceptual‑hash similarity used in the original graph construction.

The resulting hybrid system therefore:
1. Represents both adaptive weights and compressed features as multivectors.
2. Predicts by the scalar part of a geometric product.
3. Updates weights with an NLMS‑style rule applied to the vector part.
4. Builds a similarity matrix that fuses geometric‑product similarity with
   perceptual‑hash similarity, and extracts a minimum‑cost spanning tree.

The code below implements this fusion in a self‑contained, executable form.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Dict, FrozenSet, Tuple, List

# ----------------------------------------------------------------------
# Multivector (grade 0 and 1, simple geometric product)
# ----------------------------------------------------------------------
class Multivector:
    """Euclidean Clifford algebra limited to scalar (grade‑0) and vector (grade‑1)."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # keep only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------
    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def vector_part(self) -> np.ndarray:
        vec = np.zeros(self.n)
        for blade, value in self.components.items():
            if len(blade) == 1:
                idx = next(iter(blade))
                vec[idx] = value
        return vec

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, val in other.components.items():
            result[blade] = result.get(blade, 0.0) + val
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (only scalar part is exact; higher‑grade parts are kept naïvely)."""
        result: Dict[FrozenSet[int], float] = {}
        # scalar‑scalar contribution
        scalar = self.scalar_part() * other.scalar_part()
        # vector‑vector contribution to scalar (dot product)
        a_vec = self.vector_part()
        b_vec = other.vector_part()
        scalar += np.dot(a_vec, b_vec)

        # store scalar
        if abs(scalar) > 1e-15:
            result[frozenset()] = scalar

        # store vector part (simply concatenate vectors)
        for i, coeff in enumerate(a_vec):
            if abs(coeff) > 1e-15:
                result[frozenset({i})] = result.get(frozenset({i}), 0.0) + coeff
        for i, coeff in enumerate(b_vec):
            if abs(coeff) > 1e-15:
                result[frozenset({i})] = result.get(frozenset({i}), 0.0) + coeff

        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Multivector(n={self.n}, components={self.components})"


# ----------------------------------------------------------------------
# Helper: convert a NumPy vector to a Multivector (grade‑1 only)
# ----------------------------------------------------------------------
def vector_to_multivector(v: np.ndarray) -> Multivector:
    comps = {frozenset({i}): float(val) for i, val in enumerate(v)}
    return Multivector(comps, n=v.shape[0])


# ----------------------------------------------------------------------
# NLMS update expressed on Multivectors
# ----------------------------------------------------------------------
def nlms_update_mv(
    weights_mv: Multivector,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[Multivector, float]:
    """
    Predict with the scalar part of the geometric product,
    then perform an NLMS‑style update on the vector part of `weights_mv`.
    Returns the updated Multivector and the prediction error.
    """
    # prediction = <weights, x> = scalar part of geometric product
    x_mv = vector_to_multivector(x)
    pred = (weights_mv * x_mv).scalar_part()
    error = target - pred

    power = float(np.dot(x, x) + eps)
    delta = mu * error / power

    # update vector part
    w_vec = weights_mv.vector_part()
    w_vec += delta * x
    new_weights = vector_to_multivector(w_vec)
    # preserve any existing scalar component (often zero)
    if weights_mv.scalar_part() != 0.0:
        new_weights = Multivector(
            {frozenset(): weights_mv.scalar_part(), **new_weights.components}, new_weights.n
        )
    return new_weights, error


# ----------------------------------------------------------------------
# Simple Count‑Min Sketch (deterministic hash seeds)
# ----------------------------------------------------------------------
def count_min_sketch(
    vec: np.ndarray, depth: int = 4, width: int = 256, seed: int = 0
) -> np.ndarray:
    """
    Returns a count‑min sketch of `vec`.  The sketch is a 1‑D array of length `depth*width`.
    """
    rng = np.random.default_rng(seed)
    # generate `depth` independent hash salts
    salts = rng.integers(0, 2**31 - 1, size=depth, dtype=np.int64)

    sketch = np.zeros(depth * width, dtype=np.float64)
    for idx, val in enumerate(vec):
        for d, salt in enumerate(salts):
            h = (hash((idx, salt)) % width) + d * width
            sketch[h] += float(val)
    return sketch


def sketch_to_multivector(sketch: np.ndarray, n: int) -> Multivector:
    """
    Interprets each bucket of the sketch as a basis blade.
    The first `n` buckets become vector components; the remaining part is stored as higher‑grade
    (ignored for scalar predictions).
    """
    comps: Dict[FrozenSet[int], float] = {}
    limit = min(n, sketch.shape[0])
    for i in range(limit):
        comps[frozenset({i})] = float(sketch[i])
    # optional scalar part as total count
    total = float(sketch.sum())
    if abs(total) > 1e-15:
        comps[frozenset()] = total
    return Multivector(comps, n=n)


# ----------------------------------------------------------------------
# Perceptual hash utilities (copied from parent B)
# ----------------------------------------------------------------------
def compute_phash(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    avg = float(np.mean(values))
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------------
# Hybrid similarity matrix
# ----------------------------------------------------------------------
def hybrid_similarity_matrix(
    features: Dict[int, np.ndarray],
    sketch_depth: int = 4,
    sketch_width: int = 256,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    For each feature vector:
      * Build a count‑min sketch → multivector.
      * Compute scalar similarity via geometric product.
      * Compute perceptual‑hash similarity.
    The final similarity is a convex combination:
        S = α·S_geom + (1‑α)·S_phash
    """
    ids = list(features.keys())
    n = len(ids)
    # Pre‑compute multivectors and phashes
    mv_list: List[Multivector] = []
    phash_list: List[int] = []
    max_dim = max(v.shape[0] for v in features.values())
    for i in ids:
        vec = features[i]
        sketch = count_min_sketch(vec, depth=sketch_depth, width=sketch_width, seed=42)
        mv = sketch_to_multivector(sketch, n=max_dim)
        mv_list.append(mv)
        phash_list.append(compute_phash(vec))

    S = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        S[i, i] = 1.0
        for j in range(i + 1, n):
            # geometric similarity = scalar part of product, normalized
            prod = mv_list[i] * mv_list[j]
            geom = prod.scalar_part()
            # simple normalization to [0,1]
            norm = (geom - 0.0) / (abs(geom) + 1e-12)  # keep sign
            geom_sim = 0.5 * (norm + 1.0)  # map to [0,1]

            # phash similarity
            ham = hamming_distance(phash_list[i], phash_list[j])
            phash_sim = 1.0 - ham / 64.0

            # convex blend
            sim = alpha * geom_sim + (1.0 - alpha) * phash_sim
            S[i, j] = S[j, i] = sim
    return S


# ----------------------------------------------------------------------
# Graph construction using hybrid similarity
# ----------------------------------------------------------------------
def construct_hybrid_graph(
    weights: np.ndarray,
    features: Dict[int, np.ndarray],
    alpha: float = 0.6,
) -> dict:
    """
    Builds an undirected weighted graph where edge weight is the hybrid similarity.
    Node ids correspond to indices of `weights` (assumed same length as `features` keys).
    """
    sim = hybrid_similarity_matrix(features, alpha=alpha)
    graph: dict = {}
    n = sim.shape[0]
    for i in range(n):
        graph[i] = []
        for j in range(n):
            if i != j:
                graph[i].append((j, sim[i, j]))
    return graph


# ----------------------------------------------------------------------
# Minimum‑cost spanning tree (depth‑first traversal)
# ----------------------------------------------------------------------
def minimum_cost_tree(graph: dict) -> List[int]:
    """Return a DFS order that visits every node once (a cheap spanning‑tree traversal)."""
    visited = set()
    order: List[int] = []
    stack = [0]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            # push neighbors sorted by descending weight (higher similarity first)
            neighbors = sorted(graph[node], key=lambda x: -x[1])
            for nbr, _ in neighbors:
                if nbr not in visited:
                    stack.append(nbr)
    return order


# ----------------------------------------------------------------------
# Example hybrid pipeline (three public functions)
# ----------------------------------------------------------------------
def hybrid_predict(weights_mv: Multivector, x: np.ndarray) -> float:
    """Prediction using scalar part of geometric product."""
    return (weights_mv * vector_to_multivector(x)).scalar_part()


def hybrid_train_step(
    weights_mv: Multivector,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[Multivector, float]:
    """One NLMS‑style training step on multivector weights."""
    return nlms_update_mv(weights_mv, x, target, mu, eps)


def hybrid_build_and_traverse(
    weights: np.ndarray,
    features: Dict[int, np.ndarray],
    alpha: float = 0.6,
) -> List[int]:
    """Construct the hybrid graph and return a minimum‑cost traversal order."""
    graph = construct_hybrid_graph(weights, features, alpha=alpha)
    return minimum_cost_tree(graph)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic data
    dim = 8
    np.random.seed(0)
    weights_vec = np.random.randn(dim)
    weights_mv = vector_to_multivector(weights_vec)

    # generate 5 random feature vectors
    features = {i: np.random.randn(dim) for i in range(5)}

    # single training step
    x = features[0]
    target = 1.23
    new_weights_mv, err = hybrid_train_step(weights_mv, x, target)
    print(f"Training error: {err:.4f}")

    # prediction after update
    pred = hybrid_predict(new_weights_mv, x)
    print(f"Prediction after update: {pred:.4f}")

    # build graph and traverse
    order = hybrid_build_and_traverse(weights_vec, features, alpha=0.7)
    print(f"Traversal order: {order}")