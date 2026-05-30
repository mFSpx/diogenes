# DARWIN HAMMER — match 4975, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m504_s0.py (gen4)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s4.py (gen3)
# born: 2026-05-29T23:59:04Z

"""Hybrid Sheaf‑Stylometry‑Geometric Algorithm
==========================================

Parents
-------
* **Algorithm A** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m504_s0.py``  
  Provides a cellular *sheaf* over a graph and the associated Laplacian
  ``L = δᵀδ``.  The Frobenius norm ``‖L‖_F`` (the *sheaf energy*) is a
  scalar that quantifies the connectivity of the underlying graph.

* **Algorithm B** – ``hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s4.py``  
  Extracts a stylometric fingerprint from raw text as a normalized frequency
  vector over *functional‑word categories* and interprets each fingerprint as
  a point in Euclidean space.  Points are clustered by a Voronoi partition and
  each cluster is equipped with a Clifford‑algebra‑style *geometric product*
  on *blades* (sets of category indices).

Mathematical Bridge
-------------------
The bridge is the observation that the sheaf Laplacian energy is a single
scalar that can be used to *modulate* the stylometric vectors before they are
fed to the geometric‑product layer.  Concretely, for a text ``t`` we compute


v(t)   – raw stylometric vector  ∈ ℝ^k
E      – sheaf energy (‖L‖_F)
v̂(t)  = E · v(t)                – energy‑weighted fingerprint


The weighted fingerprint ``v̂(t)`` is then used for Voronoi assignment and
blade construction.  This coupling yields a unified system in which the
topology of the sheaf influences the algebraic encoding of textual data.

The module below implements the fused pipeline and provides three public
functions that demonstrate the hybrid operation.
"""

import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, FrozenSet, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Sheaf utilities (from Algorithm A)
# ---------------------------------------------------------------------------

class Sheaf:
    """Cellular sheaf over a simple undirected graph.

    * ``node_dims`` – mapping ``node_id → dimension`` (unused in this
      simplified version but kept for compatibility).
    * ``edges`` – list of ``(u, v)`` tuples; orientation is ignored.
    """

    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node_id → int
        self.edges = list(edge_list)              # list of (u, v)

    def compute_laplacian(self) -> np.ndarray:
        """Return the (combinatorial) sheaf Laplacian L = δᵀδ.

        For the toy implementation we treat δ as the signed incidence matrix
        of the underlying graph, i.e. L[u, v] = -1 if (u, v) is an edge,
        L[v, u] = 1, and diagonal entries are zero.
        """
        n = len(self.node_dims)
        L = np.zeros((n, n), dtype=float)
        for u, v in self.edges:
            L[u, v] = -1.0
            L[v, u] = 1.0
        return L

def sheaf_energy(sheaf: Sheaf) -> float:
    """Compute a scalar “energy” from the sheaf Laplacian.

    Energy is defined as the squared Frobenius norm:
        E = ‖L‖_F² = trace(Lᵀ L)
    """
    L = sheaf.compute_laplacian()
    return float(np.trace(L.T @ L))


# ---------------------------------------------------------------------------
# Stylometry utilities (from Algorithm B)
# ---------------------------------------------------------------------------

# Minimal functional‑word categories for demonstration.
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": {
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "they", "them", "their", "theirs", "themselves",
        "we", "us", "our", "ours", "ourselves"
    },
    "article": {"a", "an", "the"},
    "preposition": {
        "in", "on", "at", "by", "for", "with", "about",
        "against", "among", "between", "into", "through",
        "during", "before", "after", "above", "below"
    },
    "conjunction": {"and", "but", "or", "nor", "for", "yet", "so"}
}
CATEGORY_ORDER = list(FUNCTION_CATS.keys())  # fixed ordering for vectors


def stylometry_features(text: str) -> np.ndarray:
    """Return a normalized frequency vector over ``CATEGORY_ORDER``.

    The vector has length ``k = len(CATEGORY_ORDER)`` and sums to 1
    (or is all zeros for empty input).
    """
    tokens = [tok.strip(".,!?;:()[]\"'").lower() for tok in text.split()]
    total = len(tokens)
    if total == 0:
        return np.zeros(len(CATEGORY_ORDER), dtype=float)

    counts = np.zeros(len(CATEGORY_ORDER), dtype=float)
    for idx, cat in enumerate(CATEGORY_ORDER):
        cat_words = FUNCTION_CATS[cat]
        counts[idx] = sum(1 for w in tokens if w in cat_words)

    return counts / total


# ---------------------------------------------------------------------------
# Geometric‑product (Clifford‑algebra style) utilities
# ---------------------------------------------------------------------------

def blade_from_vector(vec: np.ndarray, top_k: int = 2) -> FrozenSet[int]:
    """Map a weighted stylometric vector to a blade.

    The ``top_k`` categories with largest magnitude are selected; their
    indices (according to ``CATEGORY_ORDER``) form a frozenset representing
    the blade.
    """
    if top_k <= 0:
        return frozenset()
    # argsort descending
    idxs = np.argsort(-np.abs(vec))[:top_k]
    return frozenset(int(i) for i in idxs)


def geometric_product(b1: FrozenSet[int], b2: FrozenSet[int]) -> Tuple[int, FrozenSet[int]]:
    """Return the geometric product of two blades.

    The product is defined as:
        sign = (-1)^{|b1 ∩ b2|}
        result = b1 Δ b2   (symmetric difference)

    The sign is +1 if the intersection size is even, -1 otherwise.
    """
    inter = b1.intersection(b2)
    sign = -1 if (len(inter) % 2) else 1
    result = b1.symmetric_difference(b2)
    return sign, frozenset(result)


def product_of_blades(blades: List[FrozenSet[int]]) -> Tuple[int, FrozenSet[int]]:
    """Iteratively multiply a list of blades, returning the overall sign
    and resulting blade.
    """
    sign = 1
    acc = frozenset()
    for b in blades:
        s, acc = geometric_product(acc, b)
        sign *= s
    return sign, acc


# ---------------------------------------------------------------------------
# Hybrid pipeline
# ---------------------------------------------------------------------------

def weighted_stylometry(texts: List[str], sheaf: Sheaf) -> List[np.ndarray]:
    """Compute energy‑weighted stylometric vectors for a list of texts.

    Each raw vector ``v`` is scaled by the sheaf energy ``E``:
        v̂ = E · v
    """
    E = sheaf_energy(sheaf)
    weighted = [E * stylometry_features(t) for t in texts]
    return weighted


def voronoi_partition(
    points: List[np.ndarray],
    seeds: List[np.ndarray]
) -> Dict[int, List[int]]:
    """Assign each point to the index of the nearest seed (Euclidean distance).

    Returns a dictionary ``seed_index → list of point indices``.
    """
    if not seeds:
        raise ValueError("At least one seed is required for Voronoi partition.")
    assignment: Dict[int, List[int]] = defaultdict(list)
    for i, p in enumerate(points):
        dists = [np.linalg.norm(p - s) for s in seeds]
        nearest = int(np.argmin(dists))
        assignment[nearest].append(i)
    return dict(assignment)


def region_blade_product(
    texts: List[str],
    sheaf: Sheaf,
    seed_texts: List[str],
    top_k: int = 2
) -> Dict[int, Tuple[int, FrozenSet[int]]]:
    """Full hybrid operation.

    1. Compute energy‑weighted stylometric vectors for ``texts``.
    2. Compute stylometric vectors for ``seed_texts`` (used as Voronoi seeds).
    3. Partition the weighted points into Voronoi cells.
    4. For each cell, map every point to a blade (top‑k categories) and
       multiply all blades to obtain a region‑level blade.

    Returns a mapping ``seed_index → (sign, blade)``.
    """
    # Step 1 – weighted vectors for the corpus
    weighted_vectors = weighted_stylometry(texts, sheaf)

    # Step 2 – seed vectors (no weighting needed for seeds)
    seed_vectors = [stylometry_features(t) for t in seed_texts]

    # Step 3 – Voronoi assignment
    partitions = voronoi_partition(weighted_vectors, seed_vectors)

    # Step 4 – blade multiplication per region
    region_products: Dict[int, Tuple[int, FrozenSet[int]]] = {}
    for seed_idx, point_idxs in partitions.items():
        blades = [
            blade_from_vector(weighted_vectors[i], top_k=top_k)
            for i in point_idxs
        ]
        sign, blade = product_of_blades(blades)
        region_products[seed_idx] = (sign, blade)

    return region_products


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Construct a tiny graph (3 nodes, 2 edges) for the sheaf.
    node_dimensions = {0: 2, 1: 2, 2: 2}
    edge_list = [(0, 1), (1, 2)]
    sheaf = Sheaf(node_dimensions, edge_list)

    # Example corpus.
    corpus = [
        "I think that you are right and I agree.",
        "She went to the market and bought an apple.",
        "They will meet us at the park after the game.",
        "The quick brown fox jumps over the lazy dog."
    ]

    # Choose two seed texts (representative stylometric fingerprints).
    seeds = [
        "I you he she we they.",
        "The a an."
    ]

    # Run the hybrid pipeline.
    region_results = region_blade_product(corpus, sheaf, seeds, top_k=2)

    # Display results.
    for seed_idx, (sgn, blade) in region_results.items():
        print(f"Region {seed_idx}: sign={sgn}, blade={sorted(blade)}")

    # Verify that the sheaf energy is a positive scalar.
    print(f"Sheaf energy: {sheaf_energy(sheaf):.4f}")

    # Demonstrate individual weighted vectors.
    weighted = weighted_stylometry(corpus, sheaf)
    for i, vec in enumerate(weighted):
        print(f"Weighted vector {i}: {vec}")