# DARWIN HAMMER — match 3165, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_distri_m1510_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1137_s0.py (gen5)
# born: 2026-05-29T23:48:09Z

"""Hybrid Algorithm: Stylometry‑Weighted Minimum‑Cost Tree + Tropical Leader Election

Parents:
- **Parent A** – Stylometry‑Weighted Ollivier‑Ricci Curvature → Epistemic Certainty → Tropical Leader Election
- **Parent B** – Hybrid Minimum‑Cost Tree with LSM‑Weighted Bayesian Update (geometric tree utilities)

Mathematical Bridge:
1. Stylometry feature vectors extracted from text (Parent A) are used to compute a similarity matrix S where
   S[i, j] = 𝑓_i·𝑓_j (dot‑product of LSM‑category frequency vectors).
2. The similarity matrix defines edge “distances” d_ij = 1 / (1 + S_ij).  These distances feed the
   Minimum‑Cost (minimum‑spanning) Tree algorithm from Parent B, yielding a weighted adjacency matrix **W** that
   respects both the geometric tree topology and the stylometric evidence.
3. The resulting weighted adjacency matrix **W** is then supplied to the tropical leader‑election routine
   (max‑plus algebra) from Parent A.  Broadcast strength b_i is iteratively updated by
   b_i ← max_j (W_ij + b_j).  Nodes whose final strength exceeds a confidence threshold become leaders.

The three core functions below demonstrate this fused pipeline:
- `stylometry_features` – extract LSM‑style counts.
- `minimum_cost_tree` – build a MST using stylometry‑derived distances.
- `tropical_leader_election` – run the max‑plus broadcast process on the tree’s weighted adjacency.

The `hybrid_leader_finder` function ties everything together.
"""

import math
import random
import sys
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers herself we our ours ourselves they them their theirs themselves".split()
    ),
    # Additional categories could be added here.
}

def stylometry_features(text: str) -> Dict[str, int]:
    """
    Count occurrences of each functional word category in *text*.
    Returns a mapping category → count.
    """
    features = Counter()
    tokens = text.lower().split()
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                features[cat] += 1
    # Ensure every known category appears (zero count if absent)
    for cat in FUNCTION_CATS:
        features.setdefault(cat, 0)
    return dict(features)


# ----------------------------------------------------------------------
# Parent B – geometric tree utilities (minimum‑cost tree)
# ----------------------------------------------------------------------
def _euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def minimum_cost_tree(
    feature_dicts: List[Dict[str, int]],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Build a Minimum‑Spanning Tree (MST) whose edge weights are derived from
    stylometry similarity.

    Steps
    -----
    1. Convert each feature dict to a dense vector `v_i`.
    2. Compute similarity matrix S where S_ij = v_i·v_j (dot product).
    3. Convert similarity to a distance metric d_ij = 1 / (1 + S_ij) – larger
       similarity → shorter distance.
    4. Run Prim's algorithm on the distance matrix to obtain the MST.
    5. Return the weighted adjacency matrix W (size N×N) where W_ij = S_ij
       if (i, j) belongs to the MST, otherwise 0, and also the list of
       tree edges.

    The adjacency matrix **W** will later serve as the tropical‑algebra weight
    matrix for leader election.
    """
    n = len(feature_dicts)
    if n == 0:
        raise ValueError("feature_dicts must contain at least one element")

    # 1. Dense vectors
    categories = sorted(FUNCTION_CATS.keys())
    vectors = np.zeros((n, len(categories)), dtype=float)
    for idx, fmap in enumerate(feature_dicts):
        vectors[idx] = [float(fmap.get(cat, 0)) for cat in categories]

    # 2. Similarity matrix (dot product)
    similarity = vectors @ vectors.T  # shape (n, n)

    # 3. Distance matrix for MST
    distance = 1.0 / (1.0 + similarity)  # avoid division by zero

    # 4. Prim's algorithm (O(n^2) – sufficient for modest n)
    in_mst = [False] * n
    edge_to = [-1] * n
    min_dist = [math.inf] * n
    min_dist[0] = 0.0
    for _ in range(n):
        # select the vertex with minimal distance to the current MST
        u = min((idx for idx in range(n) if not in_mst[idx]), key=lambda i: min_dist[i])
        in_mst[u] = True
        # update neighbours
        for v in range(n):
            if not in_mst[v] and distance[u, v] < min_dist[v]:
                min_dist[v] = distance[u, v]
                edge_to[v] = u

    # 5. Build weighted adjacency matrix using original similarity as weight
    W = np.zeros((n, n), dtype=float)
    edges = []
    for v in range(1, n):
        u = edge_to[v]
        if u == -1:
            continue  # isolated node (should not happen in a connected graph)
        weight = similarity[u, v]
        W[u, v] = weight
        W[v, u] = weight
        edges.append((u, v))

    return W, edges


# ----------------------------------------------------------------------
# Tropical Leader Election (max‑plus algebra)
# ----------------------------------------------------------------------
def tropical_leader_election(
    weight_matrix: np.ndarray,
    max_iters: int = 100,
    epsilon: float = 1e-6,
    strength_threshold: float = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Perform tropical (max‑plus) broadcast on *weight_matrix*.

    The broadcast strength vector `b` satisfies the fixed‑point equation
        b_i = max_j (W_ij + b_j)
    Starting from the zero vector, we iterate until convergence.

    Parameters
    ----------
    weight_matrix : np.ndarray
        Symmetric non‑negative matrix (size N×N).  Zero entries denote no
        direct edge.
    max_iters : int
        Upper bound on iteration count.
    epsilon : float
        Convergence tolerance (max absolute change).
    strength_threshold : float or None
        If provided, nodes with final strength ≥ threshold are returned as
        leaders.  If None, the top‑10 % strongest nodes become leaders.

    Returns
    -------
    b : np.ndarray
        Final broadcast strength for each node.
    leaders : List[int]
        Indices of nodes selected as leaders.
    """
    n = weight_matrix.shape[0]
    b = np.zeros(n, dtype=float)

    for _ in range(max_iters):
        b_new = np.max(weight_matrix + b, axis=1)
        if np.max(np.abs(b_new - b)) < epsilon:
            b = b_new
            break
        b = b_new

    # Determine leaders
    if strength_threshold is None:
        # Choose top 10 % (at least one)
        k = max(1, n // 10)
        threshold = np.partition(b, -k)[-k]
    else:
        threshold = strength_threshold

    leaders = [i for i, val in enumerate(b) if val >= threshold]
    return b, leaders


# ----------------------------------------------------------------------
# Hybrid pipeline tying both parents together
# ----------------------------------------------------------------------
def hybrid_leader_finder(texts: List[str]) -> Tuple[np.ndarray, List[int]]:
    """
    End‑to‑end hybrid algorithm:

    1. Extract stylometry feature dicts from each input *text*.
    2. Build a Minimum‑Cost Tree whose edge weights are the stylometry dot‑products.
    3. Run tropical leader election on the resulting weighted adjacency matrix.
    4. Return the broadcast strengths and the indices of elected leaders.

    The returned leader indices correspond to the positions of *texts* in the
    original list.
    """
    # Step 1 – stylometry extraction
    feature_dicts = [stylometry_features(t) for t in texts]

    # Step 2 – MST construction (provides weighted adjacency)
    W, _ = minimum_cost_tree(feature_dicts)

    # Step 3 – tropical election
    strengths, leaders = tropical_leader_election(W)

    return strengths, leaders


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I think therefore I am. You are the one who knows the truth.",
        "She loves her cat, but he hates his dog.",
        "We will meet at the park tomorrow. They are bringing snacks.",
        "You should consider the evidence carefully before deciding.",
        "He said that he would arrive early, but he was late."
    ]

    strengths, leaders = hybrid_leader_finder(sample_texts)

    print("Broadcast strengths per document:")
    for idx, val in enumerate(strengths):
        print(f"  Doc {idx}: {val:.4f}")

    print("\nElected leader document indices:", leaders)
    print("\nLeader excerpts:")
    for idx in leaders:
        print(f"--- Doc {idx} ---")
        print(sample_texts[idx])
        print()