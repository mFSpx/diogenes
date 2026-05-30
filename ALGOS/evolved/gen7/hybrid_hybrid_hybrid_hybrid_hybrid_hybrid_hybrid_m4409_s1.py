# DARWIN HAMMER — match 4409, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_vorono_m2690_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_decrea_hybrid_hybrid_infota_m1342_s0.py (gen5)
# born: 2026-05-29T23:55:25Z

"""Hybrid Voronoi‑RBF‑MinHash Graph Pruner

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_nlms_hybrid_h_m978_s1.py (Similarity Matrix RBF Kernel)
- hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s3.py (Voronoi‑RBF‑Associative Memory)
- hybrid_hybrid_decrea_hybrid_hybrid_infota_m1342_s0.py (Graph pruning with MinHash & Bayesian update)

Mathematical bridge:
Both parent A and parent B rely on Euclidean distances and Gaussian radial‑basis
functions (RBF).  Parent A uses those distances to build a similarity matrix and to
assign points to Voronoi cells, while parent B uses a MinHash‑derived similarity
as a probability‑like likelihood inside a Bayesian edge‑scoring scheme.
The fusion therefore proceeds as follows:

1. For every node we have a spatial coordinate *x* and a discrete feature set
   *F*.  
2. A Gaussian RBF `g(r)=exp(-(ε r)^2)` is evaluated on the Euclidean distance
   between node coordinates and a set of seed centroids, yielding an
   “spatial affinity” vector *w*.
3. A MinHash signature of *F* provides a Jaccard‑style similarity `s_mh`
   between two nodes.
4. The spatial affinity and the MinHash similarity are multiplied to obtain a
   joint likelihood `ℓ = w_i w_j · s_mh`.  
5. A Bayesian update with a prior survival probability `π` and a false‑positive
   rate `α` produces a posterior edge score `p = ℓ·π / (ℓ·π + α·(1‑π))`.
6. Finally a time‑dependent decreasing pruning probability discards edges whose
   posterior score falls below a decaying threshold.

The code below implements this unified pipeline with three core functions:
`hybrid_edge_score`, `prune_edges`, and `combined_similarity_matrix`.  A tiny
smoke test demonstrates end‑to‑end execution."""


import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict, Any, Set
import numpy as np

# ----------------------------------------------------------------------
# Basic geometric primitives (shared by both parents)
# ----------------------------------------------------------------------
def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial‑basis function."""
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Voronoi / RBF utilities (from Parent A)
# ----------------------------------------------------------------------
def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    """Index of the closest seed to *point*."""
    if not seeds.size:
        raise ValueError('seeds required')
    dists = np.linalg.norm(seeds - point, axis=1)
    return int(np.argmin(dists))


def assign_regions(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Binary region matrix R of shape (n_seeds, n_points) where
    R[i, j] == 1 iff point j belongs to the Voronoi cell of seed i.
    """
    n_seeds = seeds.shape[0]
    n_pts = points.shape[0]
    regions = np.zeros((n_seeds, n_pts), dtype=int)
    for j, p in enumerate(points):
        i = nearest(p, seeds)
        regions[i, j] = 1
    return regions


def rbf_affinity(point: np.ndarray, seeds: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Gaussian RBF weights of *point* to every seed."""
    dists = np.linalg.norm(seeds - point, axis=1)
    return np.vectorize(lambda r: gaussian(r, epsilon))(dists)


# ----------------------------------------------------------------------
# MinHash utilities (from Parent B)
# ----------------------------------------------------------------------
def _perm_hash(value: Any, seed: int) -> int:
    """Hash a value with a simple permutation seed."""
    return hash((value, seed))


def minhash_signature(elements: Set[Any], num_perm: int = 128) -> List[int]:
    """
    Compute a MinHash signature for a set of hashable elements.
    Each permutation is simulated by XOR‑ing the element hash with a random
    integer seed.
    """
    if not elements:
        return [0] * num_perm
    # deterministic pseudo‑random seeds for reproducibility
    random.seed(0)
    seeds = [random.randrange(1 << 30) for _ in range(num_perm)]
    signature = []
    for s in seeds:
        mins = min(_perm_hash(el, s) for el in elements)
        signature.append(mins)
    return signature


def minhash_jaccard(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("signatures must have equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


# ----------------------------------------------------------------------
# Bayesian edge scoring (from Parent B)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)."""
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_posterior(prior: float, likelihood: float, false_positive: float) -> float:
    """Posterior P(H|E) using Bayes rule."""
    marginal = bayes_marginal(prior, likelihood, false_positive)
    if marginal == 0.0:
        return 0.0
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_edge_score(
    node_a: str,
    node_b: str,
    positions: Dict[str, np.ndarray],
    features: Dict[str, Set[Any]],
    seeds: np.ndarray,
    prior: float = 0.5,
    false_positive: float = 0.01,
    epsilon: float = 1.0,
) -> float:
    """
    Compute a hybrid posterior score for the edge (node_a, node_b).

    Steps:
    1. Spatial affinity via Gaussian RBF to all seeds.
    2. MinHash Jaccard similarity of feature sets.
    3. Joint likelihood = (w_a · w_b) * s_mh.
    4. Bayesian posterior with given prior and false‑positive rate.
    """
    # 1. Spatial RBF affinities
    w_a = rbf_affinity(positions[node_a], seeds, epsilon)  # shape (n_seeds,)
    w_b = rbf_affinity(positions[node_b], seeds, epsilon)

    spatial_likelihood = float(np.dot(w_a, w_b) / (np.linalg.norm(w_a) * np.linalg.norm(w_b) + 1e-12))

    # 2. MinHash similarity
    sig_a = minhash_signature(features[node_a])
    sig_b = minhash_signature(features[node_b])
    s_mh = minhash_jaccard(sig_a, sig_b)

    # 3. Joint likelihood (bounded in [0,1])
    joint_likelihood = min(max(spatial_likelihood * s_mh, 0.0), 1.0)

    # 4. Bayesian posterior
    posterior = bayes_posterior(prior, joint_likelihood, false_positive)
    return posterior


def combined_similarity_matrix(
    node_ids: List[str],
    positions: Dict[str, np.ndarray],
    features: Dict[str, Set[Any]],
    seeds: np.ndarray,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Produce a symmetric similarity matrix S where
    S[i, j] = hybrid_edge_score(node_i, node_j, ...) for all i≠j
    and S[i, i] = 1.
    """
    n = len(node_ids)
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            score = hybrid_edge_score(
                node_ids[i],
                node_ids[j],
                positions,
                features,
                seeds,
                epsilon=epsilon,
            )
            S[i, j] = S[j, i] = score
    return S


def prune_edges(
    edges: List[Tuple[str, str]],
    scores: List[float],
    initial_threshold: float = 0.4,
    decay: float = 0.95,
    min_threshold: float = 0.05,
) -> List[Tuple[str, str]]:
    """
    Time‑dependent decreasing pruning.
    Edges are examined in order; after each inspection the threshold is
    multiplied by *decay*.  An edge survives if its score ≥ current threshold
    and the threshold stays above *min_threshold*.
    """
    if len(edges) != len(scores):
        raise ValueError("edges and scores must have the same length")
    kept = []
    thresh = initial_threshold
    for (e, sc) in zip(edges, scores):
        if thresh < min_threshold:
            break
        if sc >= thresh:
            kept.append(e)
        thresh *= decay
    return kept


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create synthetic graph
    N_NODES = 8
    node_ids = [f"N{i}" for i in range(N_NODES)]

    # Random 2‑D positions
    positions = {nid: np.random.rand(2) for nid in node_ids}

    # Random feature sets (integers 0‑99)
    features = {
        nid: set(random.sample(range(100), k=random.randint(5, 15))) for nid in node_ids
    }

    # Random seed centroids for Voronoi / RBF (choose 3 seeds)
    seeds = np.random.rand(3, 2)

    # Build all possible undirected edges
    all_edges = [(node_ids[i], node_ids[j]) for i in range(N_NODES) for j in range(i + 1, N_NODES)]

    # Compute hybrid scores
    scores = [
        hybrid_edge_score(a, b, positions, features, seeds, epsilon=1.2)
        for a, b in all_edges
    ]

    # Prune edges
    kept_edges = prune_edges(all_edges, scores, initial_threshold=0.3, decay=0.9)

    # Produce combined similarity matrix for inspection
    S = combined_similarity_matrix(node_ids, positions, features, seeds, epsilon=1.2)

    # Simple sanity prints (no external libraries)
    print("Number of original edges:", len(all_edges))
    print("Number of kept edges after pruning:", len(kept_edges))
    print("Sample of kept edges:", kept_edges[:5])
    print("Similarity matrix shape:", S.shape)
    print("Matrix diagonal (should be 1):", np.diag(S)[:5])
    sys.exit(0)