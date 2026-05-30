# DARWIN HAMMER — match 2690, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_nlms_hybrid_h_m978_s1.py (gen5)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s3.py (gen5)
# born: 2026-05-29T23:43:33Z

"""Hybrid Voronoi‑RBF‑Feature Associative Memory

Parents:
- **Parent A** (`hybrid_hybrid_hybrid_hybrid_hybrid_nlms_hybrid_h_m978_s1.py`):
  Provides Euclidean distance, Gaussian RBF, feature‑based similarity matrix,
  perceptual hashing and a Hoeffding‑bound based split decision.
- **Parent B** (`hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s3.py`):
  Provides Voronoi partitioning (nearest‑seed assignment) and geometry utilities.

Mathematical Bridge:
Both parents rely on the Euclidean distance `‖x‑y‖`.  In Parent B the distance
determines a Voronoi cell; in Parent A the same distance is fed to a Gaussian
RBF to obtain a similarity `exp(-ε²‖x‑y‖²)`.  The fusion therefore builds a
**joint similarity** that is the element‑wise product of


S_feature[i,j] = exp(-ε_f² ‖f_i‑f_j‖²)          # feature‑based RBF (Parent A)
S_geo[i,j]    = exp(-ε_g² ‖p_i‑c_j‖²)          # geometric RBF to seed centroids (Parent B)


The Voronoi assignment matrix `R` (binary, shape `n_seeds × n_points`) masks the
joint similarity so that each query point only draws memory from its own cell.
The final retrieval is a weighted sum of per‑seed associative memory vectors.

The module implements three core hybrid functions:
1. `hybrid_similarity_matrix` – builds the joint similarity.
2. `hybrid_voronoi_rbf_retrieval` – performs Voronoi‑aware, RBF‑weighted memory
   readout.
3. `hybrid_should_split` – decides whether to split a node using the Hoeffding
   bound (directly re‑using Parent A’s logic).

All code is pure Python 3 with only the allowed standard‑library imports.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Hashable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic geometry & similarity utilities (shared by both parents)
# ----------------------------------------------------------------------
Node = Hashable
FeatureVec = Sequence[float]
Graph = Dict[Node, set]

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial‑basis function."""
    return math.exp(-((epsilon * r) ** 2))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance for NumPy vectors (Parent B)."""
    return np.linalg.norm(a - b)


# ----------------------------------------------------------------------
# Feature‑based similarity (Parent A)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def similarity_matrix(features: Dict[Node, FeatureVec],
                      vram_budget_mb: int) -> Tuple[np.ndarray, List[Node]]:
    """Feature‑based Gaussian similarity (Parent A)."""
    nodes = list(features.keys())
    n = len(nodes)
    epsilon = 1.0 / (vram_budget_mb / 1024.0)  # heuristic scaling
    S = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            sim = gaussian(dist, epsilon)
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes


def rbf_kernel_matrix(features: Dict[Node, FeatureVec],
                      epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    """Pure geometric RBF kernel (Parent A)."""
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes


# ----------------------------------------------------------------------
# Voronoi utilities (Parent B)
# ----------------------------------------------------------------------
def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    """Index of the closest seed to *point*."""
    if not seeds.size:
        raise ValueError('seeds required')
    return int(np.argmin(np.linalg.norm(seeds - point, axis=1)))


def assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Binary region matrix R (n_seeds × n_points).  R[i, j] == 1 iff point j belongs
    to the Voronoi cell of seed i.
    """
    n_seeds = seeds.shape[0]
    n_pts = points.shape[0]
    regions = np.zeros((n_seeds, n_pts), dtype=int)
    for j, p in enumerate(points):
        i = nearest(p, seeds)
        regions[i, j] = 1
    return regions


# ----------------------------------------------------------------------
# Hoeffding‑bound split decision (Parent A)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def hybrid_should_split(best_gain: float,
                        second_best_gain: float,
                        r: float,
                        delta: float,
                        n: int,
                        tie_threshold: float = 0.05) -> bool:
    """
    Decide whether to split a node.  The logic mirrors Parent A's `should_split`
    but is exposed as a reusable helper.
    """
    eps = hoeffding_bound(r, delta, n)
    gain_gap = best_gain - second_best_gain
    # If the gap exceeds the bound or is within a small tie threshold we split.
    return gain_gap > eps or gain_gap < tie_threshold


# ----------------------------------------------------------------------
# Hybrid core operations
# ----------------------------------------------------------------------
def hybrid_similarity_matrix(features: Dict[Node, FeatureVec],
                             seed_nodes: List[Node],
                             vram_budget_mb: int,
                             epsilon_feature: float = None,
                             epsilon_geo: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a joint similarity matrix that fuses:

    * Feature‑based Gaussian similarity (Parent A).
    * Geometric Gaussian similarity to a *subset* of nodes used as Voronoi seeds.

    The result `S_joint` has shape (n_features, n_seeds) and contains the
    element‑wise product `S_feature[i, k] * K_geo[i, k]`.
    """
    # Full feature similarity (square matrix)
    S_feat, all_nodes = similarity_matrix(features, vram_budget_mb)
    if epsilon_feature is None:
        epsilon_feature = 1.0 / (vram_budget_mb / 1024.0)

    # Map seed identifiers to indices in the full node list
    seed_indices = [all_nodes.index(s) for s in seed_nodes]

    # Geometric RBF kernel restricted to (all_nodes × seed_nodes)
    K_geo_full, _ = rbf_kernel_matrix(features, epsilon=epsilon_geo)
    K_geo = K_geo_full[:, seed_indices]  # shape (n_all, n_seeds)

    # Joint similarity (broadcasting over seeds)
    S_joint = S_feat[:, seed_indices] * K_geo
    return S_joint, all_nodes


def hybrid_voronoi_rbf_retrieval(points: np.ndarray,
                                 seeds: np.ndarray,
                                 memory_matrices: List[np.ndarray],
                                 features: Dict[Node, FeatureVec],
                                 vram_budget_mb: int,
                                 epsilon_geo: float = 1.0,
                                 epsilon_feat: float = None) -> np.ndarray:
    """
    Perform a Voronoi‑aware, RBF‑weighted retrieval from per‑seed associative
    memory.

    Parameters
    ----------
    points : (n_points, d) array
        Query vectors.
    seeds : (n_seeds, d) array
        Seed centroids defining Voronoi cells.
    memory_matrices : list of length n_seeds
        Each entry is a 1‑D array (`output_dim`,) representing the stored memory
        for that seed.
    features : dict mapping node identifiers to raw feature vectors.
        Used to compute a *feature‑aware* weight in addition to the geometric RBF.
    vram_budget_mb : int
        Controls the scale of the feature‑based epsilon (as in Parent A).

    Returns
    -------
    retrievals : (n_points, output_dim) array
        Weighted combination of memory vectors.
    """
    n_seeds = seeds.shape[0]
    n_points = points.shape[0]
    output_dim = memory_matrices[0].shape[0]

    # 1) Voronoi region matrix (binary)
    R = assign(points, seeds)               # shape (n_seeds, n_points)

    # 2) Geometric RBF weights for every (seed, point) pair
    geo_weights = np.empty((n_seeds, n_points), dtype=np.float64)
    for i in range(n_seeds):
        for j in range(n_points):
            geo_weights[i, j] = gaussian(distance(seeds[i], points[j]), epsilon_geo)

    # 3) Feature‑based weights.
    #    We treat each point as a temporary node with a feature vector equal to its
    #    raw coordinates (converted to list).  Seeds are looked up in `features`.
    if epsilon_feat is None:
        epsilon_feat = 1.0 / (vram_budget_mb / 1024.0)

    feat_weights = np.empty((n_seeds, n_points), dtype=np.float64)
    for i in range(n_seeds):
        seed_id = i  # assume seed indices correspond to keys in `features`
        seed_feat = features.get(seed_id, seeds[i].tolist())
        for j in range(n_points):
            pt_feat = points[j].tolist()
            dist = euclidean(seed_feat, pt_feat)
            feat_weights[i, j] = gaussian(dist, epsilon_feat)

    # 4) Joint weight = Voronoi mask * geometric * feature
    joint_weights = R * geo_weights * feat_weights   # element‑wise multiplication

    # 5) Normalise per point to avoid scaling blow‑up
    col_sums = joint_weights.sum(axis=0, keepdims=True) + 1e-12
    joint_weights /= col_sums

    # 6) Retrieve: weighted sum of memory vectors
    retrievals = np.zeros((n_points, output_dim), dtype=np.float64)
    for i, mem in enumerate(memory_matrices):
        retrievals += np.outer(joint_weights[i, :], mem)

    return retrievals


def hybrid_phash_hamming(features: Dict[Node, FeatureVec]) -> Dict[Tuple[Node, Node], int]:
    """
    Compute a pairwise Hamming distance matrix on perceptual hashes of feature
    vectors.  Demonstrates integration of the hashing utilities from Parent A
    within the hybrid framework.
    """
    nodes = list(features.keys())
    hashes = {n: compute_phash(list(features[n])) for n in nodes}
    hamming = {}
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            dist = hamming_distance(hashes[a], hashes[b])
            hamming[(a, b)] = dist
            hamming[(b, a)] = dist
    return hamming


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data generation
    random.seed(0)
    np.random.seed(0)

    # 5 nodes with 3‑dimensional random feature vectors
    n_nodes = 5
    dim = 3
    features: Dict[int, List[float]] = {
        i: np.random.rand(dim).tolist() for i in range(n_nodes)
    }

    # Choose first 2 nodes as Voronoi seeds
    seed_ids = [0, 1]
    seeds_array = np.array([features[i] for i in seed_ids])

    # 8 query points (random)
    points_array = np.random.rand(8, dim)

    # Per‑seed associative memory (output_dim = 4)
    output_dim = 4
    memory = [np.random.rand(output_dim) for _ in seed_ids]

    # 1) Joint similarity matrix
    S_joint, all_nodes = hybrid_similarity_matrix(
        features,
        seed_nodes=seed_ids,
        vram_budget_mb=512
    )
    print("Joint similarity matrix shape:", S_joint.shape)

    # 2) Retrieval
    retrieved = hybrid_voronoi_rbf_retrieval(
        points=points_array,
        seeds=seeds_array,
        memory_matrices=memory,
        features=features,
        vram_budget_mb=512
    )
    print("Retrieval shape:", retrieved.shape)

    # 3) Hash‑based Hamming distances
    hamming = hybrid_phash_hamming(features)
    sample_pair = (0, 2)
    print(f"Hamming distance between nodes {sample_pair[0]} and {sample_pair[1]}:",
          hamming[sample_pair])

    # 4) Split decision demo
    split = hybrid_should_split(
        best_gain=0.12,
        second_best_gain=0.08,
        r=1.0,
        delta=0.05,
        n=100
    )
    print("Should split decision:", split)

    sys.exit(0)