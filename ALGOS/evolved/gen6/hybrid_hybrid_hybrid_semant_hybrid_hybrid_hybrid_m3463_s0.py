# DARWIN HAMMER — match 3463, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_ternar_m1293_s1.py (gen5)
# born: 2026-05-29T23:50:13Z

"""Hybrid Morphology‑Stylometry Fusion (HM‑SF)

Parents:
- hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s2.py (Algorithm A)
- hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_ternar_m1293_s1.py (Algorithm B)

Mathematical bridge:
Algorithm A provides a *recovery priority* ρ(m) derived from a morphology m
and a *semantic joint score* J(p)=s(p)·(1+z_s) where s(p) is a cosine similarity
and z_s is the z‑score of the similarity distribution across a pattern set.
Algorithm B treats stylometric feature vectors as geometric points, builds a
Voronoi partition of the feature space and evaluates similarity of a point to
its cell centroid using an SSIM‑like measure.

The fusion maps the stylometric vector of a document to a Voronoi cell,
uses the centroid similarity as the “semantic similarity” s(p), and then
feeds that similarity into the joint score J(p).  Finally the unified
priority U combines the morphological recovery priority ρ(m) with J(p) by a
convex combination:

    U = α·ρ(m) + (1‑α)·J(p) ,   0 ≤ α ≤ 1

Thus spatial (Voronoi geometry), temporal (z‑score) and morphological
information are mathematically fused into a single scalar that can drive
ranking, routing or selection decisions.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Core structures from Parent A
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized recovery priority ρ(m) ∈ [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Core utilities from Parent B (stylometry → geometry)
# ----------------------------------------------------------------------


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two 1‑D arrays."""
    if v1.shape != v2.shape:
        raise ValueError("vectors must have the same shape")
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def z_score(value: float, mean: float, std: float) -> float:
    """Standard z‑score."""
    if std == 0:
        return 0.0
    return (value - mean) / std


def build_voronoi_cells(points: np.ndarray, n_cells: int,
                        rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """
    Very lightweight Voronoi approximation:
    - Randomly pick `n_cells` seed points (centroids).
    - Assign each point to the nearest centroid (Euclidean distance).
    Returns:
        assignments: shape (N,) integer index of the centroid for each point.
        centroids:   shape (n_cells, D) the seed coordinates.
    """
    if n_cells <= 0 or n_cells > len(points):
        raise ValueError("invalid number of Voronoi cells")
    indices = rng.sample(range(len(points)), n_cells)
    centroids = points[indices].copy()
    # iterative refinement (2 iterations are enough for a toy implementation)
    for _ in range(2):
        # assign
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)
        # recompute centroids
        for i in range(n_cells):
            mask = assignments == i
            if np.any(mask):
                centroids[i] = points[mask].mean(axis=0)
    return assignments, centroids


def centroid_similarity(point: np.ndarray, centroid: np.ndarray) -> float:
    """
    SSIM‑like similarity between a point and its Voronoi centroid.
    For 1‑D vectors we reuse cosine similarity as a proxy.
    """
    return cosine_similarity(point, centroid)


# ----------------------------------------------------------------------
# Fusion functions (the new hybrid)
# ----------------------------------------------------------------------


def semantic_joint_score(similarity: float,
                         similarity_distribution: np.ndarray) -> float:
    """
    Implements J(p) = s(p)·(1+z_s) from Parent A.
    `similarity` is s(p); `similarity_distribution` contains all s(p) values
    for the current pattern set, used to compute the z‑score.
    """
    mean = similarity_distribution.mean()
    std = similarity_distribution.std()
    z = z_score(similarity, mean, std)
    return similarity * (1.0 + z)


def unified_priority(morph: Morphology,
                     doc_vector: np.ndarray,
                     all_vectors: np.ndarray,
                     n_voronoi_cells: int = 5,
                     alpha: float = 0.4,
                     rng: random.Random = random.Random(42)) -> float:
    """
    Compute the hybrid priority U = α·ρ(m) + (1‑α)·J(p).

    Steps:
    1. Build a Voronoi partition of the stylometric space (Parent B).
    2. Find the centroid of the cell containing `doc_vector`.
    3. Measure cosine similarity between `doc_vector` and its centroid
       → serves as s(p) for the joint score.
    4. Compute the distribution of similarities of all vectors to their
       respective centroids; use it to obtain the z‑score.
    5. Assemble J(p) and blend with the morphological recovery priority.
    """
    # 1. Voronoi partition
    assignments, centroids = build_voronoi_cells(all_vectors, n_voronoi_cells, rng)

    # 2. Locate the cell of the target document
    dists = np.linalg.norm(all_vectors - doc_vector, axis=1)
    target_idx = np.argmin(dists)
    cell_id = assignments[target_idx]
    centroid = centroids[cell_id]

    # 3. Cosine similarity as s(p)
    s_p = centroid_similarity(doc_vector, centroid)

    # 4. Similarity distribution across the whole set
    sim_all = np.empty(len(all_vectors))
    for i, point in enumerate(all_vectors):
        cid = assignments[i]
        sim_all[i] = centroid_similarity(point, centroids[cid])

    # 5. Joint score J(p)
    J_p = semantic_joint_score(s_p, sim_all)

    # 6. Morphology priority ρ(m)
    rho = recovery_priority(morph)

    # 7. Convex blend
    U = alpha * rho + (1.0 - alpha) * J_p
    # Clamp to [0,1] for safety
    return max(0.0, min(1.0, U))


def sample_voronoi_vectors(num_points: int = 30,
                           dim: int = 8,
                           rng: random.Random = random.Random(123)) -> np.ndarray:
    """Generate a synthetic stylometric matrix (num_points × dim)."""
    return np.array([rng.random() for _ in range(num_points * dim)]).reshape(num_points, dim)


def demo_hybrid():
    """Run a quick demonstration of the hybrid algorithm."""
    # Create a random morphology
    morph = Morphology(length=2.3, width=1.1, height=0.9, mass=4.5)

    # Generate stylometric vectors
    all_vecs = sample_voronoi_vectors(num_points=50, dim=6)

    # Pick a target document vector (the first one for reproducibility)
    target_vec = all_vecs[0].copy()

    # Compute the unified priority
    priority = unified_priority(morph, target_vec, all_vecs,
                                n_voronoi_cells=7,
                                alpha=0.35)

    print(f"Unified priority for the sample document: {priority:.4f}")


if __name__ == "__main__":
    demo_hybrid()