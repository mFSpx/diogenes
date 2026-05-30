# DARWIN HAMMER — match 4832, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1857_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2702_s3.py (gen5)
# born: 2026-05-29T23:58:23Z

"""
Hybrid Algorithm: hybrid_fusion_morphology_geometry.py

This module fuses the core topologies of two parent algorithms:

* Parent A – hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s0.py 
  (provides morphology‑driven righting‑time index, recovery priority and a 
  structural‑similarity (SSIM) based evaluation).

* Parent B – hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2702_s3.py 
  (provides a geometric_pro based evaluation and a gaussian similarity 
  framework).

**Mathematical bridge** – The bridge uses the sphericity_index from Parent A 
and the gaussian similarity from Parent B to couple the morphology‑driven 
recovery_priority with the geometric_pro based evaluation. The resulting 
hybrid_priority scales the gaussian similarity and recovery priority, 
producing a unified assessment that respects both physical stability and 
geometric reliability.

"""

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Morphology:
    """Physical description of an endpoint."""
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity used by both parents."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness metric from Parent A."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0) -> float:
    """Righting time index from Parent A."""
    return b * m.mass / (m.length * m.width)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel from Parent B."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance between two feature vectors from Parent B."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: 64‑bit based on median threshold from Parent B."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers from Parent B."""
    return (a ^ b).bit_count()


def gaussian_similarity_matrix(features: Dict[int, Tuple[float, ...]],
                               epsilon: float = 1.0) -> Tuple[np.ndarray, List[int]]:
    """
    Build a symmetric similarity matrix S where
        S[i, j] = 1 - (Hamming(phash_i, phash_j) / 64)
    and then modulate it with a Gaussian kernel based on Euclidean distance.
    Returns the matrix and the ordered list of nodes.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)

    # Pre‑compute phashes
    phashes = {node: compute_phash(list(features[node])) for node in nodes}

    for i, ni in enumerate(nodes):
        hi = phashes[ni]
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
                continue
            hj = phashes[nj]
            d_ham = hamming_distance(hi, hj)
            base_sim = 1.0 - d_ham / 64.0

            # Modulate with Gaussian kernel
            if i == j:
                S[i, j] = 1.0
            else:
                d_euc = euclidean(features[ni], features[nj])
                S[i, j] = base_sim * gaussian(d_euc, epsilon)

    return S, nodes


def hybrid_priority(morphology: Morphology, features: Tuple[float, ...],
                    epsilon: float = 1.0) -> float:
    """
    Calculate a hybrid priority that couples morphology-driven recovery
    priority with geometric_pro based evaluation.
    """
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    flatness = flatness_index(morphology.length, morphology.width, morphology.height)
    recovery_priority = righting_time_index(morphology)

    # Calculate gaussian similarity
    gaussian_sim = gaussian(euclidean(features, (0.0, 0.0)), epsilon)

    # Calculate hybrid priority
    hybrid_priority = recovery_priority * gaussian_sim * (1.0 + sphericity)

    return hybrid_priority


def hybrid_similarity_matrix(morphologies: Dict[int, Morphology],
                             features: Dict[int, Tuple[float, ...]],
                             epsilon: float = 1.0) -> Tuple[np.ndarray, List[int]]:
    """
    Build a symmetric similarity matrix S where
        S[i, j] = hybrid_priority(m_i, f_j) * gaussian_similarity_matrix(f_i, f_j)
    Returns the matrix and the ordered list of nodes.
    """
    nodes = list(morphologies.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)

    # Pre‑compute phashes and gaussian similarity matrix
    phashes = {node: compute_phash(list(features[node])) for node in nodes}
    gaussian_sim_matrix, _ = gaussian_similarity_matrix(features, epsilon)

    for i, ni in enumerate(nodes):
        hi = phashes[ni]
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
                continue

            # Calculate hybrid priority
            hybrid_pri = hybrid_priority(morphologies[ni], features[nj], epsilon)

            # Calculate gaussian similarity
            gaussian_sim = gaussian_sim_matrix[i, j]

            # Calculate hybrid similarity
            S[i, j] = hybrid_pri * gaussian_sim

    return S, nodes


if __name__ == "__main__":
    # Smoke test
    morphologies = {
        0: Morphology(1.0, 2.0, 3.0, 10.0),
        1: Morphology(4.0, 5.0, 6.0, 20.0),
    }
    features = {
        0: (0.1, 0.2, 0.3),
        1: (0.4, 0.5, 0.6),
    }
    hybrid_sim_matrix, nodes = hybrid_similarity_matrix(morphologies, features)
    print(hybrid_sim_matrix)