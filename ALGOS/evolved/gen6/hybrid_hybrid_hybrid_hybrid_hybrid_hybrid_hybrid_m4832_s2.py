# DARWIN HAMMER — match 4832, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1857_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2702_s3.py (gen5)
# born: 2026-05-29T23:58:23Z

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Physical description of an endpoint."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Core geometric helpers (shared by both parent systems)
# ----------------------------------------------------------------------


def _positive_dimensions(*dims: float) -> None:
    """Validate that all supplied dimensions are strictly positive."""
    if any(d <= 0 for d in dims):
        raise ValueError("All dimensions must be positive numbers")


def sphericity_index(length: float, width: float, height: float) -> float:
    """
    Classical sphericity: ratio of the radius of a sphere with the same volume
    to the longest dimension. Returns a value in (0, 1].
    """
    _positive_dimensions(length, width, height)
    volume = length * width * height
    return (volume ** (1.0 / 3.0)) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """
    Flatness metric – larger values indicate a flatter shape.
    """
    _positive_dimensions(length, width, height)
    return (length + width) / (2.0 * height)


def righting_time_index(morph: Morphology, b: float = 1.0 / 3.0) -> float:
    """
    A simple physics‑based estimate of the time needed for an object to right
    itself. Higher mass and smaller footprint increase the index.
    """
    _positive_dimensions(morph.length, morph.width)
    return b * morph.mass / (morph.length * morph.width)


# ----------------------------------------------------------------------
# Perceptual‑hash utilities (Parent B)
# ----------------------------------------------------------------------


def compute_phash(values: List[float]) -> int:
    """
    64‑bit perceptual hash based on the median of *values*.
    Bits are set to 1 when the corresponding value is >= median, otherwise 0.
    """
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Distance / similarity kernels
# ----------------------------------------------------------------------


def euclidean(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Morphology‑driven score
# ----------------------------------------------------------------------


def morphology_score(morph: Morphology) -> float:
    """
    Normalised morphology score in the interval (0, 1].
    Combines sphericity, flatness and righting‑time information.
    The three components are first transformed to (0, 1] via a logistic map
    and then averaged.
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flat = flatness_index(morph.length, morph.width, morph.height)
    rt = righting_time_index(morph)

    # Logistic transform to bound values between 0 and 1 (avoid division by zero)
    def logistic(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    sph_n = logistic(5 * (sph - 0.5))          # centre around 0.5, steepness 5
    flat_n = logistic(5 * (flat - 0.5))
    rt_n = logistic(5 * (rt - 0.5))

    return (sph_n + flat_n + rt_n) / 3.0


# ----------------------------------------------------------------------
# Base similarity matrix (phash + Gaussian on Euclidean distance)
# ----------------------------------------------------------------------


def base_similarity_matrix(
    features: Dict[int, Tuple[float, ...]],
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, List[int]]:
    """
    Construct a symmetric similarity matrix S where

        S[i, j] = (1 - Hamming(phash_i, phash_j) / 64) *
                  exp(- (ε * d_Euc(i, j))²)

    The diagonal is forced to 1.0.
    """
    nodes = list(features.keys())
    n = len(nodes)

    # Pre‑compute perceptual hashes
    phashes = {nid: compute_phash(list(features[nid])) for nid in nodes}

    # Allocate matrix
    S = np.empty((n, n), dtype=np.float64)

    for i, nid_i in enumerate(nodes):
        hi = phashes[nid_i]
        for j, nid_j in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]          # symmetry shortcut
                continue

            if i == j:
                S[i, j] = 1.0
                continue

            hj = phashes[nid_j]
            ham = hamming_distance(hi, hj)
            phash_sim = 1.0 - ham / 64.0

            d_euc = euclidean(features[nid_i], features[nid_j])
            gauss = gaussian(d_euc, epsilon)

            S[i, j] = phash_sim * gauss

    return S, nodes


# ----------------------------------------------------------------------
# Hybrid similarity matrix (morphology × base similarity)
# ----------------------------------------------------------------------


def hybrid_similarity_matrix(
    morphologies: Dict[int, Morphology],
    features: Dict[int, Tuple[float, ...]],
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, List[int]]:
    """
    Build a symmetric hybrid similarity matrix.

    Let `M_i` be the normalised morphology score for node *i* and
    `B_ij` the base similarity from :func:`base_similarity_matrix`.
    The hybrid similarity is defined as

        H_ij = M_i * M_j * B_ij

    This formulation guarantees symmetry (H_ij = H_ji) and cleanly
    separates the morphology contribution from the purely geometric one.
    """
    # 1️⃣ Compute morphology scores vector
    nodes = list(morphologies.keys())
    if set(nodes) != set(features.keys()):
        raise ValueError("Morphology and feature dictionaries must contain the same keys")

    morph_scores = np.array([morphology_score(morphologies[n]) for n in nodes], dtype=np.float64)

    # 2️⃣ Compute base similarity matrix (phash + Gaussian)
    base_mat, base_nodes = base_similarity_matrix(features, epsilon)

    # Ensure ordering consistency
    if nodes != base_nodes:
        # Re‑order base_mat to match `nodes`
        idx_map = {nid: i for i, nid in enumerate(base_nodes)}
        reorder = [idx_map[n] for n in nodes]
        base_mat = base_mat[np.ix_(reorder, reorder)]

    # 3️⃣ Outer product of morphology scores yields a matrix M_ij = M_i * M_j
    morph_outer = np.outer(morph_scores, morph_scores)

    # 4️⃣ Element‑wise multiplication yields the hybrid matrix
    hybrid_mat = morph_outer * base_mat

    return hybrid_mat, nodes


# ----------------------------------------------------------------------
# Simple smoke test (executed when run as a script)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Minimal example with two nodes
    morphologies = {
        0: Morphology(length=1.0, width=2.0, height=3.0, mass=10.0),
        1: Morphology(length=4.0, width=5.0, height=6.0, mass=20.0),
    }

    features = {
        0: (0.1, 0.2, 0.3),
        1: (0.4, 0.5, 0.6),
    }

    hybrid_mat, order = hybrid_similarity_matrix(morphologies, features, epsilon=1.2)
    np.set_printoptions(precision=4, suppress=True)
    print("Node order :", order)
    print("Hybrid similarity matrix:\n", hybrid_mat)