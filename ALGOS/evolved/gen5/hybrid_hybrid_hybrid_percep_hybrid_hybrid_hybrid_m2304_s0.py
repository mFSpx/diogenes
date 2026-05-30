# DARWIN HAMMER — match 2304, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hdc_serpentin_m327_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s1.py (gen3)
# born: 2026-05-29T23:41:42Z

"""Hybrid Algorithm: Perceptual-RBF Hyperdimensional Privacy‑Aware Clustering
Parents:
- hybrid_hyperdimensional_surrogate (algorithm A)
- hybrid_hybrid_privac_hybrid_endpoint (algorithm B)

Mathematical bridge:
Both parents define a ``Morphology`` dataclass and expose a *sphericity* measure.
Algorithm A uses radial‑basis functions (RBF) to compute similarity and then
creates bipolar hyper‑dimensional vectors that are clustered via perceptual
hashes. Algorithm B computes a privacy reconstruction risk and combines it with
resource (VRAM) expectations.

The fusion treats the sphericity index as a geometric scaling factor that
modulates the RBF‑derived similarity, while the privacy risk provides an
additional weight. The resulting scalar weight drives the generation of a
high‑dimensional bipolar vector; these vectors are hashed and finally clustered
using Hamming distance. This intertwines the kernel‑based similarity,
hyper‑dimensional representation, and privacy‑resource modelling in a single
mathematical pipeline."""


import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Geometric description of an entity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str


# ----------------------------------------------------------------------
# Core mathematical primitives (from both parents)
# ----------------------------------------------------------------------


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance."""
    return float(np.linalg.norm(a - b))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF kernel (Gaussian)."""
    return math.exp(-((epsilon * r) ** 2))


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity: cubic root of volume divided by longest side."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Probability that a record can be re‑identified (parent B)."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: np.ndarray, epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Differential‑privacy‑style weighted sum (parent B)."""
    # Simple Laplace‑style weighting for illustration
    return float(np.sum(values * np.exp(epsilon)) / sensitivity)


def hamming_distance(a: int, b: int) -> int:
    """Bit‑wise Hamming distance."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def rbf_similarity_matrix(morphs: List[Morphology], epsilon: float = 1.0) -> np.ndarray:
    """
    Build a symmetric similarity matrix using a Gaussian RBF kernel on the
    4‑dimensional morphological feature vectors.
    """
    n = len(morphs)
    feats = np.array([[m.length, m.width, m.height, m.mass] for m in morphs])
    sim = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(feats[i], feats[j])
            sim_ij = gaussian(dist, epsilon)
            sim[i, j] = sim_ij
            sim[j, i] = sim_ij
    return sim


def privacy_weighted_sphericity(morph: Morphology, risk: float) -> float:
    """
    Combine geometric sphericity with a privacy risk factor.
    The product is used as a scalar weight for hyper‑dimensional encoding.
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    return sph * risk


def generate_bipolar_vector(weight: float, dim: int = 1024) -> np.ndarray:
    """
    Produce a random bipolar (+1 / -1) hyper‑dimensional vector whose
    polarity is biased by ``weight`` (in [0, 1]).
    """
    # Base random bipolar vector
    vec = np.where(np.random.rand(dim) > 0.5, 1, -1).astype(int)

    # Bias: flip a proportion proportional to (1-weight)
    flip_prob = max(0.0, min(1.0, 1.0 - weight))
    mask = np.random.rand(dim) < flip_prob
    vec[mask] *= -1
    return vec


def perceptual_hash(vector: np.ndarray, bits: int = 64) -> int:
    """
    Compute a simple perceptual hash: compare each component to the mean
    and set bits accordingly (parent A style).
    """
    if bits > len(vector):
        raise ValueError("bits must not exceed vector length")
    mean_val = float(np.mean(vector))
    h = 0
    for i in range(bits):
        h = (h << 1) | int(vector[i] >= mean_val)
    return h


def hybrid_score_matrix(
    morphs: List[Morphology],
    risks: List[float],
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Combine RBF similarity with privacy‑weighted sphericity.
    For each pair (i, j):
        score_ij = sim_ij * sqrt(w_i * w_j)
    where sim_ij is the Gaussian kernel and w_i is the privacy‑weighted sphericity.
    """
    sim = rbf_similarity_matrix(morphs, epsilon)
    weights = np.array([privacy_weighted_sphericity(m, r) for m, r in zip(morphs, risks)])
    # Ensure weights are non‑negative
    weights = np.clip(weights, 0.0, None)
    # Outer product sqrt weighting
    weight_factor = np.sqrt(np.outer(weights, weights))
    return sim * weight_factor


def encode_entities(
    morphs: List[Morphology],
    risks: List[float],
    dim: int = 1024,
) -> Dict[str, np.ndarray]:
    """
    Produce a dictionary mapping entity identifiers to hyper‑dimensional vectors.
    The weight for each vector is derived from the hybrid score matrix row
    (average similarity to others) multiplied by the privacy‑weighted sphericity.
    """
    if len(morphs) != len(risks):
        raise ValueError("morphs and risks must have the same length")
    # Global hybrid scores to derive per‑entity strength
    hybrid_mat = hybrid_score_matrix(morphs, risks)
    # Row‑wise average similarity as a proxy for “prominence”
    prominence = np.mean(hybrid_mat, axis=1)
    encodings: Dict[str, np.ndarray] = {}
    for idx, (m, r) in enumerate(zip(morphs, risks)):
        base_weight = privacy_weighted_sphericity(m, r)
        final_weight = float(base_weight * (prominence[idx] + 1e-6))  # avoid zero
        # Normalise weight to [0,1] for the generator
        norm_weight = max(0.0, min(1.0, final_weight))
        encodings[f"entity_{idx}"] = generate_bipolar_vector(norm_weight, dim)
    return encodings


def cluster_by_hamming(hashes: Dict[str, int], max_distance: int = 4) -> List[List[str]]:
    """
    Group identifiers whose perceptual hashes differ by at most ``max_distance``.
    """
    clusters: List[List[str]] = []
    for key, h in hashes.items():
        placed = False
        for cluster in clusters:
            rep_key = cluster[0]
            if hamming_distance(h, hashes[rep_key]) <= max_distance:
                cluster.append(key)
                placed = True
                break
        if not placed:
            clusters.append([key])
    return clusters


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a synthetic dataset
    random.seed(42)
    np.random.seed(42)

    # Five entities with random morphology
    morphologies = [
        Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(1.0, 10.0),
        )
        for _ in range(5)
    ]

    # Simulated privacy risks (unique identifiers / total records)
    risks = [
        reconstruction_risk_score(
            unique_quasi_identifiers=random.randint(0, 100),
            total_records=200,
        )
        for _ in morphologies
    ]

    # Encode entities into hyper‑dimensional vectors
    vectors = encode_entities(morphologies, risks, dim=1024)

    # Compute perceptual hashes
    hashes = {k: perceptual_hash(v) for k, v in vectors.items()}

    # Cluster based on Hamming distance
    clusters = cluster_by_hamming(hashes, max_distance=4)

    # Output results (simple print, no external files)
    print("Risks:", ["{:.3f}".format(r) for r in risks])
    print("Hashes:", {k: f"{h:016x}" for k, h in hashes.items()})
    print("Clusters:", clusters)

    # Verify that the pipeline runs without raising exceptions
    sys.exit(0)