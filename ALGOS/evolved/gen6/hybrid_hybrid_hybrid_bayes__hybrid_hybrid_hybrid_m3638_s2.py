# DARWIN HAMMER — match 3638, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_bayes_update__hybrid_possum_filter_m210_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m2304_s0.py (gen5)
# born: 2026-05-29T23:51:00Z

"""Hybrid Bayesian‑Ricci – RBF Hyperdimensional Privacy Model

Parents:
- hybrid_hybrid_bayes_update__hybrid_possum_filter_m210_s0.py
- hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m2304_s0.py

Mathematical bridge:
The spatial‑aware privacy risk vector from the first parent is interpreted as a prior
distribution over graph nodes. Bayesian posteriors w_{ij}=P(H_i|E_j) become edge
weights of a graph on which Ollivier‑Ricci curvature κ_{ij} is evaluated.
The second parent supplies a radial‑basis‑function (RBF) similarity between the
geometric feature vectors of entities; this similarity is scaled by the
sphericity index (geometric shape) and by the privacy‑risk‑derived posterior
weights. The scaled similarity drives the construction of high‑dimensional
bipolar hypervectors, which are finally clustered by Hamming distance.

The code below intertwines the two pipelines:
1. Bayesian edge‑weight construction → adjacency matrix.
2. Approximate Ollivier‑Ricci curvature on that adjacency.
3. RBF‑based similarity modulated by sphericity and curvature → hypervectors.
4. Simple Hamming‑based clustering as a demonstration of the fused system.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from Parent B)
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
# Bayesian primitives (from Parent A)
# ----------------------------------------------------------------------


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = likelihood * prior + false_positive * (1‑prior)"""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = prior * likelihood / marginal"""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


def build_adjacency(priors: List[float],
                    likelihoods: List[float],
                    false_positives: List[float]) -> np.ndarray:
    """
    Build a square adjacency matrix where entry (i, j) is the Bayesian posterior
    w_{ij}=P(H_i|E_j).  The matrix is directed because priors differ per node.
    """
    n = len(priors)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            marginal = bayes_marginal(priors[i], likelihoods[j], false_positives[i])
            adj[i, j] = bayes_update(priors[i], likelihoods[j], marginal)
    return adj


# ----------------------------------------------------------------------
# RBF / geometry primitives (from Parent B)
# ----------------------------------------------------------------------


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance."""
    return float(np.linalg.norm(a - b))


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial‑basis‑function."""
    return math.exp(-((epsilon * r) ** 2))


def sphericity_index(length: float, width: float, height: float) -> float:
    """
    Geometric sphericity: (volume)^(1/3) / longest side.
    Returns a value in (0, 1] with 1 for a perfect sphere‑like cube.
    """
    volume = length * width * height
    longest = max(length, width, height)
    if longest == 0:
        return 0.0
    return (volume ** (1.0 / 3.0)) / longest


def morphology_vector(m: Morphology) -> np.ndarray:
    """Convert Morphology to a numeric vector (including mass)."""
    return np.array([m.length, m.width, m.height, m.mass], dtype=float)


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature approximation (new fusion)
# ----------------------------------------------------------------------


def approximate_ollivier_ricci(adj: np.ndarray) -> np.ndarray:
    """
    Very lightweight approximation of Ollivier‑Ricci curvature.
    For each directed edge i→j we compute:

        κ_{ij} = 1 - (|w_{ij} - w_{ji}|) / (w_{ij} + w_{ji} + ε)

    where w are the Bayesian posterior weights.  This captures the idea that
    symmetric edges (w_{ij}≈w_{ji}) have high curvature, while highly
    asymmetric edges have low curvature.
    """
    eps = 1e-9
    n = adj.shape[0]
    curvature = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                curvature[i, j] = 0.0
                continue
            w_ij = adj[i, j]
            w_ji = adj[j, i]
            curvature[i, j] = 1.0 - abs(w_ij - w_ji) / (w_ij + w_ji + eps)
    return curvature


# ----------------------------------------------------------------------
# Hyperdimensional vector generation (fusion of RBF & privacy)
# ----------------------------------------------------------------------


def generate_hypervector(seed: int,
                         weight: float,
                         dim: int = 10_000) -> np.ndarray:
    """
    Produce a bipolar (+1 / -1) hypervector of length `dim`.
    The random generator is seeded with `seed` to make the operation
    deterministic for a given entity.  The `weight` (0‑1) biases the proportion
    of +1 entries: probability(+1) = 0.5 + 0.5 * weight.
    """
    rng = np.random.default_rng(seed)
    prob_pos = 0.5 + 0.5 * max(min(weight, 1.0), -1.0)  # clamp to [-1,1]
    bits = rng.random(dim) < prob_pos
    return np.where(bits, 1, -1).astype(np.int8)


def hypervector_from_entity(idx: int,
                            morph: Morphology,
                            curvature: float,
                            privacy_weight: float,
                            epsilon: float = 1.0) -> np.ndarray:
    """
    Combine geometric RBF similarity, sphericity, curvature and privacy
    weight into a single scalar that drives hypervector generation.

    Steps:
    1. Compute a self‑RBF value (distance zero → 1.0) scaled by sphericity.
    2. Multiply by curvature (captures graph geometry) and privacy_weight.
    3. Feed the resulting scalar to `generate_hypervector`.
    """
    # self‑similarity is always 1.0, but we still apply the scaling factors
    sph = sphericity_index(morph.length, morph.width, morph.height)
    combined_weight = sph * curvature * privacy_weight
    # Clamp to [0,1] for stability
    combined_weight = max(0.0, min(1.0, combined_weight))
    return generate_hypervector(seed=idx, weight=combined_weight)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two bipolar vectors."""
    return int(np.sum(a != b) / 2)  # because -1 vs 1 counts as difference


# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------


def hybrid_pipeline(priors: List[float],
                    likelihoods: List[float],
                    false_positives: List[float],
                    morphologies: List[Morphology],
                    privacy_risks: List[float],
                    dim: int = 10_000) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    End‑to‑end fusion:

    1. Build Bayesian adjacency matrix from priors/likelihoods.
    2. Approximate Ollivier‑Ricci curvature on that graph.
    3. For each entity i, generate a hypervector using:
           curvature[i,i] (self‑curvature) and privacy_risks[i].
    4. Return adjacency, curvature, and the list of hypervectors.
    """
    if not (len(priors) == len(likelihoods) == len(false_positives) ==
            len(morphologies) == len(privacy_risks)):
        raise ValueError("All input lists must have the same length")

    adj = build_adjacency(priors, likelihoods, false_positives)
    curv = approximate_ollivier_ricci(adj)

    hypervectors = []
    for idx, (morph, priv) in enumerate(zip(morphologies, privacy_risks)):
        # Use self‑curvature as a proxy for local geometric consistency
        self_curv = curv[idx, idx]
        hv = hypervector_from_entity(idx, morph, self_curv, priv, epsilon=1.0)
        hypervectors.append(hv)

    return adj, curv, hypervectors


def cluster_hypervectors(vectors: List[np.ndarray],
                         threshold: int = 2000) -> List[List[int]]:
    """
    Very simple greedy clustering based on Hamming distance.
    Two vectors belong to the same cluster if their distance is below `threshold`.
    Returns a list of clusters, each cluster being a list of indices.
    """
    n = len(vectors)
    unassigned = set(range(n))
    clusters: List[List[int]] = []

    while unassigned:
        seed = unassigned.pop()
        cluster = [seed]
        to_check = list(unassigned)
        for other in to_check:
            if hamming_distance(vectors[seed], vectors[other]) <= threshold:
                cluster.append(other)
                unassigned.remove(other)
        clusters.append(cluster)

    return clusters


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a tiny synthetic scenario with three entities
    priors = [0.7, 0.4, 0.9]
    likelihoods = [0.6, 0.3, 0.8]
    false_positives = [0.3, 0.6, 0.1]

    morphologies = [
        Morphology(length=1.0, width=0.8, height=0.9, mass=2.0),
        Morphology(length=0.5, width=0.5, height=0.5, mass=1.0),
        Morphology(length=1.5, width=1.2, height=1.3, mass=3.0)
    ]

    # Privacy risk is taken as the Bayesian posterior of self‑edge for illustration
    adj_tmp = build_adjacency(priors, likelihoods, false_positives)
    privacy_risks = [adj_tmp[i, i] for i in range(3)]

    adj, curv, hypervectors = hybrid_pipeline(
        priors, likelihoods, false_positives,
        morphologies, privacy_risks,
        dim=4096
    )

    print("Adjacency matrix:\n", np.round(adj, 3))
    print("\nCurvature matrix:\n", np.round(curv, 3))

    clusters = cluster_hypervectors(hypervectors, threshold=800)
    print("\nClusters (indices):", clusters)

    # Verify that hypervectors have the correct shape and bipolar values
    for i, hv in enumerate(hypervectors):
        assert hv.shape == (4096,)
        assert set(np.unique(hv)) == {-1, 1}
        print(f"Entity {i} hypervector mean value: {hv.mean():.3f}")