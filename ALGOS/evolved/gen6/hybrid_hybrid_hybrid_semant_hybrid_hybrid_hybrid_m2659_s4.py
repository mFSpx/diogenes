# DARWIN HAMMER — match 2659, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1197_s1.py (gen5)
# born: 2026-05-29T23:43:29Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s0 (semantic neighbors, temporal motifs, morphology indices)
- hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1197_s1 (RBF similarity, Gini coefficient, Hoeffding‑style decision weighting)

Mathematical Bridge:
The bridge is built on a common vector space that represents each temporal motif both
(1) as a semantic embedding (used by `semantic_neighbors`) and
(2) as a geometric point for which an RBF similarity matrix can be constructed.
The RBF matrix supplies a dense affinity measure between motifs; the Gini coefficient
is then applied to the distribution of these affinities (the “tropical score vector”)
to quantify inequality. Finally, the morphology‑based righting‑time index
provides a physical weighting. The hybrid score for a motif is therefore

    hybrid_score = recovery_priority(morphology) *
                   mean(RBF_similarity to semantic neighbors) *
                   (1 - Gini(affinity_vector))

Thus the algorithm fuses the topological neighbor search of parent A with the
continuous similarity and inequality measures of parent B into a single
decision metric.
"""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Set, Hashable, Sequence
import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared across both parent designs)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int
    embedding: Tuple[float, ...]  # semantic vector representation

@dataclass(frozen=True)
class HybridMotif:
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    score: float

# ----------------------------------------------------------------------
# Parent‑A style morphology indices
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

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
    """Normalised righting‑time index in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

# ----------------------------------------------------------------------
# Parent‑A style semantic neighbour search
# ----------------------------------------------------------------------
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def semantic_neighbors(target: TemporalMotif,
                       corpus: List[TemporalMotif],
                       k: int = 5) -> List[Tuple[TemporalMotif, float]]:
    """Return the top‑k most similar motifs to *target* using cosine similarity."""
    similarities = [
        (motif, cosine_similarity(target.embedding, motif.embedding))
        for motif in corpus
        if motif != target
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

# ----------------------------------------------------------------------
# Parent‑B style RBF similarity matrix and Gini coefficient
# ----------------------------------------------------------------------
def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def rbf_similarity_matrix(vectors: List[Tuple[float, ...]],
                         epsilon: float = 1.0) -> np.ndarray:
    """Dense RBF similarity matrix for a list of vectors."""
    n = len(vectors)
    mat = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(vectors[i], vectors[j])
            sim = gaussian_rbf(dist, epsilon)
            mat[i, j] = sim
            mat[j, i] = sim
    return mat

def gini_coefficient(values: Sequence[float]) -> float:
    """Gini coefficient of a non‑negative value list."""
    if not values:
        return 0.0
    sorted_vals = sorted(v for v in values if v >= 0)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    cumulative = 0.0
    for i, v in enumerate(sorted_vals, 1):
        cumulative += i * v
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return gini

# ----------------------------------------------------------------------
# Hybrid operations (three core functions)
# ----------------------------------------------------------------------
def compute_hybrid_scores(motifs: List[TemporalMotif],
                          morphology: Morphology,
                          epsilon: float = 1.0,
                          k_neighbors: int = 5) -> List[HybridMotif]:
    """
    For each motif:
      1. Find its semantic neighbours.
      2. Compute mean RBF similarity to those neighbours.
      3. Compute Gini of the full RBF similarity row (tropical score vector).
      4. Combine with recovery_priority to obtain a final score.
    Returns a list of HybridMotif objects.
    """
    # Pre‑compute the RBF matrix for all embeddings
    embeddings = [m.embedding for m in motifs]
    rbf_mat = rbf_similarity_matrix(embeddings, epsilon)

    # Physical weighting from morphology
    phys_weight = recovery_priority(morphology)

    hybrid_list: List[HybridMotif] = []
    for idx, motif in enumerate(motifs):
        # 1. Semantic neighbours (using cosine similarity)
        neigh = semantic_neighbors(motif, motifs, k=k_neighbors)
        if neigh:
            mean_rbf = float(np.mean([rbf_mat[idx, motifs.index(n[0])] for n in neigh]))
        else:
            mean_rbf = 0.0

        # 2. Gini of the motif's affinity row
        gini = gini_coefficient(rbf_mat[idx, :])

        # 3. Hybrid score composition
        score = phys_weight * mean_rbf * (1.0 - gini)

        # Dummy geographic centroid (random for demonstration)
        centroid_lat = random.uniform(-90.0, 90.0)
        centroid_lon = random.uniform(-180.0, 180.0)

        hybrid = HybridMotif(
            pattern=motif.pattern,
            support=motif.support,
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            score=score,
        )
        hybrid_list.append(hybrid)
    return hybrid_list

def rank_hybrid_motifs(hybrid_motifs: List[HybridMotif]) -> List[HybridMotif]:
    """Return motifs sorted by descending hybrid score."""
    return sorted(hybrid_motifs, key=lambda hm: hm.score, reverse=True)

def filter_by_spatial_diversity(hybrid_motifs: List[HybridMotif],
                               min_distance_km: float = 50.0) -> List[HybridMotif]:
    """
    Simple spatial diversity filter: keep a motif only if it is at least
    *min_distance_km* away from any previously kept motif (using haversine).
    """
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    kept: List[HybridMotif] = []
    for motif in hybrid_motifs:
        if all(haversine(motif.centroid_lat, motif.centroid_lon,
                         k.centroid_lat, k.centroid_lon) >= min_distance_km
               for k in kept):
            kept.append(motif)
    return kept

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy morphology
    demo_morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Generate random temporal motifs with 8‑dimensional embeddings
    random.seed(42)
    motifs: List[TemporalMotif] = []
    for i in range(10):
        pattern = tuple(f"event{i}_{j}" for j in range(3))
        support = random.randint(5, 20)
        embedding = tuple(random.random() for _ in range(8))
        motifs.append(TemporalMotif(pattern=pattern, support=support, embedding=embedding))

    # Compute hybrid scores
    hybrid = compute_hybrid_scores(motifs, demo_morph, epsilon=0.8, k_neighbors=3)

    # Rank and filter
    ranked = rank_hybrid_motifs(hybrid)
    filtered = filter_by_spatial_diversity(ranked, min_distance_km=1000.0)

    # Print a concise summary
    print("Top hybrid motifs after spatial filtering:")
    for hm in filtered[:5]:
        print(f"Pattern: {hm.pattern}, Score: {hm.score:.4f}, "
              f"Location: ({hm.centroid_lat:.2f}, {hm.centroid_lon:.2f})")