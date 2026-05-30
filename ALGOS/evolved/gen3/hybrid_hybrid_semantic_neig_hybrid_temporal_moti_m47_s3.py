# DARWIN HAMMER — match 47, survivor 3
# gen: 3
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s1.py (gen2)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s1.py (gen1)
# born: 2026-05-29T23:26:30Z

"""Hybrid Semantic‑Temporal Morphology Fusion
Parents:
- hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s1.py (semantic similarity,
  morphology‑based recovery priority, endpoint circuit)
- hybrid_temporal_motifs_possum_filter_m87_s1.py (temporal motif support,
  z‑score scaling, spatial possum filter)

Mathematical bridge:
For each entity we attach a *Morphology* (from Parent A) and a *TemporalMotif*
(Parent B).  The recovery priority  R(m)  (Parent A) is used as a continuous
weight for the temporal support  s(p)  (Parent B).  The cosine similarity
between the entity’s feature vector and its *semantic neighbours* supplies a
spatial proximity factor  σ(i,j)  that replaces the binary matrix D(i,j) used by
the possum filter.  The unified score for a candidate spatio‑temporal motif
is

    S = s(p) · (1 + z_s) · R(m) · σ

where  z_s  is the z‑score of the support distribution across all motifs.
The mask M(i,j)=σ(i,j)∧(sig_i==sig_j) yields a maximal independent set,
exactly the possum‑style filtering but driven by the continuous similarity
σ.  The following module implements this fusion with three demonstration
functions.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# ---------- Data structures -------------------------------------------------

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

@dataclass(frozen=True)
class HybridMotif:
    """Entity representing a spatio‑temporal motif with morphology."""
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    morphology: Morphology
    vector: Tuple[float, ...]          # semantic feature vector
    score: float                       # unified hybrid score

# ---------- Parent A – morphology & semantic utilities -----------------------

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
    """Normalized priority in [0,1] derived from righting‑time index."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    a_arr = np.fromiter(a, dtype=float)
    b_arr = np.fromiter(b, dtype=float)
    den = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return 0.0 if den == 0 else float(a_arr @ b_arr / den)


def semantic_neighbors(doc_id: str,
                       vector: List[float],
                       pool_vectors: List[Tuple[str, List[float]]],
                       k: int = 5) -> List[Tuple[str, float]]:
    """Return the top‑k most similar (cosine) vectors excluding the query."""
    candidates = [(did, _cosine(vector, vec)) for did, vec in pool_vectors
                  if did != doc_id]
    return sorted(candidates, key=lambda x: (-x[1], x[0]))[:k]

# ---------- Parent B – temporal motif utilities ------------------------------

def z_scores(supports: List[int]) -> List[float]:
    """Return z‑score for each support value in the list."""
    if not supports:
        return []
    mu = np.mean(supports)
    sigma = np.std(supports, ddof=1) or 1.0
    return [(s - mu) / sigma for s in supports]

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great‑circle distance in kilometres."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def proximity_mask(motifs: List[HybridMotif],
                   delta_km: float,
                   signature: str) -> List[Tuple[int, int]]:
    """
    Build a list of edges (i,j) where motifs are within Δ km and share the same signature.
    The signature is a simple string derived from the pattern; here we use the first token.
    """
    edges = []
    for i, a in enumerate(motifs):
        sig_i = a.pattern[0] if a.pattern else ""
        for j in range(i + 1, len(motifs)):
            b = motifs[j]
            sig_j = b.pattern[0] if b.pattern else ""
            if sig_i != sig_j:
                continue
            if haversine(a.centroid_lat, a.centroid_lon,
                         b.centroid_lat, b.centroid_lon) <= delta_km:
                edges.append((i, j))
    return edges

def maximal_independent_set(num_nodes: int,
                            edges: List[Tuple[int, int]]) -> List[int]:
    """
    Greedy approximation of a maximal independent set.
    Nodes are selected in order of decreasing degree‑adjusted weight.
    """
    adjacency = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    # weight = 1 for all; could be extended to use scores.
    remaining = set(range(num_nodes))
    independent = []

    while remaining:
        # pick node with smallest degree (most likely to stay independent)
        node = min(remaining, key=lambda n: len(adjacency[n] & remaining))
        independent.append(node)
        # remove node and its neighbours
        to_remove = {node} | (adjacency[node] & remaining)
        remaining -= to_remove

    return independent

# ---------- Hybrid core ----------------------------------------------------

def compute_hybrid_scores(morphologies: List[Morphology],
                          motifs: List[TemporalMotif],
                          vectors: List[Tuple[str, List[float]]],
                          centroids: List[Tuple[float, float]],
                          delta_km: float = 10.0) -> List[HybridMotif]:
    """
    Build HybridMotif objects, compute the unified score
        S = support * (1 + z_s) * R(m) * σ
    where σ is the average cosine similarity to the k‑nearest semantic neighbours.
    Afterwards apply the possum‑style maximal independent set filter.
    """
    if not (len(morphologies) == len(motifs) == len(vectors) == len(centroids)):
        raise ValueError("All input lists must have equal length")

    # 1️⃣ Compute raw scores per candidate
    supports = [m.support for m in motifs]
    zlist = z_scores(supports)

    hybrid_candidates: List[HybridMotif] = []
    for idx, ((morph, motif, (doc_id, vec), (lat, lon), z)) in enumerate(
            zip(morphologies, motifs, vectors, centroids, zlist)):
        # recovery priority
        R = recovery_priority(morph)

        # semantic neighbour similarity (average of top‑k)
        neighbours = semantic_neighbors(doc_id, vec,
                                        pool_vectors=vectors,
                                        k=5)
        sigma = np.mean([sim for _, sim in neighbours]) if neighbours else 0.0

        # unified score
        score = motif.support * (1.0 + z) * R * sigma

        hybrid_candidates.append(
            HybridMotif(pattern=motif.pattern,
                        support=motif.support,
                        centroid_lat=lat,
                        centroid_lon=lon,
                        morphology=morph,
                        vector=tuple(vec),
                        score=score)
        )

    # 2️⃣ Build proximity mask (possum filter) using haversine & pattern signature
    edges = proximity_mask(hybrid_candidates, delta_km=delta_km, signature="unused")

    # 3️⃣ Select maximal independent set
    keep_idx = maximal_independent_set(len(hybrid_candidates), edges)

    return [hybrid_candidates[i] for i in keep_idx]

def top_hybrid_motifs(hybrid_motifs: List[HybridMotif], n: int = 10) -> List[HybridMotif]:
    """Return the n highest‑scoring hybrid motifs."""
    return sorted(hybrid_motifs, key=lambda x: -x.score)[:n]

def summarize_hybrid(hybrid_motifs: List[HybridMotif]) -> str:
    """Create a short textual summary of the hybrid set."""
    lines = [f"Hybrid set size: {len(hybrid_motifs)}"]
    for hm in hybrid_motifs[:5]:
        lines.append(
            f"Pattern={hm.pattern} Support={hm.support} Score={hm.score:.3f} "
            f"R={recovery_priority(hm.morphology):.2f}"
        )
    return "\n".join(lines)

# ---------- Smoke test -----------------------------------------------------

if __name__ == "__main__":
    # generate synthetic data
    random.seed(0)
    np.random.seed(0)

    num_entities = 20

    # Morphologies
    morphs = [
        Morphology(length=random.uniform(0.5, 2.0),
                   width=random.uniform(0.5, 2.0),
                   height=random.uniform(0.2, 1.0),
                   mass=random.uniform(1.0, 10.0))
        for _ in range(num_entities)
    ]

    # Temporal motifs (random patterns)
    motifs = [
        TemporalMotif(pattern=tuple(f"event{i%5}" for i in range(random.randint(2, 5))),
                      support=random.randint(5, 50))
        for _ in range(num_entities)
    ]

    # Semantic vectors (5‑dim)
    vectors = [(f"doc{i}", np.random.rand(5).tolist()) for i in range(num_entities)]

    # Random geographic centroids
    centroids = [(random.uniform(-90, 90), random.uniform(-180, 180))
                 for _ in range(num_entities)]

    hybrid = compute_hybrid_scores(morphologies=morphs,
                                   motifs=motifs,
                                   vectors=vectors,
                                   centroids=centroids,
                                   delta_km=500.0)

    top = top_hybrid_motifs(hybrid, n=5)
    print(summarize_hybrid(top))