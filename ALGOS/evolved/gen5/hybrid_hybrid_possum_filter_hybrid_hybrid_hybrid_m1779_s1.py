# DARWIN HAMMER — match 1779, survivor 1
# gen: 5
# parent_a: hybrid_possum_filter_hybrid_semantic_neig_m209_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_ternar_korpus_text_m1017_s1.py (gen4)
# born: 2026-05-29T23:38:47Z

"""hybrid_possum_textual_morphology.py
Hybrid algorithm merging:

- Parent A: possum_filter (spatial haversine distance + morphological recovery priority)
- Parent B: hybrid_hybrid_ternar_korpus_text (minhash‑based textual vectors, Euclidean cost matrix, ternary routing, Voronoi partition)

Mathematical bridge:
Each Entity is represented by a *joint feature vector* consisting of
1. Normalised spatial coordinates (lat, lon) → converted to 2‑D Euclidean points on a unit sphere.
2. Normalised textual signature (minhash + Shannon entropy) from Parent B.
3. Normalised recovery priority (morphology) from Parent A.

A composite distance matrix is built as a weighted sum of:
- Haversine distance `D_geo`
- Euclidean distance between textual vectors `D_txt`
- Absolute difference of recovery priorities `D_rec`

`C_ij = w_geo * D_geo_ij + w_txt * D_txt_ij + w_rec * D_rec_ij`

All downstream algorithms (ternary routing, Voronoi partition) operate on this hybrid cost matrix,
thereby fusing spatial diversity, textual similarity, and morphological robustness in a single
mathematical system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """Represents a spatial, textual and morphological object."""
    id: str
    lat: float               # degrees
    lon: float               # degrees
    category: str
    text: str                # raw textual description
    score: float = 0.0
    length: float = 1.0
    width: float = 1.0
    height: float = 1.0
    mass: float = 1.0


# ----------------------------------------------------------------------
# Parent‑A building blocks (spatial + morphology)
# ----------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great‑circle distance in metres between two lat/lon points."""
    R = 6371000.0  # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = φ2 - φ1
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Entity, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Entity, max_index: float = 10.0) -> float:
    """Normalised priority in [0,1] derived from morphology."""
    rti = righting_time_index(m)
    return min(rti / max_index, 1.0)


# ----------------------------------------------------------------------
# Parent‑B building blocks (textual minhash + entropy)
# ----------------------------------------------------------------------
def _shingles(text: str, width: int = 5) -> List[str]:
    return [text[i: i + width] for i in range(len(text) - width + 1)]


def minhash_signature(text: str, k: int = 64, width: int = 5, seed: int = 42) -> List[int]:
    if not text:
        return [0] * k
    sh = _shingles(text.lower(), width)
    hashes = [(hash(s) + seed) & 0xFFFFFFFFFFFFFFFF for s in sh]
    hashes.sort()
    return (hashes[:k] + [0] * k)[:k]


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    text = text[:10000]
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def text_to_vector(text: str, k: int = 64) -> np.ndarray:
    sig = minhash_signature(text, k=k)
    sig_arr = np.array(sig, dtype=np.float64) / float(0xFFFFFFFFFFFFFFFF)
    ent = shannon_entropy(text)
    ent_norm = ent / 8.0                      # entropy max for bytes ≈ 8 bits
    return np.concatenate([sig_arr, np.array([ent_norm], dtype=np.float64)])


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def entity_joint_vector(entity: Entity, txt_dim: int = 64) -> np.ndarray:
    """
    Concatenates normalised spatial coordinates, textual vector,
    and recovery priority into a single feature vector.
    """
    # Normalise lat/lon onto a unit sphere (x, y, z) and keep only x, y for simplicity
    φ = math.radians(entity.lat)
    λ = math.radians(entity.lon)
    x = math.cos(φ) * math.cos(λ)
    y = math.cos(φ) * math.sin(λ)
    spatial = np.array([x, y], dtype=np.float64)

    txt_vec = text_to_vector(entity.text, k=txt_dim)
    rec = np.array([recovery_priority(entity)], dtype=np.float64)

    return np.concatenate([spatial, txt_vec, rec])


def build_hybrid_cost_matrix(entities: List[Entity],
                             w_geo: float = 0.4,
                             w_txt: float = 0.5,
                             w_rec: float = 0.1) -> np.ndarray:
    """
    Returns a symmetric cost matrix C where
    C_ij = w_geo * haversine(i,j) +
           w_txt * Euclidean(txt_vector_i, txt_vector_j) +
           w_rec * |rec_i - rec_j|
    All components are normalised to comparable scales.
    """
    n = len(entities)
    if n == 0:
        raise ValueError("entity list must not be empty")

    # Pre‑compute components
    # 1. Geographic distance matrix (metres) → normalise by max distance
    geo_mat = np.zeros((n, n), dtype=np.float64)
    max_geo = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance(entities[i].lat, entities[i].lon,
                                   entities[j].lat, entities[j].lon)
            geo_mat[i, j] = geo_mat[j, i] = d
            if d > max_geo:
                max_geo = d
    if max_geo == 0:
        max_geo = 1.0
    geo_norm = geo_mat / max_geo

    # 2. Textual Euclidean distance matrix (already in [0, sqrt(k+1)])
    txt_vectors = [text_to_vector(e.text) for e in entities]
    stacked_txt = np.stack(txt_vectors)                     # shape (n, k+1)
    sq_norms = np.sum(stacked_txt ** 2, axis=1, keepdims=True)
    prod = stacked_txt @ stacked_txt.T
    txt_sq = sq_norms + sq_norms.T - 2 * prod
    np.maximum(txt_sq, 0.0, out=txt_sq)
    txt_dist = np.sqrt(txt_sq)                              # Euclidean
    max_txt = np.max(txt_dist) if np.max(txt_dist) > 0 else 1.0
    txt_norm = txt_dist / max_txt

    # 3. Recovery priority absolute difference matrix
    rec_vals = np.array([recovery_priority(e) for e in entities], dtype=np.float64)
    rec_mat = np.abs(rec_vals[:, None] - rec_vals[None, :])   # already in [0,1]

    # Weighted sum
    C = w_geo * geo_norm + w_txt * txt_norm + w_rec * rec_mat
    np.fill_diagonal(C, 0.0)
    return C


def hybrid_ternary_route(cost_matrix: np.ndarray,
                         source: int,
                         destination: int) -> Tuple[int, float]:
    """
    Finds an intermediate node k that minimises
    C[source, k] + C[k, destination].
    Returns the chosen k and the associated total cost.
    """
    if source == destination:
        return source, 0.0
    combined = cost_matrix[source, :] + cost_matrix[:, destination]
    # Exclude the destination itself from being selected
    mask = np.arange(combined.shape[0]) != destination
    if not np.any(mask):
        raise RuntimeError("No valid intermediate node found")
    k = int(np.argmin(combined[mask]))
    # map back to original index
    idxs = np.where(mask)[0]
    k = idxs[k]
    total = float(combined[k])
    return k, total


def hybrid_voronoi_partition(entities: List[Entity],
                             seed_ids: List[str]) -> Dict[str, List[str]]:
    """
    Assigns each entity to the nearest seed according to the hybrid cost matrix.
    Returns a mapping from seed id to list of assigned entity ids.
    """
    if not seed_ids:
        raise ValueError("seed_ids must contain at least one identifier")
    id_to_idx = {e.id: i for i, e in enumerate(entities)}
    seed_indices = [id_to_idx[sid] for sid in seed_ids if sid in id_to_idx]
    if len(seed_indices) != len(seed_ids):
        missing = set(seed_ids) - set(id_to_idx.keys())
        raise ValueError(f"Seed ids not found in entities: {missing}")

    C = build_hybrid_cost_matrix(entities)
    assignments: Dict[str, List[str]] = {sid: [] for sid in seed_ids}
    for e in entities:
        e_idx = id_to_idx[e.id]
        # compute distance to each seed
        dists = [C[e_idx, s_idx] for s_idx in seed_indices]
        nearest_seed_idx = int(np.argmin(dists))
        nearest_seed_id = seed_ids[nearest_seed_idx]
        assignments[nearest_seed_id].append(e.id)
    return assignments


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small synthetic population
    demo_entities = [
        Entity(id="A", lat=34.05, lon=-118.25, category="city",
               text="Los Angeles is a sprawling Southern California city.",
               length=3.5, width=2.0, height=1.5, mass=2.0),
        Entity(id="B", lat=40.71, lon=-74.00, category="city",
               text="New York City is the largest city in the United States.",
               length=3.0, width=2.5, height=1.8, mass=2.5),
        Entity(id="C", lat=51.51, lon=-0.13, category="city",
               text="London is the capital of England and the United Kingdom.",
               length=2.8, width=2.2, height=1.6, mass=2.2),
        Entity(id="D", lat=35.68, lon=139.69, category="city",
               text="Tokyo, Japan's busy capital, mixes the ultramodern and the traditional.",
               length=3.2, width=2.1, height=1.7, mass=2.3),
    ]

    # Build hybrid cost matrix
    C = build_hybrid_cost_matrix(demo_entities)
    print("Hybrid cost matrix:\n", C.round(3))

    # Perform ternary routing from A to C
    src = 0  # A
    dst = 2  # C
    k, total = hybrid_ternary_route(C, src, dst)
    print(f"Ternary route A->{demo_entities[k].id}->C total cost = {total:.3f}")

    # Voronoi partition using A and D as seeds
    partitions = hybrid_voronoi_partition(demo_entities, seed_ids=["A", "D"])
    print("Voronoi partitions:", partitions)