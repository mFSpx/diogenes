# DARWIN HAMMER — match 1327, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m650_s0.py (gen4)
# parent_b: hybrid_voronoi_partition_percyphon_m779_s1.py (gen1)
# born: 2026-05-29T23:35:24Z

"""Hybrid Voronoi‑Entropy‑Signature Algorithm
================================================

This module fuses two previously independent algorithms:

* **Parent A** – information‑theoretic utilities (Shannon entropy,
  min‑hash signatures, similarity and a Gaussian kernel).
* **Parent B** – spatial partitioning via a Voronoi diagram and procedural
  entity generation.

**Mathematical bridge**

The Voronoi diagram partitions a point cloud into regions *Rₖ*.  
Each region supplies a discrete count vector *cₖ* (e.g. number of points,
or any categorical feature extracted from the points).  
Parent A’s Shannon entropy *H(cₖ)* quantifies the uncertainty of that
count distribution.  By hashing a textual representation of the region we
obtain a min‑hash signature *σₖ*; pairwise Jaccard‑like similarity
*Sim(σᵢ,σⱼ)* measures overlap of region “identities”.  

Spatial proximity between region centroids *μᵢ, μⱼ* is turned into a
Gaussian weight *G(dᵢⱼ)=exp(−(ε·dᵢⱼ)²)* (Parent A).  The final hybrid
metric aggregates entropy, similarity and spatial weighting:


Score = Σₖ H(cₖ)·G(‖μₖ−μ̄‖)            # intra‑region term
      + Σ_{i<j} Sim(σᵢ,σⱼ)·G(‖μᵢ−μⱼ‖)  # inter‑region term


where *μ̄* is the global centroid of all seeds.  This unifies the
information‑theoretic core of Parent A with the geometric core of
Parent B.

The implementation below provides three public functions that embody
this hybrid operation and a small smoke test.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence, Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – information‑theoretic utilities
# ----------------------------------------------------------------------
def shannon_entropy(counts: List[int]) -> float:
    """Compute Shannon entropy from a list of non‑negative integer counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for cnt in counts:
        if cnt > 0:
            p = cnt / total
            entropy -= p * math.log2(p)
    return entropy


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used for min‑hash signatures."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """
    Produce a min‑hash signature of length *k* from an iterable of tokens.
    Empty token sets yield a signature of maximal values.
    """
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two min‑hash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Parent B – Voronoi partitioning and procedural utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """
    Voronoi assignment: map each point to the index of its nearest seed.
    Returns a dictionary ``region_index -> list_of_points``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


def centroid(points: List[Point]) -> Point:
    """Geometric centroid of a non‑empty point list."""
    if not points:
        raise ValueError("cannot compute centroid of empty set")
    xs, ys = zip(*points)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_region_features(
    regions: Dict[int, List[Point]],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    For each Voronoi region produce:

    * ``counts`` – a simple histogram of point‑density bins (here we use
      a 4‑bin radial histogram around the region centroid).
    * ``signature`` – a min‑hash signature derived from stringified point
      coordinates (rounded to 3 decimals) to act as a region identifier.

    Returns two dictionaries indexed by region id:
    ``counts_map`` and ``signature_map``.
    """
    counts_map: Dict[int, List[int]] = {}
    signature_map: Dict[int, List[int]] = {}

    for idx, pts in regions.items():
        if not pts:
            # Empty region: zero counts and a deterministic empty signature
            counts_map[idx] = [0, 0, 0, 0]
            signature_map[idx] = signature([], k=64)
            continue

        # 1) Radial histogram (4 bins) around the region centroid
        cen = centroid(pts)
        dists = [distance(p, cen) for p in pts]
        max_dist = max(dists) if dists else 1.0
        bin_edges = np.linspace(0, max_dist, 5)  # 4 bins
        hist, _ = np.histogram(dists, bins=bin_edges)
        counts_map[idx] = hist.tolist()

        # 2) Tokenise rounded coordinates for hashing
        tokens = [f"{round(p[0],3)}:{round(p[1],3)}" for p in pts]
        signature_map[idx] = signature(tokens, k=64)

    return counts_map, signature_map


def hybrid_region_score(
    seeds: List[Point],
    regions: Dict[int, List[Point]],
    epsilon: float = 1.0,
) -> float:
    """
    Compute the unified hybrid metric described in the module docstring.

    Parameters
    ----------
    seeds
        The Voronoi seed points (also the region centroids for weighting).
    regions
        Mapping ``region_index -> list_of_points`` as produced by ``assign``.
    epsilon
        Scale parameter for the Gaussian kernel.

    Returns
    -------
    float
        The aggregated hybrid score.
    """
    # Global centroid of seeds (used for intra‑region weighting)
    global_cen = centroid(seeds)

    # Feature extraction per region
    counts_map, signature_map = compute_region_features(regions)

    # Pre‑compute centroids of each region for inter‑region distances
    region_cents: Dict[int, Point] = {}
    for idx, pts in regions.items():
        region_cents[idx] = centroid(pts) if pts else seeds[idx]

    # ----- Intra‑region term ------------------------------------------------
    intra = 0.0
    for idx, counts in counts_map.items():
        ent = shannon_entropy(counts)
        dist_to_global = distance(region_cents[idx], global_cen)
        intra += ent * gaussian(dist_to_global, epsilon)

    # ----- Inter‑region term ------------------------------------------------
    inter = 0.0
    indices = list(regions.keys())
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_i, idx_j = indices[i], indices[j]
            sim = similarity(signature_map[idx_i], signature_map[idx_j])
            d = distance(region_cents[idx_i], region_cents[idx_j])
            inter += sim * gaussian(d, epsilon)

    return intra + inter


def generate_procedural_entities(
    seeds: List[Point],
    psyche_wrath_velocity: float = 0.0,
    psyche_forensic_shield_ratio: float = 0.0,
    fluid_slots: int = 88,
) -> List[dict]:
    """
    Produce a list of procedural entities – one per Voronoi seed – using
    the naming scheme from Parent B.  The function demonstrates that the
    hybrid algorithm can be coupled with the original procedural generator.
    """
    entities = []
    for idx, seed_pt in enumerate(seeds):
        seed_str = f"{seed_pt[0]:.3f},{seed_pt[1]:.3f}"
        name, alias, persona = _slot_name(seed_str, idx)
        uuid = _uuid_from_sha256(seed_str + f":{idx}")
        ternary_offset = 1 if psyche_wrath_velocity > psyche_forensic_shield_ratio else 0
        entity = {
            "slot_index": idx,
            "name": name,
            "alias": alias,
            "persona": persona,
            "uuid": uuid,
            "ternary_offset": ternary_offset,
            "position": {"x": seed_pt[0], "y": seed_pt[1]},
        }
        entities.append(entity)
    return entities


# ----------------------------------------------------------------------
# Helper functions (originally from Parent B)
# ----------------------------------------------------------------------
def _uuid_from_sha256(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _slot_name(seed: str, idx: int) -> Tuple[str, str, str]:
    h = hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest()
    name = f"Villager-{int(h[:6], 16) % 5000:04d}"
    alias = f"Alias-{h[6:10]}"
    persona = ["ledger", "runner", "witness", "archivist", "carrier", "scribe"][
        int(h[10:12], 16) % 6
    ]
    return name, alias, persona


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Generate a random set of points and seeds
    random.seed(42)
    num_points = 500
    num_seeds = 12
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_points)]
    seeds = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_seeds)]

    # 2) Voronoi assignment
    regions = assign(points, seeds)

    # 3) Compute the hybrid metric
    score = hybrid_region_score(seeds, regions, epsilon=0.05)
    print(f"Hybrid metric score: {score:.6f}")

    # 4) Generate procedural entities (demonstration of interoperability)
    entities = generate_procedural_entities(
        seeds,
        psyche_wrath_velocity=0.7,
        psyche_forensic_shield_ratio=0.3,
        fluid_slots=88,
    )
    print(f"Generated {len(entities)} procedural entities. Sample:")
    print(entities[0] if entities else "none")