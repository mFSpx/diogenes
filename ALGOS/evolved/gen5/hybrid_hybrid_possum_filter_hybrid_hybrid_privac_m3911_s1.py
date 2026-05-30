# DARWIN HAMMER — match 3911, survivor 1
# gen: 5
# parent_a: hybrid_possum_filter_hybrid_semantic_neig_m209_s0.py (gen3)
# parent_b: hybrid_hybrid_privacy_model_hybrid_hybrid_semant_m1133_s0.py (gen4)
# born: 2026-05-29T23:52:26Z

"""Hybrid Possum‑Filtered Semantic‑Resource Allocation System

Parents:
- hybrid_possum_filter_hybrid_semantic_neig_m209_s0.py (Algorithm A)
- hybrid_hybrid_privacy_model_hybrid_hybrid_semant_m1133_s0.py (Algorithm B)

Mathematical Bridge:
Algorithm A supplies a spatial‑diversity filter (via `keep_candidate`) and a
convex‑combination hybrid score  

    h(i,j) = α·c(v_i,v_j) + (1‑α)·p(m_j)

where `c` is semantic similarity and `p` is a recovery‑priority term.
Algorithm B models each entity as a resource‑consumer with a three‑dimensional
load vector  

    ℓ(m) = [ram_m, privacy_m, s(m)]ᵀ, s(m)=β·max_k c(doc_m, seed_k)

and enforces a global budget  

    L = Aᵀ·x ≤ [ram_ceiling, privacy_budget, semantic_budget]ᵀ

The fusion treats every `Entity` from A as a “model” from B, augmenting it with
resource attributes and a document vector.  The unified system first filters
candidates spatially, then computes the hybrid score, and finally selects a
subset that respects the composite resource budget using a greedy knapsack‑like
procedure.

The code below implements this hybrid pipeline.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (extended from Algorithm A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class Entity:
    """A spatial‑semantic object that also carries resource usage."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""
    morphology: Morphology | None = None
    # Extensions for the hybrid model (Algorithm B)
    ram: float = 0.0                # megabytes required
    privacy_load: float = 0.0       # abstract privacy cost
    doc_vector: np.ndarray | None = None  # semantic embedding
    recovery_priority: float = 0.0  # p(m) term used in hybrid score


# ----------------------------------------------------------------------
# Utility functions (Algorithm A)
# ----------------------------------------------------------------------


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def signature(e: Entity) -> str:
    """Canonical textual signature used for the Possum filter."""
    return (e.address_signature or e.category).strip().lower()


def keep_candidate(candidate: Entity, selected: List[Entity], delta_m: float) -> bool:
    """
    Possum‑style spatial diversity test.
    Reject `candidate` if it is too close (≤ delta_m) to an already selected
    entity of the same signature.
    """
    for existing in selected:
        same_kind = (
            signature(candidate) == signature(existing)
            or candidate.category.strip().lower() == existing.category.strip().lower()
        )
        if same_kind and haversine_m((candidate.lat, candidate.lon),
                                    (existing.lat, existing.lon)) <= delta_m:
            return False
    return True


def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimension‑based shape metric (Algorithm A)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Dimension‑based flatness metric (Algorithm A)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return min(width, height) / length


# ----------------------------------------------------------------------
# Semantic similarity utilities (shared by both parents)
# ----------------------------------------------------------------------


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity in [-1,1]; safe for zero vectors."""
    if v1 is None or v2 is None:
        return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def semantic_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Alias that rescales cosine similarity to [0,1]."""
    return (cosine_similarity(v1, v2) + 1.0) / 2.0


# ----------------------------------------------------------------------
# Hybrid score (Algorithm A) ------------------------------------------------
# ----------------------------------------------------------------------


def hybrid_score(entity_i: Entity, entity_j: Entity, alpha: float = 0.5) -> float:
    """
    Convex combination of semantic similarity and recovery priority.

        h(i,j) = α·c(v_i, v_j) + (1‑α)·p(m_j)

    `c` is the [0,1] semantic similarity, `p` is the recovery priority (already
    normalised to [0,1] by the caller).  The function is symmetric in the sense
    that the priority term always refers to the *target* entity `j`.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0,1]")
    sim = semantic_similarity(
        entity_i.doc_vector if entity_i.doc_vector is not None else np.zeros(1),
        entity_j.doc_vector if entity_j.doc_vector is not None else np.zeros(1),
    )
    # Normalise recovery_priority to [0,1] (assume caller respects this)
    p = max(0.0, min(1.0, entity_j.recovery_priority))
    return alpha * sim + (1.0 - alpha) * p


# ----------------------------------------------------------------------
# Resource matrix construction (Algorithm B)
# ----------------------------------------------------------------------


def build_resource_matrix(
    entities: List[Entity],
    seed_vectors: List[np.ndarray],
    beta: float = 1.0,
) -> Tuple[np.ndarray, List[float]]:
    """
    Construct the matrix A ∈ ℝ^{3×n} where each column corresponds to a model
    (entity) and rows are [ram, privacy_load, semantic_load].

    The semantic load for a model m is

        s(m) = β * max_k  c(doc_m, seed_k)

    Returns:
        A   – 3×n numpy array
        semantic_loads – list of the computed s(m) values (for debugging)
    """
    n = len(entities)
    A = np.zeros((3, n), dtype=float)
    semantic_loads = []

    for idx, ent in enumerate(entities):
        A[0, idx] = ent.ram
        A[1, idx] = ent.privacy_load

        # compute semantic load
        if ent.doc_vector is None or not seed_vectors:
            s_load = 0.0
        else:
            sims = [semantic_similarity(ent.doc_vector, seed) for seed in seed_vectors]
            s_load = beta * max(sims)
        A[2, idx] = s_load
        semantic_loads.append(s_load)

    return A, semantic_loads


# ----------------------------------------------------------------------
# Greedy allocation respecting composite budget (Algorithm B)
# ----------------------------------------------------------------------


def allocate_models_greedy(
    entities: List[Entity],
    ram_ceiling: float,
    privacy_budget: float,
    semantic_budget: float,
    seed_vectors: List[np.ndarray],
    beta: float = 1.0,
    alpha: float = 0.5,
) -> List[Entity]:
    """
    Greedy selection of a subset of `entities` that maximises the summed hybrid
    score while respecting the three‑dimensional resource budget.

    The algorithm:
        1. Build matrix A and compute per‑entity semantic loads.
        2. Sort entities by descending max hybrid score against any other entity.
        3. Iterate, adding an entity if the cumulative load stays within budget.
    """
    if not entities:
        return []

    # Pre‑compute resource matrix
    A, _ = build_resource_matrix(entities, seed_vectors, beta)

    # Compute a scalar utility for each entity: the best hybrid score it can
    # achieve as a *target* of any other entity.
    utilities = []
    for i, target in enumerate(entities):
        best = 0.0
        for j, source in enumerate(entities):
            if i == j:
                continue
            score = hybrid_score(source, target, alpha)
            if score > best:
                best = score
        utilities.append(best)

    # Order indices by decreasing utility
    ordered_indices = sorted(range(len(entities)), key=lambda i: utilities[i], reverse=True)

    selected: List[Entity] = []
    cumulative_load = np.zeros(3, dtype=float)

    for idx in ordered_indices:
        prospective_load = cumulative_load + A[:, idx]
        if (
            prospective_load[0] <= ram_ceiling
            and prospective_load[1] <= privacy_budget
            and prospective_load[2] <= semantic_budget
        ):
            selected.append(entities[idx])
            cumulative_load = prospective_load

    return selected


# ----------------------------------------------------------------------
# Voronoi‑style partitioning of points to semantic seeds (Algorithm B)
# ----------------------------------------------------------------------


def voronoi_partition(
    points: List[Tuple[float, float]],
    seed_points: List[Tuple[float, float]],
) -> List[int]:
    """
    Assign each point to the index of the nearest seed (Euclidean distance on
    the plane).  Returns a list `assignments` where `assignments[i]` is the seed
    index for `points[i]`.
    """
    if not seed_points:
        raise ValueError("seed_points must contain at least one element")
    assignments = []
    for px, py in points:
        dists = [(px - sx) ** 2 + (py - sy) ** 2 for sx, sy in seed_points]
        assignments.append(int(np.argmin(dists)))
    return assignments


# ----------------------------------------------------------------------
# High‑level hybrid pipeline (demonstrates the fusion)
# ----------------------------------------------------------------------


def hybrid_pipeline(
    entities: List[Entity],
    delta_m: float,
    ram_ceiling: float,
    privacy_budget: float,
    semantic_budget: float,
    seed_vectors: List[np.ndarray],
    seed_geo: List[Tuple[float, float]],
    alpha: float = 0.5,
    beta: float = 1.0,
) -> Tuple[List[Entity], List[int]]:
    """
    1. Apply the Possum spatial filter (`keep_candidate`).
    2. Partition the filtered entities into Voronoi cells using `seed_geo`.
    3. Within each cell, run the greedy resource‑aware allocation.

    Returns:
        selected_entities – list of entities finally loaded
        geo_assignments   – Voronoi cell index for each original entity
    """
    # Step 1: spatial diversity filtering
    filtered: List[Entity] = []
    for cand in entities:
        if keep_candidate(cand, filtered, delta_m):
            filtered.append(cand)

    # Step 2: geometric partitioning
    geo_points = [(e.lat, e.lon) for e in filtered]
    assignments = voronoi_partition(geo_points, seed_geo)

    # Step 3: resource‑constrained selection per cell
    selected: List[Entity] = []
    for cell_idx in range(len(seed_geo)):
        cell_entities = [e for e, a in zip(filtered, assignments) if a == cell_idx]
        cell_selected = allocate_models_greedy(
            cell_entities,
            ram_ceiling,
            privacy_budget,
            semantic_budget,
            seed_vectors,
            beta,
            alpha,
        )
        selected.extend(cell_selected)

    return selected, assignments


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a small synthetic dataset
    random.seed(42)
    np.random.seed(42)

    def rand_vec(dim: int = 5) -> np.ndarray:
        return np.random.rand(dim)

    # Generate 12 entities with random geography, resources and semantics
    entities: List[Entity] = []
    categories = ["alpha", "beta", "gamma"]
    for i in range(12):
        ent = Entity(
            id=f"E{i}",
            lat=random.uniform(-90, 90),
            lon=random.uniform(-180, 180),
            category=random.choice(categories),
            address_signature=f"addr{i % 3}",
            morphology=Morphology(
                length=random.uniform(0.5, 5.0),
                width=random.uniform(0.5, 5.0),
                height=random.uniform(0.5, 5.0),
                mass=random.uniform(10, 100),
            ),
            ram=random.uniform(10, 200),               # MB
            privacy_load=random.uniform(0.1, 5.0),      # abstract units
            doc_vector=rand_vec(),
            recovery_priority=random.uniform(0, 1),
        )
        entities.append(ent)

    # Semantic seeds (for semantic load) – three random vectors
    seed_vectors = [rand_vec() for _ in range(3)]

    # Geographic seeds for Voronoi partitioning – three arbitrary lat/lon points
    seed_geo = [(-30.0, -30.0), (0.0, 0.0), (45.0, 90.0)]

    # Run the hybrid pipeline
    selected, assignments = hybrid_pipeline(
        entities=entities,
        delta_m=1_000_000.0,          # 1000 km spatial diversity radius
        ram_ceiling=500.0,           # MB
        privacy_budget=10.0,         # abstract units
        semantic_budget=2.0,         # abstract units
        seed_vectors=seed_vectors,
        seed_geo=seed_geo,
        alpha=0.6,
        beta=1.2,
    )

    print("=== Voronoi assignments (entity index → cell) ===")
    for idx, cell in enumerate(assignments):
        print(f"{entities[idx].id:>4} → cell {cell}")

    print("\n=== Selected entities after resource‑aware allocation ===")
    for e in selected:
        print(
            f"{e.id:>4} | RAM={e.ram:6.1f}MB | Privacy={e.privacy_load:5.2f} | "
            f"Priority={e.recovery_priority:.2f} | Category={e.category}"
        )

    print(f"\nTotal selected: {len(selected)} out of {len(entities)} original entities.")
    sys.exit(0)