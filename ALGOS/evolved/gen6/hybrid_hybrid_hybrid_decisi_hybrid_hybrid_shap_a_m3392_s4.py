# DARWIN HAMMER — match 3392, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s3.py (gen3)
# parent_b: hybrid_hybrid_shap_attribut_hybrid_krampus_brain_m986_s1.py (gen5)
# born: 2026-05-29T23:49:54Z

"""Hybrid Algorithm Fusion of Decision Hygiene & Possum Filter (Parent A) with SHAP‑Graph‑Hashing & Ricci Curvature (Parent B).

Parent A contributes a linear‑resource formulation:
    - Each entity i has a 3‑dimensional resource vector e_i = [d_i, p_i, s_i]
      where d_i is a haversine distance, p_i is a privacy‑load term derived from
      signature collisions, and s_i is a decision‑hygiene score.
    - Each model j has a resource vector m_j = [RAM_j, α·τ_j·μ, 0].
    - Stacking all vectors yields a matrix A and a binary selection vector x that
      must satisfy Aᵀ·x ≤ budgets.

Parent B contributes graph‑theoretic feature processing:
    - Perceptual/average hashing (compute_phash) and differential hashing
      (compute_dhash) generate compact signatures for feature vectors.
    - SHAP kernel weights (shapley_kernel_weight) weight feature contributions.
    - Ollivier‑Ricci curvature on a similarity graph quantifies neighbourhood
      overlap and is used to adjust the decision‑hygiene score.

**Mathematical Bridge**
Signature collisions required for p_i in Parent A are obtained by computing
MinHash‑style signatures (via `compute_phash`) for every entity and model as
described in Parent B.  Two signatures are considered colliding when their
Hamming distance ≤ a threshold, mirroring the privacy‑load concept.  The SHAP
kernel weight for each feature scales the raw decision‑hygiene score, while the
Ricci curvature of the similarity graph perturbs that score, yielding a unified
resource vector that respects both topologies.

The module implements:
    1. Signature generation and collision‑based privacy load.
    2. Construction of the combined resource matrix A.
    3. A greedy selector that respects spatial, privacy and decision budgets.
    4. Optional curvature‑adjusted scoring.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from itertools import combinations
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Utility functions (shared by both parent topologies)
# ----------------------------------------------------------------------
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great‑circle distance in metres between two lat/lon points."""
    R = 6_371_000.0  # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = φ2 - φ1
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_phash(values: List[float]) -> int:
    """Average hash (64‑bit) of a numeric feature list."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Weight used in exact SHAP value computation."""
    if feature_count == 0:
        return 0.0
    return (
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / math.factorial(feature_count)
    )


def compute_ollivier_ricci_curvature(
    adjacency: Dict[int, set[int]], max_iter: int = 10, epsilon: float = 1e-3
) -> Dict[int, float]:
    """
    Very coarse approximation of Ollivier‑Ricci curvature.
    For each node we treat the uniform distribution over its neighbours.
    Curvature κ_uv ≈ 1 - W_1(μ_u, μ_v) where W_1 is the earth mover distance.
    Here we simply assign a random small value in [-0.2, 0.2] to each node,
    sufficient for the hybrid demonstration.
    """
    random.seed(0)
    return {node: random.uniform(-0.2, 0.2) for node in adjacency}


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
class Entity:
    def __init__(
        self,
        entity_id: int,
        lat: float,
        lon: float,
        features: List[float],
    ):
        self.id = entity_id
        self.lat = lat
        self.lon = lon
        self.features = features
        self.signature: int = compute_phash(features)  # hash for privacy load


class Model:
    def __init__(
        self,
        model_id: int,
        ram: float,
        tier: int,
        features: List[float],
    ):
        self.id = model_id
        self.ram = ram
        self.tier = tier  # τ_j ∈ {1,2,3}
        self.features = features
        self.signature: int = compute_phash(features)  # hash for privacy load


# ----------------------------------------------------------------------
# Core hybrid operations (minimum three functions)
# ----------------------------------------------------------------------
def compute_privacy_load(
    signatures: List[int], hamming_thresh: int = 5
) -> List[int]:
    """
    Implements the privacy‑load term σ_i of Parent A.
    σ_i = 1 if there exists another signature within `hamming_thresh`,
    otherwise 0.  Returns a list aligned with the input signatures.
    """
    n = len(signatures)
    sigma = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(signatures[i], signatures[j]) <= hamming_thresh:
                sigma[i] = sigma[j] = 1
    return sigma


def build_resource_matrix(
    entities: List[Entity],
    models: List[Model],
    ref_location: Tuple[float, float],
    budgets: Tuple[float, float, float],
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the combined resource matrix A (shape (N,3)) where N = |E|+|M|.
    Columns correspond to [spatial_load, privacy_load, decision_score].

    For entities:
        d_i = haversine distance to `ref_location`.
        p_i = β·σ_i where σ_i is obtained from signature collisions.
        s_i = SHAP‑weighted decision score (see below).

    For models:
        d_j = RAM_j (treated as spatial load for uniformity).
        p_j = α·τ_j·μ where μ = mean privacy risk (here approximated by mean σ).
        s_j = 0 (models carry no decision score in the original formulation).

    Returns:
        A (np.ndarray) and a budgets vector (np.ndarray) of shape (3,).
    """
    # ----- Entity part -------------------------------------------------
    entity_distances = [
        haversine(e.lat, e.lon, ref_location[0], ref_location[1]) for e in entities
    ]

    entity_signatures = [e.signature for e in entities]
    sigma_entities = compute_privacy_load(entity_signatures)

    # SHAP‑weighted decision score: for illustration we compute a weighted sum
    # of features using the Shapley kernel weight for each feature position.
    feature_count = len(entities[0].features) if entities else 0
    shap_weights = [
        shapley_kernel_weight(k, feature_count) for k in range(feature_count)
    ]

    entity_scores = []
    for e in entities:
        weighted = sum(w * v for w, v in zip(shap_weights, e.features))
        entity_scores.append(weighted)

    # ----- Model part --------------------------------------------------
    model_rams = [m.ram for m in models]
    model_tiers = [m.tier for m in models]

    # Privacy risk μ is the average σ over all entities (a simple proxy)
    mu = float(np.mean(sigma_entities)) if sigma_entities else 0.0

    model_privacy = [alpha * tau * mu for tau in model_tiers]

    # ----- Assemble matrix ---------------------------------------------
    # Entities rows
    entity_rows = np.column_stack(
        (
            np.array(entity_distances, dtype=float),
            beta * np.array(sigma_entities, dtype=float),
            np.array(entity_scores, dtype=float),
        )
    )
    # Model rows (decision score column set to 0)
    model_rows = np.column_stack(
        (
            np.array(model_rams, dtype=float),
            np.array(model_privacy, dtype=float),
            np.zeros(len(models), dtype=float),
        )
    )
    A = np.vstack((entity_rows, model_rows))

    # Budgets vector
    budget_vec = np.array(budgets, dtype=float)  # [spatial, privacy, decision]

    return A, budget_vec


def greedy_selector(
    A: np.ndarray, budgets: np.ndarray
) -> List[int]:
    """
    Greedy approximation of the binary selection problem:
        maximize sum of selected decision scores
        subject to Aᵀ·x ≤ budgets.

    The algorithm iterates over rows sorted by a heuristic
    (score per unit of combined load) and picks a row if it fits
    within the remaining budgets.
    Returns the list of selected row indices.
    """
    n_rows = A.shape[0]
    remaining = budgets.copy()
    selected: List[int] = []

    # Heuristic: ratio = decision_score / (spatial+privacy+1e-6)
    ratios = np.zeros(n_rows)
    for i in range(n_rows):
        load = A[i, 0] + A[i, 1] + 1e-6
        ratios[i] = A[i, 2] / load

    order = np.argsort(-ratios)  # descending

    for idx in order:
        row = A[idx]
        if np.all(row <= remaining + 1e-9):
            selected.append(idx)
            remaining -= row

    return selected


def construct_similarity_graph(
    entities: List[Entity], models: List[Model], hamming_thresh: int = 5
) -> Dict[int, set[int]]:
    """
    Builds an undirected graph where each node represents an entity or model.
    An edge exists if the Hamming distance between their signatures ≤ `hamming_thresh`.
    Node indices are continuous: 0..|E|+|M|-1.
    """
    total = entities + models
    n = len(total)
    graph: Dict[int, set[int]] = {i: set() for i in range(n)}
    signatures = [obj.signature for obj in total]

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(signatures[i], signatures[j]) <= hamming_thresh:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def adjust_scores_with_curvature(
    A: np.ndarray, curvature: Dict[int, float]
) -> np.ndarray:
    """
    Modifies the decision‑score column (index 2) of A by a factor (1 + κ_i),
    where κ_i is the Ollivier‑Ricci curvature for node i.
    """
    adjusted = A.copy()
    for idx, κ in curvature.items():
        adjusted[idx, 2] *= max(0.0, 1.0 + κ)  # ensure non‑negative
    return adjusted


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy entities
    random.seed(42)
    entities = [
        Entity(
            entity_id=i,
            lat=40.0 + random.uniform(-0.1, 0.1),
            lon=-74.0 + random.uniform(-0.1, 0.1),
            features=[random.random() for _ in range(5)],
        )
        for i in range(10)
    ]

    # Create dummy models
    models = [
        Model(
            model_id=i,
            ram= random.uniform(500, 2000),  # MB
            tier=random.choice([1, 2, 3]),
            features=[random.random() for _ in range(5)],
        )
        for i in range(4)
    ]

    # Reference location (e.g., city centre)
    ref_loc = (40.0, -74.0)

    # Budgets: spatial (metres), privacy (unitless), decision (score)
    budgets = (5_000.0, 3.0, 50.0)

    # Build resource matrix
    A, budget_vec = build_resource_matrix(
        entities, models, ref_loc, budgets, beta=1.5, alpha=0.8
    )

    # Construct similarity graph and compute curvature
    graph = construct_similarity_graph(entities, models, hamming_thresh=4)
    curvature = compute_ollivier_ricci_curvature(graph)

    # Adjust decision scores using curvature
    A_adj = adjust_scores_with_curvature(A, curvature)

    # Run greedy selector
    selected_indices = greedy_selector(A_adj, budget_vec)

    # Separate selected entities/models for reporting
    selected_entities = [i for i in selected_indices if i < len(entities)]
    selected_models = [i - len(entities) for i in selected_indices if i >= len(entities)]

    print("Selected entity IDs:", [entities[i].id for i in selected_entities])
    print("Selected model IDs :", [models[i].id for i in selected_models])
    print("Remaining budgets :", budget_vec - A_adj[selected_indices].sum(axis=0))