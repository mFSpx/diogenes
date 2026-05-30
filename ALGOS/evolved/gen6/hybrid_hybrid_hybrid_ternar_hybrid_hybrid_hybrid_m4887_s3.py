# DARWIN HAMMER — match 4887, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_hybrid_m1231_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_krampus_brain_m295_s0.py (gen3)
# born: 2026-05-29T23:58:32Z

"""Hybrid Algorithm: Fusion of
- hybrid_hybrid_ternary_route_hybrid_hybrid_hybrid_m1231_s4.py (ternary routing + regret‑weighted strategy + Ollivier‑Ricci curvature)
- hybrid_hybrid_hybrid_hard_t_hybrid_krampus_brain_m295_s0.py (stylometry feature vectors, model‑tier RAM optimisation, Ollivier‑Ricci curvature on brain‑map projections)

Mathematical Bridge
Both parents employ Ollivier‑Ricci curvature as a geometric filter on a weighted graph.
Parent A generates routing weights via a ternary router; Parent B builds high‑dimensional feature vectors from entity morphology and stylometry.
The hybrid therefore:
1. Constructs a probability‑weight matrix **W** from ternary routing.
2. Augments **W** with feature similarity derived from stylometry/morphology (dot‑product similarity).
3. Uses the combined matrix to define random‑walk measures µ_i on each node.
4. Computes Ollivier‑Ricci curvature κ_{ij}=1−W₁(µ_i,µ_j)/d_{ij}, where d_{ij} is Euclidean distance between nodes.
5. Filters entities and selects model tiers respecting a RAM ceiling, guided by curvature‑weighted scores.

The implementation below provides three core functions demonstrating this integration:
- `ternary_router`
- `feature_matrix`
- `ollivier_ricci_curvature`
and a higher‑level orchestrator `hybrid_select_entities`.

"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Data structures (shared from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""
    morphology: Morphology = None

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

class ModelPool:
    """Simple RAM‑constrained pool."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.models: List[ModelTier] = []

    def add_model(self, model: ModelTier) -> None:
        self.models.append(model)

    def optimal_subset(self) -> List[ModelTier]:
        """Greedy knapsack: maximise total RAM usage without exceeding ceiling."""
        sorted_models = sorted(self.models, key=lambda m: m.ram_mb, reverse=True)
        used, subset = 0, []
        for m in sorted_models:
            if used + m.ram_mb <= self.ram_ceiling_mb:
                subset.append(m)
                used += m.ram_mb
        return subset

# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def ternary_router(route_command: str, num_routes: int) -> np.ndarray:
    """
    Produce a 1‑D array of routing probabilities.
    - \"route_all\": uniform distribution.
    - \"route_one\": a single random route gets probability 1.
    - any other command: Dirichlet‑like random split.
    """
    if num_routes <= 0:
        raise ValueError("num_routes must be positive")
    weights = np.zeros(num_routes, dtype=float)

    if route_command == "route_all":
        weights[:] = 1.0 / num_routes
    elif route_command == "route_one":
        idx = random.randrange(num_routes)
        weights[idx] = 1.0
    else:
        # random ternary split using three exponential draws
        draws = np.random.exponential(scale=1.0, size=num_routes)
        weights = draws / draws.sum()
    return weights

def feature_matrix(entities: List[Entity]) -> np.ndarray:
    """
    Build a (N, D) feature matrix where D = 4 (morphology) + C (category one‑hot).
    Categories are collected from the input list.
    """
    categories = sorted({e.category for e in entities})
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    D = 4 + len(categories)  # morphology (4) + one‑hot categories
    N = len(entities)
    mat = np.zeros((N, D), dtype=float)

    for i, e in enumerate(entities):
        if e.morphology:
            mat[i, 0] = e.morphology.length
            mat[i, 1] = e.morphology.width
            mat[i, 2] = e.morphology.height
            mat[i, 3] = e.morphology.mass
        # one‑hot encode category
        cat_idx = cat_to_idx.get(e.category, None)
        if cat_idx is not None:
            mat[i, 4 + cat_idx] = 1.0
    # normalise each row to unit length (needed for similarity)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms
    return mat

def distance_matrix(entities: List[Entity]) -> np.ndarray:
    """Haversine‑style Euclidean distance on lat/lon (approx for small scales)."""
    N = len(entities)
    coords = np.array([[e.lat, e.lon] for e in entities], dtype=float)
    dmat = np.zeros((N, N), dtype=float)
    for i in range(N):
        diff = coords[i] - coords  # broadcast
        dmat[i] = np.sqrt(np.sum(diff ** 2, axis=1))
    return dmat

def ollivier_ricci_curvature(
    routing_weights: np.ndarray,
    feature_sim: np.ndarray,
    distances: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Compute a curvature matrix κ (N×N) using:
        κ_ij = 1 - W1(µ_i, µ_j) / d_ij
    where µ_i is a probability distribution over neighbours obtained by
    element‑wise product of routing_weights and feature similarity, then normalised.
    """
    N = routing_weights.shape[0]
    # Build transition matrix P where P_ij ∝ w_i * sim_ij
    P = routing_weights[:, None] * feature_sim  # shape (N,N)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums

    # Wasserstein‑1 distance between two discrete distributions on the same metric space:
    # For each pair (i,j) we solve a simple 1‑step transport: W1 = Σ_k |P_ik - P_jk| * d_ik
    # This is an approximation sufficient for the hybrid demo.
    W1 = np.zeros((N, N), dtype=float)
    for i in range(N):
        diff = np.abs(P[i][:, None] - P)  # (N,N)
        W1[i] = (diff * distances[i][:, None]).sum(axis=0)

    curv = 1.0 - W1 / (distances + epsilon)
    np.fill_diagonal(curv, 1.0)  # self‑curvature defined as 1
    return curv

def hybrid_select_entities(
    entities: List[Entity],
    model_pool: ModelPool,
    route_command: str = "route_all",
    curvature_threshold: float = 0.2
) -> Tuple[List[Entity], List[ModelTier], np.ndarray]:
    """
    End‑to‑end hybrid pipeline:
    1. Compute ternary routing weights.
    2. Build feature matrix and similarity matrix (dot product).
    3. Compute pairwise distances.
    4. Evaluate Ollivier‑Ricci curvature.
    5. Keep entities whose average curvature ≥ threshold.
    6. Choose a RAM‑feasible subset of models (greedy).
    Returns (selected_entities, selected_models, curvature_matrix).
    """
    if not entities:
        return [], [], np.array([])

    N = len(entities)

    # 1. routing
    w = ternary_router(route_command, N)  # shape (N,)

    # 2. feature similarity
    F = feature_matrix(entities)               # (N,D)
    sim = F @ F.T                               # cosine similarity because rows are unit‑normed
    sim = np.clip(sim, 0.0, 1.0)                # keep it in [0,1]

    # 3. distances
    dmat = distance_matrix(entities)

    # 4. curvature
    curv = ollivier_ricci_curvature(w, sim, dmat)

    # 5. filter by average curvature per node
    avg_curv = curv.mean(axis=1)
    keep_mask = avg_curv >= curvature_threshold
    selected_entities = [e for e, keep in zip(entities, keep_mask) if keep]

    # 6. model selection
    selected_models = model_pool.optimal_subset()

    return selected_entities, selected_models, curv

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # create dummy entities
    dummy_entities = []
    categories = ["alpha", "beta", "gamma"]
    for i in range(10):
        morph = Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.2, 1.0),
            height=random.uniform(0.1, 0.8),
            mass=random.uniform(10, 100)
        )
        ent = Entity(
            id=f"E{i}",
            lat=random.uniform(35.0, 36.0),
            lon=random.uniform(-120.0, -119.0),
            category=random.choice(categories),
            morphology=morph
        )
        dummy_entities.append(ent)

    # create a model pool with varied RAM footprints
    pool = ModelPool(ram_ceiling_mb=2500)
    for name, ram, tier in [
        ("tiny", 300, "low"),
        ("small", 800, "mid"),
        ("medium", 1200, "mid"),
        ("large", 1800, "high"),
        ("huge", 2500, "high")
    ]:
        pool.add_model(ModelTier(name=name, ram_mb=ram, tier=tier))

    # run hybrid selection
    sel_entities, sel_models, curvature = hybrid_select_entities(
        dummy_entities,
        pool,
        route_command="custom",   # triggers random Dirichlet split
        curvature_threshold=0.15
    )

    print("=== Selected Entities ===")
    for e in sel_entities:
        print(f"{e.id} ({e.category}) lat={e.lat:.3f} lon={e.lon:.3f}")

    print("\n=== Selected Models (RAM constrained) ===")
    for m in sel_models:
        print(f"{m.name} – {m.ram_mb} MB – tier {m.tier}")

    print("\nCurvature matrix shape:", curvature.shape)
    print("Average curvature of kept entities:",
          curvature[[i for i, e in enumerate(dummy_entities) if e in sel_entities]].mean())