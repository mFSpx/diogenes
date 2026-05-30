# DARWIN HAMMER — match 5200, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_hybrid_m1231_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_hybrid_m2156_s0.py (gen4)
# born: 2026-05-30T00:00:34Z

"""Hybrid Algorithm integrating:
- Parent A: Hybrid Ternary Router with Shapley Attribution and Ollivier‑Ricci curvature.
- Parent B: Sheaf Cohomology coboundary regularization with Radial‑Basis Function surrogate.

Mathematical bridge:
Both parents rely on Euclidean distances between points.
* In Parent A the distances feed the Ollivier‑Ricci curvature, which then
  supplies regret‑weighted routing scores.
* In Parent B the same distances weight the Gaussian kernel of the RBF
  surrogate and appear as coefficients of the coboundary operator that
  regularizes the surrogate loss.

The fusion therefore uses a single distance matrix to:
1. Compute curvature‑derived regret weights.
2. Build an RBF surrogate whose kernel is distance‑scaled.
3. Apply a coboundary regularizer whose strength is also distance‑scaled.
4. Distribute final routing scores with Shapley attribution over actions.

The resulting system yields a hybrid “route‑score” for each action that
encodes geometric curvature, surrogate prediction, cohomological
regularization and cooperative game‑theoretic attribution.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from itertools import combinations, chain
from typing import List, Dict, Tuple, Sequence, Callable

import numpy as np

# ----------------------------------------------------------------------
# Basic data structures
# ----------------------------------------------------------------------
Vector = Sequence[float]

@dataclass(frozen=True)
class Entity:
    """Spatial entity used by both parents."""
    id: str
    lat: float
    lon: float
    category: str = ""
    score: float = 0.0

@dataclass(frozen=True)
class MathAction:
    """Action on which routing/Shapley is performed."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

# ----------------------------------------------------------------------
# Core geometric utilities (shared by both parents)
# ----------------------------------------------------------------------
def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def distance_matrix(entities: List[Entity]) -> np.ndarray:
    """Return a symmetric matrix of Euclidean distances between entities."""
    n = len(entities)
    mat = np.zeros((n, n), dtype=float)
    coords = [(e.lat, e.lon) for e in entities]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(coords[i], coords[j])
            mat[i, j] = mat[j, i] = d
    return mat


# ----------------------------------------------------------------------
# Parent‑A components
# ----------------------------------------------------------------------
def ollivier_ricci_curvature(dist_mat: np.ndarray) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature κ_{ij} = 1 - (W_{ij} / d_{ij}),
    where W_{ij} is a simple transport cost approximated by the average
    distance from i's neighbours to j and vice‑versa.
    """
    n = dist_mat.shape[0]
    curvature = np.zeros_like(dist_mat)
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = dist_mat[i, j] + 1e-12  # avoid division by zero
            # neighbour‑average transport cost
            transport = (np.mean(dist_mat[i, :]) + np.mean(dist_mat[j, :])) / 2.0
            k = 1.0 - (transport / d_ij)
            curvature[i, j] = curvature[j, i] = k
    return curvature


def regret_weights(actions: List[MathAction],
                   curvature: np.ndarray,
                   entity_indices: Dict[str, int]) -> Dict[str, float]:
    """
    Map each action to a regret weight.
    The weight is the product of the action's expected value and the mean
    curvature of the entities it influences (here we mock a link via the
    action id containing an entity id).
    """
    weights: Dict[str, float] = {}
    for act in actions:
        # extract entity id from action id if present, else fallback to mean curvature
        ent_id = act.id.split("_")[-1]  # convention: actionX_entityY
        if ent_id in entity_indices:
            idx = entity_indices[ent_id]
            mean_curv = np.mean(curvature[idx, :])
        else:
            mean_curv = np.mean(curvature)
        w = act.expected_value * (1.0 + mean_curv)  # regret amplifies expected value
        weights[act.id] = w
    return weights


def shapley_attribution(actions: List[MathAction]) -> Dict[str, float]:
    """
    Compute exact Shapley values for a cooperative game where the value of a
    coalition S is the sum of expected values of its members.
    """
    n = len(actions)
    ids = [a.id for a in actions]
    values = {a.id: a.expected_value for a in actions}
    shapley: Dict[str, float] = {aid: 0.0 for aid in ids}
    factorial = math.factorial

    for aid in ids:
        marginal_sum = 0.0
        for r in range(n):
            coeff = factorial(r) * factorial(n - r - 1) / factorial(n)
            for coalition in combinations([x for x in ids if x != aid], r):
                coalition_value = sum(values[x] for x in coalition)
                marginal = values[aid] + coalition_value - coalition_value
                marginal_sum += coeff * marginal
        shapley[aid] = marginal_sum
    # Normalise to sum to 1
    total = sum(shapley.values()) + 1e-12
    for k in shapley:
        shapley[k] /= total
    return shapley


# ----------------------------------------------------------------------
# Parent‑B components
# ----------------------------------------------------------------------
def gaussian_kernel(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def rbf_surrogate_predict(centers: List[Vector],
                          coeffs: List[float],
                          point: Vector,
                          epsilon: float = 1.0) -> float:
    """Predict with a radial‑basis function surrogate."""
    if len(centers) != len(coeffs):
        raise ValueError("centers and coeffs must have same length")
    pred = 0.0
    for c, w in zip(centers, coeffs):
        r = euclidean(c, point)
        pred += w * gaussian_kernel(r, epsilon)
    return pred


def coboundary_regularizer(sheaf_matrix: np.ndarray,
                           distances: np.ndarray,
                           lam: float = 0.1) -> float:
    """
    Compute a regularization term λ * ||δᵗ D δ||₂,
    where δ is the coboundary matrix (sheaf_matrix) and D is a diagonal
    matrix of pairwise distances (here we use the average distance per node).
    """
    avg_dist = np.mean(distances, axis=1)
    D = np.diag(avg_dist)
    term = sheaf_matrix.T @ D @ sheaf_matrix
    reg = lam * np.linalg.norm(term, ord=2)
    return reg


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_route_score(actions: List[MathAction],
                       entities: List[Entity],
                       rbf_centers: List[Vector],
                       rbf_coeffs: List[float],
                       sheaf_matrix: np.ndarray) -> List[Tuple[str, float]]:
    """
    Produce a hybrid route score for each action by:
    1. Computing the distance matrix of entities.
    2. Deriving Ollivier‑Ricci curvature.
    3. Building regret weights from curvature.
    4. Obtaining Shapley attribution over actions.
    5. Predicting a surrogate value for each action's associated entity.
    6. Adding the coboundary regularization term.
    7. Combining the three contributions multiplicatively.
    """
    # 1‑2
    dist_mat = distance_matrix(entities)
    curvature = ollivier_ricci_curvature(dist_mat)

    # mapping entity id → index for quick lookup
    ent_index = {e.id: idx for idx, e in enumerate(entities)}

    # 3
    regret = regret_weights(actions, curvature, ent_index)

    # 4
    shapley = shapley_attribution(actions)

    # 5 – surrogate prediction per action
    # we assume each action references an entity; use that entity's coordinates
    surrogate_vals: Dict[str, float] = {}
    for act in actions:
        ent_id = act.id.split("_")[-1]
        if ent_id in ent_index:
            point = (entities[ent_index[ent_id]].lat,
                     entities[ent_index[ent_id]].lon)
        else:
            # fallback to centre of all entities
            point = (np.mean([e.lat for e in entities]),
                     np.mean([e.lon for e in entities]))
        surrogate_vals[act.id] = rbf_surrogate_predict(rbf_centers,
                                                      rbf_coeffs,
                                                      point,
                                                      epsilon=1.0)

    # 6 – single regularization scalar for the whole system
    reg_term = coboundary_regularizer(sheaf_matrix, dist_mat, lam=0.05)

    # 7 – combine
    scores: List[Tuple[str, float]] = []
    for act in actions:
        base = regret[act.id] * shapley[act.id] * surrogate_vals[act.id]
        final_score = base - reg_term  # regularization penalises
        scores.append((act.id, final_score))
    # sort descending by score
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def sample_rbf_model(num_centers: int = 5) -> Tuple[List[Vector], List[float]]:
    """Generate a toy RBF surrogate with random centers and coefficients."""
    centers = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(num_centers)]
    coeffs = [random.uniform(-1, 1) for _ in range(num_centers)]
    return centers, coeffs


def sample_sheaf_matrix(num_entities: int) -> np.ndarray:
    """
    Create a random coboundary matrix for a simplicial complex of dimension 1.
    For simplicity we generate a (num_entities‑1) × num_entities matrix with
    entries -1, 0, 1 mimicking an oriented edge incidence matrix.
    """
    rows = max(1, num_entities - 1)
    mat = np.zeros((rows, num_entities), dtype=float)
    for r in range(rows):
        mat[r, r] = -1
        mat[r, r + 1] = 1
    return mat


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small set of entities
    entities = [
        Entity(id="E1", lat=0.0, lon=0.0),
        Entity(id="E2", lat=1.0, lon=2.0),
        Entity(id="E3", lat=-1.5, lon=0.5),
        Entity(id="E4", lat=2.0, lon=-1.0),
    ]

    # Define actions that reference entities via naming convention
    actions = [
        MathAction(id="act_E1", expected_value=10.0, cost=2.0, risk=0.1),
        MathAction(id="act_E2", expected_value=8.0, cost=1.5, risk=0.2),
        MathAction(id="act_E3", expected_value=12.0, cost=3.0, risk=0.05),
        MathAction(id="act_E4", expected_value=7.0, cost=1.0, risk=0.15),
    ]

    # Sample RBF surrogate and sheaf matrix
    rbf_centers, rbf_coeffs = sample_rbf_model(num_centers=6)
    sheaf_mat = sample_sheaf_matrix(num_entities=len(entities))

    # Run the hybrid routing
    scores = hybrid_route_score(actions,
                                entities,
                                rbf_centers,
                                rbf_coeffs,
                                sheaf_mat)

    print("Hybrid route scores (sorted):")
    for aid, sc in scores:
        print(f"  {aid}: {sc:.4f}")