# DARWIN HAMMER — match 1617, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_krampus_brain_ttt_linear_m4_s0.py (gen2)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_hybrid_m515_s0.py (gen5)
# born: 2026-05-29T23:37:55Z

import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, Tuple


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _ensure_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric version of *matrix* by averaging with its transpose."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    sym = (matrix + matrix.T) / 2.0
    np.fill_diagonal(sym, 0.0)  # no self‑loops for curvature calculations
    return sym


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row‑normalize *matrix* to obtain a stochastic matrix."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    # avoid division by zero – rows with sum 0 become uniform (or stay zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(row_sums == 0, 0, matrix / row_sums)
    return norm


def _shortest_path_distances(cost: np.ndarray) -> np.ndarray:
    """Floyd‑Warshall algorithm for all‑pairs shortest path on a dense cost matrix."""
    n = cost.shape[0]
    dist = cost.copy()
    np.fill_diagonal(dist, 0.0)
    for k in range(n):
        # broadcasting version of dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
        dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])
    return dist


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Hybrid algorithms
# ----------------------------------------------------------------------
def hybrid_ollivier_ricci_curvature(
    adjacency: np.ndarray,
    edge_costs: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> float:
    """
    Compute the average Ollivier‑Ricci curvature of an undirected weighted graph.

    The graph is described by ``adjacency`` (non‑negative weights) and a
    ground‑metric ``edge_costs`` that quantifies the transport cost between
    directly connected vertices.  The curvature between two adjacent vertices
    *i* and *j* is

        κ(i, j) = 1 - W₁(μ_i, μ_j) / d(i, j)

    where μ_i is the probability distribution of a random walk starting at *i*
    (row‑normalized adjacency) and d(i, j) is the shortest‑path distance
    induced by ``edge_costs``.  The function returns the mean curvature over all
    existing edges.

    Parameters
    ----------
    adjacency:
        Square matrix of edge weights (non‑negative).  Must be symmetric.
    edge_costs:
        Square matrix of transport costs.  Must be symmetric and have zeros on
        the diagonal.
    epsilon:
        Small constant to avoid division by zero.

    Returns
    -------
    float
        The average curvature (value in [-∞, 1]).
    """
    # ------------------------------------------------------------------
    # Validation & preprocessing
    # ------------------------------------------------------------------
    if adjacency.shape != edge_costs.shape:
        raise ValueError("Adjacency and edge_costs must have the same shape.")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    # Ensure symmetry and non‑negativity
    adjacency = _ensure_symmetric(adjacency)
    edge_costs = _ensure_symmetric(edge_costs)
    if np.any(adjacency < 0) or np.any(edge_costs < 0):
        raise ValueError("Matrices must contain only non‑negative entries.")

    # Row‑normalize adjacency to obtain the random‑walk measures μ_i
    mu = _row_normalize(adjacency)

    # Compute all‑pairs shortest‑path distances using edge_costs as ground metric
    distances = _shortest_path_distances(edge_costs)

    n = adjacency.shape[0]
    curvature_sum = 0.0
    edge_count = 0

    # ------------------------------------------------------------------
    # Curvature evaluation
    # ------------------------------------------------------------------
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] <= 0:
                continue  # not an edge, skip

            d_ij = distances[i, j]
            if d_ij < epsilon:
                # Edge exists but distance is numerically zero – treat curvature as 0
                curvature = 0.0
            else:
                # Compute the 1‑Wasserstein distance between μ_i and μ_j
                # For dense graphs the exact linear program is expensive;
                # we use the Kantorovich–Rubinstein dual formulation with the
                # ground metric distances.
                diff = mu[i] - mu[j]  # vector of length n
                # The optimal transport cost reduces to sum_k |diff_k| * d(i,k) / 2
                # when the ground metric is a tree metric; we approximate with
                # the shortest‑path distances.
                transport = np.sum(np.abs(diff) * distances[i]) / 2.0
                curvature = 1.0 - transport / d_ij

            curvature_sum += curvature
            edge_count += 1

    if edge_count == 0:
        raise ValueError("The adjacency matrix contains no edges.")
    return curvature_sum / edge_count


def hybrid_lsm_score(
    morphology: Morphology,
    adjacency: np.ndarray,
    edge_contributions: np.ndarray,
    *,
    weight_factor: float = 1.0,
) -> float:
    """
    Compute a morphology‑aware score that blends the expected edge contribution
    with the physical volume of the object.

    The expected contribution of an edge (i, j) is taken as the product of the
    adjacency weight and the supplied ``edge_contributions`` matrix.
    The total is then scaled by the object's volume (length·width·height) and
    optionally by its mass.

    Parameters
    ----------
    morphology:
        Physical parameters of the object.
    adjacency:
        Symmetric adjacency matrix (weights ≥ 0).
    edge_contributions:
        Same shape as ``adjacency``; values represent the belief that an edge
        contributes positively to the score.
    weight_factor:
        Global multiplier to tune the influence of the edge term.

    Returns
    -------
    float
        The hybrid LSM score.
    """
    adjacency = _ensure_symmetric(adjacency)
    edge_contributions = _ensure_symmetric(edge_contributions)

    # Expected edge weight = adjacency * contribution
    expected = adjacency * edge_contributions
    total_expected = np.triu(expected, k=1).sum()  # count each undirected edge once

    volume = morphology.length * morphology.width * morphology.height
    mass_term = morphology.mass if morphology.mass > 0 else 1.0

    return weight_factor * total_expected * volume * mass_term


def hybrid_decision(
    curvature: float,
    lsm_score: float,
    *,
    curvature_weight: float = 0.6,
    lsm_weight: float = 0.4,
    threshold: float = 0.0,
) -> bool:
    """
    Make a binary decision by linearly combining curvature and LSM score.

    The combined metric is

        Z = w_c * curvature + w_s * log(1 + lsm_score)

    The decision is ``True`` iff Z exceeds ``threshold``.

    Parameters
    ----------
    curvature:
        Average Ollivier‑Ricci curvature (typically in [-1, 1]).
    lsm_score:
        Hybrid LSM score (non‑negative).
    curvature_weight, lsm_weight:
        Non‑negative weights that sum to 1 (the function normalises them).
    threshold:
        Decision boundary.

    Returns
    -------
    bool
        ``True`` if the combined metric is above the threshold.
    """
    if curvature_weight < 0 or lsm_weight < 0:
        raise ValueError("Weights must be non‑negative.")
    total = curvature_weight + lsm_weight
    if total == 0:
        raise ValueError("At least one weight must be positive.")
    w_c = curvature_weight / total
    w_s = lsm_weight / total

    combined = w_c * curvature + w_s * np.log1p(lsm_score)
    return combined > threshold


# ----------------------------------------------------------------------
# Example usage (kept lightweight for import safety)
# ----------------------------------------------------------------------
def _example() -> Tuple[bool, float, float]:
    """Run a tiny reproducible example and return (decision, curvature, lsm_score)."""
    rng = np.random.default_rng(seed=42)

    # Small synthetic graph
    size = 6
    raw_adj = rng.random((size, size))
    adjacency = _ensure_symmetric(raw_adj) * (raw_adj > 0.7)  # sparsify
    edge_costs = _ensure_symmetric(rng.random((size, size))) + 0.1  # avoid zeros

    # Edge contributions (beliefs)
    edge_contributions = _ensure_symmetric(rng.random((size, size)))

    # Morphology
    morph = Morphology(length=12.0, width=7.5, height=3.0, mass=2.0)

    curvature = hybrid_ollivier_ricci_curvature(adjacency, edge_costs)
    lsm = hybrid_lsm_score(morph, adjacency, edge_contributions, weight_factor=0.8)
    decision = hybrid_decision(curvature, lsm, curvature_weight=0.7, lsm_weight=0.3)

    return decision, curvature, lsm


if __name__ == "__main__":
    dec, curv, lsm = _example()
    print(f"Decision: {dec}")
    print(f"Avg. Ollivier‑Ricci curvature: {curv:.4f}")
    print(f"Hybrid LSM score: {lsm:.4f}")