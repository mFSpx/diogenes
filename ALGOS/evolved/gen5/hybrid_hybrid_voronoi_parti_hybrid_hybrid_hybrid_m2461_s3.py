# DARWIN HAMMER — match 2461, survivor 3
# gen: 5
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m765_s0.py (gen4)
# born: 2026-05-29T23:42:27Z

"""
Hybrid Voronoi‑Associative‑Motif System
======================================

Parent algorithms
-----------------
* **A – hybrid_voronoi_partition_hybrid_hybrid_m325_s2.py**  
  Provides a `Sheaf` abstraction, an energy function for a Dense Associative
  Memory (DAM) and a `hybrid_retrieve` routine that retrieves a stored pattern
  by minimizing the DAM energy.

* **B – hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m765_s0.py**  
  Supplies Bayesian updates for temporal motifs, stylometry‑based feature vectors,
  and a model‑selection scheme that uses an Ollivier‑Ricci curvature‑like
  weighting of model‑graph edges.

Mathematical bridge
-------------------
Both parents operate on **vectors** and **linear transformations**:

* In A the DAM energy `E(x) = -log Σ exp(β M x) + ½‖x‖²` is a scalar function of a
  vector `x` and a matrix `M`.
* In B the stylometry representation of a text is a vector `f ∈ ℝⁿ`; similarity
  between two feature vectors is measured with the Euclidean (or cosine)
  distance, which can be turned into a **graph edge weight**.  Ollivier‑Ricci
  curvature is a function of these edge weights.

The fusion therefore proceeds as follows:

1. **Voronoi partition** the set of stylometry vectors using a set of seed
   vectors.  Each Voronoi cell groups together “similar” inputs.
2. Inside each cell we treat the collection of vectors as a **sheaf section**.
   The DAM matrix `M` is built from the vectors of that cell (e.g. by stacking
   them) and a query vector is retrieved with `hybrid_retrieve`.
3. The **temporal‑motif** prior for each cell is updated in a Bayesian fashion
   after each retrieval, allowing the seeds to drift over time.
4. A **model‑selection graph** is built from the available models; edge
   curvature is approximated from the feature similarity of the models’
   representative vectors.  The curvature weights the RAM cost, yielding a
   curvature‑aware ranking of models.

The code below implements this pipeline with three public functions that
demonstrate the hybrid operation:
`voronoi_partition`, `dam_retrieve_in_cell`, and `curvature_weighted_model_choice`.  A
smoke test at the end exercises the whole system."""

import math
import random
import sys
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core structures from Parent A
# ----------------------------------------------------------------------
class Sheaf:
    """Container for node dimensions, edge restrictions and node sections."""
    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims: Dict[Any, int] = dict(node_dims)
        self.edges: List[Tuple[Any, Any]] = list(edges)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    def set_restriction(
        self,
        edge: Tuple[Any, Any],
        src_map: np.ndarray,
        dst_map: np.ndarray,
    ) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    def set_section(self, node: Any, value: np.ndarray) -> None:
        self._sections[node] = np.asarray(value, dtype=float)

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def _lse(z: np.ndarray) -> float:
    m = z.max()
    return float(m + np.log(np.exp(z - m).sum()))

def energy(xi: np.ndarray, M: np.ndarray, beta: float = 1.0) -> float:
    """Dense Associative Memory energy."""
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * np.sum(xi ** 2)
    return -lse_term + quadratic_term

def hybrid_retrieve(M: np.ndarray, query: np.ndarray, beta: float = 1.0,
                    steps: int = 50, lr: float = 0.1) -> np.ndarray:
    """
    Gradient descent on the DAM energy to retrieve a pattern close to ``query``.
    """
    x = np.asarray(query, dtype=float).copy()
    for _ in range(steps):
        grad = -beta * M.T @ _softmax(beta * (M @ x)) + x
        x -= lr * grad
    return x

# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def bayesian_update(prior: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Dirichlet‑conjugate Bayesian update for motif counts."""
    posterior = prior + observation
    return posterior / posterior.sum()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1]."""
    a, b = a.astype(float), b.astype(float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def approximate_ricci_curvature(v_i: np.ndarray, v_j: np.ndarray) -> float:
    """
    Very rough Ollivier‑Ricci curvature approximation based on feature
    similarity.  Returns a value in (0, 1] where higher means more “curved”.
    """
    dist = np.linalg.norm(v_i - v_j)
    norm_sum = np.linalg.norm(v_i) + np.linalg.norm(v_j) + 1e-12
    return max(0.0, 1.0 - dist / norm_sum)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def voronoi_partition(points: np.ndarray, seeds: np.ndarray) -> List[np.ndarray]:
    """
    Assign each point to the nearest seed (Euclidean distance) and return a
    list of arrays, one per seed, containing the points belonging to that cell.
    """
    points = np.asarray(points, dtype=float)
    seeds = np.asarray(seeds, dtype=float)
    assignments = np.argmin(
        np.linalg.norm(points[:, None, :] - seeds[None, :, :], axis=2),
        axis=1,
    )
    cells = [points[assignments == i] for i in range(seeds.shape[0])]
    return cells

def dam_retrieve_in_cell(cell_points: np.ndarray,
                         query: np.ndarray,
                         beta: float = 1.0) -> np.ndarray:
    """
    Build a DAM matrix from the vectors inside a Voronoi cell (stacked as rows)
    and retrieve a pattern for ``query`` using gradient descent.
    """
    if cell_points.size == 0:
        raise ValueError("Voronoi cell is empty – cannot build DAM matrix.")
    M = cell_points  # shape (k, d)
    retrieved = hybrid_retrieve(M, query, beta=beta)
    return retrieved

def curvature_weighted_model_choice(models: List[Tuple[str, int, np.ndarray]],
                                    feature_vec: np.ndarray) -> str:
    """
    Choose a model based on RAM cost weighted by Ollivier‑Ricci curvature
    between the model's representative feature vector and the input ``feature_vec``.
    ``models`` is a list of (name, ram_mb, rep_vector).
    Returns the name of the selected model.
    """
    scores = []
    for name, ram_mb, rep_vec in models:
        curv = approximate_ricci_curvature(feature_vec, rep_vec)
        # Higher curvature should reduce the effective RAM penalty.
        weighted_cost = ram_mb / (curv + 1e-6)
        scores.append((weighted_cost, name))
    scores.sort()
    return scores[0][1]

# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_process(
    data_points: np.ndarray,
    seed_vectors: np.ndarray,
    query_vector: np.ndarray,
    motif_prior: np.ndarray,
    motif_observation: np.ndarray,
    model_catalog: List[Tuple[str, int, np.ndarray]],
    beta: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    1. Partition ``data_points`` into Voronoi cells using ``seed_vectors``.
    2. Retrieve a pattern from the cell that contains ``query_vector``.
    3. Update the motif distribution with a Bayesian step.
    4. Select a model using curvature‑weighted RAM cost.

    Returns
    -------
    retrieved : np.ndarray
        Pattern retrieved by the DAM inside the appropriate cell.
    posterior : np.ndarray
        Updated motif distribution.
    chosen_model : str
        Name of the model selected by curvature‑aware ranking.
    """
    cells = voronoi_partition(data_points, seed_vectors)

    # Find the cell whose centroid is closest to the query (fallback if empty)
    centroids = np.array([c.mean(axis=0) if c.size else np.full(data_points.shape[1], np.inf)
                         for c in cells])
    nearest_cell_idx = int(np.argmin(np.linalg.norm(centroids - query_vector, axis=1)))
    target_cell = cells[nearest_cell_idx]

    retrieved = dam_retrieve_in_cell(target_cell, query_vector, beta=beta)

    posterior = bayesian_update(motif_prior, motif_observation)

    chosen_model = curvature_weighted_model_choice(model_catalog, query_vector)

    return retrieved, posterior, chosen_model

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic stylometry vectors (100 samples, 8‑dimensional)
    np.random.seed(42)
    data = np.random.randn(100, 8)

    # Random seed vectors for Voronoi partition (5 cells)
    seeds = np.random.randn(5, 8)

    # Random query vector
    query = np.random.randn(8)

    # Motif prior (Dirichlet with 4 categories)
    prior = np.array([0.25, 0.25, 0.25, 0.25])

    # Simulated observation (one‑hot like)
    observation = np.array([0, 1, 0, 0])

    # Model catalog: (name, ram_mb, representative feature vector)
    model_catalog = [
        ("tiny", 200, np.random.randn(8)),
        ("small", 500, np.random.randn(8)),
        ("medium", 1000, np.random.randn(8)),
        ("large", 2000, np.random.randn(8)),
    ]

    retrieved_pattern, posterior_motif, selected_model = hybrid_process(
        data_points=data,
        seed_vectors=seeds,
        query_vector=query,
        motif_prior=prior,
        motif_observation=observation,
        model_catalog=model_catalog,
        beta=1.2,
    )

    print("Retrieved pattern (first 5 entries):", retrieved_pattern[:5])
    print("Posterior motif distribution:", posterior_motif)
    print("Selected model:", selected_model)