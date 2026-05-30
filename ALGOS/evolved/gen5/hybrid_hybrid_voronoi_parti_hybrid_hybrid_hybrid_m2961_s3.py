# DARWIN HAMMER — match 2961, survivor 3
# gen: 5
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1952_s0.py (gen4)
# born: 2026-05-29T23:47:03Z

"""Hybrid Voronoi‑Associative‑Probabilistic System
================================================

This module fuses the core ideas of the two parent algorithms:

* **Parent A** – Voronoi partitioning together with a *Sheaf* that stores
  linear restriction maps between nodes.
* **Parent B** – Probabilistic primitives (broadcast and acceptance
  probabilities), a Hoeffding‑style bound and a variational free‑energy
  formulation that are used to guide graph construction and belief
  updates.

**Mathematical bridge**

Voronoi cells provide a natural spatial discretisation: each seed becomes
a *node* of a graph and the set of points assigned to a seed defines the
node’s *section* in the sheaf.  Edges are created between neighbouring
cells; their weights are derived from Euclidean distances of the cell
centroids.  The probabilistic primitives from Parent B are then used to
(1) decide whether an edge is *broadcast* (i.e. kept) and (2) accept or
reject a proposed sheaf update based on a *free‑energy* term that
incorporates a curvature‑like quantity (a simple Ollivier‑Ricci proxy).

The resulting system can be used for data‑organisation, pattern retrieval
and stochastic belief propagation in a single unified code base.
"""

import sys
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Voronoi utilities
# ----------------------------------------------------------------------


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    """Index of the seed closest to *point*."""
    if not seeds.size:
        raise ValueError("seeds required")
    return int(np.argmin(np.linalg.norm(seeds - point, axis=1)))


def assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Return a binary region matrix R where R[i, j] == 1 iff point j belongs
    to seed i.
    """
    regions = np.zeros((seeds.shape[0], points.shape[0]), dtype=int)
    for j, p in enumerate(points):
        i = nearest(p, seeds)
        regions[i, j] = 1
    return regions


# ----------------------------------------------------------------------
# Parent B – Probabilistic primitives
# ----------------------------------------------------------------------


def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """
    Probability that a candidate edge is kept during the *current_phase*
    of graph construction.
    """
    if total_phases < 1 or current_phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """
    Metropolis‑style acceptance probability for an energy change *delta_e*
    at a given *temperature*.
    """
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def hoeffding_bound(mean: float, range_width: float, n: int, delta: float = 0.05) -> float:
    """
    Simple Hoeffding bound for the deviation of the empirical mean.
    Returns the radius ε such that P(|X̄ - μ| ≥ ε) ≤ δ.
    """
    if n <= 0:
        raise ValueError("sample size must be positive")
    return range_width * math.sqrt(math.log(2 / delta) / (2 * n))


# ----------------------------------------------------------------------
# Sheaf structure (from Parent A)
# ----------------------------------------------------------------------


class Sheaf:
    """
    A sheaf stores a vector space (section) on each node and linear
    restriction maps on each directed edge (u → v).  Sections are updated
    by propagating through the restriction maps.
    """

    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims: Dict[Any, int] = dict(node_dims)
        self.edges: List[Tuple[Any, Any]] = list(edges)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    # ------------------------------------------------------------------
    # API for building the sheaf
    # ------------------------------------------------------------------

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
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: Any, value: np.ndarray) -> None:
        if value.shape != (self.node_dims[node],):
            raise ValueError(f"section shape mismatch for node {node}")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    # ------------------------------------------------------------------
    # Propagation (single synchronous sweep)
    # ------------------------------------------------------------------

    def propagate(self) -> None:
        """
        Perform one synchronous propagation step: each node receives a
        contribution from all incoming edges, transformed by the
        corresponding restriction maps, and averages them.
        """
        new_sections: Dict[Any, List[np.ndarray]] = {n: [] for n in self.node_dims}
        for (u, v), (src_map, dst_map) in self._restrictions.items():
            src_sec = self.get_section(u)
            # Linear transformation: src_map @ src_sec  (row‑wise)
            transformed = src_map @ src_sec
            # Map into destination space via dst_map (pseudo‑inverse style)
            # Solve dst_map.T * y = transformed for y (least squares)
            y, *_ = np.linalg.lstsq(dst_map.T, transformed, rcond=None)
            new_sections[v].append(y)

        for node, contributions in new_sections.items():
            if contributions:
                avg = np.mean(contributions, axis=0)
                self._sections[node] = avg


# ----------------------------------------------------------------------
# Hybrid functions (the required three+ demonstrations)
# ----------------------------------------------------------------------


def build_voronoi_graph(
    points: np.ndarray,
    seeds: np.ndarray,
    total_phases: int = 3,
) -> Tuple[Dict[int, Set[int]], np.ndarray]:
    """
    Construct a graph where each seed index is a node.
    Edges are added between a seed and its *k* nearest neighbour seeds
    (k = 3 by default).  Whether an edge survives a phase is decided by
    ``broadcast_probability``.
    Returns the adjacency dictionary and the matrix of seed centroids
    (the centroids of the points assigned to each seed).
    """
    # Assign points to seeds
    region_matrix = assign(points, seeds)          # shape (n_seeds, n_points)

    # Compute centroids of each Voronoi cell
    centroids = np.zeros_like(seeds, dtype=float)
    for i in range(seeds.shape[0]):
        assigned = points[region_matrix[i] == 1]
        if assigned.size:
            centroids[i] = assigned.mean(axis=0)
        else:
            centroids[i] = seeds[i]  # fallback to seed position

    # Build k‑nearest neighbour adjacency
    n_seeds = seeds.shape[0]
    k = min(3, n_seeds - 1)
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n_seeds)}
    for i in range(n_seeds):
        dists = np.linalg.norm(centroids - centroids[i], axis=1)
        nearest_idxs = np.argsort(dists)[1:k + 1]  # skip self
        for phase in range(1, total_phases + 1):
            prob = broadcast_probability(total_phases, phase)
            for j in nearest_idxs:
                if random.random() < prob:
                    adjacency[i].add(j)
                    adjacency[j].add(i)  # undirected

    return adjacency, centroids


def ollivier_ricci_curvature(
    adjacency: Dict[int, Set[int]],
    centroids: np.ndarray,
) -> Dict[Tuple[int, int], float]:
    """
    Compute a simple Ollivier‑Ricci curvature proxy for each edge.
    For edge (i, j):
        κ_ij = 1 - d(i,j) / (average distance of i to its neighbours)
    The curvature lies in (-∞, 1]; higher values indicate tighter coupling.
    """
    curvature: Dict[Tuple[int, int], float] = {}
    for i, neighbours in adjacency.items():
        if not neighbours:
            continue
        avg_dist_i = np.mean([distance(centroids[i], centroids[n]) for n in neighbours])
        for j in neighbours:
            if i < j:  # store each undirected edge once
                d_ij = distance(centroids[i], centroids[j])
                kappa = 1.0 - (d_ij / avg_dist_i) if avg_dist_i != 0 else 0.0
                curvature[(i, j)] = kappa
    return curvature


def variational_free_energy_update(
    sheaf: Sheaf,
    adjacency: Dict[int, Set[int]],
    curvature: Dict[Tuple[int, int], float],
    temperature: float = 1.0,
) -> None:
    """
    Perform a Metropolis‑style update of the sheaf sections.
    For each edge we compute an energy term ΔE = -κ_ij (so higher curvature
    lowers energy).  The update is accepted with probability given by
    ``acceptance_probability``.  Accepted updates replace the destination
    section with the transformed source section (using the stored
    restriction maps).
    """
    for (u, v), (src_map, dst_map) in sheaf._restrictions.items():
        # Ensure the edge exists in the graph (ignore otherwise)
        edge_key = (u, v) if (u, v) in curvature else ((v, u) if (v, u) in curvature else None)
        if edge_key is None:
            continue

        kappa = curvature[edge_key]
        delta_e = -kappa  # negative curvature = lower energy
        if random.random() < acceptance_probability(delta_e, temperature):
            src_sec = sheaf.get_section(u)
            transformed = src_map @ src_sec
            # Solve for destination section (least‑squares)
            new_sec, *_ = np.linalg.lstsq(dst_map.T, transformed, rcond=None)
            sheaf._sections[v] = new_sec


def hybrid_step(
    points: np.ndarray,
    seeds: np.ndarray,
    sheaf: Sheaf,
    total_phases: int = 3,
    temperature: float = 1.0,
) -> Tuple[Dict[int, Set[int]], Dict[Tuple[int, int], float]]:
    """
    One complete hybrid iteration:
    1. Build / update the Voronoi‑graph.
    2. Compute curvature on the graph.
    3. Update the sheaf via variational free‑energy dynamics.
    Returns the adjacency and curvature dictionaries for inspection.
    """
    adjacency, centroids = build_voronoi_graph(points, seeds, total_phases)
    curvature = ollivier_ricci_curvature(adjacency, centroids)
    variational_free_energy_update(sheaf, adjacency, curvature, temperature)
    return adjacency, curvature


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic data
    n_points = 200
    dim = 2
    points = np.random.randn(n_points, dim)

    # Choose 5 random seeds
    n_seeds = 5
    seed_indices = np.random.choice(n_points, n_seeds, replace=False)
    seeds = points[seed_indices]

    # Initialise a sheaf: each node holds a vector of size `dim`
    node_dims = {i: dim for i in range(n_seeds)}
    edges = [(i, j) for i in range(n_seeds) for j in range(i + 1, n_seeds)]
    sheaf = Sheaf(node_dims, edges)

    # Random linear restrictions for each edge
    for (u, v) in edges:
        src_map = np.eye(dim) + 0.1 * np.random.randn(dim, dim)
        dst_map = np.eye(dim) + 0.1 * np.random.randn(dim, dim)
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Initialise sections with the centroid of each Voronoi cell
    region_mat = assign(points, seeds)
    for i in range(n_seeds):
        assigned_pts = points[region_mat[i] == 1]
        if assigned_pts.size:
            sheaf.set_section(i, assigned_pts.mean(axis=0))
        else:
            sheaf.set_section(i, seeds[i])

    # Run a few hybrid steps
    for step in range(4):
        adj, curv = hybrid_step(points, seeds, sheaf, total_phases=3, temperature=0.5)
        print(f"Step {step+1}:")
        print(f"  Adjacency: { {k: sorted(v) for k, v in adj.items()} }")
        print(f"  Curvature (sample): {list(curv.items())[:3]}")
        # Propagate sheaf once more to smooth sections
        sheaf.propagate()

    print("Final sheaf sections:")
    for node in range(n_seeds):
        print(f"  Node {node}: {sheaf.get_section(node)}")