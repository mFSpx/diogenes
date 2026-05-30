# DARWIN HAMMER — match 3266, survivor 2
# gen: 6
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_rbf_su_m1815_s1.py (gen5)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s1.py (gen2)
# born: 2026-05-29T23:48:59Z

"""
Hybrid Voronoi‑RBF‑VRAM‑Ricci module
===================================

Parent A: ``hybrid_voronoi_partition_hybrid_hybrid_rbf_su_m1815_s1.py``  
Parent B: ``hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s1.py``  

Mathematical bridge
-------------------
The Voronoi tessellation supplies a spatial decomposition of a set of
points.  Each Voronoi cell induces a sub‑graph of a global interaction
network.  For a cell we compute Ollivier‑Ricci curvature on its induced
graph – a matrix‑valued operation that yields scalar curvature values
per edge.  The statistical descriptors of these curvatures (e.g. mean,
variance) together with the size of the cell form a feature vector.
A radial‑basis‑function (RBF) surrogate model predicts the **memory
footprint** that a curvature computation for that cell will require.
The predicted footprint is fed to a VRAM planner that registers the
cell‑model as an artifact and evicts the least‑important artifact when
the total budget is exceeded.

Thus the core topology of the hybrid algorithm is:


points ──Voronoi──► cells
cells  ──induce──► sub‑graphs
sub‑graphs ──Ricci──► curvature statistics
statistics + cell size ──RBF──► memory prediction
predictions ──VRAM planner──► load / evict decisions


The implementation below weaves together the essential equations from
both parents (Voronoi assignment, Gaussian RBF, linear system solve,
VRAM bookkeeping, and Ollivier‑Ricci curvature) into a single,
self‑contained module.
"""

import math
import random
import sys
import json
import os
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

# ----------------------------------------------------------------------
# Geometry utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Vector = Sequence[float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance in 2‑D."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed nearest to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Return a mapping seed_index → list of points belonging to that cell."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean norm between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ----------------------------------------------------------------------
# RBF surrogate (from Parent A)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis."""
    return math.exp(-((epsilon * r) ** 2))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Gauss‑Jordan elimination for a dense square system."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(n):
        # pivot selection
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]

        # normalize pivot row
        div = m[col][col]
        m[col] = [v / div for v in m[col]]

        # eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]

    return [row[-1] for row in m]

@dataclass(frozen=True)
class RBFSurrogate:
    """Thin RBF surrogate predicting a scalar (e.g. memory usage)."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def _phi(self, x: Vector) -> List[float]:
        return [gaussian(euclidean(x, c), self.epsilon) for c in self.centers]

    def predict(self, x: Vector) -> float:
        phi = self._phi(x)
        return sum(w * p for w, p in zip(self.weights, phi))

# ----------------------------------------------------------------------
# Simple VRAM planner (inspired by Parent B)
# ----------------------------------------------------------------------
@dataclass
class VRAMArtifact:
    name: str
    size_mb: float
    priority: float  # higher = more important

@dataclass
class VRAMPlanner:
    """Keeps registered artifacts inside a memory budget."""
    budget_mb: float
    reserve_mb: float = 0.0
    _registry: List[VRAMArtifact] = field(default_factory=list)

    @property
    def used_mb(self) -> float:
        return sum(a.size_mb for a in self._registry)

    @property
    def free_mb(self) -> float:
        return max(0.0, self.budget_mb - self.used_mb - self.reserve_mb)

    def register(self, artifact: VRAMArtifact) -> None:
        """Attempt to register *artifact*; evict low‑priority items if needed."""
        self._registry.append(artifact)
        self._evict_if_necessary()

    def _evict_if_necessary(self) -> None:
        while self.free_mb < 0 and self._registry:
            # evict the artifact with the smallest priority / size ratio
            victim = min(self._registry, key=lambda a: a.priority / (a.size_mb + 1e-6))
            self._registry.remove(victim)

    def snapshot(self) -> List[Dict[str, Any]]:
        return [asdict(a) for a in self._registry]

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature (high‑level version, Parent B flavor)
# ----------------------------------------------------------------------
def _neighbor_distribution(G: Dict[int, List[int]], node: int) -> Dict[int, float]:
    """Uniform probability over node itself + its neighbors."""
    neigh = [node] + G.get(node, [])
    prob = 1.0 / len(neigh)
    return {n: prob for n in neigh}

def _wasserstein_distance(p: Dict[int, float], q: Dict[int, float], G: Dict[int, List[int]]) -> float:
    """
    Approximate 1‑Wasserstein distance for two discrete distributions on the
    same node set using shortest‑path distances in *G*.
    """
    # Build distance matrix (BFS since graph is unweighted)
    nodes = set(p) | set(q)
    dist: Dict[Tuple[int, int], int] = {}
    for src in nodes:
        # BFS
        visited = {src: 0}
        frontier = [src]
        while frontier:
            cur = frontier.pop(0)
            for nb in G.get(cur, []):
                if nb not in visited:
                    visited[nb] = visited[cur] + 1
                    frontier.append(nb)
        for dst in nodes:
            dist[(src, dst)] = visited.get(dst, sys.maxsize)

    # Earth mover's distance (optimal transport) for uniform distributions
    # Since both are small, use a naive O(|V|^2) formulation
    total = 0.0
    for i, pi in p.items():
        for j, qj in q.items():
            total += pi * qj * dist[(i, j)]
    return total

def ricci_curvature_edge(G: Dict[int, List[int]], u: int, v: int) -> float:
    """
    Ollivier‑Ricci curvature κ(u,v) = 1 - W1(m_u, m_v) / d(u,v)
    Here d(u,v)=1 for adjacent nodes.
    """
    pu = _neighbor_distribution(G, u)
    pv = _neighbor_distribution(G, v)
    w1 = _wasserstein_distance(pu, pv, G)
    return 1.0 - w1  # d(u,v)=1

def compute_region_curvature_stats(G: Dict[int, List[int]]) -> Tuple[float, float]:
    """
    Return (mean_curvature, variance) over all edges of *G*.
    Empty graph yields (0.0, 0.0).
    """
    curvs: List[float] = []
    seen = set()
    for u, nbrs in G.items():
        for v in nbrs:
            if (v, u) in seen:
                continue
            curvs.append(ricci_curvature_edge(G, u, v))
            seen.add((u, v))
    if not curvs:
        return 0.0, 0.0
    mean = sum(curvs) / len(curvs)
    var = sum((c - mean) ** 2 for c in curvs) / len(curvs)
    return mean, var

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_region_features(cell_points: List[Point],
                           subgraph: Dict[int, List[int]]) -> List[float]:
    """
    Build a feature vector for an RBF prediction:
    - number of points in the cell
    - average Euclidean distance from cell centroid
    - mean Ollivier‑Ricci curvature of the subgraph
    - variance of curvature
    """
    n = len(cell_points)
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0]

    cx = sum(p[0] for p in cell_points) / n
    cy = sum(p[1] for p in cell_points) / n
    centroid = (cx, cy)

    avg_dist = sum(distance(p, centroid) for p in cell_points) / n
    mean_curv, var_curv = compute_region_curvature_stats(subgraph)

    return [float(n), avg_dist, mean_curv, var_curv]

def hybrid_predict_memory(cell_features: List[float],
                          surrogate: RBFSurrogate) -> float:
    """
    Use the RBF surrogate to predict memory consumption (MiB) for a region.
    """
    return max(0.0, surrogate.predict(cell_features))

def hybrid_process(points: List[Point],
                  seeds: List[Point],
                  global_graph: Dict[int, List[int]],
                  surrogate: RBFSurrogate,
                  planner: VRAMPlanner) -> Dict[int, Any]:
    """
    Full hybrid pipeline:
    1. Voronoi partition points.
    2. For each cell, induce a sub‑graph (nodes are point indices).
    3. Compute curvature statistics, build RBF features, predict memory.
    4. Register the region with the VRAM planner.
    Returns a dict mapping cell index → metadata.
    """
    regions = assign(points, seeds)
    results: Dict[int, Any] = {}

    # Build a mapping from point index to its coordinates for easy lookup
    point_index_map = {i: p for i, p in enumerate(points)}

    for cell_idx, cell_pts in regions.items():
        # Induce sub‑graph: keep edges whose both endpoints belong to this cell
        cell_node_ids = [i for i, p in point_index_map.items() if p in cell_pts]
        subgraph: Dict[int, List[int]] = {}
        cell_set = set(cell_node_ids)
        for u in cell_node_ids:
            subgraph[u] = [v for v in global_graph.get(u, []) if v in cell_set]

        features = hybrid_region_features(cell_pts, subgraph)
        pred_mb = hybrid_predict_memory(features, surrogate)

        artifact = VRAMArtifact(
            name=f"region_{cell_idx}",
            size_mb=pred_mb,
            priority=features[0]  # prioritize by number of points
        )
        planner.register(artifact)

        results[cell_idx] = {
            "points": cell_pts,
            "features": features,
            "predicted_mb": pred_mb,
            "registered": any(a.name == artifact.name for a in planner._registry)
        }

    return results

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate random 2‑D points and seeds
    random.seed(0)
    pts = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(200)]
    seeds = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(5)]

    # 2. Build a random undirected graph on point indices
    N = len(pts)
    G: Dict[int, List[int]] = {i: [] for i in range(N)}
    edge_prob = 0.05
    for i in range(N):
        for j in range(i + 1, N):
            if random.random() < edge_prob:
                G[i].append(j)
                G[j].append(i)

    # 3. Simple RBF surrogate: use synthetic centers and weights
    #    Feature dimension = 4 (as defined in hybrid_region_features)
    centers = [
        (10.0, 5.0, 0.2, 0.01),
        (50.0, 10.0, 0.0, 0.05),
        (100.0, 20.0, -0.1, 0.1)
    ]
    # Solve linear system to obtain weights that map centers to plausible memory sizes
    A = [[gaussian(euclidean(c, d)) for d in centers] for c in centers]
    b = [20.0, 80.0, 150.0]  # imagined memory footprints in MiB
    weights = solve_linear(A, b)
    surrogate = RBFSurrogate(centers=centers, weights=weights, epsilon=0.1)

    # 4. VRAM planner with a modest budget
    planner = VRAMPlanner(budget_mb=300.0, reserve_mb=20.0)

    # 5. Run hybrid pipeline
    out = hybrid_process(pts, seeds, G, surrogate, planner)

    # 6. Print summary
    total_used = planner.used_mb
    print(f"VRAM used after registration: {total_used:.2f} MiB (budget {planner.budget_mb} MiB)")
    for cid, meta in out.items():
        print(f"Cell {cid}: points={len(meta['points'])}, pred={meta['predicted_mb']:.2f}MiB, registered={meta['registered']}")
    # Ensure the program exits cleanly
    sys.exit(0)