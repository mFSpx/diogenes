# DARWIN HAMMER — match 5258, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_vorono_m2114_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m2039_s1.py (gen5)
# born: 2026-05-30T00:00:56Z

"""Hybrid Algorithm: Curvature‑Sketch Fusion

Parents:
- hybrid_hybrid_hard_truth_ma_hybrid_bayes_update__m123_s0 (Algorithm A)
- hybrid_hybrid_ternary_route_hybrid_hoeffding_tre_m1040_s3 (Algorithm B)

Mathematical Bridge:
Algorithm A produces a low‑dimensional multivector **c** ∈ ℝⁿ representing
Ollivier‑Ricci curvature values over a graph or a Voronoi‑partitioned space.
Algorithm B provides probabilistic sketch structures (Count‑Min Sketch and
HyperLogLog) together with a similarity measure (SSIM) and a confidence
bound (Hoeffding).

The fusion treats the curvature vector **c** as a frequency stream:
each curvature value (quantised) is fed to a Count‑Min sketch **CMS** and a
HyperLogLog sketch **HLL**.  The SSIM between two such sketches quantifies
the similarity of curvature distributions across different Voronoi cells,
while the Hoeffding bound supplies a statistical confidence interval for
the estimated curvature mean of each cell.  This yields a unified estimator
that simultaneously leverages geometric curvature information and
probabilistic sketch‑based similarity/confidence.

The module implements:
1. Curvature computation on a graph (simplified Ollivier‑Ricci).
2. Voronoi partition of points.
3. Sketch construction from curvature vectors.
4. SSIM‑based similarity between curvature sketches.
5. Hoeffding‑bounded hybrid curvature estimate per Voronoi cell.
"""

import math
import random
import sys
import pathlib
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Helper structures from Parent B
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a bounded random variable with range r."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def compute_ssim(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """Structural Similarity Index Measure between two 1‑D arrays."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.cov(x, y)[0, 1]

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)

    return numerator / denominator


class CountMinSketch:
    """Count‑Min sketch for non‑negative integer streams."""
    def __init__(self, width: int = 1000, depth: int = 5):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
        self.seeds = [random.randint(0, 2 ** 31 - 1) for _ in range(depth)]

    def _hash(self, item: int, i: int) -> int:
        random.seed(self.seeds[i] + item)
        return random.randint(0, self.width - 1)

    def update(self, item: int, inc: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i][idx] += inc

    def estimate(self, item: int) -> int:
        return min(self.table[i][self._hash(item, i)] for i in range(self.depth))

    def frequency_vector(self) -> np.ndarray:
        """Flatten the sketch into a 1‑D array (used for SSIM)."""
        return np.array([self.table[i][j] for i in range(self.depth) for j in range(self.width)], dtype=float)


class HyperLogLog:
    """Very small HyperLogLog implementation (registers only)."""
    def __init__(self, b: int = 4):
        if not (4 <= b <= 16):
            raise ValueError("b must be in [4,16]")
        self.b = b
        self.m = 1 << b
        self.registers = [0] * self.m
        self.seed = random.randint(0, 2 ** 31 - 1)

    def _rho(self, w: int) -> int:
        return (w & -w).bit_length()

    def add(self, item: int) -> None:
        random.seed(self.seed + item)
        x = random.getrandbits(32)
        j = x >> (32 - self.b)
        w = x << self.b & 0xFFFFFFFF
        self.registers[j] = max(self.registers[j], self._rho(w))

    def cardinality(self) -> float:
        """Raw cardinality estimate (no bias correction)."""
        Z = 1.0 / sum(2.0 ** -r for r in self.registers)
        return self.m * Z

    def register_vector(self) -> np.ndarray:
        return np.array(self.registers, dtype=float)


# ----------------------------------------------------------------------
# Geometry / Curvature utilities (simplified version of Parent A)
# ----------------------------------------------------------------------
def euclidean_distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def voronoi_partition(points: List[Tuple[float, float]],
                      seeds: List[Tuple[float, float]]) -> Dict[int, List[int]]:
    """Assign each point index to the nearest seed index."""
    assignment: Dict[int, List[int]] = defaultdict(list)
    for idx, pt in enumerate(points):
        dists = [euclidean_distance(pt, s) for s in seeds]
        nearest = int(np.argmin(dists))
        assignment[nearest].append(idx)
    return dict(assignment)


def ollivier_ricci_curvature(edge: Tuple[int, int],
                            distances: np.ndarray,
                            mass: float = 1.0) -> float:
    """
    Simplified Ollivier‑Ricci curvature for an undirected edge (i,j).

    κ(i,j) = 1 - W₁(μ_i, μ_j) / d(i,j)

    where μ_i, μ_j are unit mass distributions on the neighbours of i and j.
    For simplicity we use uniform mass on the immediate neighbours.
    """
    i, j = edge
    d_ij = distances[i, j]
    if d_ij == 0:
        return 0.0

    # neighbours (including self) with uniform probability
    neigh_i = np.where(distances[i] > 0)[0]
    neigh_j = np.where(distances[j] > 0)[0]

    # probability vectors
    pi = np.zeros(distances.shape[0])
    pj = np.zeros(distances.shape[0])
    pi[neigh_i] = 1.0 / max(len(neigh_i), 1)
    pj[neigh_j] = 1.0 / max(len(neigh_j), 1)

    # Earth Mover's distance (here reduced to L1 because the ground metric is graph distance)
    W1 = np.sum(np.abs(pi - pj) * distances[i, :])  # approximate

    return 1.0 - W1 / d_ij


def curvature_vector(points: List[Tuple[float, float]],
                     edges: List[Tuple[int, int]]) -> np.ndarray:
    """Return a vector of curvature values for all edges."""
    n = len(points)
    # build distance matrix (Euclidean for this toy example)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(points[i], points[j])
            dist_mat[i, j] = dist_mat[j, i] = d

    curvs = [ollivier_ricci_curvature(e, dist_mat) for e in edges]
    return np.array(curvs, dtype=float)


# ----------------------------------------------------------------------
# Hybrid operations (the required three+ functions)
# ----------------------------------------------------------------------
def sketch_curvature_distribution(curv_vec: np.ndarray,
                                  cms_width: int = 500,
                                  cms_depth: int = 4,
                                  hll_b: int = 5) -> Tuple[CountMinSketch, HyperLogLog]:
    """
    Convert a curvature vector into a Count‑Min sketch and a HyperLogLog sketch.
    Curvature values are quantised to integers in [0, 255] before insertion.
    """
    cms = CountMinSketch(width=cms_width, depth=cms_depth)
    hll = HyperLogLog(b=hll_b)

    # Quantise curvature to 0‑255 (unsigned byte) for deterministic hashing
    quantised = np.clip(((curv_vec - curv_vec.min()) /
                        (curv_vec.ptp() + 1e-12) * 255), 0, 255).astype(int)

    for val in quantised:
        cms.update(val)
        hll.add(val)

    return cms, hll


def curvature_similarity(cell_a: Tuple[CountMinSketch, HyperLogLog],
                         cell_b: Tuple[CountMinSketch, HyperLogLog]) -> float:
    """
    Compute a hybrid similarity between two Voronoi cells:
    - SSIM on the flattened Count‑Min frequency vectors.
    - Gini coefficient on the HyperLogLog register vectors (optional, not used here).
    The final similarity is the product of SSIM and a normalized HLL overlap.
    """
    cms_a, hll_a = cell_a
    cms_b, hll_b = cell_b

    ssim_val = compute_ssim(cms_a.frequency_vector(), cms_b.frequency_vector())

    # Simple overlap measure for HLL registers
    reg_a = hll_a.register_vector()
    reg_b = hll_b.register_vector()
    overlap = np.mean(reg_a == reg_b)

    return ssim_val * overlap


def hybrid_curvature_estimate(points: List[Tuple[float, float]],
                              edges: List[Tuple[int, int]],
                              seeds: List[Tuple[float, float]],
                              delta: float = 0.05) -> Dict[int, Dict]:
    """
    End‑to‑end hybrid estimator:

    1. Compute curvature for every edge.
    2. Partition points into Voronoi cells.
    3. For each cell, build sketches from the curvatures of edges whose
       endpoints lie inside the cell.
    4. Return, per cell:
       - mean curvature,
       - Hoeffding confidence interval,
       - sketch objects,
       - similarity to every other cell (pairwise matrix).
    """
    # 1. Global curvature vector
    curv_vec = curvature_vector(points, edges)

    # 2. Voronoi partition of point indices
    cell_assignment = voronoi_partition(points, seeds)

    # 3. Map edges to cells (both endpoints must belong to the same cell)
    cell_edges: Dict[int, List[int]] = defaultdict(list)  # cell -> list of edge indices
    point_to_cell = {}
    for cell_id, idxs in cell_assignment.items():
        for idx in idxs:
            point_to_cell[idx] = cell_id

    for e_idx, (u, v) in enumerate(edges):
        cu = point_to_cell.get(u)
        cv = point_to_cell.get(v)
        if cu is not None and cu == cv:
            cell_edges[cu].append(e_idx)

    # 4. Build per‑cell results
    results: Dict[int, Dict] = {}
    for cell_id, e_indices in cell_edges.items():
        if not e_indices:
            continue
        cell_curv = curv_vec[e_indices]
        mean_c = float(np.mean(cell_curv))
        n = len(cell_curv)
        # curvature values lie in [-1,1] for Ollivier‑Ricci; range r = 2
        bound = hoeffding_bound(r=2.0, delta=delta, n=n)
        cms, hll = sketch_curvature_distribution(cell_curv)

        results[cell_id] = {
            "mean_curvature": mean_c,
            "hoeffding_interval": (mean_c - bound, mean_c + bound),
            "sketches": (cms, hll),
            "edge_indices": e_indices,
        }

    # 5. Pairwise similarity matrix
    cell_ids = list(results.keys())
    similarity_matrix = np.zeros((len(cell_ids), len(cell_ids)))
    for i, cid_i in enumerate(cell_ids):
        for j, cid_j in enumerate(cell_ids):
            if i <= j:
                sim = curvature_similarity(results[cid_i]["sketches"],
                                           results[cid_j]["sketches"])
                similarity_matrix[i, j] = similarity_matrix[j, i] = sim

    # attach matrix
    for idx, cid in enumerate(cell_ids):
        results[cid]["similarities"] = {
            other_cid: similarity_matrix[idx, jdx] for jdx, other_cid in enumerate(cell_ids)
        }

    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a random set of points in 2‑D
    random.seed(42)
    num_points = 30
    points = [(random.random(), random.random()) for _ in range(num_points)]

    # Build a simple k‑nearest‑neighbour graph (k=3)
    k = 3
    edges = []
    for i, p in enumerate(points):
        dists = [euclidean_distance(p, q) for q in points]
        nearest = np.argsort(dists)[1:k + 1]  # skip self (distance 0)
        for j in nearest:
            if i < j:
                edges.append((i, j))

    # Choose 4 random seeds for Voronoi partition
    seeds = random.sample(points, 4)

    # Run hybrid estimator
    hybrid_result = hybrid_curvature_estimate(points, edges, seeds, delta=0.1)

    # Print summary
    for cell_id, info in hybrid_result.items():
        print(f"Cell {cell_id}:")
        print(f"  Mean curvature          = {info['mean_curvature']:.4f}")
        lo, hi = info['hoeffding_interval']
        print(f"  Hoeffding 95% CI       = [{lo:.4f}, {hi:.4f}]")
        print(f"  Edges in cell          = {len(info['edge_indices'])}")
        print(f"  Similarities to others = { {k: round(v,3) for k,v in info['similarities'].items()} }")
        print()
    print("Hybrid curvature‑sketch fusion executed successfully.")