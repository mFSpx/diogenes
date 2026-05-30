# DARWIN HAMMER — match 3002, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_hybrid_geomet_m1460_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2421_s2.py (gen5)
# born: 2026-05-29T23:47:10Z

"""Hybrid algorithm combining VRAM curvature scheduling (Parent A) and sheaf-based feature aggregation (Parent B).

Mathematical bridge:
- Parent A builds a covariance matrix whose off‑diagonal entries are Ollivier‑Ricci curvature weights
  `exp(-scale*|i‑j|)`.
- Parent B defines a cellular sheaf whose nodes carry feature vectors and whose edges encode
  linear restrictions.

The hybrid algorithm:
1. Partition a set of 2‑D points into Voronoi cells (Parent A).
2. For each cell aggregate textual feature counts (evidence, planning, delay) extracted with the
   regexes from Parent B.
3. Create a sheaf where each node corresponds to a Voronoi cell and its vector space is the
   aggregated feature vector.
4. Use the same curvature‑weight function to populate the sheaf adjacency and to construct a
   global covariance matrix that couples the node‑wise feature statistics.

Thus the curvature‑based metric from the VRAM scheduler becomes the edge‑weighting scheme
for the sheaf, while the sheaf supplies the multivariate data that feeds the covariance
construction. The result is a unified representation that can be used for downstream
probabilistic or geometric reasoning.
"""

import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (curvature, Voronoi partitioning)
# ----------------------------------------------------------------------
def curvature_weight(i: int, j: int, scale: float = 0.1) -> float:
    """Ollivier‑Ricci style weight between indices i and j."""
    distance = abs(i - j)
    return math.exp(-scale * distance)


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance in 2‑D."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    """Return the index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def voronoi_assign(points: List[Tuple[float, float]],
                  seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """Assign each point to the nearest seed, yielding a Voronoi partition."""
    assignment: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for pt in points:
        idx = nearest(pt, seeds)
        assignment[idx].append(pt)
    return assignment


# ----------------------------------------------------------------------
# Parent B utilities (feature extraction, sheaf definition)
# ----------------------------------------------------------------------
_EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
_PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
_DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)


def extract_combined_features(text: str) -> Dict[str, int]:
    """Count evidence, planning and delay cues in *text* using both parents' regexes."""
    return {
        "evidence": len(_EVIDENCE_RE.findall(text)),
        "planning": len(_PLANNING_RE.findall(text)),
        "delay": len(_DELAY_RE.findall(text)),
    }


# ----------------------------------------------------------------------
# Sheaf implementation (adapted from Parent B)
# ----------------------------------------------------------------------
class Sheaf:
    """Cellular sheaf on a simple undirected graph.

    Each node carries a vector (the aggregated feature vector of its Voronoi cell).
    Edges are weighted by curvature_weight(i, j).
    """

    def __init__(self) -> None:
        self._graph: Dict[int, List[int]] = {}
        self._node_vectors: Dict[int, np.ndarray] = {}

    def add_node(self, node: int, vector: np.ndarray) -> None:
        self._graph[node] = []
        self._node_vectors[node] = vector

    def add_edge(self, node1: int, node2: int) -> None:
        if node1 not in self._graph or node2 not in self._graph:
            raise KeyError("both nodes must be added before an edge")
        self._graph[node1].append(node2)
        self._graph[node2].append(node1)

    def adjacency_matrix(self) -> np.ndarray:
        """Curvature‑weighted adjacency matrix A where A[i,j]=weight if edge exists else 0."""
        n = len(self._graph)
        A = np.zeros((n, n), dtype=float)
        for i, neighbors in self._graph.items():
            for j in neighbors:
                A[i, j] = curvature_weight(i, j)
        return A

    def laplacian(self) -> np.ndarray:
        """Unnormalized graph Laplacian L = D - A."""
        A = self.adjacency_matrix()
        D = np.diag(A.sum(axis=1))
        return D - A

    def node_matrix(self) -> np.ndarray:
        """Stack node vectors row‑wise into a matrix V (n x d)."""
        n = len(self._graph)
        d = next(iter(self._node_vectors.values())).shape[0]
        V = np.zeros((n, d), dtype=float)
        for i, vec in self._node_vectors.items():
            V[i] = vec
        return V


# ----------------------------------------------------------------------
# Hybrid construction utilities
# ----------------------------------------------------------------------
def build_hybrid_sheaf(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    texts: List[str],
) -> Sheaf:
    """Create a sheaf whose nodes are Voronoi cells.

    For each seed:
      * collect its points,
      * concatenate the corresponding texts,
      * extract combined feature counts,
      * store the counts as a 3‑dimensional node vector.

    Edges are added between each node and its two nearest neighbours in seed space
    (providing a sparse connectivity reminiscent of a Delaunay graph).
    """
    if len(texts) != len(points):
        raise ValueError("texts list must parallel points list")

    # Partition points
    assignment = voronoi_assign(points, seeds)

    # Aggregate texts per cell
    cell_texts: Dict[int, List[str]] = {i: [] for i in range(len(seeds))}
    for pt, txt in zip(points, texts):
        idx = nearest(pt, seeds)
        cell_texts[idx].append(txt)

    sheaf = Sheaf()

    # Create nodes with feature vectors
    for i in range(len(seeds)):
        aggregated = " ".join(cell_texts[i])
        feats = extract_combined_features(aggregated)
        vector = np.array([feats["evidence"], feats["planning"], feats["delay"]], dtype=float)
        sheaf.add_node(i, vector)

    # Connect each node to its two nearest seed neighbours
    for i, seed_i in enumerate(seeds):
        # compute distances to other seeds
        dists = [(j, distance(seed_i, seeds[j])) for j in range(len(seeds)) if j != i]
        dists.sort(key=lambda x: x[1])
        for neighbor_idx, _ in dists[:2]:  # connect to two closest neighbours
            # avoid duplicate edge insertion
            if neighbor_idx not in sheaf._graph[i]:
                sheaf.add_edge(i, neighbor_idx)

    return sheaf


def build_prior_from_sheaf(sheaf: Sheaf) -> Tuple[np.ndarray, np.ndarray]:
    """Construct mean vector and curvature‑weighted covariance matrix from a sheaf.

    *Mean* is the per‑node L2 norm of its feature vector.
    *Covariance* uses curvature_weight(i, j) for off‑diagonal entries and a small
    variance term (5 % of the mean) on the diagonal, mirroring Parent A's `build_prior`.
    """
    V = sheaf.node_matrix()
    mean = np.linalg.norm(V, axis=1)  # shape (n,)

    n = mean.shape[0]
    cov = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                cov[i, j] = mean[i] * 0.05  # 5 % variance of the mean
            else:
                cov[i, j] = curvature_weight(i, j)
    return mean, cov


def hybrid_curvature_laplacian(sheaf: Sheaf) -> np.ndarray:
    """Compute L = D - A where A uses curvature weights (the sheaf's Laplacian)."""
    return sheaf.laplacian()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic data
    random.seed(42)
    np.random.seed(42)

    # 30 random 2‑D points
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(30)]
    # 5 random seed locations
    seeds = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(5)]
    # Dummy texts (some contain keywords)
    sample_phrases = [
        "We have evidence from the log files.",
        "The plan includes three steps.",
        "We need to wait before proceeding.",
        "No relevant keywords here.",
        "Evidence and plan are both present.",
    ]
    texts = [random.choice(sample_phrases) for _ in range(30)]

    # Build hybrid sheaf
    sheaf = build_hybrid_sheaf(points, seeds, texts)

    # Compute hybrid covariance
    mean_vec, cov_mat = build_prior_from_sheaf(sheaf)

    # Compute Laplacian for diagnostic output
    L = hybrid_curvature_laplacian(sheaf)

    # Simple sanity prints
    print("Node feature matrix (V):")
    print(sheaf.node_matrix())
    print("\nMean vector (from node norms):")
    print(mean_vec)
    print("\nCovariance matrix (curvature‑weighted):")
    print(cov_mat)
    print("\nCurvature‑weighted Laplacian:")
    print(L)

    # Verify that covariance is positive semi‑definite (eigenvalues >= 0 within tolerance)
    eigs = np.linalg.eigvalsh(cov_mat)
    print("\nCovariance eigenvalues:", eigs)
    assert np.all(eigs >= -1e-8), "Covariance matrix not PSD"
    print("\nSmoke test completed successfully.")