# DARWIN HAMMER — match 787, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_semantic_neig_hybrid_caputo_fracti_m258_s1.py (gen3)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_hybrid_ternar_m55_s2.py (gen4)
# born: 2026-05-29T23:30:54Z

"""Hybrid Algorithm integrating:
- Parent A: hybrid_hybrid_semantic_neig_hybrid_caputo_fracti_m258_s1 (morphology recovery priority, Caputo fractional kernel)
- Parent B: hybrid_hybrid_krampus_brain_hybrid_hybrid_ternar_m55_s2 (graph sheaf, Ollivier‑Ricci curvature)

Mathematical bridge:
Each Morphology instance becomes a node in a graph. The node attribute is the
**recovery priority** p_i = recovery_priority(m_i).  Edges are created between
nodes whose geometric descriptors are close; the edge weight w_{ij} is the
**Ollivier‑Ricci curvature** approximated by 1‑cosine_similarity(feature_i,
feature_j).  The sheaf stores the feature vectors as sections and the curvature
as restriction maps.  A discrete Caputo fractional operator of order α is then
applied to the vector of priorities using the edge‑weight matrix, yielding a
fractional diffusion that respects both the semantic‑recovery topology (A) and
the curvature‑filtered sheaf structure (B)."""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ---------- Parent A components ----------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] derived from righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if den == 0 else float(np.dot(a, b) / den)

# ---------- Parent B components ----------
class Sheaf:
    """Simple sheaf storing sections (node vectors) and linear restrictions (edges)."""
    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node -> dimension of its section
        self.edges = list(edge_list)              # list of (u, v)
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[int, int], src_map: np.ndarray, dst_map: np.ndarray):
        u, v = edge
        self._restrictions[(u, v)] = (np.array(src_map, dtype=float),
                                      np.array(dst_map, dtype=float))

    def set_section(self, node: int, value: np.ndarray):
        self._sections[node] = np.array(value, dtype=float)

    def get_section(self, node: int) -> np.ndarray:
        return self._sections[node]

    def restriction(self, edge: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        return self._restrictions[edge]

# ---------- Hybrid Functions ----------
def build_hybrid_graph(morphologies: List[Morphology],
                       distance_thresh: float = 0.3) -> Tuple[Dict[int, float],
                                                            Dict[Tuple[int, int], float],
                                                            Sheaf]:
    """
    Constructs:
    - node_priorities: dict node_id -> recovery priority (A)
    - edge_curvatures: dict (i,j) -> Ollivier‑Ricci curvature approximation (B)
    - sheaf: stores feature sections (raw geometry) and curvature‑derived restrictions.
    Edges are created when Euclidean distance between normalized geometry vectors
    is below `distance_thresh`.
    """
    n = len(morphologies)
    node_priorities: Dict[int, float] = {}
    feature_vectors: Dict[int, np.ndarray] = {}

    # Build node attributes
    for idx, m in enumerate(morphologies):
        node_priorities[idx] = recovery_priority(m)
        # Feature vector: normalized geometric descriptor
        vec = np.array([m.length, m.width, m.height, m.mass], dtype=float)
        norm = np.linalg.norm(vec)
        feature_vectors[idx] = vec / norm if norm != 0 else vec

    # Edge creation + curvature
    edge_curvatures: Dict[Tuple[int, int], float] = {}
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(feature_vectors[i] - feature_vectors[j])
            if dist <= distance_thresh:
                edges.append((i, j))
                # Ollivier‑Ricci curvature ≈ 1 - cosine similarity of feature vectors
                curvature = 1.0 - _cos(feature_vectors[i], feature_vectors[j])
                edge_curvatures[(i, j)] = curvature

    # Build sheaf
    node_dims = {i: feature_vectors[i].size for i in range(n)}
    sheaf = Sheaf(node_dims, edges)
    for i, vec in feature_vectors.items():
        sheaf.set_section(i, vec)

    for (u, v), curv in edge_curvatures.items():
        # Use curvature to define linear maps: identity scaled by (1‑curv)
        scale = 1.0 - curv  # between 0 and 1
        I = np.identity(sheaf.node_dims[u])
        sheaf.set_restriction((u, v), I * scale, I * scale)

    return node_priorities, edge_curvatures, sheaf

def caputo_fractional_step(priorities: np.ndarray,
                           edge_weights: np.ndarray,
                           alpha: float = 0.5,
                           dt: float = 1.0) -> np.ndarray:
    """
    One step of a discrete Caputo fractional diffusion on the graph.
    The Laplacian L = D - W, where W contains edge_weights and D is the degree matrix.
    The Caputo fractional derivative of order α is approximated by:
        p_{t+1} = p_t - dt^α / Γ(1-α) * L * p_t
    This simple scheme respects the fractional scaling while staying explicit.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1) for a Caputo fractional order")
    # Degree matrix
    deg = np.sum(edge_weights, axis=1)
    L = np.diag(deg) - edge_weights
    coeff = (dt ** alpha) / math.gamma(1 - alpha)
    return priorities - coeff * L.dot(priorities)

def hybrid_propagate(morphologies: List[Morphology],
                     alpha: float = 0.5,
                     dt: float = 1.0,
                     distance_thresh: float = 0.3,
                     steps: int = 3) -> List[float]:
    """
    Full hybrid pipeline:
    1. Build graph + sheaf from morphologies.
    2. Assemble weighted adjacency matrix using curvature as edge weight.
    3. Apply a few Caputo fractional diffusion steps to the priority vector.
    Returns the final priority list.
    """
    node_priorities, edge_curvatures, _ = build_hybrid_graph(
        morphologies, distance_thresh=distance_thresh)

    n = len(morphologies)
    W = np.zeros((n, n), dtype=float)  # adjacency matrix
    for (i, j), w in edge_curvatures.items():
        # Higher curvature (more “overlap”) should increase coupling;
        # we use weight = 1 - curvature to keep values in [0,1].
        weight = 1.0 - w
        W[i, j] = weight
        W[j, i] = weight

    p = np.array([node_priorities[i] for i in range(n)], dtype=float)

    for _ in range(steps):
        p = caputo_fractional_step(p, W, alpha=alpha, dt=dt)
        # Clamp to [0,1] after each step to keep a valid priority scale
        p = np.clip(p, 0.0, 1.0)

    return p.tolist()

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Generate a small random population of morphologies
    random.seed(42)
    pop: List[Morphology] = []
    for _ in range(8):
        m = Morphology(
            length= random.uniform(0.5, 2.0),
            width=  random.uniform(0.5, 2.0),
            height= random.uniform(0.2, 1.0),
            mass=   random.uniform(0.1, 5.0)
        )
        pop.append(m)

    final_priorities = hybrid_propagate(pop,
                                        alpha=0.6,
                                        dt=1.0,
                                        distance_thresh=0.25,
                                        steps=5)

    print("Final recovery priorities after hybrid fractional diffusion:")
    for idx, pr in enumerate(final_priorities):
        print(f"Node {idx}: {pr:.4f}")
    sys.exit(0)