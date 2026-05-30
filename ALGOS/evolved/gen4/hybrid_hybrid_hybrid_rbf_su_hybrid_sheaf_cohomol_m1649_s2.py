# DARWIN HAMMER — match 1649, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s4.py (gen3)
# parent_b: hybrid_sheaf_cohomology_percyphon_m2_s1.py (gen1)
# born: 2026-05-29T23:38:01Z

"""Hybrid RBF–Sheaf–Hoeffding Algorithm
Parents:
- hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s4.py (RBF kernel + perceptual similarity + tropical algebra)
- hybrid_sheaf_cohomology_percyphon_m2_s1.py (Cellular sheaf, restriction maps, procedural slots)

Mathematical bridge:
The RBF similarity matrix K∈ℝ^{n×n} and the perceptual‑hash similarity matrix S∈[0,1]^{n×n}
are fused by the tropical matrix product

    C = K ⊗ S ,   C_{ij}= max_k ( K_{ik} + S_{kj} ).

Each entry C_{ij} is then interpreted as a linear scaling factor for the
restriction map of a cellular sheaf defined on the same graph.  The sheaf
carries a vector space (the feature space) at every node; restriction maps
between adjacent nodes are simple diagonal matrices whose diagonal entries are
the corresponding C‑values.  The Hoeffding bound is finally applied to the
row‑wise gains derived from C to decide whether a node should be “promoted’’
(i.e. split) in a streaming‑style decision process.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, Hashable, List, Sequence, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Parent A utilities (RBF & perceptual similarity)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Very simple perceptual hash: 1‑bit per value (threshold at median)."""
    if not values:
        return 0
    median = np.median(values)
    bits = [(1 if v > median else 0) for v in values]
    # pack bits into an integer
    h = 0
    for b in bits:
        h = (h << 1) | b
    return h


def rbf_kernel_matrix(features: List[FeatureVec], epsilon: float = 1.0) -> np.ndarray:
    """Compute dense RBF kernel K_{ij}=exp(-ε²‖f_i-f_j‖²)."""
    n = len(features)
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist = euclidean(features[i], features[j])
            K[i, j] = gaussian(dist, epsilon)
    return K


def perceptual_similarity_matrix(phashes: List[int]) -> np.ndarray:
    """Similarity based on Hamming overlap of perceptual hashes.
    Normalised to [0,1] where 1 means identical hashes."""
    n = len(phashes)
    S = np.empty((n, n), dtype=float)
    max_bits = max(phashes).bit_length() if phashes else 1
    for i in range(n):
        for j in range(n):
            xor = phashes[i] ^ phashes[j]
            # Hamming distance = number of differing bits
            hd = bin(xor).count("1")
            S[i, j] = 1.0 - hd / max_bits
    return S


def tropical_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tropical matrix product C = A ⊗ B,  C_{ij}=max_k (A_{ik}+B_{kj})."""
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible shapes for tropical product")
    n, m = A.shape[0], B.shape[1]
    C = np.full((n, m), -np.inf, dtype=float)
    for i in range(n):
        for j in range(m):
            # vectorized max over k
            C[i, j] = np.max(A[i, :] + B[:, j])
    return C


def combined_similarity(features: List[FeatureVec],
                       phashes: List[int],
                       epsilon: float = 1.0) -> np.ndarray:
    """Fuse geometric and perceptual similarities via tropical algebra."""
    K = rbf_kernel_matrix(features, epsilon)
    S = perceptual_similarity_matrix(phashes)
    C = tropical_product(K, S)
    return C

# ----------------------------------------------------------------------
# Parent B utilities (Sheaf & ProceduralSlot)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Sheaf:
    """Cellular sheaf over a graph with linear restriction maps derived from a
    similarity matrix."""

    def __init__(self, node_dims: Dict[Node, int], edge_list: List[Tuple[Node, Node]]):
        self.node_dims = dict(node_dims)          # node -> dimension
        self.edges = list(edge_list)              # oriented edges (u,v)
        self._restrictions: Dict[Tuple[Node, Node],
                                 Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Node, np.ndarray] = {}

    def set_restriction(self,
                        edge: Tuple[Node, Node],
                        src_map: np.ndarray,
                        dst_map: np.ndarray) -> None:
        """Assign linear restriction maps for an oriented edge."""
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column size does not match dimension of node u")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column size does not match dimension of node v")
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node: Node, value: np.ndarray) -> None:
        """Assign a local section (vector) to a node."""
        value = np.array(value, dtype=float)
        if value.shape != (self.node_dims[node],):
            raise ValueError(f"Section shape {value.shape} inconsistent with node dimension {self.node_dims[node]}")
        self._sections[node] = value

    def get_section(self, node: Node) -> np.ndarray:
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def restriction(self, edge: Tuple[Node, Node]) -> Tuple[np.ndarray, np.ndarray]:
        return self._restrictions[edge]

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_sheaf_from_graph_and_similarity(
    graph: Graph,
    similarity: np.ndarray,
    feature_dim: int = 1
) -> Sheaf:
    """
    Construct a Sheaf where each node carries a vector space of dimension `feature_dim`.
    For each oriented edge (u,v) we create diagonal restriction maps
    diag_src and diag_dst whose diagonal entries are the similarity values
    C_{uv} (scaled to a reasonable range).
    """
    nodes = list(graph.keys())
    node_index = {node: i for i, node in enumerate(nodes)}
    node_dims = {node: feature_dim for node in nodes}
    edge_list = []

    for u, neigh in graph.items():
        for v in neigh:
            edge_list.append((u, v))

    sheaf = Sheaf(node_dims, edge_list)

    # Build diagonal maps; we keep them 1‑D for simplicity (feature_dim == 1)
    for (u, v) in edge_list:
        i, j = node_index[u], node_index[v]
        scale = similarity[i, j] if similarity.size else 0.0
        # Clamp to non‑negative values to keep maps well‑behaved
        scale = max(scale, 0.0)
        src_map = np.array([[scale]])  # maps R^{1} → R^{1}
        dst_map = np.array([[scale]])  # same for destination
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Initialise sections with raw feature values (first component of feature vector)
    for node in nodes:
        idx = node_index[node]
        # For demonstration we take the first component of the feature vector
        # (or zero if unavailable)
        section_val = similarity[idx, idx] if similarity.size else 0.0
        sheaf.set_section(node, np.array([section_val]))
    return sheaf


def hoeffding_gain_split(
    similarity: np.ndarray,
    node_idx: int,
    n_observations: int,
    delta: float = 0.05
) -> bool:
    """
    Decide whether to split (promote) a node based on the Hoeffding bound.
    The gain is the average similarity of the node to all others.
    If the empirical gain exceeds the Hoeffding bound by a margin, we split.
    """
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    gains = similarity[node_idx, :]
    empirical_mean = float(np.mean(gains))

    # Hoeffding bound for values in [0,1]
    epsilon = math.sqrt(math.log(2 / delta) / (2 * n_observations))

    # Simple decision rule
    split = empirical_mean - epsilon > 0.5  # 0.5 is a heuristic threshold
    return split


def hybrid_step(
    features: List[FeatureVec],
    graph: Graph,
    epsilon_rbf: float = 1.0,
    delta_hoeffding: float = 0.05,
    n_obs: int = 30
) -> Tuple[Sheaf, List[Node]]:
    """
    Execute one iteration of the hybrid algorithm:
    1. Compute perceptual hashes from raw features.
    2. Fuse RBF and perceptual similarities via tropical product.
    3. Build a sheaf whose restriction maps are derived from the fused matrix.
    4. Apply Hoeffding‑based split decisions to each node.
    Returns the constructed sheaf and the list of nodes that were split.
    """
    # 1. Perceptual hashes (one hash per node)
    phashes = [compute_phash(list(map(float, feat))) for feat in features]

    # 2. Combined similarity matrix C
    C = combined_similarity(features, phashes, epsilon=epsilon_rbf)

    # 3. Sheaf construction
    sheaf = build_sheaf_from_graph_and_similarity(graph, C, feature_dim=1)

    # 4. Split decisions
    split_nodes: List[Node] = []
    nodes = list(graph.keys())
    for idx, node in enumerate(nodes):
        if hoeffding_gain_split(C, idx, n_obs, delta=delta_hoeffding):
            split_nodes.append(node)

    return sheaf, split_nodes

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic graph with 4 nodes
    test_graph: Graph = {
        "A": {"B", "C"},
        "B": {"A", "D"},
        "C": {"A", "D"},
        "D": {"B", "C"},
    }

    # Random 3‑dimensional feature vectors
    random.seed(42)
    test_features: List[FeatureVec] = [
        [random.random() for _ in range(3)] for _ in range(4)
    ]

    sheaf_obj, splits = hybrid_step(test_features, test_graph,
                                    epsilon_rbf=0.8,
                                    delta_hoeffding=0.1,
                                    n_obs=50)

    print("Constructed Sheaf restriction maps (sample):")
    for edge in sheaf_obj.edges[:2]:
        src_map, dst_map = sheaf_obj.restriction(edge)
        print(f"Edge {edge}: src_map={src_map}, dst_map={dst_map}")

    print("\nNodes selected for split:", splits)
    print("\nSample sections:")
    for node in list(test_graph.keys())[:2]:
        print(f"Node {node}: section={sheaf_obj.get_section(node)}")