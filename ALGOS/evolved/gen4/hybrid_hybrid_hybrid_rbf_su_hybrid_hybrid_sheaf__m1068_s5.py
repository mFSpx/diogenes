# DARWIN HAMMER — match 1068, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_fisher_m317_s0.py (gen3)
# parent_b: hybrid_hybrid_sheaf_cohomol_hybrid_ternary_lens__m23_s0.py (gen2)
# born: 2026-05-29T23:32:46Z

"""Hybrid Sheaf-RBF Algorithm
Parents:
- hybrid_hybrid_rbf_surrogate_hybrid_hybrid_fisher_m317_s0.py (Gaussian similarity & Fisher scoring)
- hybrid_hybrid_sheaf_cohomol_hybrid_ternary_lens__m23_s0.py (Sheaf cohomology with restriction maps)

Mathematical Bridge:
The similarity matrix S derived from perceptual‑hash Hamming distances (Parent A) provides a scalar weight for every pair of nodes.
We embed these scalar weights as 1‑dimensional linear restriction maps on the edges of a sheaf (Parent B). 
Thus each edge restriction is a Gaussian‑scaled map, and the Fisher score computed from the same Gaussian
parameters supplies a probabilistic pruning metric for the sheaf sections.  The resulting system fuses
RBF‑style uncertainty modelling with sheaf cohomology structure."""
import math
import random
import sys
from pathlib import Path
import numpy as np

Node = int
Graph = dict[Node, set[Node]]
FeatureVec = tuple[float, float]

# ---------- Parent A utilities ----------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: list[float]) -> int:
    """Simple perceptual hash based on average threshold."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Bitwise Hamming distance."""
    return (a ^ b).bit_count()

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam used for pruning probabilities."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher‑information‑like score derived from a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    # derivative of Gaussian w.r.t. theta (up to a constant)
    return intensity * (-(theta - center) / (width ** 2))

def similarity_matrix(features: dict[Node, FeatureVec]) -> tuple[np.ndarray, list[Node]]:
    """
    Build a symmetric similarity matrix S where
    S_ij = 1 - (Hamming distance of perceptual hashes) / 64.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(list(features[ni]))
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = compute_phash(list(features[nj]))
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes

# ---------- Parent B sheaf definition ----------
class Sheaf:
    """Minimal sheaf structure with scalar node spaces."""
    def __init__(self, node_dims: dict[Node, int], edge_list: list[tuple[Node, Node]]):
        self.node_dims = dict(node_dims)          # dimension of each node space
        self.edges = list(edge_list)              # list of (u, v)
        self._restrictions: dict[tuple[Node, Node], tuple[np.ndarray, np.ndarray]] = {}
        self._sections: dict[Node, np.ndarray] = {}

    def set_restriction(self, edge: tuple[Node, Node], src_map: np.ndarray, dst_map: np.ndarray):
        """Define linear restriction maps for an edge."""
        u, v = edge
        if src_map.shape != (self.node_dims[u], self.node_dims[v]):
            raise ValueError("src_map shape mismatch")
        if dst_map.shape != (self.node_dims[v], self.node_dims[u]):
            raise ValueError("dst_map shape mismatch")
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node: Node, value: np.ndarray):
        """Assign a section (vector) to a node."""
        if value.shape != (self.node_dims[node],):
            raise ValueError("section shape mismatch")
        self._sections[node] = value

    def get_section(self, node: Node) -> np.ndarray | None:
        return self._sections.get(node)

    def edge_restriction(self, edge: tuple[Node, Node]) -> tuple[np.ndarray, np.ndarray]:
        return self._restrictions[edge]

# ---------- Hybrid Functions ----------
def build_sheaf_from_graph(
    graph: Graph,
    features: dict[Node, FeatureVec],
    S: np.ndarray,
    node_order: list[Node],
) -> Sheaf:
    """
    Construct a Sheaf where each node carries a 1‑dimensional scalar space.
    Edge restriction maps are 1×1 matrices whose single entry equals the
    Gaussian‑scaled similarity weight between the incident nodes.
    """
    # All node spaces are 1‑dimensional
    node_dims = {n: 1 for n in graph}
    edge_list = [(u, v) for u, nbrs in graph.items() for v in nbrs if u < v]

    sheaf = Sheaf(node_dims, edge_list)

    # Map node index for quick lookup in S
    idx = {n: i for i, n in enumerate(node_order)}

    for (u, v) in edge_list:
        i, j = idx[u], idx[v]
        sim = S[i, j]                     # similarity in [0,1]
        # Use Gaussian beam to turn similarity into a weight centered at 0.5
        weight = gaussian_beam(sim, center=0.5, width=0.2)
        src_map = np.array([[weight]], dtype=float)   # 1×1
        dst_map = np.array([[weight]], dtype=float)   # symmetric
        sheaf.set_restriction((u, v), src_map, dst_map)
    return sheaf

def assign_feature_based_sections(sheaf: Sheaf, features: dict[Node, FeatureVec]) -> None:
    """
    For each node, compute a scalar section value using a Gaussian RBF
    centred at the origin on the Euclidean norm of its feature vector.
    """
    for node, vec in features.items():
        norm = euclidean(vec, (0.0, 0.0))
        value = gaussian(norm, epsilon=1.0)   # scalar in (0,1]
        sheaf.set_section(node, np.array([value], dtype=float))

def compute_edge_fisher_scores(
    sheaf: Sheaf,
    S: np.ndarray,
    node_order: list[Node],
    center: float = 0.5,
    width: float = 0.2,
) -> dict[tuple[Node, Node], float]:
    """
    Apply the Fisher‑score formula to each edge's similarity value.
    The result can be interpreted as a confidence weight for that edge.
    """
    idx = {n: i for i, n in enumerate(node_order)}
    scores: dict[tuple[Node, Node], float] = {}
    for (u, v) in sheaf.edges:
        i, j = idx[u], idx[v]
        sim = S[i, j]
        scores[(u, v)] = fisher_score(sim, center, width)
    return scores

def prune_sheaf_sections(
    sheaf: Sheaf,
    prune_prob_center: float = 0.5,
    prune_prob_width: float = 0.15,
    rng: random.Random | None = None,
) -> None:
    """
    Probabilistically remove node sections.
    The keep‑probability for a node is a Gaussian beam evaluated on the
    current section value; low values are more likely to be pruned.
    """
    if rng is None:
        rng = random
    to_delete = []
    for node, sec in sheaf._sections.items():
        val = float(sec[0])                     # scalar section
        keep_prob = gaussian_beam(val, prune_prob_center, prune_prob_width)
        if rng.random() > keep_prob:
            to_delete.append(node)
    for node in to_delete:
        del sheaf._sections[node]

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Build a small random graph
    num_nodes = 6
    rng = random.Random(42)
    graph: Graph = {i: set() for i in range(num_nodes)}
    # Random edges (undirected)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < 0.4:
                graph[i].add(j)
                graph[j].add(i)

    # Random 2‑D feature vectors per node
    features: dict[Node, FeatureVec] = {
        i: (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for i in range(num_nodes)
    }

    # Parent A: similarity matrix
    S, node_order = similarity_matrix(features)

    # Hybrid construction
    sheaf = build_sheaf_from_graph(graph, features, S, node_order)

    # Assign sections based on features (RBF)
    assign_feature_based_sections(sheaf, features)

    # Compute Fisher scores for diagnostic output
    scores = compute_edge_fisher_scores(sheaf, S, node_order)

    # Prune sections using Gaussian‑beam probability
    prune_sheaf_sections(sheaf)

    # Simple verification prints
    print("Graph adjacency:")
    for n, nbrs in graph.items():
        print(f" {n}: {sorted(nbrs)}")
    print("\nSimilarity matrix (rounded):")
    print(np.round(S, 3))
    print("\nEdge Fisher scores (rounded):")
    for e, sc in scores.items():
        print(f" {e}: {round(sc, 4)}")
    print("\nRemaining sections after pruning:")
    for n, sec in sheaf._sections.items():
        print(f" Node {n}: {sec}")
    print("\nSmoke test completed without error.")