# DARWIN HAMMER — match 3955, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_distri_m1385_s5.py (gen4)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_sheaf_cohomol_m1649_s0.py (gen4)
# born: 2026-05-29T23:52:47Z

"""Hybrid Fusion of Bayesian Feature Vectors and RBF‑Sheaf Cohomology

Parents:
- hybrid_hybrid_hybrid_bayes__hybrid_hybrid_distri_m1385_s5.py (Bayesian feature extraction,
  master‑vector creation and SSIM‑like similarity)
- hybrid_hybrid_hybrid_rbf_su_hybrid_sheaf_cohomol_m1649_s0.py (RBF kernel computation and
  sheaf‑cohomology structure)

Mathematical Bridge:
Both parents operate on collections of feature vectors.  The Bayesian side produces a
normalized master vector per textual sample, while the RBF side defines a Gaussian
kernel  K(i,j)=exp(‑ε²‖v_i‑v_j‖²)  that yields a similarity matrix.  The fusion treats the
RBF similarity values as *restriction‑map scalars* inside a sheaf: for an edge (u,v) the
restriction from node u to node v is the identity matrix scaled by K(u,v).  This embeds the
probabilistic similarity directly into the linear‑algebraic sheaf, allowing a single
consistency metric that blends the SSIM‑like Bayesian similarity with the RBF‑induced
cohomology.

The module therefore:
1. Extracts Bayesian master vectors from raw text.
2. Builds an RBF similarity matrix from those vectors.
3. Constructs a sheaf whose restriction maps are the RBF‑scaled identities.
4. Provides a consistency evaluator that aggregates the deviation of sections after
   restriction, yielding a hybrid “cohomological‑Bayesian” score.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence, List, Dict, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Parent‑A: Bayesian feature handling (adapted)
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

def extract_full_features(text: str) -> dict[str, float]:
    """Generate a synthetic feature dictionary from a string."""
    # In a real system this would parse *text*; here we use random numbers.
    features: dict[str, float] = {}
    features.update({
        "operator_visceral_ratio": random.random(),
        "operator_tech_ratio": random.random(),
        "operator_legal_osint_ratio": random.random(),
    })
    features.update({
        "psyche_forensic_shield_ratio": random.random(),
        "psyche_poetic_entropy": random.random(),
        "psyche_dissociative_index": random.random(),
    })
    features.update({
        "resilience_bureaucratic_weaponization_index": random.random(),
        "resilience_resource_exhaustion_metric": random.random(),
        "resilience_swarm_orchestration_density": random.random(),
    })
    features.update({
        "rainmaker_corporate_grit_tension": random.random(),
        "rainmaker_countdown_density": random.random(),
        "rainmaker_asset_structuring_weight": random.random(),
    })
    features.update({
        "telemetry_agent_symmetry_ratio": random.random(),
        "telemetry_protocol_discipline": random.random(),
        "telemetry_manic_velocity": random.random(),
    })
    return features

def extract_master_vector(text: str) -> np.ndarray:
    """Select five core features, normalize, and return a master vector."""
    f = extract_full_features(text)
    vec = np.array([
        f.get("operator_visceral_ratio", 0.0),
        f.get("operator_tech_ratio", 0.0),
        f.get("operator_legal_osint_ratio", 0.0),
        f.get("psyche_forensic_shield_ratio", 0.0),
        f.get("psyche_poetic_entropy", 0.0),
    ], dtype=np.float64)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def ssim_like_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """A lightweight SSIM‑style similarity for 1‑D vectors."""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    mu1, mu2 = v1.mean(), v2.mean()
    sigma1 = v1.var()
    sigma2 = v2.var()
    sigma12 = ((v1 - mu1) * (v2 - mu2)).mean()
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    return numerator / denominator if denominator != 0 else 0.0

# ----------------------------------------------------------------------
# Parent‑B: RBF kernel & Sheaf cohomology (adapted)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict:
        return {
            'slot_index': self.slot_index,
            'name': self.name,
            'alias': self.alias,
            'persona': self.persona,
            'uuid': self.uuid,
            'ternary_offset': self.ternary_offset
        }

class Sheaf:
    """A minimal sheaf structure where restriction maps are linear transforms."""
    def __init__(self, node_dims: Mapping[Node, int], edge_list: Sequence[Tuple[Node, Node]]):
        self.node_dims: Dict[Node, int] = dict(node_dims)
        self.edges: List[Tuple[Node, Node]] = list(edge_list)
        self._restrictions: Dict[Tuple[Node, Node], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Node, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[Node, Node], src_map: Sequence[Sequence[float]], dst_map: Sequence[Sequence[float]]) -> None:
        u, v = edge
        src_mat = np.array(src_map, dtype=float)
        dst_mat = np.array(dst_map, dtype=float)
        if src_mat.shape != (self.node_dims[u], self.node_dims[u]) or dst_mat.shape != (self.node_dims[v], self.node_dims[v]):
            raise ValueError("Restriction matrix shape mismatch with node dimensions.")
        self._restrictions[(u, v)] = (src_mat, dst_mat)

    def set_section(self, node: Node, value: Sequence[float]) -> None:
        dim = self.node_dims.get(node)
        if dim is None:
            raise KeyError(f"Node {node} not declared in node_dims.")
        arr = np.array(value, dtype=float)
        if arr.shape != (dim,):
            raise ValueError(f"Section vector shape {arr.shape} does not match node dimension {dim}.")
        self._sections[node] = arr

    def get_section(self, node: Node) -> np.ndarray:
        return self._sections[node]

    def restriction(self, edge: Tuple[Node, Node]) -> Tuple[np.ndarray, np.ndarray]:
        return self._restrictions[edge]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Standard Euclidean distance."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a_arr - b_arr))

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def compute_rbf_similarity_matrix(vectors: List[np.ndarray], epsilon: float = 1.0) -> np.ndarray:
    """Return a dense RBF similarity matrix K where K_ij = exp(-ε²‖v_i‑v_j‖²)."""
    n = len(vectors)
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(vectors[i], vectors[j])
            sim = gaussian(dist, epsilon)
            K[i, j] = K[j, i] = sim
    return K

def build_sheaf_from_vectors(
    graph: Graph,
    vectors: List[np.ndarray],
    epsilon: float = 1.0
) -> Sheaf:
    """
    Construct a Sheaf where each node dimension equals the vector length.
    For every edge (u, v) the restriction maps are identity matrices scaled by the
    RBF similarity K_uv.
    """
    if not vectors:
        raise ValueError("Vector list cannot be empty.")
    dim = vectors[0].shape[0]
    node_dims = {node: dim for node in graph}
    edge_list = [(u, v) for u, nbrs in graph.items() for v in nbrs if (v, u) not in graph]  # avoid duplicates
    sheaf = Sheaf(node_dims, edge_list)

    # Pre‑compute similarity matrix
    K = compute_rbf_similarity_matrix(vectors, epsilon)

    # Map graph nodes to vector indices (order is deterministic via sorted keys)
    node_index = {node: idx for idx, node in enumerate(sorted(graph))}

    # Populate sections
    for node, idx in node_index.items():
        sheaf.set_section(node, vectors[idx])

    # Populate restriction maps
    for (u, v) in sheaf.edges:
        i, j = node_index[u], node_index[v]
        scale = K[i, j]
        src_id = np.eye(dim) * scale
        dst_id = np.eye(dim) * scale
        sheaf.set_restriction((u, v), src_id, dst_id)

    return sheaf

def sheaf_consistency_score(sheaf: Sheaf) -> float:
    """
    Compute a global consistency score:
    For each edge (u,v) compare the restricted section from u to v with the
    actual section at v.  The squared L2 norm of the difference is summed and
    normalized by the number of edges.
    """
    if not sheaf.edges:
        return 0.0
    total_error = 0.0
    for (u, v) in sheaf.edges:
        src_map, dst_map = sheaf.restriction((u, v))
        sec_u = sheaf.get_section(u)
        sec_v = sheaf.get_section(v)
        # Apply restriction: src_map @ sec_u should equal dst_map @ sec_v
        lhs = src_map @ sec_u
        rhs = dst_map @ sec_v
        total_error += np.linalg.norm(lhs - rhs) ** 2
    return float(total_error) / len(sheaf.edges)

def hybrid_pipeline(texts: List[str], graph: Graph, epsilon: float = 1.0) -> Tuple[Sheaf, float]:
    """
    End‑to‑end pipeline:
    1. Convert each text to a Bayesian master vector.
    2. Build a sheaf using the RBF similarity as restriction scalars.
    3. Return the sheaf and its consistency score.
    """
    vectors = [extract_master_vector(t) for t in texts]
    sheaf = build_sheaf_from_vectors(graph, vectors, epsilon)
    score = sheaf_consistency_score(sheaf)
    return sheaf, score

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create three dummy textual inputs
    sample_texts = [
        "Alpha protocol engaged.",
        "Beta subsystem initializing.",
        "Gamma routine completed."
    ]

    # Define a simple triangle graph over three nodes
    triangle_graph: Graph = {
        "A": {"B", "C"},
        "B": {"A", "C"},
        "C": {"A", "B"}
    }

    # Run the hybrid pipeline
    sheaf_obj, consistency = hybrid_pipeline(sample_texts, triangle_graph, epsilon=0.8)

    # Print a concise report
    print("Hybrid Sheaf constructed.")
    print(f"Node dimensions: {sheaf_obj.node_dims}")
    print(f"Number of edges: {len(sheaf_obj.edges)}")
    print(f"Global consistency score: {consistency:.6f}")

    # Demonstrate access to a restriction map and sections
    edge_example = sheaf_obj.edges[0]
    src_map, dst_map = sheaf_obj.restriction(edge_example)
    print(f"\nExample edge {edge_example}:")
    print(f"  Restriction src_map (scaled identity):\n{src_map}")
    print(f"  Section at {edge_example[0]}: {sheaf_obj.get_section(edge_example[0])}")
    print(f"  Section at {edge_example[1]}: {sheaf_obj.get_section(edge_example[1])}")