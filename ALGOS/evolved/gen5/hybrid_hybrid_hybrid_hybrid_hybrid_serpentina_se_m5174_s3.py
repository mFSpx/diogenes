# DARWIN HAMMER — match 5174, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s5.py (gen4)
# parent_b: hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s2.py (gen4)
# born: 2026-05-30T00:00:27Z

"""Hybrid Sheaf‑RBF‑Morphology Algorithm
Parents:
- hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s5.py
- hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s2.py

Mathematical bridge:
1. Parent A supplies a Gaussian radial‑basis‑function similarity S_ij based on Euclidean
   distances of feature vectors and a perceptual‑hash Hamming‑distance weight
   g_hamming = exp(‑(ε·d_hamming)²).
2. Parent B supplies morphology‑driven Gaussian‑beam parameters:
   * centre c = sphericity index σ (compactness of the object),
   * width w = flatness index φ (degree of flattening),
   * energy scale R = righting‑time index τ (mass‑weighted recovery priority).

We embed the scalar weight w_ij = S_ij·g_hamming·B_ij as a 1‑D linear restriction map on the
edge (i→j) of a sheaf, where B_ij = gaussian_beam(θ=τ_i, centre=c_i, width=w_i) is the
morphology‑driven beam intensity evaluated at the source node i.  The Fisher
information of that beam furnishes a probabilistic pruning metric for sections of the
sheaf.  The resulting system fuses RBF‑style uncertainty, perceptual similarity,
and sheaf‑cohomology structure with morphology‑parameterised optics."""
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Set, Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric / similarity utilities (Parent A)
# ----------------------------------------------------------------------


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial‑basis‑function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: 1‑bit per entry, thresholded by the mean."""
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


def gaussian_hamming(d: int, epsilon: float = 1.0) -> float:
    """Gaussian weighting of a Hamming distance."""
    return math.exp(-((epsilon * d) ** 2))


# ----------------------------------------------------------------------
# Morphology primitives (Parent B)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Compactness proxy: ratio of geometric mean to maximal edge."""
    geo_mean = (length * width * height) ** (1 / 3)
    max_edge = max(length, width, height)
    return geo_mean / max_edge if max_edge != 0 else 0.0


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness proxy: normalised spread of dimensions."""
    dims = np.array([length, width, height])
    spread = dims.max() - dims.min()
    return spread / dims.max() if dims.max() != 0 else 0.0


def righting_time_index(mass: float, length: float, width: float, height: float) -> float:
    """Energy‑scale factor: mass divided by total linear size."""
    total = length + width + height
    return mass / total if total != 0 else 0.0


def gaussian_beam(theta: float, centre: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - centre) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, centre: float, width: float, eps: float = 1e-12) -> float:
    """Fisher‑information‑like score derived from a Gaussian beam."""
    intensity = max(gaussian_beam(theta, centre, width), eps)
    # derivative of log‑likelihood w.r.t. theta for a Gaussian is (theta‑centre)/width²
    derivative = (theta - centre) / (width ** 2)
    return (derivative ** 2) / intensity


# ----------------------------------------------------------------------
# Data structures merging both parents
# ----------------------------------------------------------------------


Node = int
FeatureVec = Tuple[float, ...]


@dataclass
class NodeData:
    """Container for all information attached to a graph node."""
    features: FeatureVec
    morphology: Morphology
    phash: int = 0  # filled after construction


def initialise_node_data(features: FeatureVec, morph: Morphology) -> NodeData:
    """Create NodeData and compute its perceptual hash."""
    ph = compute_phash(list(features))
    return NodeData(features=features, morphology=morph, phash=ph)


# ----------------------------------------------------------------------
# Hybrid core operations
# ----------------------------------------------------------------------


def hybrid_morph_beam(node: NodeData) -> float:
    """
    Morphology‑driven Gaussian‑beam intensity.

    centre  = sphericity_index(length, width, height)
    width   = flatness_index(length, width, height)
    theta   = righting_time_index(mass, length, width, height)
    """
    m = node.morphology
    centre = sphericity_index(m.length, m.width, m.height)
    width = flatness_index(m.length, m.width, m.height)
    theta = righting_time_index(m.mass, m.length, m.width, m.height)
    return gaussian_beam(theta, centre, width)


def hybrid_edge_weight(
    src: NodeData,
    dst: NodeData,
    epsilon_feat: float = 1.0,
    epsilon_hash: float = 1.0,
) -> float:
    """
    Combined restriction weight for the directed edge src → dst.

    w = RBF_similarity * Gaussian_hamming * Morphology_beam
    """
    # 1. RBF similarity on feature space
    r = euclidean(src.features, dst.features)
    sim = gaussian_rbf(r, epsilon_feat)

    # 2. Hamming‑based Gaussian weight
    d_ham = hamming_distance(src.phash, dst.phash)
    ham = gaussian_hamming(d_ham, epsilon_hash)

    # 3. Morphology‑driven beam evaluated at the source node
    beam = hybrid_morph_beam(src)

    return sim * ham * beam


def build_restriction_matrix(
    graph: Dict[Node, Set[Node]],
    node_data: Dict[Node, NodeData],
    epsilon_feat: float = 1.0,
    epsilon_hash: float = 1.0,
) -> np.ndarray:
    """
    Construct the sheaf restriction matrix R of shape (N, N) where
    R[i, j] = weight of edge i→j (zero if no edge).

    The matrix encodes the linear restriction maps of the hybrid sheaf.
    """
    nodes = sorted(graph.keys())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    R = np.zeros((N, N), dtype=float)

    for i in nodes:
        for j in graph[i]:
            w = hybrid_edge_weight(
                src=node_data[i],
                dst=node_data[j],
                epsilon_feat=epsilon_feat,
                epsilon_hash=epsilon_hash,
            )
            R[idx[i], idx[j]] = w
    return R


def hybrid_fisher_prune(
    node: NodeData,
    epsilon_fisher: float = 1e-12,
) -> float:
    """
    Compute a pruning probability for a node based on its morphology‑driven
    Fisher information.  The probability lies in (0, 1]; larger values indicate
    higher confidence (less pruning).
    """
    m = node.morphology
    centre = sphericity_index(m.length, m.width, m.height)
    width = flatness_index(m.length, m.width, m.height)
    theta = righting_time_index(m.mass, m.length, m.width, m.height)
    score = fisher_score(theta, centre, width, eps=epsilon_fisher)
    # Map the Fisher score to a probability via a sigmoid‑like transform
    prob = 1.0 / (1.0 + math.exp(-score))
    return prob


def propagate_sections(
    graph: Dict[Node, Set[Node]],
    node_data: Dict[Node, NodeData],
    steps: int = 5,
    epsilon_feat: float = 1.0,
    epsilon_hash: float = 1.0,
) -> Dict[Node, float]:
    """
    Simple sheaf‑section propagation.

    1. Initialise a scalar field `x` on nodes (random in [0,1]).
    2. At each step, replace x_i by the weighted average of incoming neighbours
       using the restriction matrix R (i.e. x ← Rᵀ·x).
    3. After the final step, apply morphology‑driven Fisher pruning to each node.
    Returns the final field as a dict {node: value}.
    """
    R = build_restriction_matrix(graph, node_data, epsilon_feat, epsilon_hash)
    # Normalise rows to obtain stochastic matrix for averaging
    row_sums = R.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    R_norm = R / row_sums

    # Random initial field
    x = np.random.rand(R.shape[0])

    for _ in range(steps):
        x = R_norm.T @ x  # aggregate from predecessors

    # Apply pruning probabilities
    nodes = sorted(graph.keys())
    pruned = {}
    for i, n in enumerate(nodes):
        prob = hybrid_fisher_prune(node_data[n])
        pruned[n] = x[i] * prob
    return pruned


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny undirected graph (treated as directed both ways)
    G: Dict[Node, Set[Node]] = {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1},
    }

    # Random feature vectors (3‑dimensional) and random morphologies
    random.seed(42)
    node_data: Dict[Node, NodeData] = {}
    for n in G:
        feats = tuple(random.random() for _ in range(3))
        morph = Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(1.0, 5.0),
        )
        node_data[n] = initialise_node_data(feats, morph)

    # Demonstrate core functions
    print("Hybrid morph‑beam intensities:")
    for n, nd in node_data.items():
        print(f" node {n}: {hybrid_morph_beam(nd):.4f}")

    print("\nEdge weights (restriction matrix):")
    R = build_restriction_matrix(G, node_data)
    print(R)

    print("\nPropagation result (pruned values):")
    result = propagate_sections(G, node_data, steps=10)
    for n, val in result.items():
        print(f" node {n}: {val:.4f}")