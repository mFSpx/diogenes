# DARWIN HAMMER — match 5174, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s5.py (gen4)
# parent_b: hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s2.py (gen4)
# born: 2026-05-30T00:00:27Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Set, Tuple, List
import numpy as np

# ----------------------------------------------------------------------
# Basic geometric / similarity utilities
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
# Morphology primitives
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
    r = euclidean(src.features, dst.features)
    sim = gaussian_rbf(r, epsilon_feat)

    d_ham = hamming_distance(src.phash, dst.phash)
    ham = gaussian_hamming(d_ham, epsilon_hash)

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
    prob = 1.0 / (1.0 + math.exp(-score))
    return prob


def propagate_sections(
    graph: Dict[Node, Set[Node]],
    node_data: Dict[Node, NodeData],
    steps: int = 5,
    epsilon_feat: float = 1.0,
    epsilon_hash: float = 1.0,
    epsilon_fisher: float = 1e-12,
) -> np.ndarray:
    R = build_restriction_matrix(graph, node_data, epsilon_feat, epsilon_hash)
    N = R.shape[0]

    sections = np.random.rand(N, steps)
    for _ in range(steps):
        sections = np.dot(R, sections)

    pruning_probs = np.array(
        [hybrid_fisher_prune(node_data[i], epsilon_fisher) for i in range(N)]
    )

    return np.multiply(sections, pruning_probs[:, np.newaxis])


def main():
    # Example usage
    graph = {
        0: {1, 2},
        1: {2, 3},
        2: {3, 4},
        3: {4},
        4: set(),
    }

    node_data = {
        0: initialise_node_data((1.0, 2.0, 3.0), Morphology(1.0, 2.0, 3.0, 1.0)),
        1: initialise_node_data((4.0, 5.0, 6.0), Morphology(4.0, 5.0, 6.0, 2.0)),
        2: initialise_node_data((7.0, 8.0, 9.0), Morphology(7.0, 8.0, 9.0, 3.0)),
        3: initialise_node_data((10.0, 11.0, 12.0), Morphology(10.0, 11.0, 12.0, 4.0)),
        4: initialise_node_data((13.0, 14.0, 15.0), Morphology(13.0, 14.0, 15.0, 5.0)),
    }

    sections = propagate_sections(graph, node_data)
    print(sections)


if __name__ == "__main__":
    main()