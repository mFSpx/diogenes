# DARWIN HAMMER — match 3811, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_nlms_hybrid_h_m978_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1731_s1.py (gen5)
# born: 2026-05-29T23:51:51Z

"""Hybrid Kernel-RandomWalk-Bayesian Module
Parent Algorithms:
- hybrid_hybrid_hybrid_hybrid_hybrid_nlms_hybrid_h_m978_s1.py (RBF kernel, Gaussian similarity, Hoeffding bound)
- hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1731_s1.py (Krampus curvature feature, lazy random walk, Bayesian update)

Mathematical Bridge:
Both parents operate on a graph of nodes with feature vectors.
Parent B introduces a curvature scalar κᵢ that is injected as an extra dimension
into a 3‑D coordinate pᵢ = (xᵢ, yᵢ, κᵢ).  
Parent A builds an RBF kernel K_{ij}=exp(-ε²‖fᵢ−fⱼ‖²).

The hybrid algorithm augments each original feature vector with its curvature
value, computes a Gaussian RBF kernel in this augmented space, and then
weights the resulting similarity matrix by a lazy random‑walk prior over the
graph.  The weighted similarity serves as a likelihood which is combined with
the random‑walk prior in a simple Bayesian update, yielding a posterior
similarity matrix that fuses both parent behaviours.
"""

import math
import random
import sys
from pathlib import Path
from typing import Hashable, Sequence, List, Dict, Set, Tuple

import numpy as np

Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Perceptual hash of a list of floats (first 64 values)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound used by Parent A."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Utilities from Parent B (adapted)
# ----------------------------------------------------------------------
def extract_full_features(_: str) -> dict[str, float]:
    """Placeholder feature extractor – returns a fixed dictionary."""
    # In a real system this would parse *text*; here we return a deterministic map.
    return {
        "visceral_ratio": 0.5,
        "tech_ratio": 0.3,
        "legal_osint_ratio": 0.2,
        "ledger_density": 0.1,
        "recursion_score": 0.4,
        "directive_ratio": 0.6,
        "target_density": 0.7,
        "forensic_shield_ratio": 0.8,
        "poetic_entropy": 0.9,
        "dissociative_index": 0.1,
        "wrath_velocity": 0.2,
        "bureaucratic_weaponization_index": 0.3,
        "resource_exhaustion_metric": 0.4,
        "swarm_orchestration_density": 0.5,
        "logic_crucifixion_index": 0.6,
        "conspiracy_grounding_ratio": 0.7,
        "chaotic_good_tax": 0.8,
        "corporate_grit_tension": 0.9,
        "countdown_density": 0.1,
        "asset_structuring_weight": 0.2,
        "pitch_formatting_ratio": 0.3,
        "agent_symmetry_ratio": 0.4,
        "protocol_discipline": 0.5,
        "manic_velocity": 0.6,
    }


def lazy_rw_distribution(adj: Graph, node: Node, alpha: float = 0.5) -> Dict[Node, float]:
    """
    Lazy random walk distribution centred at *node*.
    With probability *alpha* we stay, otherwise we move uniformly to a neighbour.
    Returns a probability distribution over the immediate neighbourhood (including self).
    """
    neighbours = list(adj.get(node, []))
    deg = len(neighbours)
    dist: Dict[Node, float] = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = spread
    return dist


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def curvature_from_features(feat: Dict[str, float]) -> float:
    """
    Compute a scalar curvature κᵢ for a node from its feature dictionary.
    Here we define κᵢ as the standard deviation of the feature values,
    providing a simple measure of local “shape”.
    """
    values = list(feat.values())
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def augment_features(
    raw_features: Dict[Node, FeatureVec],
    curvature_map: Dict[Node, float],
) -> Tuple[Dict[Node, FeatureVec], List[Node]]:
    """
    Append curvature κᵢ as an extra dimension to each feature vector.
    Returns a new feature dict and the ordered node list.
    """
    nodes = list(raw_features.keys())
    augmented: Dict[Node, FeatureVec] = {}
    for n in nodes:
        base = list(raw_features[n])
        augmented[n] = base + [curvature_map.get(n, 0.0)]
    return augmented, nodes


def hybrid_rbf_kernel(
    features: Dict[Node, FeatureVec],
    vram_budget_mb: int,
    epsilon_factor: float = 1.0,
) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute an RBF kernel on curvature‑augmented feature vectors.
    The epsilon scaling follows Parent A (inverse of VRAM budget) and can be
    further tuned by *epsilon_factor*.
    """
    nodes = list(features.keys())
    n = len(nodes)
    epsilon = epsilon_factor / (vram_budget_mb / 1024.0)  # same heuristic as Parent A
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            K[i, j] = gaussian(dist, epsilon)
            K[j, i] = K[i, j]
    return K, nodes


def bayesian_posterior(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Simple element‑wise Bayesian update:
        posterior ∝ prior * likelihood
    The result is normalized so each row sums to 1 (interpreted as a
    conditional distribution over target nodes given a source node).
    """
    unnorm = prior * likelihood
    # Avoid division by zero – rows that sum to zero become uniform.
    row_sums = unnorm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior = np.where(row_sums == 0, 1.0 / unnorm.shape[1], unnorm / row_sums)
    return posterior


def hybrid_similarity_matrix(
    raw_features: Dict[Node, FeatureVec],
    adjacency: Graph,
    vram_budget_mb: int,
    alpha_rw: float = 0.5,
) -> Tuple[np.ndarray, List[Node]]:
    """
    Full hybrid pipeline:
    1. Derive curvature κᵢ from raw feature vectors.
    2. Augment each vector with κᵢ (Parent B bridge).
    3. Build an RBF kernel on the augmented vectors (Parent A kernel).
    4. Construct a lazy‑random‑walk prior matrix P where P_{ij}
       is the probability of moving from i to j in one step.
    5. Perform a Bayesian update: Posterior = Bayes(P, K).
    Returns the posterior similarity matrix and the node ordering.
    """
    # 1. curvature
    curvature_map = {
        n: curvature_from_features(
            {"f{}".format(idx): v for idx, v in enumerate(vec)}
        )
        for n, vec in raw_features.items()
    }

    # 2. augment
    aug_features, nodes = augment_features(raw_features, curvature_map)

    # 3. kernel (likelihood)
    kernel, _ = hybrid_rbf_kernel(aug_features, vram_budget_mb)

    # 4. lazy random walk prior matrix
    n = len(nodes)
    prior = np.zeros((n, n), dtype=np.float64)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    for i, node in enumerate(nodes):
        rw_dist = lazy_rw_distribution(adjacency, node, alpha=alpha_rw)
        for target, prob in rw_dist.items():
            j = node_index.get(target)
            if j is not None:
                prior[i, j] = prob
    # rows that are all zero (isolated nodes) become uniform prior
    zero_rows = prior.sum(axis=1) == 0
    prior[zero_rows] = 1.0 / n

    # 5. Bayesian posterior
    posterior = bayesian_posterior(prior, kernel)
    return posterior, nodes


# ----------------------------------------------------------------------
# Additional Demonstrative Functions
# ----------------------------------------------------------------------
def node_hashes(features: Dict[Node, FeatureVec]) -> Dict[Node, int]:
    """Compute perceptual hashes for all nodes (Parent A utility)."""
    return {n: compute_phash(list(vec)) for n, vec in features.items()}


def hamming_similarity_matrix(hashes: Dict[Node, int]) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a similarity matrix based on inverse Hamming distance.
    Similarity = 1 - (dist / max_bits).
    """
    nodes = list(hashes.keys())
    n = len(nodes)
    max_bits = max(hashes.values()).bit_length() or 1
    S = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dist = hamming_distance(hashes[nodes[i]], hashes[nodes[j]])
            sim = 1.0 - (dist / max_bits)
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes


def should_split(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> bool:
    """
    Decision rule from Parent A using Hoeffding bound.
    Returns True if the observed gain gap is statistically significant.
    """
    eps = hoeffding_bound(r, delta, n)
    gain_gap = best_gain - second_best_gain
    return gain_gap > max(eps, tie_threshold * best_gain)


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic graph with 4 nodes.
    nodes = ["A", "B", "C", "D"]
    # Random 3‑dimensional feature vectors.
    random.seed(42)
    raw_feats: Dict[Node, FeatureVec] = {
        n: [random.random() for _ in range(3)] for n in nodes
    }

    # Simple adjacency (undirected) with a couple of edges.
    adjacency: Graph = {
        "A": {"B", "C"},
        "B": {"A", "D"},
        "C": {"A"},
        "D": {"B"},
    }

    # Run the hybrid similarity pipeline.
    posterior_matrix, order = hybrid_similarity_matrix(
        raw_features=raw_feats,
        adjacency=adjacency,
        vram_budget_mb=512,  # modest budget
        alpha_rw=0.6,
    )

    print("Node order:", order)
    print("Posterior similarity matrix:")
    print(posterior_matrix)

    # Verify that the matrix rows sum to ~1 (probability distribution).
    row_sums = posterior_matrix.sum(axis=1)
    print("Row sums (should be 1.0):", row_sums)

    # Demonstrate hash‑based similarity as a secondary view.
    hashes = node_hashes(raw_feats)
    hamming_sim, _ = hamming_similarity_matrix(hashes)
    print("Hamming‑based similarity matrix:")
    print(hamming_sim)

    # Simple split decision example.
    decision = should_split(best_gain=0.12, second_best_gain=0.08, r=1.0, delta=0.05, n=100)
    print("Should split decision:", decision)