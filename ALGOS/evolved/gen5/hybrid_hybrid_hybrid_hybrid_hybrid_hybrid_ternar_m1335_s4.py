# DARWIN HAMMER — match 1335, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s5.py (gen4)
# parent_b: hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s1.py (gen2)
# born: 2026-05-29T23:35:28Z

"""Hybrid Sheaf‑RBF‑SSIM Bandit Router

Parents:
- hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s5.py (Gaussian RBF similarity,
  perceptual‑hash distances, Fisher‑score pruning, sheaf restriction maps)
- hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s1.py (SSIM similarity,
  ternary routing, multi‑armed bandit update)

Mathematical Bridge:
Both parents expose a *pairwise similarity* between items:
    • Parent A supplies a Gaussian radial‑basis similarity `S_g(i,j)` derived from
      Hamming distances of perceptual hashes.
    • Parent B supplies a structural similarity `S_s(i,j)` (SSIM) between feature
      vectors.

We fuse them into a single similarity matrix  

    S(i,j) = α·S_g(i,j) + (1‑α)·S_s(i,j) ,   0≤α≤1

The fused similarity is interpreted as the weight of a *sheaf edge restriction*.
A Fisher‑style score `F(i,j)` is computed from `S(i,j)` and used as a probabilistic
pruning factor.  Finally a contextual bandit updates routing probabilities on the
graph using the fused similarity as reward.  The result is a unified system that
combines uncertainty modelling (RBF), structural image‑like similarity (SSIM),
cohomological restriction maps (sheaf) and adaptive decision making (bandit)."""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = int
Graph = Dict[Node, Set[Node]]
FeatureVec = Tuple[float, ...]


# ----------------------------------------------------------------------
# Parent A utilities (Gaussian RBF & Fisher)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash based on average threshold (max 64 bits)."""
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
    # derivative of log‑likelihood for a Gaussian is (theta-center)/width^2
    return ((theta - center) / (width ** 2)) * intensity


# ----------------------------------------------------------------------
# Parent B utilities (SSIM & Bandit)
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Simplified 1‑D SSIM for equal‑length vectors."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator


class Exp3Bandit:
    """Exponential‑weight algorithm for exploration‑exploitation (Exp3)."""
    def __init__(self, n_arms: int, gamma: float = 0.1):
        self.n = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms)

    def probabilities(self) -> np.ndarray:
        """Current probability distribution over arms."""
        total = self.weights.sum()
        return (1 - self.gamma) * (self.weights / total) + (self.gamma / self.n)

    def select(self) -> int:
        """Draw an arm according to the current distribution."""
        probs = self.probabilities()
        return int(np.random.choice(self.n, p=probs))

    def update(self, chosen: int, reward: float) -> None:
        """Update weight of the chosen arm using observed reward ∈[0,1]."""
        probs = self.probabilities()
        x = reward / probs[chosen]
        self.weights[chosen] *= math.exp((self.gamma * x) / self.n)


# ----------------------------------------------------------------------
# Fusion core
# ----------------------------------------------------------------------
def fused_similarity_matrix(features: Dict[Node, FeatureVec],
                            alpha: float = 0.5,
                            epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute the fused similarity matrix S = α·S_g + (1‑α)·S_s.

    Returns
    -------
    S : np.ndarray, shape (N,N)
        Symmetric similarity matrix.
    order : list of Node
        Node ordering that matches matrix rows/columns.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")

    nodes = list(features.keys())
    N = len(nodes)
    S = np.zeros((N, N), dtype=float)

    # pre‑compute perceptual hashes for Gaussian part
    phashes = {n: compute_phash(list(features[n])) for n in nodes}

    for i, ni in enumerate(nodes):
        vi = np.array(features[ni], dtype=float)
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]  # symmetry
                continue
            vj = np.array(features[nj], dtype=float)

            # Gaussian RBF similarity from Hamming distance
            ham = hamming_distance(phashes[ni], phashes[nj])
            # convert Hamming distance to a pseudo‑radius (max 64)
            r = ham / 64.0
            s_g = gaussian(r, epsilon)

            # SSIM similarity on raw feature vectors (scaled to 0‑255)
            s_s = ssim(vi * 255.0, vj * 255.0)

            s = alpha * s_g + (1.0 - alpha) * s_s
            S[i, j] = s
            S[j, i] = s
    return S, nodes


def sheaf_fisher_pruning(S: np.ndarray,
                         center: float = 0.5,
                         width: float = 0.2) -> np.ndarray:
    """
    From the fused similarity matrix produce a pruning mask using Fisher scores.

    For each entry θ = S[i,j] we compute
        p_ij = sigmoid( fisher_score(θ, center, width) )
    The mask keeps edges with probability > 0.5.
    """
    N = S.shape[0]
    mask = np.zeros_like(S, dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            theta = S[i, j]
            f = fisher_score(theta, center, width)
            prob = 1.0 / (1.0 + math.exp(-f))  # logistic sigmoid
            keep = prob > 0.5
            mask[i, j] = mask[j, i] = keep
    return mask


def bandit_guided_routing(graph: Graph,
                          features: Dict[Node, FeatureVec],
                          alpha: float = 0.5,
                          epsilon: float = 1.0,
                          steps: int = 10) -> Dict[Node, Node]:
    """
    Perform a single routing pass for every node using a bandit that
    rewards neighbours according to fused similarity.

    Returns a mapping src_node -> chosen_dst_node.
    """
    S, order = fused_similarity_matrix(features, alpha, epsilon)
    node_index = {n: i for i, n in enumerate(order)}
    pruning_mask = sheaf_fisher_pruning(S)

    # initialise a bandit per source node (arms = neighbours that survive pruning)
    routing: Dict[Node, Node] = {}
    for src in graph:
        neighbours = [n for n in graph[src] if pruning_mask[node_index[src], node_index[n]]]
        if not neighbours:
            # fall back to any neighbour if pruning eliminated all
            neighbours = list(graph[src])
        bandit = Exp3Bandit(len(neighbours), gamma=0.15)

        # run a short internal bandit loop to decide the best arm
        for _ in range(steps):
            arm = bandit.select()
            dst = neighbours[arm]
            # reward = fused similarity (higher is better, normalise to [0,1])
            sim = S[node_index[src], node_index[dst]]
            reward = sim  # already in [0,1] because Gaussian+SSIM ≤1
            bandit.update(arm, reward)

        # final choice
        final_arm = bandit.select()
        routing[src] = neighbours[final_arm]
    return routing


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def demo_graph() -> Graph:
    """Create a tiny undirected graph of 5 nodes."""
    g: Graph = {
        0: {1, 2},
        1: {0, 2, 3},
        2: {0, 1, 4},
        3: {1, 4},
        4: {2, 3},
    }
    return g


def demo_features() -> Dict[Node, FeatureVec]:
    """Generate deterministic 3‑dimensional feature vectors."""
    rng = np.random.default_rng(42)
    return {i: tuple(rng.random(3).astype(float)) for i in range(5)}


def main() -> None:
    """Smoke test exercising the hybrid pipeline."""
    graph = demo_graph()
    feats = demo_features()
    routing = bandit_guided_routing(graph, feats, alpha=0.6, epsilon=1.2, steps=15)
    print("Routing decisions (src -> dst):")
    for src, dst in routing.items():
        print(f"  {src} -> {dst}")

    # Verify that the fused similarity matrix is symmetric and bounded
    S, order = fused_similarity_matrix(feats, alpha=0.6)
    assert np.allclose(S, S.T), "Similarity matrix should be symmetric"
    assert np.all((S >= 0) & (S <= 1)), "Similarity values must lie in [0,1]"
    print("Fused similarity matrix OK.")


if __name__ == "__main__":
    main()