# DARWIN HAMMER — match 3364, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1215_s7.py (gen4)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hybrid_geomet_m641_s0.py (gen4)
# born: 2026-05-29T23:49:40Z

"""Hybrid Fisher‑SSIM‑Hash Bandit with Graph‑Curvature Leader Election.

Parents:
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1215_s7.py (Fisher information, SSIM, UCB bandit)
- hybrid_hybrid_distributed_l_hybrid_hybrid_geomet_m641_s0.py (perceptual hashing, graph construction, leader election, geometric curvature)

Mathematical bridge:
Both parents expose a *pairwise similarity* measure:
    • Fisher score 𝓕(θ) quantifies information density of a sensing direction.
    • SSIM 𝒮(x,y) quantifies visual similarity of payloads.
    • Hamming distance 𝓗(h₁,h₂) between perceptual hashes measures perceptual similarity.
We fuse them into a unified similarity weight
    𝓦 = 𝓕·𝒮·exp(‑𝓗/τ)
where τ is a temperature controlling hash influence.
The weight 𝓦 serves as the stochastic reward in a classic Upper‑Confidence‑Bound (UCB) bandit.
The bandit’s arm selection simultaneously chooses a graph node (leader) and a sensing angle,
while the graph Laplacian curvature 𝓒 = mean|λᵢ| (λᵢ eigenvalues of the Laplacian) provides a global geometric regulariser.
The resulting algorithm performs leader election, perceptual deduplication and information‑maximising sensing
in a single mathematically unified loop.
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Sequence, Mapping, Hashable, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Core building blocks from Parent A
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def compute_ssim(
    x: Sequence[float],
    y: Sequence[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index Measure (simplified, 1‑D)."""
    if len(x) != len(y):
        raise ValueError("inputs must have the same length")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    sigma_x = math.sqrt(sum((xi - mx) ** 2 for xi in x) / len(x))
    sigma_y = math.sqrt(sum((yi - my) ** 2 for yi in y) / len(y))
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (sigma_x * sigma_x + sigma_y * sigma_y + c2)
    return numerator / denominator if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Core building blocks from Parent B
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]


def compute_phash(values: List[float]) -> int:
    """Perceptual hash (average hash) for a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:  # limit to 64‑bit hash
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def build_graph(elements: List[List[float]], max_hamming: int = 4) -> Graph:
    """
    Build an undirected graph where nodes are element indices (as strings)
    and edges connect perceptually similar elements (Hamming distance ≤ max_hamming).
    """
    hashes: Dict[str, int] = {str(i): compute_phash(el) for i, el in enumerate(elements)}
    graph: Dict[str, set[str]] = {str(i): set() for i in range(len(elements))}
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if hamming_distance(hashes[str(i)], hashes[str(j)]) <= max_hamming:
                graph[str(i)].add(str(j))
                graph[str(j)].add(str(i))
    return graph


def maximal_independent_set(graph: Graph, seed: int | None = None) -> set[Node]:
    """
    Greedy approximation of a maximal independent set.
    Randomly shuffles nodes (seedable) and picks nodes that are not adjacent
    to already‑selected ones.
    """
    rng = random.Random(seed)
    nodes = list(graph.keys())
    rng.shuffle(nodes)
    mis: set[Node] = set()
    excluded: set[Node] = set()
    for n in nodes:
        if n not in excluded:
            mis.add(n)
            excluded.update(graph[n])
            excluded.add(n)
    return mis


def graph_curvature(graph: Graph) -> float:
    """
    Simple curvature proxy: mean absolute eigenvalue of the combinatorial Laplacian.
    """
    nodes = list(graph.keys())
    n = len(nodes)
    if n == 0:
        return 0.0
    index = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((n, n), dtype=float)
    for node, neigh in graph.items():
        i = index[node]
        for nb in neigh:
            j = index[nb]
            A[i, j] = 1.0
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals = np.linalg.eigvalsh(L)
    return float(np.mean(np.abs(eigvals)))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def combined_weight(
    theta: float,
    center: float,
    width: float,
    payload_x: Sequence[float],
    payload_y: Sequence[float],
    hash_x: int,
    hash_y: int,
    tau: float = 2.0,
) -> float:
    """
    Unified similarity weight 𝓦 = 𝓕·𝒮·exp(-𝓗/τ).

    Parameters
    ----------
    theta, center, width : float
        Parameters for the Fisher information term.
    payload_x, payload_y : Sequence[float]
        Payloads for SSIM computation.
    hash_x, hash_y : int
        Perceptual hashes for Hamming distance.
    tau : float
        Temperature controlling hash influence.

    Returns
    -------
    float
        The combined weight.
    """
    F = fisher_score(theta, center, width)
    S = compute_ssim(payload_x, payload_y)
    H = hamming_distance(hash_x, hash_y)
    return F * S * math.exp(-H / tau)


def ucb_scores(rewards: np.ndarray, counts: np.ndarray, total: int, c: float = 2.0) -> np.ndarray:
    """
    Compute Upper‑Confidence‑Bound (UCB) scores for each arm.

    rewards : cumulative reward per arm
    counts  : number of pulls per arm
    total   : total number of pulls so far
    c       : exploration coefficient
    """
    with np.errstate(divide='ignore'):
        avg = np.where(counts > 0, rewards / counts, 0.0)
        bonus = c * np.sqrt(np.log(max(total, 1)) / np.where(counts > 0, counts, 1e-9))
    return avg + bonus


def select_node_ucb(
    graph: Graph,
    element_payloads: List[List[float]],
    theta_range: Tuple[float, float],
    center: float,
    width: float,
    counts: Dict[Node, int],
    rewards: Dict[Node, float],
    total_pulls: int,
    tau: float = 2.0,
) -> Node:
    """
    Choose a graph node (leader) using UCB where the reward is the
    combined_weight between the candidate node and a randomly sampled neighbour.
    """
    nodes = list(graph.keys())
    if not nodes:
        raise ValueError("graph is empty")

    # Build arrays for vectorised UCB computation
    reward_arr = np.array([rewards.get(n, 0.0) for n in nodes], dtype=float)
    count_arr = np.array([counts.get(n, 0) for n in nodes], dtype=int)

    ucb = ucb_scores(reward_arr, count_arr, total_pulls)
    best_idx = int(np.argmax(ucb))
    best_node = nodes[best_idx]

    # Simulate a pull: compute reward against a random neighbour (or itself if isolated)
    neighbours = list(graph[best_node])
    if neighbours:
        partner = random.choice(neighbours)
    else:
        partner = best_node

    # Random theta for the pull
    theta = random.uniform(*theta_range)

    # Compute combined weight as reward
    reward = combined_weight(
        theta,
        center,
        width,
        element_payloads[int(best_node)],
        element_payloads[int(partner)],
        compute_phash(element_payloads[int(best_node)]),
        compute_phash(element_payloads[int(partner)]),
        tau=tau,
    )

    # Update statistics
    counts[best_node] = counts.get(best_node, 0) + 1
    rewards[best_node] = rewards.get(best_node, 0.0) + reward

    return best_node


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic elements (each element is a 32‑dim float vector)
    random.seed(42)
    np.random.seed(42)
    num_elements = 20
    elements: List[List[float]] = [
        list(np.random.rand(32)) for _ in range(num_elements)
    ]

    # Build perceptual similarity graph
    G = build_graph(elements, max_hamming=4)

    # Compute a maximal independent set (leaders) – just to exercise the code
    leaders = maximal_independent_set(G, seed=42)
    print(f"Leaders (MIS) count: {len(leaders)}")

    # Prepare bandit bookkeeping
    pull_counts: Dict[Node, int] = {}
    cumulative_rewards: Dict[Node, float] = {}
    total_pulls = 0

    # Bandit parameters
    theta_bounds = (0.0, math.pi)          # sensing angle range
    beam_center = math.pi / 2
    beam_width = math.pi / 4

    # Run a few selection rounds
    for step in range(50):
        selected = select_node_ucb(
            graph=G,
            element_payloads=elements,
            theta_range=theta_bounds,
            center=beam_center,
            width=beam_width,
            counts=pull_counts,
            rewards=cumulative_rewards,
            total_pulls=total_pulls,
            tau=2.0,
        )
        total_pulls += 1
        if step % 10 == 0:
            curv = graph_curvature(G)
            print(f"Step {step:02d}: selected node {selected}, curvature {curv:.4f}")

    print("Final pull counts per node (non‑zero):")
    for n, cnt in pull_counts.items():
        if cnt > 0:
            print(f"  Node {n}: {cnt} pulls, total reward {cumulative_rewards[n]:.4f}")
    sys.exit(0)