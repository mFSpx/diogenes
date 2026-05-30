# DARWIN HAMMER — match 773, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_infota_m346_s1.py (gen3)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# born: 2026-05-29T23:30:56Z

import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple
import numpy as np

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


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # identify the edge direction used for length lookup
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist


def jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity coefficient."""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """
    Compute MinHash signature for a list of tokens.
    """
    MAX64 = (1 << 64) - 1
    signature = []
    for seed in range(k):
        min_hash = float('inf')
        for token in tokens:
            hash_value = (hash(token) + seed) % MAX64
            min_hash = min(min_hash, hash_value)
        signature.append(min_hash)
    return signature


def hybrid_fisher_minhash(theta: float, center: float, width: float, tokens1: List[str], tokens2: List[str], k: int = 128) -> float:
    """
    Hybrid Fisher-MinHash metric, combining Fisher information score and MinHash similarity.
    """
    fisher = fisher_score(theta, center, width)
    set1 = set(tokens1)
    set2 = set(tokens2)
    jaccard = jaccard_similarity(set1, set2)
    return fisher * jaccard


def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    """
    Vectorised marginal:  P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """
    Vectorised posterior:  P
    """
    return (likelihood * prior) / marginal


def hybrid_bayes_tree(prior: np.ndarray, likelihood: np.ndarray, false_positive: float, nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str) -> np.ndarray:
    """
    Hybrid Bayesian tree metric, combining Bayesian update and minimum-cost tree.
    """
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    min_cost_tree = np.array([min(edge_len[(node, neighbor)] for neighbor in adj[node]) for node in nodes])
    return posterior * np.array(min_cost_tree)


def expected_entropy(p_hit: float, p_miss: float) -> float:
    """Expected entropy."""
    return -p_hit * math.log(p_hit, 2) - p_miss * math.log(p_miss, 2)


def optimal_sensing_angle(theta: float, center: float, width: float, tokens1: List[str], tokens2: List[str]) -> float:
    """
    Optimal sensing angle.
    """
    fisher = fisher_score(theta, center, width)
    set1 = set(tokens1)
    set2 = set(tokens2)
    jaccard = jaccard_similarity(set1, set2)
    p_hit = jaccard
    p_miss = 1 - p_hit
    expected_ent = expected_entropy(p_hit, p_miss)
    return fisher * expected_ent


if __name__ == "__main__":
    theta = 0.5
    center = 0.0
    width = 1.0
    tokens1 = ["token1", "token2", "token3"]
    tokens2 = ["token2", "token3", "token4"]
    prior = np.array([0.2, 0.3, 0.5])
    likelihood = np.array([0.4, 0.6, 0.0])
    false_positive = 0.1
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"

    print(hybrid_fisher_minhash(theta, center, width, tokens1, tokens2))
    print(hybrid_bayes_tree(prior, likelihood, false_positive, nodes, edges, root))
    print(optimal_sensing_angle(theta, center, width, tokens1, tokens2))