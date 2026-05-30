# DARWIN HAMMER — match 773, survivor 3
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
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = length(nodes[a], nodes[b])

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)

    return adj, edge_len, dist

def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    return (likelihood * prior) / marginal

def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    MAX64 = (1 << 64) - 1
    signature = []
    for seed in range(k):
        min_hash = float('inf')
        for token in tokens:
            hash_value = (hash(token) + seed) % MAX64
            min_hash = min(min_hash, hash_value)
        signature.append(min_hash)
    return signature

def jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)

def hybrid_fisher_minhash(theta: float, center: float, width: float, tokens1: List[str], tokens2: List[str], k: int = 128) -> float:
    fisher = fisher_score(theta, center, width)
    similarity = jaccard_similarity(tokens1, tokens2)
    return fisher * similarity

def hybrid_bayes_tree(prior: np.ndarray, likelihood: np.ndarray, false_positive: float, nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str) -> np.ndarray:
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    return posterior * len(dist)

def hybrid_integration(theta: float, center: float, width: float, tokens1: List[str], tokens2: List[str], k: int, prior: np.ndarray, likelihood: np.ndarray, false_positive: float, nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str) -> float:
    fisher_minhash = hybrid_fisher_minhash(theta, center, width, tokens1, tokens2, k)
    bayes_tree = np.sum(hybrid_bayes_tree(prior, likelihood, false_positive, nodes, edges, root))
    return fisher_minhash * bayes_tree

if __name__ == "__main__":
    theta = 0.5
    center = 0.0
    width = 1.0
    tokens1 = ["token1", "token2", "token3"]
    tokens2 = ["token1", "token4", "token5"]
    prior = np.array([0.2, 0.3, 0.5])
    likelihood = np.array([0.4, 0.6, 0.0])
    false_positive = 0.1
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"

    print(hybrid_integration(theta, center, width, tokens1, tokens2, 128, prior, likelihood, false_positive, nodes, edges, root))