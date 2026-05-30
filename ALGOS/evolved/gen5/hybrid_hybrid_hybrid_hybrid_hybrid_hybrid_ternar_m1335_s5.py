# DARWIN HAMMER — match 1335, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s5.py (gen4)
# parent_b: hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s1.py (gen2)
# born: 2026-05-29T23:35:28Z

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
import numpy as np

Node = int
Graph = Dict[Node, Set[Node]]
FeatureVec = Tuple[float, ...]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    return ((theta - center) / (width ** 2)) * intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
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
    def __init__(self, n_arms: int, gamma: float = 0.1):
        self.n = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms)

    def probabilities(self) -> np.ndarray:
        total = self.weights.sum()
        return (1 - self.gamma) * (self.weights / total) + (self.gamma / self.n)

    def select(self) -> int:
        probs = self.probabilities()
        return int(np.random.choice(self.n, p=probs))

    def update(self, chosen: int, reward: float) -> None:
        probs = self.probabilities()
        x = reward / probs[chosen]
        self.weights[chosen] *= math.exp((self.gamma * x) / self.n)

def fused_similarity_matrix(features: Dict[Node, FeatureVec],
                            alpha: float = 0.5,
                            epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")

    nodes = list(features.keys())
    N = len(nodes)
    S = np.zeros((N, N), dtype=float)

    phashes = {n: compute_phash(list(features[n])) for n in nodes}

    for i, ni in enumerate(nodes):
        vi = np.array(features[ni], dtype=float)
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]  
                continue
            vj = np.array(features[nj], dtype=float)

            ham = hamming_distance(phashes[ni], phashes[nj])
            r = ham / 64.0
            s_g = gaussian(r, epsilon)

            s_s = ssim(vi * 255.0, vj * 255.0)

            s = alpha * s_g + (1.0 - alpha) * s_s
            S[i, j] = s
            S[j, i] = s
    return S, nodes

def sheaf_fisher_pruning(S: np.ndarray,
                         center: float = 0.5,
                         width: float = 0.2) -> np.ndarray:
    N = S.shape[0]
    mask = np.zeros_like(S, dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            theta = S[i, j]
            f = fisher_score(theta, center, width)
            prob = 1.0 / (1.0 + math.exp(-f))  
            keep = prob > 0.5
            mask[i, j] = mask[j, i] = keep
    return mask

def bandit_guided_routing(graph: Graph,
                          features: Dict[Node, FeatureVec],
                          alpha: float = 0.5,
                          epsilon: float = 1.0,
                          steps: int = 10) -> Dict[Node, Node]:
    S, order = fused_similarity_matrix(features, alpha, epsilon)
    node_index = {n: i for i, n in enumerate(order)}
    pruning_mask = sheaf_fisher_pruning(S)

    routing_map = {}
    for node in graph:
        bandit = Exp3Bandit(len(graph[node]))
        for _ in range(steps):
            chosen = bandit.select()
            chosen_node = list(graph[node])[chosen]
            reward = S[node_index[node], node_index[chosen_node]]
            bandit.update(chosen, reward)
        chosen = bandit.select()
        chosen_node = list(graph[node])[chosen]
        routing_map[node] = chosen_node
    return routing_map