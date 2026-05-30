# DARWIN HAMMER — match 5302, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ssim_h_m1837_s2.py (gen6)
# born: 2026-05-30T00:01:14Z

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
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
        d = length(nodes[a], nodes[b])
        edge_len[(a, b)] = d
        edge_len[(b, a)] = d

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
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior = np.where(marginal > 0, (likelihood * prior) / marginal, 0.0)
    return posterior

def calculate_pheromone_probabilities(limit: int) -> List[float]:
    pheromones = np.random.rand(limit)
    total = pheromones.sum()
    if total == 0:
        raise ValueError("Pheromone sum cannot be zero")
    return (pheromones / total).tolist()

def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = [p / total for p in probabilities if p > 0]
    return -sum(p * math.log(max(p, eps)) for p in probs)

def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    if not 0 <= p_hit <= 1:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * entropy(hit_state) + (1 - p_hit) * entropy(miss_state)

def simple_ssim(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator if denominator != 0 else 0.0

def fisher_information_from_pheromones(
    probs: List[float],
    center: float = 0.5,
    width: float = 0.15,
) -> np.ndarray:
    scores = np.array(
        [fisher_score(p, center, width) for p in probs], dtype=float
    )
    weights = np.array([gaussian_beam(p, center, width) for p in probs], dtype=float)
    return scores * weights

def update_edge_beliefs(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    pheromone_limit: int = 10,
    false_positive: float = 0.01,
) -> Dict[Tuple[str, str], float]:
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    pher_probs = calculate_pheromone_probabilities(pheromone_limit)
    theta_vals = np.array([dist[n] / max(dist.values()) if dist else 0 for n in nodes])
    likelihood_vec = fisher_information_from_pheromones(theta_vals.tolist())
    likelihood = np.array(
        [likelihood_vec[list(nodes.keys()).index(src)] for src, _ in edges],
        dtype=float,
    )
    prior = np.full_like(likelihood, 1.0 / len(likelihood))
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    posterior_dict: Dict[Tuple[str, str], float] = {}
    for (src, dst), prob in zip(edges, posterior):
        posterior_dict[(src, dst)] = prob
        posterior_dict[(dst, src)] = prob
    return posterior_dict

def ssim_scaled_multivector(
    attr_matrix_a: np.ndarray,
    attr_matrix_b: np.ndarray,
    base_vector: np.ndarray,
) -> np.ndarray:
    ssim_val = simple_ssim(attr_matrix_a, attr_matrix_b)
    return base_vector * ssim_val

def improved_update_edge_beliefs(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    pheromone_limit: int = 10,
    false_positive: float = 0.01,
    attr_matrix_a: np.ndarray = None,
    attr_matrix_b: np.ndarray = None,
    base_vector: np.ndarray = None,
) -> Dict[Tuple[str, str], float]:
    posterior_dict = update_edge_beliefs(nodes, edges, root, pheromone_limit, false_positive)
    if attr_matrix_a is not None and attr_matrix_b is not None and base_vector is not None:
        ssim_val = simple_ssim(attr_matrix_a, attr_matrix_b)
        posterior_dict = {edge: prob * ssim_val for edge, prob in posterior_dict.items()}
    return posterior_dict

if __name__ == "__main__":
    nodes_demo = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (2.0, 0.0),
        "D": (3.0, 0.0),
        "E": (4.0, 0.0),
    }
    edges_demo = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]
    root_demo = "A"
    posterior_demo = improved_update_edge_beliefs(nodes_demo, edges_demo, root_demo)
    print(posterior_demo)