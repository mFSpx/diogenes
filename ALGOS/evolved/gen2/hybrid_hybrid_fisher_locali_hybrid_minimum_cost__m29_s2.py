# DARWIN HAMMER — match 29, survivor 2
# gen: 2
# parent_a: hybrid_fisher_localization_krampus_chrono_m17_s0.py (gen1)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s1.py (gen1)
# born: 2026-05-29T23:25:14Z

"""
This module integrates the Fisher information scoring from hybrid_fisher_localization_krampus_chrono_m17_s0.py and the minimum-cost tree scoring from hybrid_minimum_cost_tree_bayes_update_m6_s1.py.
The mathematical bridge between the two structures is the use of probabilistic scoring functions in both algorithms. 
In hybrid_fisher_localization_krampus_chrono_m17_s0.py, a Gaussian beam is used to model the intensity of a signal, while in hybrid_minimum_cost_tree_bayes_update_m6_s1.py, a Bayesian update rule is used to update probabilities based on new evidence.
This module combines these concepts to create a hybrid algorithm that uses Gaussian distributions to model and smooth out chronological data and applies Bayesian update rules to update probabilities in a minimum-cost tree.
"""

import math
import numpy as np
import random
import sys
import pathlib
from datetime import datetime

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def parse_loose_datetime(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None

def chrono_candidates_for_path(path: pathlib.Path, text_sample: str = "") -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for year in range(1900, 2100):
        for month in range(1, 13):
            for day in range(1, 32):
                raw = f"{year}-{month:02d}-{day:02d}"
                parsed = parse_loose_datetime(raw)
                if parsed:
                    candidates.append({
                        "timestamp": raw,
                        "source": "path",
                        "raw": raw,
                    })
    return candidates

def gaussian_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    return np.array([gaussian_beam(x, 0, sigma) for x in data])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

Point = tuple[float, float]
Edge = tuple[str, str]

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: dict[str, Point], edges: list[Edge], root: str, path_weight: float = 0.2) -> float:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def hybrid_tree_cost(nodes: dict[str, Point], edges: list[Edge], root: str, edge_priors: dict[Edge, float], path_weight: float = 0.2) -> float:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b]) * edge_priors[(a, b)]
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def hybrid_chrono_fisher(candidates: list[dict[str, str]], center: float, width: float) -> float:
    scores = []
    for candidate in candidates:
        timestamp = datetime.fromisoformat(candidate["timestamp"])
        score = fisher_score(timestamp.timestamp(), center, width)
        scores.append(score)
    return np.mean(scores)

def hybrid_bayes_chrono_tree(nodes: dict[str, Point], edges: list[Edge], root: str, edge_priors: dict[Edge, float], candidates: list[dict[str, str]], center: float, width: float) -> float:
    tree_score = hybrid_tree_cost(nodes, edges, root, edge_priors)
    chrono_score = hybrid_chrono_fisher(candidates, center, width)
    return tree_score * chrono_score

def hybrid_bayes_chrono_update(nodes: dict[str, Point], edges: list[Edge], root: str, edge_priors: dict[Edge, float], candidates: list[dict[str, str]], center: float, width: float, prior: float, likelihood: float, false_positive: float) -> float:
    tree_score = hybrid_tree_cost(nodes, edges, root, edge_priors)
    chrono_score = hybrid_chrono_fisher(candidates, center, width)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    update = bayes_update(prior, likelihood, marginal)
    return tree_score * chrono_score * update

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    edge_priors = {("A", "B"): 0.5, ("B", "C"): 0.6, ("C", "A"): 0.7}
    candidates = [{"timestamp": "2022-01-01", "source": "path", "raw": "2022-01-01"}]
    center = 0.0
    width = 1.0
    prior = 0.8
    likelihood = 0.9
    false_positive = 0.1

    print(hybrid_bayes_chrono_tree(nodes, edges, root, edge_priors, candidates, center, width))
    print(hybrid_bayes_chrono_update(nodes, edges, root, edge_priors, candidates, center, width, prior, likelihood, false_positive))