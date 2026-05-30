# DARWIN HAMMER — match 1924, survivor 1
# gen: 3
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# parent_b: hybrid_sketches_hybrid_bandit_router_m31_s1.py (gen2)
# born: 2026-05-29T23:39:53Z

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np

Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][h] += 1
    return table

def sketch_query(sketch: List[List[int]], item: str, width: int = 64, depth: int = 4) -> int:
    estimates = []
    for d in range(depth):
        h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        estimates.append(sketch[d][h])
    return min(estimates)

def hybrid_edge_cost(
    a: str,
    b: str,
    nodes: Dict[str, Point],
    prior_probabilities: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    sketch: List[List[int]],
    alpha: float = 0.2,
    beta: float = 0.5,
    gamma: float = 0.1
) -> float:
    geom = length(nodes[a], nodes[b])
    edge_key = (a, b) if (a, b) in likelihoods else (b, a)
    prior = prior_probabilities.get(a, 0.5)  
    like = likelihoods[edge_key]
    fp = false_positives[edge_key]
    marginal = bayes_marginal(prior, like, fp)
    post = bayes_update(prior, like, marginal)
    bayes_part = alpha * post
    sketch_key = f"{a}->{b}"
    freq_est = sketch_query(sketch, sketch_key)
    sketch_part = beta * math.log1p(freq_est)  
    novelty_part = gamma * (1 / (1 + freq_est))
    return geom + bayes_part + sketch_part + novelty_part

def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    prior_probabilities: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    sketch: List[List[int]],
    alpha: float = 0.2,
    beta: float = 0.5,
    gamma: float = 0.1
) -> float:
    total = 0.0
    for a, b in edges:
        total += hybrid_edge_cost(
            a, b, nodes, prior_probabilities, likelihoods, false_positives, sketch, alpha, beta, gamma
        )
    return total

@dataclass(frozen=True)
class EdgeAction:
    edge: Edge
    reward_estimate: float
    confidence: float
    algorithm: str

def select_edge_bandit(
    context: Dict[str, float],
    candidate_edges: List[Edge],
    nodes: Dict[str, Point],
    prior_probabilities: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    sketch: List[List[int]],
    algorithm: str = "epsilon_greedy",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
    alpha: float = 0.2,
    beta: float = 0.5,
    gamma: float = 0.1
) -> EdgeAction:
    if not candidate_edges:
        raise ValueError("No candidate edges provided")
    rng = random.Random(seed)

    rewards = []
    for e in candidate_edges:
        cost = hybrid_edge_cost(
            e[0],
            e[1],
            nodes,
            prior_probabilities,
            likelihoods,
            false_positives,
            sketch,
            alpha,
            beta,
            gamma
        )
        reward = -cost  
        freq = sketch_query(sketch, f"{e[0]}->{e[1]}")
        confidence = 1.0 / (1 + freq)
        rewards.append((e, reward, confidence))

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen_edge, reward, confidence = rng.choice(rewards)
    elif algorithm == "thompson":
        sampled = [
            (e, rng.betavariate(1 + max(0, reward), 1 + max(0, -reward)), conf)
            for (e, reward, conf) in rewards
        ]
        chosen_edge, reward, confidence = max(sampled, key=lambda x: x[1])
    else:  
        scale = math.sqrt(
            sum(float(v) * float(v) for v in context.values())
        ) if context else 1.0
        ucb_values = [
            (e, r + 0.1 * scale / math.sqrt(1 + c)) for (e, r, c) in rewards
        ]
        chosen_edge, reward = max(ucb_values, key=lambda x: x[1])
        confidence = next(c for (e, r, c) in rewards if e == chosen_edge)

    return EdgeAction(edge=chosen_edge, reward_estimate=reward, confidence=confidence, algorithm=algorithm)

if __name__ == "__main__":
    nodes = {
        "A": (0, 0),
        "B": (3, 4),
        "C": (6, 8),
        "D": (9, 12)
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    prior_probabilities = {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5}
    likelihoods = {("A", "B"): 0.8, ("B", "C"): 0.7, ("C", "D"): 0.9}
    false_positives = {("A", "B"): 0.1, ("B", "C"): 0.2, ("C", "D"): 0.1}
    sketch = count_min_sketch([f"{a}->{b}" for a, b in edges])
    print(select_edge_bandit({}, edges, nodes, prior_probabilities, likelihoods, false_positives, sketch))