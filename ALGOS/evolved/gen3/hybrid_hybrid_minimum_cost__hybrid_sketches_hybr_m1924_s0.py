# DARWIN HAMMER — match 1924, survivor 0
# gen: 3
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# parent_b: hybrid_sketches_hybrid_bandit_router_m31_s1.py (gen2)
# born: 2026-05-29T23:39:53Z

"""Hybrid Minimum-Cost Bayesian Tree with Sketch‑Based Bandit Routing

Parents:
- hybrid_minimum_cost_tree_bayes_update_m6_s0.py (minimum‑cost tree + Bayesian edge update)
- hybrid_sketches_hybrid_bandit_router_m31_s1.py (count‑min sketch + bandit action selection)

Mathematical bridge:
Each edge *e* has a physical length ℓ(e) and a Bayesian‑updated relevance weight
w_B(e)=P(H|E)= prior·likelihood / marginal, where marginal = likelihood·prior + fp·(1‑prior).
A count‑min sketch S maintains an estimate ĉ(e) of how often the edge has been
selected by the routing policy. The hybrid cost of a tree T is

    C(T) = Σ_{e∈T} [ ℓ(e) + α·w_B(e) + β·log(1 + ĉ(e)) ]

where α,β>0 balance Bayesian belief and exploration pressure.
The bandit router treats the negative hybrid cost as a reward and uses the sketch
to bias selection toward less‑explored edges, thus fusing the two topologies
into a single adaptive system.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric and Bayesian utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) for a Bayesian update."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior probability P(H|E) given prior, likelihood and marginal."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Count‑Min Sketch utilities (from Parent B)
# ----------------------------------------------------------------------
def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Create a count‑min sketch table for a list of string items."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][h] += 1
    return table

def sketch_query(sketch: List[List[int]], item: str, width: int = 64, depth: int = 4) -> int:
    """Estimate the frequency of *item* using the given sketch."""
    estimates = []
    for d in range(depth):
        h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        estimates.append(sketch[d][h])
    return min(estimates)

# ----------------------------------------------------------------------
# Hybrid cost computation integrating Bayesian weights and sketch counts
# ----------------------------------------------------------------------
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
) -> float:
    """
    Compute the cost of a single edge using:
        - Euclidean length
        - Bayesian updated relevance (scaled by alpha)
        - Exploration penalty based on sketch frequency (scaled by beta)
    """
    # geometric part
    geom = length(nodes[a], nodes[b])

    # Bayesian part (symmetrize edge key)
    edge_key = (a, b) if (a, b) in likelihoods else (b, a)
    prior = prior_probabilities.get(a, 0.5)  # fallback prior if missing
    like = likelihoods[edge_key]
    fp = false_positives[edge_key]
    marginal = bayes_marginal(prior, like, fp)
    post = bayes_update(prior, like, marginal)
    bayes_part = alpha * post

    # Sketch part
    sketch_key = f"{a}->{b}"
    freq_est = sketch_query(sketch, sketch_key)
    sketch_part = beta * math.log1p(freq_est)  # log(1+count) smooths growth

    return geom + bayes_part + sketch_part

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
) -> float:
    """
    Simple tree cost: sum of hybrid_edge_cost over all edges.
    The function assumes *edges* already form a spanning tree rooted at *root*.
    """
    total = 0.0
    for a, b in edges:
        total += hybrid_edge_cost(
            a, b, nodes, prior_probabilities, likelihoods, false_positives, sketch, alpha, beta
        )
    return total

# ----------------------------------------------------------------------
# Bandit‑style edge selection using sketch‑estimated frequencies
# ----------------------------------------------------------------------
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
) -> EdgeAction:
    """
    Choose an edge from *candidate_edges* using a bandit policy.
    Reward estimate is the negative hybrid cost (higher is better).
    """
    if not candidate_edges:
        raise ValueError("No candidate edges provided")
    rng = random.Random(seed)

    # Compute reward estimates (negative cost) and a simple confidence term
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
        )
        reward = -cost  # lower cost ⇒ higher reward
        # confidence grows with inverse sketch frequency (encourage exploration)
        freq = sketch_query(sketch, f"{e[0]}->{e[1]}")
        confidence = 1.0 / (1 + freq)
        rewards.append((e, reward, confidence))

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen_edge, reward, confidence = rng.choice(rewards)
    elif algorithm == "thompson":
        # Sample from a Beta distribution using reward as pseudo‑successes
        sampled = [
            (e, rng.betavariate(1 + max(0, reward), 1 + max(0, -reward)), conf)
            for (e, reward, conf) in rewards
        ]
        chosen_edge, reward, confidence = max(sampled, key=lambda x: x[1])
    else:  # default: Upper Confidence Bound style
        scale = math.sqrt(
            sum(float(v) * float(v) for v in context.values())
        ) if context else 1.0
        ucb_values = [
            (e, r + 0.1 * scale / math.sqrt(1 + c)) for (e, r, c) in rewards
        ]
        chosen_edge, reward = max(ucb_values, key=lambda x: x[1])
        confidence = next(c for (e, r, c) in rewards if e == chosen_edge)

    return EdgeAction(edge=chosen_edge, reward_estimate=reward, confidence=confidence, algorithm=algorithm)

# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    root = "A"

    # Bayesian parameters (synthetic)
    prior_probabilities = {"A": 0.6, "B": 0.5, "C": 0.4, "D": 0.7}
    likelihoods = {e: 0.8 for e in edges}
    false_positives = {e: 0.1 for e in edges}

    # Initialise a sketch with some historical selections
    historic_selections = ["A->B", "A->C", "A->B", "B->D"]
    sketch = count_min_sketch(historic_selections)

    # Compute total hybrid tree cost
    total_cost = hybrid_tree_cost(
        nodes,
        edges,
        root,
        prior_probabilities,
        likelihoods,
        false_positives,
        sketch,
    )
    print(f"Hybrid tree cost: {total_cost:.4f}")

    # Context vector for bandit (dummy values)
    context = {"feature1": 0.3, "feature2": 0.7}

    # Choose next edge to explore
    action = select_edge_bandit(
        context,
        candidate_edges=edges,
        nodes=nodes,
        prior_probabilities=prior_probabilities,
        likelihoods=likelihoods,
        false_positives=false_positives,
        sketch=sketch,
        algorithm="epsilon_greedy",
        epsilon=0.2,
    )
    print(f"Selected edge: {action.edge}, reward_estimate: {action.reward_estimate:.4f}, confidence: {action.confidence:.4f}")

    # Update sketch with the chosen edge (simulating a selection)
    chosen_key = f"{action.edge[0]}->{action.edge[1]}"
    sketch = count_min_sketch(historic_selections + [chosen_key])

    # Re‑compute cost after the update (should be slightly higher due to increased sketch penalty)
    new_total_cost = hybrid_tree_cost(
        nodes,
        edges,
        root,
        prior_probabilities,
        likelihoods,
        false_positives,
        sketch,
    )
    print(f"New hybrid tree cost after selection: {new_total_cost:.4f}")