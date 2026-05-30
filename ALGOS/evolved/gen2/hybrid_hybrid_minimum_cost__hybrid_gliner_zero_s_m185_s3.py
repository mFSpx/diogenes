# DARWIN HAMMER — match 185, survivor 3
# gen: 2
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s3.py (gen1)
# born: 2026-05-29T23:26:07Z

import math
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List, Set, Iterable

# Basic geometric utilities ----------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Bayesian utilities -----------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """
    Compute the marginal probability P(E) = P(E|H)P(H) + P(E|¬H)P(¬H).

    ``false_positive`` is interpreted as P(E|¬H).
    """
    if not 0.0 <= prior <= 1.0 or not 0.0 <= likelihood <= 1.0 or not 0.0 <= false_positive <= 1.0:
        raise ValueError("All probability arguments must lie in [0, 1].")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """
    Return the posterior P(H|E) = P(E|H)P(H) / P(E).
    """
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0.")
    return prior * likelihood / marginal

# Simple label scoring ---------------------------------------------------------
def simple_label_score(text: str, label: str) -> float:
    """
    Very lightweight fallback scoring: count case‑insensitive occurrences
    of the label in the text, normalised by the length of the text.
    """
    if not text:
        return 0.0
    count = text.lower().count(label.lower())
    return count / len(text.split())

def aggregate_label_scores(text: str, label_dict: Dict[str, float]) -> float:
    """
    Combine multiple label scores for a single edge.
    The combination uses a weighted sum where each label's weight is its
    own score (i.e. higher‑scoring labels influence the sum more).
    """
    if not label_dict:
        return 0.0
    scores = [simple_label_score(text, lbl) * w for lbl, w in label_dict.items()]
    return float(np.mean(scores))

# Edge cost computation --------------------------------------------------------
def edge_cost(
    a: str,
    b: str,
    nodes: Dict[str, Point],
    prior: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    edge_texts: Dict[Edge, str],
    label_scores: Dict[Edge, Dict[str, float]],
    distance_weight: float = 0.5,
    bayes_weight: float = 0.3,
    label_weight: float = 0.2,
) -> float:
    """
    Compute a unified cost for an edge (a, b) that blends:
      * geometric distance,
      * Bayesian posterior (the higher the posterior, the *lower* the cost),
      * aggregated label relevance.
    All three components are normalised to comparable scales before blending.
    """
    # 1. geometric component
    dist = euclidean(nodes[a], nodes[b])

    # 2. Bayesian component – we treat a high posterior as a *reward*,
    #    therefore we subtract it from the distance after scaling.
    marginal = bayes_marginal(prior[a], likelihoods[(a, b)], false_positives[(a, b)])
    posterior = bayes_update(prior[a], likelihoods[(a, b)], marginal)

    # 3. Label component – higher aggregated label relevance should also reduce cost.
    label_agg = aggregate_label_scores(edge_texts[(a, b)], label_scores[(a, b)])

    # Normalisation (simple min‑max across the three raw values)
    # In practice these would be pre‑computed over the whole graph;
    # here we use heuristics to keep the function self‑contained.
    max_dist = max(dist, 1.0)          # avoid division by zero
    max_post = 1.0                    # posterior already in [0,1]
    max_lbl  = 1.0                    # aggregated label relevance in [0,1]

    norm_dist = dist / max_dist
    norm_post = posterior / max_post
    norm_lbl  = label_agg / max_lbl

    # Blend – lower cost is better.
    blended = (
        distance_weight * norm_dist
        - bayes_weight * norm_post
        - label_weight * norm_lbl
    )
    # Ensure non‑negative cost for algorithms that assume positivity.
    return max(blended, 0.0)

# Prim's algorithm for a minimum‑cost spanning tree ---------------------------
def hybrid_minimum_cost_tree(
    nodes: Dict[str, Point],
    edges: List[Edge],
    prior_probabilities: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positives: Dict[Edge, float],
    edge_texts: Dict[Edge, str],
    label_scores: Dict[Edge, Dict[str, float]],
    distance_weight: float = 0.5,
    bayes_weight: float = 0.3,
    label_weight: float = 0.2,
) -> Tuple[float, Set[Edge]]:
    """
    Return the total cost and the set of edges forming the minimum‑cost spanning tree.
    The algorithm is a direct implementation of Prim's algorithm where the edge
    weight is given by :func:`edge_cost`.
    """
    if not nodes:
        raise ValueError("The graph must contain at least one node.")
    if len(edges) < len(nodes) - 1:
        raise ValueError("Not enough edges to form a spanning tree.")

    # Build adjacency list for fast neighbour lookup.
    adjacency: Dict[str, List[str]] = defaultdict(list)
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    # Initialise the tree with an arbitrary start node.
    start = next(iter(nodes))
    visited: Set[str] = {start}
    candidate_edges: List[Tuple[float, Edge]] = []

    # Helper to push candidate edges from a newly visited node.
    def push_candidates(node: str) -> None:
        for nbr in adjacency[node]:
            if nbr in visited:
                continue
            cost = edge_cost(
                node,
                nbr,
                nodes,
                prior_probabilities,
                likelihoods,
                false_positives,
                edge_texts,
                label_scores,
                distance_weight,
                bayes_weight,
                label_weight,
            )
            candidate_edges.append((cost, (node, nbr)))

    push_candidates(start)

    total_cost = 0.0
    tree_edges: Set[Edge] = set()

    while len(visited) < len(nodes):
        if not candidate_edges:
            raise RuntimeError("Graph is disconnected; cannot form a spanning tree.")
        # Select the edge with the smallest cost.
        candidate_edges.sort(key=lambda x: x[0])
        cost, (u, v) = candidate_edges.pop(0)

        if v in visited:
            # Edge points to an already visited node – skip.
            continue

        visited.add(v)
        tree_edges.add((u, v))
        total_cost += cost
        push_candidates(v)

    return total_cost, tree_edges

# -------------------------------------------------------------------------------
# Example usage (smoke test)
if __name__ == "__main__":
    # Define a tiny graph.
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 1.0),
        "C": (2.0, 0.0),
        "D": (0.0, 2.0),
    }

    edges = [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    ]

    prior_probabilities = {n: 0.6 for n in nodes}
    likelihoods = {e: 0.8 for e in edges}
    false_positives = {e: 0.1 for e in edges}
    edge_texts = {
        ("A", "B"): "alpha beta",
        ("A", "C"): "gamma delta",
        ("A", "D"): "epsilon zeta",
        ("B", "C"): "eta theta",
        ("B", "D"): "iota kappa",
        ("C", "D"): "lambda mu",
    }
    label_scores = {
        ("A", "B"): {"alpha": 1.0, "beta": 0.5},
        ("A", "C"): {"gamma": 0.8},
        ("A", "D"): {"epsilon": 0.6, "zeta": 0.4},
        ("B", "C"): {"eta": 0.9},
        ("B", "D"): {"iota": 0.7},
        ("C", "D"): {"lambda": 0.5, "mu": 0.3},
    }

    total, tree = hybrid_minimum_cost_tree(
        nodes,
        edges,
        prior_probabilities,
        likelihoods,
        false_positives,
        edge_texts,
        label_scores,
    )
    print(f"Total cost of hybrid MST: {total:.4f}")
    print("Edges in the tree:")
    for e in tree:
        print(e)