# DARWIN HAMMER — match 185, survivor 1
# gen: 2
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s3.py (gen1)
# born: 2026-05-29T23:26:07Z

import math
import numpy as np

Point = tuple[float, float]
Edge = tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def label_score(text: str, label: str) -> float:
    """Compute the score of a label on the text using a more robust literal fallback algorithm."""
    # Improved literal fallback matching algorithm
    labels = [label]
    spans = []
    for i in range(len(text)):
        for j in range(i + 1, len(text) + 1):
            substring = text[i:j]
            if substring in labels:
                spans.append((substring, i, j))
    scores = [1.0 if span[0] == label else 0.0 for span in spans]
    return sum(scores) / len(spans) if spans else 0.0

def hybrid_tree_cost(nodes: dict[str, Point], edges: list[Edge], root: str, 
                     prior_probabilities: dict[str, float], likelihoods: dict[Edge, float], 
                     false_positives: dict[Edge, float], label_scores: dict[Edge, dict[str, float]], 
                     path_weight: float = 0.2) -> float:
    """Calculate the cost of the tree incorporating Bayesian update and label scoring in edge weights."""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    bayes_weights = {}
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
        marginal = bayes_marginal(prior_probabilities[a], likelihoods[(a, b)], false_positives[(a, b)])
        updated_weight = bayes_update(prior_probabilities[a], likelihoods[(a, b)], marginal)
        # Use a more robust label scoring method
        label_score_sum = sum(label_scores[(a, b)].values())
        label_score_std = np.std(list(label_scores[(a, b)].values()))
        label_score_var = np.var(list(label_scores[(a, b)].values()))
        label_scoring_penalty = -label_score_var / (label_score_std + 1e-6)
        bayes_weighted_label_score = -(updated_weight * label_score_sum + label_scoring_penalty)
        edges_cost = length(nodes[a], nodes[b]) * (1 - path_weight) + bayes_weighted_label_score * path_weight
        material += edges_cost
        bayes_weights[(a, b)] = bayes_weighted_label_score
    return material

def hybrid_minimum_cost_tree(n_nodes: int, nodes: dict[str, Point], edges: list[Edge], 
                             prior_probabilities: dict[str, float], likelihoods: dict[Edge, float], 
                             false_positives: dict[Edge, float], label_scores: dict[Edge, dict[str, float]]) -> float:
    """Compute the minimum cost tree using Prim's algorithm."""
    # Initialize the minimum spanning tree
    mst = []
    visited = set()
    visited.add(next(iter(nodes)))
    while len(visited) < n_nodes:
        min_cost = float('inf')
        min_edge = None
        for u in visited:
            for v in [node for node in nodes if node not in visited and (u, node) in edges or (node, u) in edges]:
                edge = (u, v) if (u, v) in edges else (v, u)
                cost = hybrid_tree_cost(nodes, [edge], u, prior_probabilities, likelihoods, false_positives, label_scores)
                if cost < min_cost:
                    min_cost = cost
                    min_edge = edge
        mst.append(min_edge)
        visited.add(min_edge[1])
    return sum(hybrid_tree_cost(nodes, [edge], next(iter(nodes)), prior_probabilities, likelihoods, false_positives, label_scores) for edge in mst)

if __name__ == "__main__":
    # Smoke test
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C"), ("A", "C")]
    prior_probabilities = {"A": 0.5, "B": 0.7, "C": 0.3}
    likelihoods = {("A", "B"): 0.9, ("B", "C"): 0.8, ("A", "C"): 0.7}
    false_positives = {("A", "B"): 0.1, ("B", "C"): 0.2, ("A", "C"): 0.3}
    label_scores = {("A", "B"): {"label1": 0.7, "label2": 0.3}, ("B", "C"): {"label3": 0.9, "label4": 0.1}, ("A", "C"): {"label5": 0.6, "label6": 0.4}}
    print(hybrid_minimum_cost_tree(len(nodes), nodes, edges, prior_probabilities, likelihoods, false_positives, label_scores))