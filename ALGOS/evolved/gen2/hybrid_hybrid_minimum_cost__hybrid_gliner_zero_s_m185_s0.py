# DARWIN HAMMER — match 185, survivor 0
# gen: 2
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s3.py (gen1)
# born: 2026-05-29T23:26:07Z

"""
This module represents a hybrid algorithm, combining the principles of minimum-cost tree scoring 
from minimum_cost_tree.py and Bayesian evidence update from bayes_update.py. The exact mathematical
bridge between these two systems is established by incorporating the Bayesian update rules into 
the edge weights of the minimum-cost tree, while also utilizing the label scoring from 
gliner_zero_shot_ext.py. This fusion enables the tree to not only consider the physical distances 
between nodes but also the probabilistic relevance of the paths connecting them and the relevance 
of labels to these paths.

The core idea is to use the Bayesian update function to modify the path weights in the tree scoring 
function, while also considering the score of labels on these paths. This dynamic system where 
the tree structure and the Bayesian probabilities inform each other is integrated with the 
relevance of labels to the paths in the tree. The label scoring is achieved by applying the literal 
fallback matching algorithm from gliner_zero_shot_ext.py to the text on the edges of the tree.
"""

import math
import numpy as np
import random
import sys
import pathlib

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
    """Compute the score of a label on the text using the literal fallback algorithm."""
    labels = parse_labels(label)
    spans = literal_fallback(text, labels, case_sensitive=False)
    return sum(span.score for span in spans)

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
        label_score_sum = sum(label_scores[(a, b)][label] for label in label_scores[(a, b)])
        weighted_label_score_sum = sum(label_scores[(a, b)][label] * label_scores[(a, b)][label] for label in label_scores[(a, b)])
        label_score_std = np.std(list(label_scores[(a, b)].values()))
        label_score_var = np.var(list(label_scores[(a, b)].values()))
        label_scoring_penalty = -label_score_var / (label_score_std + 1e-6)
        bayes_weighted_label_score = -(updated_weight * label_score_sum + label_scoring_penalty)
        edges_cost = length(nodes[a], nodes[b]) + bayes_weighted_label_score
        material += edges_cost
        bayes_weights[(a, b)] = bayes_weighted_label_score
    return material

def hybrid_minimum_cost_tree(n_nodes: int, edge_weights: list[tuple[str, str]], 
                             prior_probabilities: dict[str, float], likelihoods: dict[tuple[str, str], float], 
                             false_positives: dict[tuple[str, str], float], label_scores: dict[tuple[str, str], dict[str, float]]) -> float:
    """Compute the minimum cost tree."""
    cost = np.full(n_nodes, np.inf)
    visited = np.zeros(n_nodes, dtype=bool)
    visited[0] = True
    for _ in range(n_nodes - 1):
        min_cost = np.inf
        min_node = None
        for node in range(n_nodes):
            if visited[node]:
                for neighbor in range(n_nodes):
                    if neighbor not in visited and (node, neighbor) in edge_weights:
                        edge_cost = hybrid_tree_cost({}, edge_weights, node, prior_probabilities, likelihoods, false_positives, label_scores, path_weight=0.2)
                        if edge_cost < min_cost:
                            min_cost = edge_cost
                            min_node = neighbor
        if min_node is None:
            break
        visited[min_node] = True
        cost[min_node] = min_cost
    return cost[-1]

if __name__ == "__main__":
    # Smoke test
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C")]
    prior_probabilities = {"A": 0.5, "B": 0.7, "C": 0.3}
    likelihoods = {("A", "B"): 0.9, ("B", "C"): 0.8}
    false_positives = {("A", "B"): 0.1, ("B", "C"): 0.2}
    label_scores = {("A", "B"): {"label1": 0.7, "label2": 0.3}, ("B", "C"): {"label3": 0.9, "label4": 0.1}}
    print(hybrid_minimum_cost_tree(len(nodes), edges, prior_probabilities, likelihoods, false_positives, label_scores))