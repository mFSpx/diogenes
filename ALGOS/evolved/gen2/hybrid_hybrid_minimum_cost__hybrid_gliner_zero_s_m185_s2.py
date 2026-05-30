# DARWIN HAMMER — match 185, survivor 2
# gen: 2
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s3.py (gen1)
# born: 2026-05-29T23:26:07Z

import math
import numpy as np

Point = tuple[float, float]
Edge = tuple[str, str]

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def label_score(text: str, label: str) -> float:
    # placeholder for the literal fallback algorithm
    return 1.0

def hybrid_tree_cost(nodes: dict[str, Point], edges: list[Edge], root: str, 
                     prior_probabilities: dict[str, float], likelihoods: dict[Edge, float], 
                     false_positives: dict[Edge, float], label_scores: dict[Edge, dict[str, float]], 
                     path_weight: float = 0.2) -> float:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    bayes_weights = {}
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
        marginal = bayes_marginal(prior_probabilities[a], likelihoods[(a, b)], false_positives[(a, b)])
        updated_weight = bayes_update(prior_probabilities[a], likelihoods[(a, b)], marginal)
        label_score_sum = np.mean(list(label_scores[(a, b)].values()))
        label_score_var = np.var(list(label_scores[(a, b)].values()))
        label_scoring_penalty = -label_score_var / (np.std(list(label_scores[(a, b)].values())) + 1e-6)
        bayes_weighted_label_score = -(updated_weight * label_score_sum + label_scoring_penalty)
        edges_cost = length(nodes[a], nodes[b]) + bayes_weighted_label_score
        material += edges_cost
        bayes_weights[(a, b)] = bayes_weighted_label_score
    return material

def hybrid_minimum_cost_tree(n_nodes: int, edge_weights: list[tuple[str, str]], 
                             prior_probabilities: dict[str, float], likelihoods: dict[tuple[str, str], float], 
                             false_positives: dict[tuple[str, str], float], label_scores: dict[tuple[str, str], dict[str, float]]) -> float:
    nodes = list(prior_probabilities.keys())
    edges = [(nodes[i], nodes[j]) for i in range(n_nodes) for j in range(i+1, n_nodes) if (nodes[i], nodes[j]) in edge_weights]
    return hybrid_tree_cost({node: (i, i) for i, node in enumerate(nodes)}, edges, nodes[0], prior_probabilities, likelihoods, false_positives, label_scores)

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C")]
    prior_probabilities = {"A": 0.5, "B": 0.7, "C": 0.3}
    likelihoods = {("A", "B"): 0.9, ("B", "C"): 0.8}
    false_positives = {("A", "B"): 0.1, ("B", "C"): 0.2}
    label_scores = {("A", "B"): {"label1": 0.7, "label2": 0.3}, ("B", "C"): {"label3": 0.9, "label4": 0.1}}
    print(hybrid_minimum_cost_tree(len(nodes), edges, prior_probabilities, likelihoods, false_positives, label_scores))