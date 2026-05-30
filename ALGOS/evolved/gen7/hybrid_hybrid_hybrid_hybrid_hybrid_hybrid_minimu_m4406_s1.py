# DARWIN HAMMER — match 4406, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m1451_s1.py (gen6)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_hybrid_hybrid_m1333_s2.py (gen5)
# born: 2026-05-29T23:55:32Z

import numpy as np
import math
import random
from typing import Tuple, Dict, List

Point = Tuple[float, float]
Edge = Tuple[str, str]

def gini_coefficient(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    values = values.astype(float)
    if np.all(values == 0):
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: Dict[str, Point], edges: List[Edge], root: str, path_weight: float = 0.2) -> float:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
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

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def hybrid_score(nodes: Dict[str, Point], edges: List[Edge], root: str, 
                prior: float, likelihood: float, false_positive: float) -> float:
    tree_cost_val = tree_cost(nodes, edges, root)
    edge_costs = [length(nodes[a], nodes[b]) for a, b in edges]
    gini_val = gini_coefficient(np.array(edge_costs))
    marginal_val = bayes_marginal(prior, likelihood, false_positive)
    return tree_cost_val * (1 - gini_val) * marginal_val * np.log(marginal_val + 1)

def build_adjacency(matrix: np.ndarray, threshold: float = 1.0) -> List[Tuple[int, int]]:
    num_nodes = len(matrix)
    adj_list = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes) 
                 if np.linalg.norm(matrix[i] - matrix[j]) < threshold]
    return adj_list

def node_curvature(adj_list: List[Tuple[int, int]], matrix: np.ndarray) -> np.ndarray:
    num_nodes = len(matrix)
    curvature = np.zeros((num_nodes,))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list or (j, i) in adj_list]
        if neighbors:
            kappa_sum = 0.0
            for j in neighbors:
                dist = np.linalg.norm(matrix[i] - matrix[j])
                kappa_sum += 1 / (dist + 1)
            curvature[i] = kappa_sum / len(neighbors)
    return curvature

def improved_hybrid_score(nodes: Dict[str, Point], edges: List[Edge], root: str, 
                         prior: float, likelihood: float, false_positive: float) -> float:
    tree_cost_val = tree_cost(nodes, edges, root)
    edge_costs = [length(nodes[a], nodes[b]) for a, b in edges]
    gini_val = gini_coefficient(np.array(edge_costs))
    curvature = node_curvature(build_adjacency(np.array([list(node) for node in nodes.values()])), 1.5)
    node_curvature_val = np.mean(curvature)
    marginal_val = bayes_marginal(prior, likelihood, false_positive)
    return tree_cost_val * (1 - gini_val) * marginal_val * np.log(marginal_val + 1) * node_curvature_val

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 0), 'C': (1, 1), 'D': (0, 1)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'
    prior = 0.5
    likelihood = 0.8
    false_positive = 0.2
    score = improved_hybrid_score(nodes, edges, root, prior, likelihood, false_positive)
    print(score)

    matrix = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    threshold = 1.5
    adj_list = build_adjacency(matrix, threshold)
    curvature = node_curvature(adj_list, matrix)
    print(curvature)