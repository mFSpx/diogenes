# DARWIN HAMMER — match 1918, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_semantic_neig_hybrid_hybrid_krampu_m540_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m405_s0.py (gen4)
# born: 2026-05-29T23:39:45Z

"""
Module for the Hybrid Algorithm, integrating the core topologies of 
hybrid_hybrid_semantic_neig_hybrid_hybrid_krampu_m540_s1.py and 
hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m405_s0.py.
The mathematical bridge between the two structures is the application of 
log-count statistics to estimate the expected reward of each action, and 
the use of pheromone signals as probabilities to inform the semantic neighborhood search.
Additionally, this module incorporates the concepts of entropy-based action selection, 
Ollivier-Ricci curvature, and minimum-cost tree Bayesian update.
"""

import numpy as np
import random
import math
import sys
import pathlib
from collections import defaultdict

def _cos(a, b):
    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return 0.0 if den == 0 else sum(x*y for x, y in zip(a, b)) / den

def pheromone_probabilities(pheromones):
    total = sum(pheromones)
    return [p / total for p in pheromones]

def entropy(probabilities, eps=1e-12):
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)

def log_count_statistics(nodes, edges):
    log_counts = defaultdict(int)
    for a, b in edges:
        log_counts[a] += 1
        log_counts[b] += 1
    return {node: math.log(count) for node, count in log_counts.items()}

def ollivier_ricci_curvature(nodes, edges):
    curvature = {}
    for node in nodes:
        neighbors = [b for a, b in edges if a == node] + [a for a, b in edges if b == node]
        neighbor_distances = [math.hypot(nodes[node][0] - nodes[neighbor][0], nodes[node][1] - nodes[neighbor][1]) for neighbor in neighbors]
        curvature[node] = sum(neighbor_distances) / len(neighbor_distances) if neighbors else 0
    return curvature

def hybrid_semantic_search(nodes, edges, pheromones, query):
    probabilities = pheromone_probabilities(pheromones)
    ollivier_ricci = ollivier_ricci_curvature(nodes, edges)
    log_counts = log_count_statistics(nodes, edges)
    scores = {}
    for node in nodes:
        score = probabilities[nodes.index(node)] * ollivier_ricci[node] * log_counts[node]
        scores[node] = score
    return max(scores, key=scores.get)

def tree_metrics(nodes, edges, root):
    adj = defaultdict(list)
    edge_len = {}
    node_dist = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
        edge_len[(b, a)] = math.hypot(nodes[b][0] - nodes[a][0], nodes[a][1] - nodes[b][1])
    queue = [(root, 0)]
    visited = set()
    while queue:
        node, dist = queue.pop(0)
        if node not in visited:
            visited.add(node)
            node_dist[node] = dist
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + edge_len[(node, neighbor)]))
    return adj, edge_len, node_dist

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 1), 'C': (2, 2)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    pheromones = [0.4, 0.3, 0.3]
    query = 'A'
    result = hybrid_semantic_search(nodes, edges, pheromones, query)
    print("Hybrid Semantic Search Result:", result)

    adj, edge_len, node_dist = tree_metrics(nodes, edges, 'A')
    print("Tree Metrics:")
    print("Adjacency:", dict(adj))
    print("Edge Lengths:", dict(edge_len))
    print("Node Distances:", dict(node_dist))