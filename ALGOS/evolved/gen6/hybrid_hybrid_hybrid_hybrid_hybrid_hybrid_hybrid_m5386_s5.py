# DARWIN HAMMER — match 5386, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_physar_m1398_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2471_s0.py (gen5)
# born: 2026-05-30T00:01:41Z

import numpy as np
import math
from typing import Tuple, Dict, List

def stylometry_feature_vector(text_data: str) -> np.ndarray:
    words = text_data.split()
    feature_vector = np.zeros((len(words), 3))
    first_person_pronouns = ["i", "me", "my", "mine", "myself"]
    second_person_pronouns = ["you", "your", "yours", "yourself"]
    third_person_pronouns = ["he", "him", "his", "himself"]
    for i, word in enumerate(words):
        if word.lower() in first_person_pronouns:
            feature_vector[i, 0] = 1
        elif word.lower() in second_person_pronouns:
            feature_vector[i, 1] = 1
        elif word.lower() in third_person_pronouns:
            feature_vector[i, 2] = 1
    return np.mean(feature_vector, axis=0)

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    dist: Dict[str, float] = {root: 0.0}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = edge_len[(a, b)]
    
    stack = [(root, 0.0)]
    while stack:
        node, d = stack.pop()
        for neighbor in adj[node]:
            if neighbor not in dist:
                dist[neighbor] = d + edge_len[(node, neighbor)]
                stack.append((neighbor, dist[neighbor]))
    return adj, edge_len, dist

def allocate_workshare_ssim(
    x: np.ndarray, 
    y: np.ndarray, 
    nodes: Dict[str, Tuple[float, float]], 
    edges: List[Tuple[str, str]], 
    root: str,
    *, 
    total_units: float
) -> np.ndarray:
    ssim = np.sum(np.abs(x - y)) / (np.sum(np.abs(x)) + np.sum(np.abs(y)))
    adj, edge_len, _ = tree_metrics(nodes, edges, root)
    workshare = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        if node == root:
            continue
        path = []
        current_node = node
        while current_node != root:
            path.append(current_node)
            for neighbor in adj[current_node]:
                if neighbor not in path:
                    current_node = neighbor
                    break
        path.append(root)
        path.reverse()
        for j in range(len(path) - 1):
            edge = (path[j], path[j+1])
            workshare[i] += edge_len[edge] * ssim
    return workshare / np.sum(workshare) * total_units

def hybrid_conductance_update(
    conductance: np.ndarray, 
    feature_vector: np.ndarray, 
    ssim: float, 
    nodes: Dict[str, Tuple[float, float]], 
    edges: List[Tuple[str, str]], 
    root: str
) -> np.ndarray:
    adj, edge_len, _ = tree_metrics(nodes, edges, root)
    tree_conductance = np.zeros_like(conductance)
    for node in nodes:
        if node == root:
            continue
        path = []
        current_node = node
        while current_node != root:
            path.append(current_node)
            for neighbor in adj[current_node]:
                if neighbor not in path:
                    current_node = neighbor
                    break
        path.append(root)
        path.reverse()
        for j in range(len(path) - 1):
            edge = (path[j], path[j+1])
            tree_conductance += edge_len[edge] * ssim
    return np.maximum(0.0, conductance + tree_conductance * feature_vector)

if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0)}
    edges = [("A", "B"), ("B", "C")]
    root = "A"
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    feature_vector = stylometry_feature_vector("Hello world!")
    ssim = np.sum(np.abs(x - y)) / (np.sum(np.abs(x)) + np.sum(np.abs(y)))
    conductance = np.array([1.0, 2.0, 3.0])
    print(allocate_workshare_ssim(x, y, nodes, edges, root, total_units=10.0))
    print(hybrid_conductance_update(conductance, feature_vector, ssim, nodes, edges, root))