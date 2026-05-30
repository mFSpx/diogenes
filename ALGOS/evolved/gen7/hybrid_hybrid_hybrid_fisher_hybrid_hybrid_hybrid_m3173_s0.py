# DARWIN HAMMER — match 3173, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_vorono_m1224_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1233_s0.py (gen6)
# born: 2026-05-29T23:48:10Z

import numpy as np
import math
import random
import sys
import pathlib

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def krampus_curvature(graph: Dict[str, List[str]]) -> Dict[str, float]:
    curvature: Dict[str, float] = {}
    for node in graph:
        neighbours = graph[node]
        curvature[node] = len(neighbours) / (len(graph) - 1)
    return curvature

def hash_node(node: Tuple[float, float]) -> int:
    return int(hash(node[0]) * 1000 + hash(node[1]))

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    adj = {node: [] for node in nodes}
    for edge in edges:
        adj[edge[0]].append(edge[1])
        adj[edge[1]].append(edge[0])
    edge_lengths = {}
    for edge in edges:
        edge_lengths[edge] = np.linalg.norm(np.array(nodes[edge[0]]) - np.array(nodes[edge[1]]))
    distances = {}
    stack = [(root, 0)]
    while stack:
        node, dist = stack.pop()
        distances[node] = dist
        for neighbor in adj[node]:
            if neighbor not in distances:
                stack.append((neighbor, dist + edge_lengths[(node, neighbor)]))
    return adj, edge_lengths, distances

def hybrid_krampus_fisher_score(
    theta: float,
    center: float,
    width: float,
    curvature: Dict[str, float],
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> float:
    intensity = max(gaussian_beam(theta, center, width), 1e-12)
    derivative = intensity * (-(theta - center) / (width * width))
    krampus_term = 0
    for node in nodes:
        krampus_term += curvature[node] * (np.linalg.norm(np.array(nodes[node]) - np.array([theta, center])))
    return (derivative * derivative) / intensity + krampus_term

def hybrid_voronoi_krampus_curvature(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    curvature: Dict[str, float],
) -> Dict[str, float]:
    adj, edge_lengths, distances = tree_metrics(nodes, edges, root)
    voronoi_curvature = {}
    for node in nodes:
        voronoi_curvature[node] = np.mean([curvature[neighbor] for neighbor in adj[node]])
    return voronoi_curvature

def hybrid_fisher_voronoi(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    curvature: Dict[str, float],
    width: float,
) -> Dict[str, float]:
    adj, edge_lengths, distances = tree_metrics(nodes, edges, root)
    voronoi_curvature = {}
    for node in nodes:
        voronoi_curvature[node] = np.mean([fisher_score(theta, center, width) for theta, center in [np.array(nodes[node])]])
    return {node: voronoi_curvature[node] + curvature[node] for node in nodes}

if __name__ == "__main__":
    # Smoke test
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    curvature = {"A": 1.0, "B": 2.0, "C": 3.0}
    width = 0.5
    print(hybrid_krampus_fisher_score(0.5, 0.5, width, curvature, nodes, edges, root))
    print(hybrid_voronoi_krampus_curvature(nodes, edges, root, curvature))
    print(hybrid_fisher_voronoi(nodes, edges, root, curvature, width))