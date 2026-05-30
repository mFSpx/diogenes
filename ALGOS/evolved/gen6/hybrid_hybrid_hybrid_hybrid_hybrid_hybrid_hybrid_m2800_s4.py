# DARWIN HAMMER — match 2800, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_indy_learning_m440_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_model__m1800_s1.py (gen4)
# born: 2026-05-29T23:46:03Z

import math
import random
import sys
from datetime import date
from pathlib import Path
import numpy as np
from scipy import integrate

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493118e-7
])

Point = tuple[float, float]
Edge = tuple[str, str]

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
) -> tuple[dict[str, list[str]], dict[Edge, float], dict[str, float]]:
    adj: dict[str, list[str]] = {}
    edge_len: dict[Edge, float] = {}
    node_dist: dict[str, float] = {}

    for a, b in edges:
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    for a, b in edges:
        if a not in adj:
            adj[a] = []
        if b not in adj:
            adj[b] = []
        adj[a].append(b)
        adj[b].append(a)

    node_dist[root] = 0.0
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbour in adj[node]:
            if neighbour not in node_dist:
                node_dist[neighbour] = node_dist[node] + edge_len[(node, neighbour)]
                queue.append(neighbour)

    return adj, edge_len, node_dist

def caputo_derivative(f: callable, alpha: float, t: float) -> float:
    return (1 / math.gamma(1 - alpha)) * integrate.quad(lambda x: (t - x)**(alpha - 1) * f(x), 0, t)[0]

def ltc_network(input_value: float, alpha: float) -> float:
    return caputo_derivative(lambda t: input_value, alpha, 1.0)

def bandit_router(action_propensity: float) -> int:
    return np.random.choice([0, 1], p=[1 - action_propensity, action_propensity])

def hybrid_llm_allocation(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    llm_base: float,
    alpha: float,
    input_value: float,
) -> float:
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    tau_sys = ltc_network(input_value, alpha)

    scaled_edge_len = {edge: edge_len[edge] * tau_sys for edge in edge_len}

    loss = sum(scaled_edge_len.values())

    llm_allocation = llm_base * (loss / len(scaled_edge_len))

    return llm_allocation

def hybrid_chunking(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    action_propensity: float,
) -> list[tuple[str, str]]:
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    chunks = []

    for node in adj:
        if bandit_router(action_propensity) == 1:
            if adj[node]:
                chunks.append((node, np.random.choice(adj[node])))

    return chunks

def hybrid_operation(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    llm_base: float,
    alpha: float,
    input_value: float,
    action_propensity: float,
) -> tuple[float, list[tuple[str, str]]]:
    llm_allocation = hybrid_llm_allocation(nodes, edges, root, llm_base, alpha, input_value)
    chunks = hybrid_chunking(nodes, edges, root, action_propensity)

    return llm_allocation, chunks

if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    edges = [("A", "B"), ("A", "C")]
    root = "A"
    llm_base = 0.5
    alpha = 0.7
    input_value = 0.8
    action_propensity = 0.9

    llm_allocation, chunks = hybrid_operation(nodes, edges, root, llm_base, alpha, input_value, action_propensity)
    print(f"LLM allocation: {llm_allocation:.6f}")
    print(f"Chunks: {chunks}")