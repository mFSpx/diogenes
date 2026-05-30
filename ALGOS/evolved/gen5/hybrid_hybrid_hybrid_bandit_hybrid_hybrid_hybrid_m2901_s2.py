# DARWIN HAMMER — match 2901, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_fisher_locali_m1102_s0.py (gen4)
# born: 2026-05-29T23:46:40Z

import json
import math
import random
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import List

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Simple in-memory policy store
# ----------------------------------------------------------------------
_POLICY: dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

# ----------------------------------------------------------------------
# Fisher-Voronoi utilities
# ----------------------------------------------------------------------
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.linalg.norm(x - y)

def voronoi_partition(seeds: np.ndarray, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    distances = np.zeros((seeds.shape[0], points.shape[0]))
    for i, seed in enumerate(seeds):
        for j, point in enumerate(points):
            distances[i, j] = euclidean_distance(seed, point)
    return distances

def fisher_score(packet: np.ndarray) -> float:
    return np.sum(packet ** 2)

# ----------------------------------------------------------------------
# Ollivier-Ricci Curvature
# ----------------------------------------------------------------------
def compute_ollivier_ricci_curvature(graph: np.ndarray) -> np.ndarray:
    n = graph.shape[0]
    curvature = np.zeros(n)
    for i in range(n):
        neighbors = np.where(graph[i, :] > 0)[0]
        if len(neighbors) == 0:
            curvature[i] = 0
            continue
        neighbor_curvature = np.zeros(len(neighbors))
        for j, neighbor in enumerate(neighbors):
            neighbor_curvature[j] = 1 - np.dot(graph[i, :], graph[neighbor, :]) / (np.linalg.norm(graph[i, :]) * np.linalg.norm(graph[neighbor, :]))
        curvature[i] = np.mean(neighbor_curvature)
    return curvature

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_bandit_reward(node: np.ndarray, graph: np.ndarray) -> float:
    node_curvature = compute_node_curvature(graph, node)
    return node_curvature

def compute_node_curvature(graph: np.ndarray, node: np.ndarray) -> float:
    ollivier_ricci_curvature = compute_ollivier_ricci_curvature(graph)
    return ollivier_ricci_curvature[np.argmax(graph[node, :])]

def compute_node_similarity(graph: np.ndarray, node: np.ndarray) -> float:
    return np.sum(graph[node, :])

def hybrid_update_policy(updates: List[BanditUpdate], graph: np.ndarray, seeds: np.ndarray, points: np.ndarray) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0
        node_curvature = hybrid_bandit_reward(np.array([int(u.action_id)]), graph)
        edge_costs = np.zeros((points.shape[0], seeds.shape[0]))
        for i, point in enumerate(points):
            for j, seed in enumerate(seeds):
                edge_costs[i, j] = euclidean_distance(point, seed) + 0.1 * node_curvature * fisher_score(point)
        graph[np.argmin(edge_costs, axis=1), :] += 0.1 * edge_costs[np.argmin(edge_costs, axis=1), :]

def hybrid_routing(points: np.ndarray, seeds: np.ndarray, graph: np.ndarray) -> List[str]:
    edge_costs = np.zeros((points.shape[0], seeds.shape[0]))
    for i, point in enumerate(points):
        for j, seed in enumerate(seeds):
            node_curvature = compute_node_curvature(graph, seed)
            edge_costs[i, j] = euclidean_distance(point, seed) + 0.1 * node_curvature * fisher_score(point)
    return [np.argmin(edge_costs[i, :]) for i in range(points.shape[0])]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    points = np.random.rand(10, 2)
    seeds = np.random.rand(5, 2)
    graph = np.random.rand(10, 10)
    graph = graph + graph.T
    np.fill_diagonal(graph, 0)
    updates = [BanditUpdate("context1", "0", 1.0, 0.5), BanditUpdate("context1", "1", 0.5, 0.5)]
    hybrid_update_policy(updates, graph, seeds, points)
    print(hybrid_routing(points, seeds, graph))