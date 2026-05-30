# DARWIN HAMMER — match 4577, survivor 3
# gen: 7
# parent_a: hybrid_infotaxis_hybrid_semantic_neig_m739_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1263_s2.py (gen6)
# born: 2026-05-29T23:56:42Z

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple
from pathlib import Path

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0   # inflow gain
    beta: float = 1.0    # outflow gain
    dt: float = 1.0
    base: float = 1.0    # unused but kept for compatibility
    gamma: float = 1.0

@dataclass(frozen=True)
class Node:
    id: int
    weight: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting time index to a normalized priority in [0,1]."""
    rti = righting_time_index(m)
    return max(0.0, min(1.0, rti / max_index))

def predict(weights: np.ndarray, x: np.ndarray) -> float:
    return np.dot(weights, x)

def update(weights: np.ndarray, x: np.ndarray, target: float, mu: float = 0.5, eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    y = predict(weights, x)
    error = target - y
    power = np.dot(x, x) + eps
    next_weights = weights + mu * error * x / power
    return next_weights, error

def construct_graph(weights: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    graph = {}
    for i in range(len(weights)):
        node = Node(i, weights[i])
        graph[node.id] = []
        for j in range(len(weights)):
            if i != j:
                similarity = np.exp(-((i-j)**2) / (2 * 1**2)) 
                graph[node.id].append((j, similarity))
    return graph

def hybrid_infotaxis_bandit(graph: Dict[int, List[Tuple[int, float]]], store_state: StoreState, morphology: Morphology) -> float:
    priority = recovery_priority(morphology)
    inflow = sum([similarity for _, (_, similarity) in [(n, edge) for n, edges in graph.items() for edge in edges] if _ == 0])  
    outflow = sum([weight for _, (_, weight) in [(n, edge) for n, edges in graph.items() for edge in edges] if _ == 0])  
    store_state.level = store_state.level + store_state.alpha * inflow - store_state.beta * outflow
    dance_signal = store_state.level
    for node_id, edges in graph.items():
        for i, (neighbor_id, _) in enumerate(edges):
            graph[node_id][i] = (neighbor_id, dance_signal * _)
    return store_state.level

def hybrid_affinity_bandit(graph: Dict[int, List[Tuple[int, float]]], store_state: StoreState, morphology: Morphology) -> float:
    priority = recovery_priority(morphology)
    expected_entropy = sum([similarity for _, (_, similarity) in [(n, edge) for n, edges in graph.items() for edge in edges]])
    hybrid_affinity = expected_entropy * priority
    outflow = sum([weight for _, (_, weight) in [(n, edge) for n, edges in graph.items() for edge in edges]])
    store_state.level = store_state.level + store_state.alpha * hybrid_affinity - store_state.beta * outflow
    for node_id, edges in graph.items():
        for i, (neighbor_id, _) in enumerate(edges):
            graph[node_id][i] = (neighbor_id, hybrid_affinity * _)
    return store_state.level

def hybrid_infotaxis_semantic_neighbor(store_state: StoreState, morphology: Morphology, graph: Dict[int, List[Tuple[int, float]]]) -> float:
    priority = recovery_priority(morphology)
    expected_entropy = sum([similarity for _, (_, similarity) in [(n, edge) for n, edges in graph.items() for edge in edges]])
    hybrid_affinity = expected_entropy * priority
    outflow = sum([weight for _, (_, weight) in [(n, edge) for n, edges in graph.items() for edge in edges]])
    store_state.level = store_state.level + store_state.alpha * hybrid_affinity - store_state.beta * outflow
    for node_id, edges in graph.items():
        for i, (neighbor_id, _) in enumerate(edges):
            graph[node_id][i] = (neighbor_id, hybrid_affinity * _)
    return store_state.level

if __name__ == "__main__":
    morphology = Morphology(length=1.0, width=2.0, height=3.0, mass=4.0)
    graph = construct_graph(np.array([1.0, 2.0, 3.0]))
    store_state = StoreState(level=0.0, alpha=1.0, beta=1.0, dt=1.0, base=1.0, gamma=1.0)
    result = hybrid_infotaxis_bandit(graph, store_state, morphology)
    print(result)
    result = hybrid_affinity_bandit(graph, store_state, morphology)
    print(result)
    result = hybrid_infotaxis_semantic_neighbor(store_state, morphology, graph)
    print(result)