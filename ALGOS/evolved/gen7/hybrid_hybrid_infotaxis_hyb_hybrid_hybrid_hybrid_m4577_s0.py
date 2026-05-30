# DARWIN HAMMER — match 4577, survivor 0
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


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


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
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


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
                similarity = 1  # This value is not used in the original code, so we set it to 1
                graph[node.id].append((j, similarity))
    return graph

def hybrid_infotaxis_bandit(graph: Dict[int, List[Tuple[int, float]]], store_state: StoreState, morphology: Morphology) -> float:
    """Hybrid infotaxis-bandit algorithm."""
    # Calculate recovery priority
    priority = recovery_priority(morphology)
    
    # Calculate inflow and outflow for store update equation
    inflow = np.sum([similarity for _, (neighbor, similarity) in graph.items() if neighbor == 0])  # This is a placeholder for the LSM similarity
    outflow = np.sum([weight for neighbor, (neighbor_id, weight) in graph.items() if neighbor_id == 0])  # This is a placeholder for the edge lengths
    
    # Update store state using store dynamics equation
    store_state.level = store_state.level + store_state.alpha * inflow - store_state.beta * outflow
    
    # Calculate dance signal
    dance_signal = store_state.level
    
    # Modulate bandit propensities with dance signal
    for action in graph[0]:
        action.propensity = action.propensity * dance_signal
    
    return store_state.level

def hybrid_affinity_bandit(graph: Dict[int, List[Tuple[int, float]]], store_state: StoreState, morphology: Morphology) -> float:
    """Hybrid affinity-bandit algorithm."""
    # Calculate recovery priority
    priority = recovery_priority(morphology)
    
    # Calculate expected entropy
    expected_entropy = np.sum([similarity for _, (neighbor, similarity) in graph.items()])
    
    # Calculate hybrid affinity
    hybrid_affinity = expected_entropy * priority
    
    # Update store state using store dynamics equation
    store_state.level = store_state.level + store_state.alpha * hybrid_affinity - store_state.beta * np.sum([weight for _, (neighbor, weight) in graph.items()])
    
    # Modulate bandit propensities with hybrid affinity
    for action in graph[0]:
        action.propensity = action.propensity * hybrid_affinity
    
    return store_state.level

def hybrid_infotaxis_semantic_neighbor(store_state: StoreState, morphology: Morphology, graph: Dict[int, List[Tuple[int, float]]]) -> float:
    """Hybrid infotaxis-semantic neighbor algorithm."""
    # Calculate recovery priority
    priority = recovery_priority(morphology)
    
    # Calculate expected entropy
    expected_entropy = np.sum([similarity for _, (neighbor, similarity) in graph.items()])
    
    # Calculate hybrid affinity
    hybrid_affinity = expected_entropy * priority
    
    # Update store state using store dynamics equation
    store_state.level = store_state.level + store_state.alpha * hybrid_affinity - store_state.beta * np.sum([weight for _, (neighbor, weight) in graph.items()])
    
    # Modulate bandit propensities with hybrid affinity
    for action in graph[0]:
        action.propensity = action.propensity * hybrid_affinity
    
    return store_state.level

if __name__ == "__main__":
    # Create a sample morphology
    morphology = Morphology(length=1.0, width=2.0, height=3.0, mass=4.0)
    
    # Create a sample graph
    graph = construct_graph(np.array([1.0, 2.0, 3.0]))
    
    # Create a sample store state
    store_state = StoreState(level=0.0, alpha=1.0, beta=1.0, dt=1.0, base=1.0, gamma=1.0)
    
    # Run the hybrid infotaxis-bandit algorithm
    result = hybrid_infotaxis_bandit(graph, store_state, morphology)
    print(result)
    
    # Run the hybrid affinity-bandit algorithm
    result = hybrid_affinity_bandit(graph, store_state, morphology)
    print(result)
    
    # Run the hybrid infotaxis-semantic neighbor algorithm
    result = hybrid_infotaxis_semantic_neighbor(store_state, morphology, graph)
    print(result)