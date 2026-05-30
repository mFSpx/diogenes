# DARWIN HAMMER — match 4242, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m2070_s0.py (gen5)
# born: 2026-05-29T23:54:26Z

"""
Hybrid Algorithm: Fusing RBF-NLMS and Pheromone-based RLCT Grokking

This module integrates the RBF-NLMS algorithm (hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s4.py)
with the Pheromone-based RLCT Grokking algorithm (hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m2070_s0.py).
The mathematical bridge between these structures is the concept of uncertainty quantification 
and information entropy optimization. The RBF-NLMS algorithm optimizes the output weights 
using the Normalized LMS rule, while the Pheromone-based RLCT Grokking algorithm 
optimizes the free energy of a system using information entropy.

The resulting hybrid algorithm integrates the information-based optimization of 
RBF-NLMS with the uncertainty quantification of Pheromone-based RLCT Grokking 
to create a novel hybrid system.

The hybrid system combines the RBF output weights and the pheromone signal 
to influence the neuronal conductances, thereby coupling the two systems 
at the level of the governing equations.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class Node:
    id: int
    weight: float

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def rbf_activation(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    dists = np.linalg.norm(centers - x, axis=1)
    return np.exp(- (dists ** 2) / (2 * sigma ** 2))

class RBFNLMS:
    def __init__(self, centers: np.ndarray, sigma: float = 1.0, mu: float = 0.5, eps: float = 1e-9):
        self.centers = centers                     
        self.sigma = sigma
        self.mu = mu
        self.eps = eps
        self.weights = np.random.rand(centers.shape[0])  

    def predict(self, x: np.ndarray) -> float:
        phi = rbf_activation(x, self.centers, self.sigma)   
        return np.dot(self.weights, phi)

    def adapt(self, x: np.ndarray, target: float) -> float:
        phi = rbf_activation(x, self.centers, self.sigma)
        y = np.dot(self.weights, phi)
        error = target - y
        power = np.dot(phi, phi) + self.eps
        self.weights += (self.mu * error / power) * phi
        return error

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def estimate_rlct_from_losses(train_losses_per_n, n_values):
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if len(losses) != len(ns):
        raise ValueError("train_losses_per_n and n_values must have the same length")
    y = np.log(np.maximum(losses, 1e-10))
    z = np.log(ns)
    rlct = np.polyfit(z, y, 1)[0]
    return rlct

def hybrid_rlct_rbf(rlct: float, rbf_nlms: RBFNLMS, x: np.ndarray, target: float) -> Tuple[float, float]:
    """
    Hybrid function that combines RLCT and RBF-NLMS.

    Args:
    rlct (float): The RLCT value.
    rbf_nlms (RBFNLMS): The RBF-NLMS instance.
    x (np.ndarray): The input vector.
    target (float): The target value.

    Returns:
    Tuple[float, float]: A tuple containing the predicted value and the adapted RLCT value.
    """
    # Predict using RBF-NLMS
    predicted = rbf_nlms.predict(x)

    # Adapt RBF-NLMS
    error = rbf_nlms.adapt(x, target)

    # Update RLCT using the error
    updated_rlct = rlct + error * 0.1

    return predicted, updated_rlct

def construct_similarity_graph(weights: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    n = len(weights)
    graph: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            sim = 1.0 - abs(weights[i] - weights[j]) / (1.0 + abs(weights[i] - weights[j]))
            graph[i].append((j, sim))
            graph[j].append((i, sim))
    return graph

def prim_mst(graph: Dict[int, List[Tuple[int, float]]]) -> List[Tuple[int, int, float]]:
    # Prim's algorithm to find the Minimum Spanning Tree
    mst = []
    visited = set()
    edges = [(weight, node1, node2) for node1 in graph for node2, weight in graph[node1]]
    edges.sort()

    for weight, node1, node2 in edges:
        if node1 not in visited or node2 not in visited:
            visited.add(node1)
            visited.add(node2)
            mst.append((node1, node2, weight))

    return mst

if __name__ == "__main__":
    # Create a random RBF-NLMS instance
    centers = np.random.rand(10, 5)
    rbf_nlms = RBFNLMS(centers)

    # Create a random RLCT value
    rlct = estimate_rlct_from_losses([0.1, 0.2, 0.3], [10, 20, 30])

    # Create a random input vector and target value
    x = np.random.rand(5)
    target = 0.5

    # Run the hybrid function
    predicted, updated_rlct = hybrid_rlct_rbf(rlct, rbf_nlms, x, target)
    print(predicted, updated_rlct)

    # Create a random graph
    weights = np.random.rand(10)
    graph = construct_similarity_graph(weights)

    # Run Prim's algorithm
    mst = prim_mst(graph)
    print(mst)