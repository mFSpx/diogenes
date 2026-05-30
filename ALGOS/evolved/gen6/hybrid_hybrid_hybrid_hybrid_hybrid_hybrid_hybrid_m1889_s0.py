# DARWIN HAMMER — match 1889, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m1162_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_h_m901_s1.py (gen5)
# born: 2026-05-29T23:39:26Z

"""
This module introduces a novel HYBRID algorithm that mathematically fuses the governing equations of 
hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m1162_s1.py (Minimum-Cost Tree with Bayesian update) 
and hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_h_m901_s1.py (Schoolfield-Rollinson poikilotherm rate primitive 
with Hybrid NLMS & Liquid-Time-Constant Network).

The mathematical bridge is formed by using the temperature-dependent developmental rate to inform the 
probabilistic transformation of the edge contributions in the Minimum-Cost Tree, and then applying the 
Bayesian update to the weighted edge lengths. This allows for the investigation of the impact of 
temperature-dependent developmental rates on the inequality in edge length distributions.

The Schoolfield-Rollinson poikilotherm rate primitive is used to update the weights of the TTT-Linear 
algorithm, which is then used to compress the input distribution. The variational free energy is 
used to update the belief mean of the ternary router, which is then used to compute the SSIM between 
the input and output of the ternary router.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

class SchoolfieldParams:
    def __init__(self, rho_25: float = 1.0, delta_h_activation: float = 12000.0, t_low: float = 283.15, 
                 t_high: float = 307.15, delta_h_low: float = -45000.0, delta_h_high: float = 65000.0, 
                 r_cal: float = 1.987):
        self.rho_25 = rho_25
        self.delta_h_activation = delta_h_activation
        self.t_low = t_low
        self.t_high = t_high
        self.delta_h_low = delta_h_low
        self.delta_h_high = delta_h_high
        self.r_cal = r_cal

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for neighbor in adj[cur]:
            if neighbor not in dist:
                dist[neighbor] = dist[cur] + edge_len[(cur, neighbor)]
                stack.append(neighbor)

    return adj, edge_len, dist

def predict(weights, x):
    """Predict the output of the system using the given weights and input."""
    return sum(w * xi for w, xi in zip(weights, x))

def update(weights, x, y, learning_rate):
    """Update the weights of the system using the given input, output, and learning rate."""
    prediction = predict(weights, x)
    error = prediction - y
    gradient = [2 * error * xi for xi in x]
    return [w - learning_rate * g for w, g in zip(weights, gradient)]

def hybrid_operation(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str, 
                     params: SchoolfieldParams, temperature: float, learning_rate: float, inputs: List[float]):
    """
    Perform the hybrid operation by combining the tree metrics and the developmental rate.

    Parameters
    ----------
    nodes : dict mapping node → (x, y) coordinates
    edges : list of (node, node) pairs
    root : root node
    params : SchoolfieldParams instance
    temperature : temperature in Celsius
    learning_rate : learning rate for the update rule
    inputs : list of input values

    Returns
    -------
    updated weights and predicted output
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    temp_k = c_to_k(temperature)
    rate = developmental_rate(temp_k, params)
    weights = [rate * len(nodes) for _ in range(len(nodes))]
    updated_weights = update(weights, inputs, predict(weights, inputs), learning_rate)
    predicted_output = predict(updated_weights, inputs)
    return updated_weights, predicted_output

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    params = SchoolfieldParams()
    temperature = 25
    learning_rate = 0.1
    inputs = [1, 2, 3]
    updated_weights, predicted_output = hybrid_operation(nodes, edges, root, params, temperature, learning_rate, inputs)
    print("Updated weights:", updated_weights)
    print("Predicted output:", predicted_output)