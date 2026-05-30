# DARWIN HAMMER — match 211, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_bandit_m53_s1.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s0.py (gen2)
# born: 2026-05-29T23:27:31Z

"""
This module integrates the Hybrid Bandit-Koopman-Linear Fusion from hybrid_hybrid_hybrid_bandit_hybrid_hybrid_bandit_m53_s1.py 
and the Fisher localization with minimum cost tree and Bayesian evidence update from hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s0.py.
The mathematical bridge between the two structures is the application of Gaussian distributions and probability updates.
In hybrid_hybrid_hybrid_bandit_hybrid_hybrid_bandit_m53_s1.py, a bandit algorithm uses empirical mean and count to calculate propensity scores,
while in hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s0.py, a Gaussian beam is used to model the intensity of a signal.
This module combines these concepts to create a hybrid algorithm that uses Gaussian distributions to model and smooth out chronological data,
and updates the bandit propensity scores based on the Fisher information score and minimum cost tree cost function.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

_POLICY: dict = {}  # action_id → [total_reward, count]
_STORE: float = 0.0  # scalar store influencing confidence
_MEAN_HISTORY: list = []  # list of μ vectors over time
_W: np.ndarray = np.array([])  # linear weight matrix (A×A)
_ETA: float = 1.0  # exploration scaling
_ALPHA: float = 0.5  # mixing factor for hybrid index
_NODES: dict = {}  # nodes for minimum cost tree
_EDGES: list = []  # edges for minimum cost tree
_ROOT: str = ""  # root node for minimum cost tree

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def length(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: dict, edges: list, root: str, path_weight: float = 0.2) -> float:
    adj = {n: [] for n in nodes}
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

def update_propensity_scores(action_id: int, reward: float, center: float, width: float) -> float:
    global _POLICY, _STORE
    if action_id not in _POLICY:
        _POLICY[action_id] = [0.0, 0]
    _POLICY[action_id][0] += reward
    _POLICY[action_id][1] += 1
    _STORE += reward
    propensity_score = _POLICY[action_id][0] / _POLICY[action_id][1] + _ETA * fisher_score(_POLICY[action_id][0] / _POLICY[action_id][1], center, width)
    return propensity_score

def update_hybrid_index(action_id: int, propensity_score: float, koopman_forecast: float) -> float:
    global _ALPHA
    hybrid_index = _ALPHA * koopman_forecast + (1 - _ALPHA) * propensity_score
    return hybrid_index

def get_koopman_forecast(mean_history: list) -> float:
    global _MEAN_HISTORY
    _MEAN_HISTORY = mean_history
    if len(_MEAN_HISTORY) < 2:
        return 0.0
    X = np.array(_MEAN_HISTORY[:-1])
    Y = np.array(_MEAN_HISTORY[1:])
    K = np.dot(Y, X.T) / np.dot(X, X.T)
    return np.dot(K, _MEAN_HISTORY[-1])

def calculate_minimum_cost_tree(action_id: int, nodes: dict, edges: list, root: str) -> float:
    global _NODES, _EDGES, _ROOT
    _NODES = nodes
    _EDGES = edges
    _ROOT = root
    return tree_cost(_NODES, _EDGES, _ROOT)

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C")]
    root = "A"
    print(calculate_minimum_cost_tree(0, nodes, edges, root))
    print(update_propensity_scores(0, 1.0, 0.5, 1.0))
    print(get_koopman_forecast([0.1, 0.2, 0.3]))