# DARWIN HAMMER — match 4759, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2151_s3.py (gen6)
# parent_b: hybrid_state_space_duality_hybrid_hybrid_liquid_m1992_s0.py (gen3)
# born: 2026-05-29T23:57:57Z

"""
This module fuses the core mathematics of two parent algorithms:
- **Parent A – Schoolfield Model with Tree Metrics**
  Provides a mathematical framework for modeling developmental rates and tree metrics.
- **Parent B – Hybrid State Space Duality and Liquid Time Constant Diffusion Forcing (LTC-DF)**
  Implements a hybrid system that combines state space duality with liquid time constant diffusion forcing.

The mathematical bridge between the two algorithms is found in the integration of the developmental rate equation with the state update equation of the hybrid LTC-DF system. 
The developmental rate equation is used to modulate the state update equation, allowing for temperature-dependent state transitions.
The tree metrics are used to compute the distances between nodes in the state space, which are then used to inform the diffusion forcing process.
"""

import math
import numpy as np
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def temperature_dependent_state_transition(A: np.ndarray, temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    rate = developmental_rate(temp_k, params)
    return rate * A

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
        edge_len[(a, b)] = math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)

    return adj, edge_len, dist

def minhash_similarity(tokens1: List[str], tokens2: List[str], k: int = 128) -> float:
    signature1 = [min(hash(seed + token) for token in tokens1) for seed in range(k)]
    signature2 = [min(hash(seed + token) for token in tokens2) for seed in range(k)]
    return sum(1 for a, b in zip(signature1, signature2) if a == b) / k

def hybrid_state_update(
    A: np.ndarray,
    B: np.ndarray,
    x: np.ndarray,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> np.ndarray:
    rate = developmental_rate(temp_k, params)
    return rate * A @ x + B @ x

def hybrid_diffusion_forcing(
    x: np.ndarray,
    t: float,
    alpha: float,
    epsilon: np.ndarray,
) -> np.ndarray:
    return math.sqrt(alpha) * x + math.sqrt(1 - alpha) * epsilon

def main():
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (0.5, 1.0),
    }
    edges = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    root = 'A'
    adj, edge_len, dist = tree_metrics(nodes, edges, root)

    A = np.array([[0.5, 0.3], [0.2, 0.7]])
    B = np.array([[0.1, 0.2], [0.3, 0.1]])
    x = np.array([1.0, 1.0])
    temp_k = c_to_k(25.0)
    params = SchoolfieldParams()

    h = hybrid_state_update(A, B, x, temp_k, params)
    print(h)

    tokens1 = ['hello', 'world']
    tokens2 = ['hello', 'foo']
    similarity = minhash_similarity(tokens1, tokens2)
    print(similarity)

    alpha = 0.5
    epsilon = np.array([0.1, 0.1])
    x_noisy = hybrid_diffusion_forcing(x, temp_k, alpha, epsilon)
    print(x_noisy)

if __name__ == "__main__":
    main()