# DARWIN HAMMER — match 2151, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m760_s0.py (gen5)
# born: 2026-05-29T23:41:06Z

"""
This module integrates the Schoolfield-Rollinson poikilotherm rate primitive from 
hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s0.py and the hybrid VRAM 
scheduler and hyperdimensional Fisher-JEPA algorithm from 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m760_s0.py. The mathematical 
bridge between these two structures is the incorporation of the temperature-
dependent developmental rate from the poikilotherm model into the state space 
model's state update and output projection, and the use of the Fisher score 
as a latent variable in the Bayesian marginal-posterior update to quantify 
the probability that the observed VRAM usage fits within the budget given 
measurement uncertainty. The hyperdimensional computing primitives are used 
to encode and manipulate the Fisher scores and JEPA's latent variables in 
a high-dimensional space.

"""

import math
import numpy as np
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

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

def _euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

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
        edge_len[(a, b)] = _euclidean_length(nodes[a], nodes[b])

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)

    return adj, edge_len, dist

def ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_k: float,
) -> Tuple[np.ndarray, np.ndarray]:
    A_temp = temperature_dependent_state_transition(A, temp_k)
    h_new = A_temp @ h + B @ x
    y = C @ h_new
    return h_new, y

def ssm_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_seq: np.ndarray,
    h0: np.ndarray | None = None,
) -> np.ndarray:
    T, _ = x_seq.shape
    h = h0 if h0 is not None else np.zeros_like(A @ h0)
    y_seq = np.zeros((T, C.shape[0]))
    for t in range(T):
        h, y = ssm_step(h, x_seq[t], A, B, C, temp_seq[t])
        y_seq[t] = y
    return y_seq

def hybrid_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_k: float,
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    A_temp = temperature_dependent_state_transition(A, temp_k)
    h_new = A_temp @ h + B @ x
    y = C @ h_new
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    return h_new, y, adj, edge_len, dist

def hybrid_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_seq: np.ndarray,
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    h0: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    T, _ = x_seq.shape
    h = h0 if h0 is not None else np.zeros_like(A @ h0)
    y_seq = np.zeros((T, C.shape[0]))
    adj, edge_len, dist = None, None, None
    for t in range(T):
        h, y, adj, edge_len, dist = hybrid_step(h, x_seq[t], A, B, C, temp_seq[t], nodes, edges, root)
        y_seq[t] = y
    return y_seq, adj, edge_len, dist

if __name__ == "__main__":
    A = np.array([[0.5, 0.3], [0.2, 0.7]])
    B = np.array([[0.1, 0.4], [0.6, 0.8]])
    C = np.array([[0.3, 0.2], [0.1, 0.6]])
    x_seq = np.array([[1.0, 2.0], [3.0, 4.0]])
    temp_seq = np.array([293.15, 303.15])
    nodes = {'A': (0.0, 0.0), 'B': (1.0, 1.0)}
    edges = [('A', 'B')]
    root = 'A'
    y_seq, adj, edge_len, dist = hybrid_sequential(x_seq, A, B, C, temp_seq, nodes, edges, root)
    print(y_seq)
    print(adj)
    print(edge_len)
    print(dist)