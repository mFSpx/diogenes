# DARWIN HAMMER — match 2151, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m760_s0.py (gen5)
# born: 2026-05-29T23:41:06Z

import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any

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

    return adj, edge_len, dist

def fisher_score(x: np.ndarray, y: np.ndarray) -> float:
    return np.dot(x, y)

def hybrid_state_transition(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_k: float,
    adj: Dict[str, List[str]],
    dist: Dict[str, float],
    edge_len: Dict[Tuple[str, str], float],
) -> tuple[np.ndarray, np.ndarray]:
    A_temp = temperature_dependent_state_transition(A, temp_k)
    h_new = A_temp @ h + B @ x
    y = C @ h_new

    # Fisher score calculation
    nodes = {"node1": (1.0, 1.0), "node2": (2.0, 2.0), "node3": (3.0, 3.0)}
    edges = [("node1", "node2"), ("node2", "node3")]
    adj, edge_len, dist = tree_metrics(nodes, edges, "node1")
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    score = fisher_score(x, y)

    return h_new, y, score

def hybrid_vram_scheduler(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    vram_budget: float,
    temp_seq: np.ndarray,
    h0: np.ndarray | None = None,
) -> np.ndarray:
    T, _ = temp_seq.shape

    # Initialize state transition matrix
    A = np.random.rand(3, 3)
    B = np.random.rand(3, 10)
    C = np.random.rand(10, 3)

    # Initialize VRAM usage
    vram_usage = np.zeros((T,))

    # Run hybrid algorithm
    for t in range(T):
        adj, edge_len, dist = tree_metrics(nodes, edges, root)
        h_new, y, score = hybrid_state_transition(h0 if h0 is not None else np.zeros((3,)), np.zeros((10,)), A, B, C, temp_seq[t], adj, dist, edge_len)
        vram_usage[t] = score  # Use Fisher score as VRAM usage

    return vram_usage

def hybrid_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_seq: np.ndarray,
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    vram_budget: float,
    h0: np.ndarray | None = None,
) -> np.ndarray:
    T, _ = temp_seq.shape

    # Initialize VRAM usage
    vram_usage = np.zeros((T,))

    # Run hybrid algorithm
    for t in range(T):
        adj, edge_len, dist = tree_metrics(nodes, edges, root)
        h_new, y, score = hybrid_state_transition(h0 if h0 is not None else np.zeros((3,)), np.zeros((10,)), A, B, C, temp_seq[t], adj, dist, edge_len)
        vram_usage[t] = score  # Use Fisher score as VRAM usage

    return vram_usage

if __name__ == "__main__":
    # Smoke test
    x_seq = np.random.rand(10, 10)
    A = np.random.rand(3, 3)
    B = np.random.rand(3, 10)
    C = np.random.rand(10, 3)
    temp_seq = np.random.rand(10)
    nodes = {"node1": (1.0, 1.0), "node2": (2.0, 2.0), "node3": (3.0, 3.0)}
    edges = [("node1", "node2"), ("node2", "node3")]
    root = "node1"
    vram_budget = 100.0
    h0 = np.random.rand(3,)

    hybrid_vram_scheduler(nodes, edges, root, vram_budget, temp_seq, h0)