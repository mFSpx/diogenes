# DARWIN HAMMER — match 2781, survivor 7
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_poikilotherm__m1522_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2043_s0.py (gen4)
# born: 2026-05-29T23:45:51Z

"""HybridTreeRBFBandit
Integrates:
- Parent A (hybrid_hybrid_minimu_hybrid_poikilotherm): tree construction, Euclidean metrics,
  and temperature‑dependent scaling ρ(T) for posterior‐like updates.
- Parent B (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit): radial‑basis‑function (RBF)
  surrogate for estimating edge‑wise uncertainties, Hoeffding‑based confidence bounds,
  and a simple multi‑armed‑bandit reward updater.

Mathematical bridge:
The temperature factor ρ(T) (from the Schoolfield‑Rollinson model) is used as a
global physiological scaler that modulates the raw RBF prediction p(e) for each
edge e. The temperature‑adjusted prediction
    ĉ(e) = ρ(T)·p(e)
acts as a surrogate “cost” (or negative reward) for the edge.  A Hoeffding bound
provides a confidence interval around ĉ(e); the upper confidence bound is fed
to a bandit‑style exponential‑moving‑average reward estimator.  Thus the tree
metric, the temperature physics, the surrogate uncertainty, and the bandit
learning loop are fused into a single coherent system.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Sequence
import numpy as np

# ----------------------------------------------------------------------
# Types
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – tree utilities ------------------------------------------------

def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def build_tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency list, compute edge lengths, and root‑to‑node distances.

    Returns
    -------
    adj : dict node → list of neighbours
    edge_len : dict (u, v) → Euclidean length (order as supplied)
    dist : dict node → distance from *root* along the unique path
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = euclidean_distance(nodes[u], nodes[v])

    # BFS to compute root distances
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[key]
                stack.append(nxt)

    return adj, edge_len, dist

# ----------------------------------------------------------------------
# Parent B – RBF surrogate & linear solver ---------------------------------

def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b using plain Gaussian elimination (no pivoting beyond max)."""
    n = len(b)
    # augment matrix
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # pivot on largest absolute value for numerical stability
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("Singular system in surrogate solve")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

class RBFSurrogate:
    """
    Simple RBF interpolant:
        f(x) = Σ_i w_i * φ(‖x - c_i‖)
    where φ is the Gaussian RBF.
    """
    def __init__(self, centers: List[Tuple[float, ...]], weights: List[float], epsilon: float = 1.0):
        if len(centers) != len(weights):
            raise ValueError("centers and weights must have same length")
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: Vector) -> float:
        """Evaluate the surrogate at point x."""
        total = 0.0
        for c, w in zip(self.centers, self.weights):
            r = math.sqrt(sum((xi - ci) ** 2 for xi, ci in zip(x, c)))
            total += w * gaussian_rbf(r, self.epsilon)
        return total

# ----------------------------------------------------------------------
# Fusion – temperature scaling, edge cost, bandit update ------------------

def temperature_rate(T: float, T_ref: float = 25.0, E: float = 0.65, k: float = 0.008617) -> float:
    """
    Schoolfield‑Rollinson temperature scaling ρ(T).

    Parameters
    ----------
    T : float
        Ambient temperature in °C.
    T_ref : float, default 25°C
        Reference temperature.
    E : float
        Activation energy (eV). Chosen arbitrarily for demonstration.
    k : float
        Boltzmann constant in eV·K⁻¹.

    Returns
    -------
    rho : float
        Dimensionless scaling factor, ρ(T_ref) = 1.
    """
    Tk = T + 273.15
    Tk_ref = T_ref + 273.15
    return math.exp(-E / (k * Tk)) / math.exp(-E / (k * Tk_ref))

def edge_surrogate_cost(
    u: str,
    v: str,
    nodes: Dict[str, Tuple[float, float]],
    rbf: RBFSurrogate,
    T: float,
) -> float:
    """
    Compute temperature‑adjusted surrogate cost for edge (u, v).

    The surrogate is evaluated at the midpoint of the edge; the raw prediction
    is then multiplied by ρ(T).
    """
    x1, y1 = nodes[u]
    x2, y2 = nodes[v]
    midpoint = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    raw = rbf.predict(midpoint)
    return temperature_rate(T) * raw

def hoeffding_upper_confidence(mean: float, n: int, delta: float = 0.05) -> float:
    """
    Hoeffding bound for bounded [0,1] rewards.
    Returns an upper confidence bound for the true mean.
    """
    if n == 0:
        return 1.0  # maximum possible reward before any observation
    radius = math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    return min(1.0, mean + radius)

class EdgeBandit:
    """
    Simple multi‑armed bandit over edges.
    Stores empirical mean reward and count; uses Hoeffding UCB for selection.
    """
    def __init__(self, edges: List[Tuple[str, str]]):
        self.counts: Dict[Tuple[str, str], int] = {e: 0 for e in edges}
        self.means: Dict[Tuple[str, str], float] = {e: 0.0 for e in edges}

    def update(self, edge: Tuple[str, str], reward: float):
        """Incremental update of empirical mean."""
        c = self.counts[edge]
        m = self.means[edge]
        new_c = c + 1
        new_m = (m * c + reward) / new_c
        self.counts[edge] = new_c
        self.means[edge] = new_m

    def ucb(self, edge: Tuple[str, str], delta: float = 0.05) -> float:
        """Upper confidence bound for the given edge."""
        return hoeffding_upper_confidence(self.means[edge], self.counts[edge], delta)

# ----------------------------------------------------------------------
# High‑level hybrid functions ----------------------------------------------

def compute_hybrid_edge_costs(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    rbf: RBFSurrogate,
    T: float,
) -> Dict[Tuple[str, str], float]:
    """
    For every edge, compute:
        cost(e) = length(e) + temperature‑scaled RBF prediction.
    The length term ensures geometric fidelity; the surrogate term injects
    learned (or simulated) physiological cost.
    """
    costs: Dict[Tuple[str, str], float] = {}
    for u, v in edges:
        length = euclidean_distance(nodes[u], nodes[v])
        surrogate = edge_surrogate_cost(u, v, nodes, rbf, T)
        costs[(u, v)] = length + surrogate
    return costs

def run_hybrid_bandit_iteration(
    edges: List[Tuple[str, str]],
    costs: Dict[Tuple[str, str], float],
    bandit: EdgeBandit,
    exploration_rate: float = 0.1,
) -> Tuple[Tuple[str, str], float]:
    """
    Perform a single decision step:
        - With probability `exploration_rate` pick a random edge.
        - Otherwise pick edge with highest UCB (i.e. lowest cost‑adjusted reward).
    Simulated reward = 1 / (1 + cost) to keep it in (0,1].
    """
    if random.random() < exploration_rate:
        chosen = random.choice(edges)
    else:
        # convert cost to a reward-like quantity (higher is better)
        ucb_vals = {e: bandit.ucb(e) - costs[e] for e in edges}
        chosen = max(ucb_vals, key=ucb_vals.get)

    reward = 1.0 / (1.0 + costs[chosen])  # bounded in (0,1]
    bandit.update(chosen, reward)
    return chosen, reward

def build_hybrid_tree(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    rbf: RBFSurrogate,
    T: float,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Construct the tree metrics while embedding hybrid costs.
    Returns adjacency, edge_costs (length+surrogate), and root distances.
    """
    adj, edge_len, dist = build_tree_metrics(nodes, edges, root)
    hybrid_costs = compute_hybrid_edge_costs(nodes, edges, rbf, T)

    # Overwrite the plain Euclidean lengths with hybrid costs for downstream use
    for e in edge_len:
        # edge_len uses ordered key; hybrid_costs may be stored (u,v) as supplied.
        # Ensure both orderings map.
        cost = hybrid_costs.get(e) or hybrid_costs.get((e[1], e[0]))
        if cost is not None:
            edge_len[e] = cost
    return adj, edge_len, dist

# ----------------------------------------------------------------------
# Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    # Simple synthetic graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")]
    root = "A"

    # Create a tiny RBF surrogate: centers are the node coordinates, weights random
    random.seed(42)
    centers = [nodes[n] for n in nodes]
    weights = [random.uniform(-0.5, 0.5) for _ in centers]
    rbf = RBFSurrogate(centers, weights, epsilon=1.5)

    # Temperature (°C)
    T = 30.0

    # Build hybrid tree
    adj, hybrid_edge_len, dist = build_hybrid_tree(nodes, edges, root, rbf, T)

    # Initialise bandit over edges
    bandit = EdgeBandit(edges)

    # Run a few bandit iterations
    for it in range(10):
        chosen_edge, reward = run_hybrid_bandit_iteration(edges,
                                                          {e: hybrid_edge_len.get(e, hybrid_edge_len.get((e[1], e[0]))) for e in edges},
                                                          bandit,
                                                          exploration_rate=0.2)
        print(f"Iter {it+1:02d}: chosen {chosen_edge}, reward {reward:.4f}")

    # Display final hybrid costs
    print("\nHybrid edge costs (length + temp‑scaled surrogate):")
    for e, c in hybrid_edge_len.items():
        print(f"  {e}: {c:.4f}")

    # Verify distances from root using the original Euclidean lengths (unchanged)
    print("\nRoot‑to‑node Euclidean distances:")
    for node, d in dist.items():
        print(f"  {node}: {d:.4f}")