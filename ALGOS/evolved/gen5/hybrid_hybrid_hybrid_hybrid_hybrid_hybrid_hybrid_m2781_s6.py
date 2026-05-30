# DARWIN HAMMER — match 2781, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_poikilotherm__m1522_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2043_s0.py (gen4)
# born: 2026-05-29T23:45:51Z

"""Hybrid Tree-RBF Cost Model
Parent A: hybrid_hybrid_hybrid_minimu_hybrid_poikilotherm__m1522_s1.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2043_s0.py

Mathematical Bridge
-------------------
The bridge is the edge‑wise posterior weight wₑ.  
Parent A supplies a tree topology and Euclidean edge lengths Lₑ.  
Parent B supplies an RBF surrogate that, given the geometric descriptors of an edge,
produces a raw posterior estimate ŷₑ.  

We interpret the temperature‑dependent developmental rate ρ(T) from the
Schoolfield‑Rollinson model (Parent A) as a global physiological scaling factor.
The final hybrid edge cost is

    Cₑ(T) = Lₑ · ρ(T) · ŷₑ .

Summing Cₑ over all edges yields a temperature‑aware, data‑driven cost functional
that fuses both parent algorithms. The implementation below provides the core
operations, a linear‑system trainer for the RBF weights, and a smoke test."""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Sequence
import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
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
    edge_len : dict mapping edge (as supplied) → length
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
        for nxt in adj[cur]:
            if nxt not in dist:
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist

# ----------------------------------------------------------------------
# Temperature‑dependent developmental rate (Schoolfield‑Rollinson)
# ----------------------------------------------------------------------
def schoolfield_rate(
    T: float,
    E: float = 0.65,
    T_ref: float = 293.15,
    T_opt: float = 303.15,
    delta_H: float = 0.2,
) -> float:
    """
    Compute the temperature‑dependent rate ρ(T).

    Parameters
    ----------
    T : float
        Absolute temperature in Kelvin.
    E, T_ref, T_opt, delta_H : float
        Empirical parameters (default values are placeholders).

    Returns
    -------
    rho : float
        Scaling factor (dimensionless, 0 < rho ≤ 1).
    """
    k = 8.617333262e-5  # Boltzmann constant in eV·K⁻¹
    # Simple Arrhenius‑type formulation with an optimal temperature cutoff
    arrhenius = math.exp(-E / (k * T))
    opt_factor = 1.0 / (1.0 + math.exp(delta_H * (1.0 / T_opt - 1.0 / T)))
    rho = arrhenius * opt_factor
    # Normalise to max 1 for stability
    return min(rho, 1.0)

# ----------------------------------------------------------------------
# RBF Surrogate from Parent B
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b via Gauss‑Jordan elimination (no external deps)."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # Pivot selection
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
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
    Radial‑Basis Function surrogate.

    The model predicts a scalar ŷ = Σ w_i · φ(‖x‑c_i‖) where φ is Gaussian.
    """
    def __init__(self, centers: List[Tuple[float, ...]], weights: List[float], epsilon: float = 1.0):
        if len(centers) != len(weights):
            raise ValueError("centers and weights must have same length")
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: Vector) -> float:
        """Return the RBF prediction for input vector x."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for c, w in zip(self.centers, self.weights)
        )

    @classmethod
    def train(cls, X: List[Vector], y: List[float], epsilon: float = 1.0) -> "RBFSurrogate":
        """
        Train an RBF surrogate by solving the linear system Φw = y,
        where Φ_{ij} = φ(‖X_i‑c_j‖) and we choose centers = X.
        """
        n = len(X)
        Phi = [[gaussian(euclidean(x_i, x_j), epsilon) for x_j in X] for x_i in X]
        w = solve_linear(Phi, y)
        return cls(centers=X, weights=w, epsilon=epsilon)

# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def edge_feature_vector(nodes: Dict[str, Tuple[float, float]], edge: Tuple[str, str]) -> List[float]:
    """
    Construct a feature vector for an edge by concatenating the coordinates
    of its two endpoints: [x1, y1, x2, y2].
    """
    a, b = edge
    return [nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1]]

def hybrid_edge_cost(
    edge: Tuple[str, str],
    nodes: Dict[str, Tuple[float, float]],
    edge_len: Dict[Tuple[str, str], float],
    surrogate: RBFSurrogate,
    temperature: float,
) -> float:
    """
    Compute the hybrid cost Cₑ(T) = Lₑ · ρ(T) · ŷₑ.
    """
    # 1. Euclidean length
    L = edge_len[edge] if edge in edge_len else edge_len[(edge[1], edge[0])]
    # 2. RBF posterior estimate
    x = edge_feature_vector(nodes, edge)
    y_hat = surrogate.predict(x)
    # 3. Temperature scaling
    rho = schoolfield_rate(temperature)
    return L * rho * max(y_hat, 0.0)   # enforce non‑negative posterior

def hybrid_tree_total_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    surrogate: RBFSurrogate,
    temperature: float,
) -> float:
    """
    Compute the total hybrid cost for the whole tree.
    """
    _, edge_len, _ = tree_metrics(nodes, edges, root)
    total = 0.0
    for e in edges:
        total += hybrid_edge_cost(e, nodes, edge_len, surrogate, temperature)
    return total

def train_surrogate_from_edge_lengths(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> RBFSurrogate:
    """
    Use the raw Euclidean edge lengths as target values y and train an RBF
    surrogate that maps geometric edge descriptors to length.  This provides
    a learned posterior estimate that can later be modulated by temperature.
    """
    X = [edge_feature_vector(nodes, e) for e in edges]
    # Obtain true lengths via tree_metrics (ensures consistency with adjacency)
    _, edge_len, _ = tree_metrics(nodes, edges, root)
    y = [edge_len[e] if e in edge_len else edge_len[(e[1], e[0])] for e in edges]
    return RBFSurrogate.train(X, y, epsilon=0.5)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple tree: a root with two children and a grand‑child
    nodes = {
        "root": (0.0, 0.0),
        "A": (1.0, 0.0),
        "B": (0.0, 1.0),
        "C": (1.0, 1.0),
    }
    edges = [("root", "A"), ("root", "B"), ("A", "C")]
    root = "root"

    # Train surrogate on the geometric edge data
    surrogate = train_surrogate_from_edge_lengths(nodes, edges, root)

    # Choose a biologically plausible temperature (e.g., 298 K ≈ 25 °C)
    temperature = 298.15

    total_cost = hybrid_tree_total_cost(nodes, edges, root, surrogate, temperature)
    print(f"Hybrid total cost at T={temperature:.2f} K: {total_cost:.6f}")

    # Verify that individual edge costs are positive and finite
    _, edge_len, _ = tree_metrics(nodes, edges, root)
    for e in edges:
        c = hybrid_edge_cost(e, nodes, edge_len, surrogate, temperature)
        assert math.isfinite(c) and c >= 0.0, f"Invalid cost for edge {e}"
    print("Smoke test passed.")