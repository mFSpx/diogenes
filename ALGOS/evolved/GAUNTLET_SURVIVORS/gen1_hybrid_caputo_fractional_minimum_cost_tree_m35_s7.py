# DARWIN HAMMER — match 35, survivor 7
# gen: 1
# parent_a: caputo_fractional.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:25:21Z

"""Hybrid fractional‑memory tree scoring.

This module merges two distinct algorithms:

* **Caputo fractional derivative** (parent A) – provides a power‑law memory
  kernel ϕ(t‑τ;α)= (t‑τ)^{α‑1}/Γ(α) that weights past contributions with a
  slowly decaying algebraic factor.
* **Minimum‑cost tree scoring** (parent B) – computes a static cost
  material + path_weight·∑dist(root, v) for a given undirected tree.

The mathematical bridge is the *temporal ordering* of edge insertions.
Each edge (and the induced change in root‑to‑node distances) is a
“past event”.  By applying the Caputo kernel to the sequence of incremental
cost contributions we obtain a *fractional‑memory tree cost* that
remembers the whole construction history with algebraic decay rather than
the usual instantaneous evaluation.

The core hybrid operation therefore consists of:

1. Compute the incremental material and path contributions for each
   edge as it is added.
2. Form the Caputo weights w_k = ϕ(T‑1‑k;α)/∑_j ϕ(T‑1‑j;α) where T is the total
   number of edges.
3. Return the weighted sum Σ_k w_k·(Δ_material_k + Δ_path_k).

The resulting cost reduces to the classic tree cost when α→1
(the kernel collapses to a Kronecker delta) and to a uniform average
when α→0.  This hybrid can be used wherever a history‑aware evaluation of
tree‑like structures is desirable (e.g. evolving phylogenies, incremental
network design, or recurrent state‑space models with spatial topology)."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Lanczos Gamma approximation and Caputo utilities
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of Γ(z) for real z>0.

    For 0<z<0.5 the reflection formula is used to keep the argument
    in the stable region.
    """
    if z < 0.5:
        # reflection formula Γ(z) = π / (sin(πz)·Γ(1‑z))
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1.0 - z))
    z_minus_1 = z - 1.0
    x = _LANCZOS_C[0]
    for i in range(1, len(_LANCZOS_C)):
        x += _LANCZOS_C[i] / (z_minus_1 + i)
    t = z_minus_1 + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z_minus_1 + 0.5) * math.exp(-t) * x

def caputo_weights(num_steps: int, alpha: float) -> np.ndarray:
    """Return the normalized Caputo kernel weights w_k for k=0…num_steps‑1.

    The kernel is ϕ(t‑τ;α) = (t‑τ)^{α‑1} / Γ(α) with t = num_steps‑1.
    A tiny epsilon avoids the singularity at τ=t.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    if num_steps == 0:
        return np.array([], dtype=float)

    t = float(num_steps - 1)
    eps = 1e-12
    phi = np.empty(num_steps, dtype=float)
    gamma_alpha = gamma_lanczos(alpha)
    for k in range(num_steps):
        delta = max(t - k, eps)  # avoid zero
        phi[k] = delta ** (alpha - 1.0) / gamma_alpha
    weight_sum = phi.sum()
    return phi / weight_sum

# ----------------------------------------------------------------------
# Parent B – basic geometric utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def incremental_contributions(
    nodes: Dict[str, Point],
    edge_sequence: List[Edge],
    root: str,
    path_weight: float = 0.2,
) -> Tuple[List[float], List[float]]:
    """Return two lists:

    * material_contribs[k] – length of the k‑th added edge.
    * path_contribs[k]      – path_weight·∑_v dist(root, v) after the k‑th addition.
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material_contribs: List[float] = []
    path_contribs: List[float] = []

    for a, b in edge_sequence:
        # edge length (material)
        l = length(nodes[a], nodes[b])
        material_contribs.append(l)

        # update adjacency
        adj[a].append(b)
        adj[b].append(a)

        # recompute distances from root via BFS
        dist: Dict[str, float] = {root: 0.0}
        q: deque[str] = deque([root])
        visited: set[str] = {root}
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    dist[nb] = dist[cur] + length(nodes[cur], nodes[nb])
                    q.append(nb)

        path_contribs.append(path_weight * sum(dist.values()))

    return material_contribs, path_contribs

def fractional_tree_cost(
    nodes: Dict[str, Point],
    edge_sequence: List[Edge],
    root: str,
    alpha: float = 0.5,
    path_weight: float = 0.2,
) -> float:
    """Hybrid cost that applies a Caputo fractional memory kernel to the
    incremental material and path contributions of a growing tree.

    Parameters
    ----------
    nodes : dict
        Mapping node identifier → (x, y) coordinate.
    edge_sequence : list of (str, str)
        Edges in the order they are added (must eventually form a tree).
    root : str
        Identifier of the root node.
    alpha : float, default 0.5
        Fractional order controlling memory decay (0 < α ≤ 1).
    path_weight : float, default 0.2
        Weight of the summed root‑to‑node distances.

    Returns
    -------
    float
        The fractional‑memory tree cost.
    """
    material, path = incremental_contributions(nodes, edge_sequence, root, path_weight)
    T = len(material)
    if T == 0:
        return 0.0
    w = caputo_weights(T, alpha)
    total = sum(w_i * (m_i + p_i) for w_i, m_i, p_i in zip(w, material, path))
    return total

def caputo_derivative_discrete(
    values: List[float],
    alpha: float,
    dt: float = 1.0,
) -> List[float]:
    """Discrete Caputo derivative of a 1‑D signal using the left‑endpoint
    trapezoidal rule.  Returns a list of the same length where the first
    element is defined as zero (no history).
    """
    n = len(values)
    if n == 0:
        return []
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    gamma_val = gamma_lanczos(1.0 - alpha)
    deriv = [0.0]  # D^α f(0) = 0 by convention
    for i in range(1, n):
        accum = 0.0
        for j in range(i):
            coeff = ( (i - j) ** (-alpha) - (i - j - 1) ** (-alpha) )
            accum += coeff * (values[j + 1] - values[j])
        deriv.append( accum / (gamma_val * dt ** alpha) )
    return deriv

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic test: a line of 4 points and 3 edges added sequentially.
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (2.0, 0.0),
        "D": (3.0, 0.0),
    }
    edge_seq = [("A", "B"), ("B", "C"), ("C", "D")]
    root_node = "A"

    # Classic tree cost (α=1 → weights collapse to last step)
    classic_cost = fractional_tree_cost(nodes, edge_seq, root_node, alpha=1.0, path_weight=0.2)
    print(f"Classic tree cost (α=1.0): {classic_cost:.6f}")

    # Fractional cost with strong memory (α=0.4)
    frac_cost = fractional_tree_cost(nodes, edge_seq, root_node, alpha=0.4, path_weight=0.2)
    print(f"Fractional tree cost (α=0.4): {frac_cost:.6f}")

    # Demonstrate discrete Caputo derivative on a simple signal
    signal = [math.sin(0.1 * i) for i in range(10)]
    deriv = caputo_derivative_discrete(signal, alpha=0.6)
    print("Signal:", ["{:.4f}".format(v) for v in signal])
    print("Caputo derivative (α=0.6):", ["{:.4f}".format(v) for v in deriv])