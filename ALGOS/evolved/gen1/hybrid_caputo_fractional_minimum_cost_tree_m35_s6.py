# DARWIN HAMMER — match 35, survivor 6
# gen: 1
# parent_a: caputo_fractional.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:25:21Z

"""Hybrid fractional-memory tree cost module.

This module fuses two parent algorithms:

* **Caputo Fractional Derivative (caputo_fractional.py)** – provides a power‑law
  memory kernel  φ(t;α)=t^{α‑1}/Γ(α)  and the Lanczos approximation for Γ.
* **Minimum‑cost tree scoring (minimum_cost_tree.py)** – computes a material
  length of a tree plus a linear path‑weight term based on distances from a
  root.

The mathematical bridge is the *summation over history* that both algorithms
share.  The tree cost already contains a sum of edge lengths and a sum of
root‑to‑node distances.  By replacing the plain sum of distances with a
Caputo‑weighted sum we obtain a **fractional‑memory tree cost**:

    C_frac = material
             + λ * Σ_{k=1}^{T} w_k(α) · d_k ,

where `d_k` are the distances from the root after the k‑th edge insertion,
`w_k(α)` are the normalized fractional kernel values
`φ(t‑k;α) = (t‑k)^{α‑1} / Γ(α)`, and `λ` is the original `path_weight`.

The implementation below provides:

* `gamma_lanczos` – Lanczos approximation of the Gamma function.
* `caputo_weights` – compute normalized Caputo kernel weights for a history.
* `fractional_weighted_sum` – apply the weights to an arbitrary numeric history.
* `length` – Euclidean edge length (from the tree module).
* `incremental_fractional_tree_cost` – builds the tree edge‑by‑edge, updates
  distances, and evaluates the hybrid cost using the fractional memory term.
* `fractional_ssm_step` – a generic state‑space update that also uses the same
  Caputo weighting, illustrating the deeper algebraic connection.

Running the module as a script executes a small smoke test that builds a
simple tree, computes the hybrid cost for a chosen fractional order `α`, and
prints the result without raising errors.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

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
    """Lanczos approximation of Γ(z) for real z > 0.

    For 0 < z < 0.5 the reflection formula is used to stay in the stable region.
    Accuracy is ~15 decimal digits for the supported range.
    """
    if z < 0.5:
        # Reflection formula Γ(z) = π / (sin(πz) * Γ(1‑z))
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1.0 - z))
    z_minus_1 = z - 1.0
    x = _LANCZOS_C[0]
    for i in range(1, len(_LANCZOS_C)):
        x += _LANCZOS_C[i] / (z_minus_1 + i)
    t = z_minus_1 + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t**(z_minus_1 + 0.5) * math.exp(-t) * x

def caputo_weights(history_len: int, alpha: float) -> np.ndarray:
    """Return normalized Caputo kernel weights w_k for a history of length *history_len*.

    The kernel φ(Δ;α) = Δ^{α‑1} / Γ(α) is evaluated for Δ = 1,2,…,history_len
    (Δ=0 would be singular and is omitted).  The resulting vector is normalized
    so that Σ w_k = 1.
    """
    if history_len == 0:
        return np.array([], dtype=float)
    # Δ runs from history_len down to 1 (most recent has Δ=1)
    deltas = np.arange(history_len, 0, -1, dtype=float)
    phi = deltas ** (alpha - 1.0) / gamma_lanczos(alpha)
    w = phi / phi.sum()
    return w

def fractional_weighted_sum(values: List[float], alpha: float) -> float:
    """Apply Caputo weights to *values* and return the weighted sum."""
    if not values:
        return 0.0
    w = caputo_weights(len(values), alpha)
    return float(np.dot(w, np.asarray(values, dtype=float)))

# ----------------------------------------------------------------------
# Parent B – tree utilities (length and basic cost)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def fractional_ssm_step(
    A: np.ndarray,
    B: np.ndarray,
    h_history: List[np.ndarray],
    x_t: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """One fractional state‑space step.

    The new hidden state h_t is a Caputo‑weighted combination of past
    linear updates A·h_k + B·x_t.
    """
    if not h_history:
        raise ValueError("h_history must contain at least one previous state.")
    # Build the list of contributions for each past step
    contributions = [A @ h_k + B @ x_t for h_k in h_history]
    # Apply the same fractional weighting used for the tree distances
    weighted = fractional_weighted_sum(contributions, alpha)
    return np.asarray(weighted, dtype=float)

def incremental_fractional_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    alpha: float,
    path_weight: float = 0.2,
) -> float:
    """Compute the hybrid tree cost while inserting edges sequentially.

    The *material* term is the ordinary sum of edge lengths.
    The *path* term uses a Caputo‑weighted sum of the root‑to‑node distances
    after each insertion, thereby giving older distances a power‑law decay.
    """
    # adjacency list for dynamic graph
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    # store the distance from root after each insertion
    distance_history: List[float] = []

    for idx, (a, b) in enumerate(edges, start=1):
        # update adjacency
        adj[a].append(b)
        adj[b].append(a)

        # material contribution
        l = length(nodes[a], nodes[b])
        material += l

        # recompute distances from root (BFS – cheap for small graphs)
        dist: Dict[str, float] = {root: 0.0}
        stack = [root]
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in dist:
                    dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                    stack.append(nxt)

        # collect distances (excluding the root itself)
        distances = [d for node, d in dist.items() if node != root]
        distance_history.extend(distances)  # grow the history with all new distances

        # fractional path term based on the full history up to now
        frac_path = fractional_weighted_sum(distance_history, alpha)

        # total cost after this insertion (optional: could be stored per step)
        total_cost = material + path_weight * frac_path

        # For debugging / demonstration we could print intermediate values,
        # but the function returns only the final cost.
        # print(f"Step {idx}: material={material:.4f}, frac_path={frac_path:.4f}, total={total_cost:.4f}")

    # Final hybrid cost
    return material + path_weight * fractional_weighted_sum(distance_history, alpha)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic graph for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Define 5 nodes placed on a unit square
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
        "E": (0.5, 0.5),
    }

    # Edge insertion order (creates a spanning tree)
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
    ]

    root = "A"
    alpha = 0.6          # fractional order (0 < α < 1)
    path_weight = 0.3

    hybrid_cost = incremental_fractional_tree_cost(
        nodes, edges, root, alpha, path_weight
    )
    print(f"Hybrid fractional tree cost (α={alpha}): {hybrid_cost:.6f}")

    # Demonstrate the fractional SSM step with tiny matrices
    A = np.array([[0.9]])
    B = np.array([[0.1]])
    h_hist = [np.array([0.0])]
    x_t = np.array([1.0])  # pretend the latest edge length is the input
    h_new = fractional_ssm_step(A, B, h_hist, x_t, alpha)
    print(f"Fractional SSM new state: {h_new}")