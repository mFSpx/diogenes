# DARWIN HAMMER — match 3214, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_tropical_maxp_hybrid_hybrid_hybrid_m2209_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rlct_g_m1723_s1.py (gen5)
# born: 2026-05-29T23:48:30Z

"""Hybrid Algorithm: Tropical‑Fisher NLMS

This module fuses the two parent algorithms:

* **Parent A** – tropical (max‑plus) algebra primitives and tree‑metric
  computation.
* **Parent B** – Gaussian‑beam based Fisher information and a Normalised
  Least‑Mean‑Squares (NLMS) adaptive filter.

**Mathematical bridge**

The bridge is the shared use of matrix‑vector operations.  
We first embed a tree into a tropical feature space using the max‑plus
semiring (`t_add`, `t_mul`, `t_matvec_mul`).  The resulting tropical
feature vector is then fed to an NLMS predictor.  The NLMS step‑size
`μ` is not static; it is adapted on‑line by the Fisher information of a
Gaussian‑beam model (`fisher_score`).  Thus the tropical transformation
provides the input representation while the Fisher‑informed NLMS supplies
the learning dynamics – a true hybrid of the two topologies.

The implementation below provides three core hybrid functions:

1. `tree_tropical_features` – compute tree distances and build a tropical
   matrix that maps raw node coordinates to tropical features.
2. `fisher_adaptive_mu` – compute a scalar step‑size from Fisher information
   of the tropical features.
3. `tropical_nlms_step` – perform one NLMS adaptation step on the tropical
   features using the Fisher‑adapted step‑size.

All code is pure Python 3 with only the allowed standard‑library modules
and NumPy."""

import math
import random
import sys
import pathlib
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A – tropical primitives and tree utilities
# ----------------------------------------------------------------------
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (plus)."""
    return np.add(x, y)


def t_matvec_mul(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Tropical matrix–vector multiplication.
    (A ⊗ v)_i = max_j (A_ij + v_j)
    """
    A = np.asarray(A, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.max(A + v, axis=1)


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Compute adjacency, edge lengths and root‑to‑node distances for an
    undirected tree.
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        d = length(nodes[u], nodes[v])
        edge_len[(u, v)] = d
        edge_len[(v, u)] = d

    dist: Dict[str, float] = {root: 0.0}
    queue: List[str] = [root]
    visited = {root}
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                dist[nb] = dist[cur] + edge_len[(cur, nb)]
                visited.add(nb)
                queue.append(nb)

    return adj, edge_len, dist


# ----------------------------------------------------------------------
# Parent B – Gaussian beam, Fisher information and NLMS
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single observation θ under the Gaussian beam model.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalised Least‑Mean‑Squares (NLMS) weight update.
    Returns (new_weights, error).
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")
    y = nlms_predict(weights, x)
    error = target - y
    power = float(np.dot(x, x)) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error


# ----------------------------------------------------------------------
# Hybrid functionality
# ----------------------------------------------------------------------
def tree_tropical_features(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a tropical matrix `T` that maps a raw coordinate vector
    (ordered by node name) to tropical features derived from tree distances.

    Returns:
        T (ndarray): shape (N, N) where N = number of nodes.
        order (list): node names in the order used to construct T.
    """
    _, _, dist = tree_metrics(nodes, edges, root)
    order = sorted(nodes.keys())
    N = len(order)

    # Tropical matrix: each row i contains the distance from node i to every node j.
    T = np.empty((N, N), dtype=float)
    for i, ni in enumerate(order):
        for j, nj in enumerate(order):
            # Tropical entry = distance(ni, root) + distance(root, nj)
            # This respects the max‑plus semiring structure.
            T[i, j] = dist[ni] + dist[nj]
    return T, order


def fisher_adaptive_mu(
    feature_vec: np.ndarray,
    center: float,
    width: float,
    base_mu: float = 0.5,
) -> float:
    """
    Compute an adaptive NLMS step‑size μ from the Fisher information of the
    tropical feature vector.
    """
    infos = np.array([fisher_score(theta, center, width) for theta in feature_vec])
    # Normalise Fisher information to a sensible range and blend with base_mu.
    info_mean = float(np.mean(infos))
    mu = base_mu * (1.0 + info_mean)  # ensures mu stays positive
    # Clip to NLMS admissible interval (0, 2)
    return max(min(mu, 1.9), 0.01)


def tropical_nlms_step(
    weights: np.ndarray,
    tropical_matrix: np.ndarray,
    raw_input: np.ndarray,
    target: float,
    center: float,
    width: float,
) -> Tuple[np.ndarray, float, float]:
    """
    One hybrid adaptation step:
      1. Transform the raw input with tropical matrix multiplication.
      2. Compute a Fisher‑adapted step‑size μ.
      3. Perform an NLMS update using the transformed features.

    Returns:
        new_weights, prediction, error
    """
    # 1) Tropical transformation
    x_trop = t_matvec_mul(tropical_matrix, raw_input)

    # 2) Adaptive step size from Fisher information
    mu = fisher_adaptive_mu(x_trop, center, width)

    # 3) NLMS adaptation
    new_weights, error = nlms_update(weights, x_trop, target, mu=mu)
    prediction = nlms_predict(weights, x_trop)
    return new_weights, prediction, error


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple tree with 4 nodes
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    root = "A"

    # Build tropical matrix from tree
    T, order = tree_tropical_features(nodes, edges, root)

    # Random initial NLMS weights (size matches number of nodes)
    rng = np.random.default_rng(seed=42)
    w = rng.normal(size=len(order))

    # Raw input: use the Euclidean coordinates stacked as a vector
    raw_vec = np.array([nodes[name][0] for name in order], dtype=float)

    # Target scalar (synthetic)
    target = 2.5

    # Gaussian beam parameters for Fisher adaptation
    centre = 0.5
    width = 0.3

    # Perform a hybrid NLMS step
    w_new, pred, err = tropical_nlms_step(
        weights=w,
        tropical_matrix=T,
        raw_input=raw_vec,
        target=target,
        center=centre,
        width=width,
    )

    print("Old weights :", w)
    print("New weights :", w_new)
    print("Prediction  :", pred)
    print("Error       :", err)