# DARWIN HAMMER — match 1965, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m7_s0.py (gen4)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s2.py (gen2)
# born: 2026-05-29T23:40:10Z

"""Hybrid Algorithm: RBF‑Surrogate + NLMS‑Adapted Minimum‑Cost Tree
Parent A – RBF surrogate (Gaussian kernel, Euclidean distance, linear solve)
Parent B – Normalized Least‑Mean‑Squares (NLMS) weight adaptation + Minimum‑Cost Tree (MST)

Mathematical Bridge
-------------------
The RBF surrogate learns a linear mapping  α  from a kernel matrix **K** (Gaussian of
pairwise Euclidean distances of MinHash‑derived signature vectors) to target
similarities **y** :

    K α = y                            (1)

Instead of solving (1) once, we treat the coefficient vector **α** as an
online‑learnable weight vector.  The NLMS update (Parent B) provides a normalized
gradient step that keeps the surrogate stable under streaming data:

    ŷ = αᵀ φ, e = t − ŷ,
    α←α + μ e φ / (‖φ‖² + ε)            (2)

where **φ** is a kernel feature (a column of **K**) and *t* the new target.
After each NLMS adaptation the current surrogate predictions are interpreted as
edge costs for a complete graph of the signatures.  A Minimum‑Cost Spanning Tree
(MST) built on these costs (Prim’s algorithm) yields a hierarchical “span”
structure that can be used for downstream extraction.

The three core functions below realise:
1. Construction of the RBF kernel matrix and training of **α**.
2. NLMS‑driven online refinement of **α**.
3. Generation of an MST from the surrogate‑predicted pairwise costs.

Only the Python standard library and NumPy are used.
"""

import math
import random
import sys
import pathlib
import hashlib
from typing import List, Sequence, Tuple

import numpy as np

Vector = Sequence[float]


# ----------------------------------------------------------------------
# Utilities from Parent A (RBF surrogate)
# ----------------------------------------------------------------------


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rbf_kernel_matrix(X: List[Vector], epsilon: float = 1.0) -> np.ndarray:
    """Build the symmetric Gaussian RBF kernel matrix K_{ij}=exp(-ε²‖x_i‑x_j‖²)."""
    n = len(X)
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        K[i, i] = 1.0  # distance to itself is zero → exp(0)=1
        for j in range(i + 1, n):
            d = euclidean(X[i], X[j])
            val = gaussian(d, epsilon)
            K[i, j] = K[j, i] = val
    return K


def solve_linear(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve K α = y with NumPy’s linear solver (fallback to pseudo‑inverse)."""
    try:
        alpha = np.linalg.solve(K, y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(K) @ y
    return alpha


# ----------------------------------------------------------------------
# Utilities from Parent B (NLMS + Minimum‑Cost Tree)
# ----------------------------------------------------------------------


def nlms_update(
    weights: np.ndarray,
    phi: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS step.
    Returns the updated weight vector and the prediction error.
    """
    pred = float(weights @ phi)
    error = target - pred
    norm = float(phi @ phi) + eps
    new_weights = weights + mu * error * phi / norm
    return new_weights, error


def prim_mst(cost_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Compute a Minimum‑Cost Spanning Tree of a fully connected undirected graph
    given its symmetric cost matrix (zero diagonal).  Returns a list of edges
    (u, v, cost) in the tree.
    """
    n = cost_matrix.shape[0]
    if n == 0:
        return []

    visited = [False] * n
    visited[0] = True
    edges = []

    # cheapest connection from the visited set to each unvisited node
    cheap = [(cost_matrix[0, j], 0, j) for j in range(1, n)]

    while len(edges) < n - 1:
        cheap.sort(key=lambda x: x[0])  # O(n log n); fine for modest n
        cost, u, v = cheap.pop(0)
        if visited[v]:
            continue
        visited[v] = True
        edges.append((u, v, cost))
        # update cheap list with new connections from v
        for w in range(n):
            if not visited[w]:
                cheap.append((cost_matrix[v, w], v, w))

    return edges


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------


def signature_from_bytes(data: bytes, dim: int = 8) -> List[float]:
    """
    Produce a deterministic pseudo‑random vector from a byte string.
    The hash is split into `dim` chunks and each chunk is mapped to [0,1].
    """
    h = hashlib.sha256(data).digest()
    # repeat the digest if needed
    needed = dim * 4
    while len(h) < needed:
        h += hashlib.sha256(h).digest()
    vec = []
    for i in range(dim):
        chunk = int.from_bytes(h[i * 4 : (i + 1) * 4], "big")
        vec.append(chunk / 0xFFFFFFFF)
    return vec


def train_surrogate_with_nlms(
    signatures: List[Vector],
    targets: List[float],
    epsilon: float = 1.0,
    mu: float = 0.4,
    epochs: int = 1,
) -> np.ndarray:
    """
    Initialise the RBF surrogate coefficients α by solving Kα = y,
    then refine α online with NLMS using the same (signatures, targets) pairs.
    """
    K = rbf_kernel_matrix(signatures, epsilon)
    y = np.array(targets, dtype=float)
    alpha = solve_linear(K, y)

    # Online NLMS refinement – each epoch cycles over the data
    for _ in range(epochs):
        for i, phi in enumerate(K):
            alpha, _ = nlms_update(alpha, phi, y[i], mu=mu)
    return alpha


def surrogate_predict(
    signatures: List[Vector],
    alpha: np.ndarray,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Compute the surrogate similarity predictions for every pair of signatures.
    Returns a symmetric matrix P where P_{ij}=αᵀ k_i_j .
    """
    K = rbf_kernel_matrix(signatures, epsilon)
    # prediction for each training point is simply K @ α
    preds = K @ alpha
    # expand to full pairwise matrix using outer product of kernel columns
    # Since K is symmetric, we can reuse it:
    pairwise = np.empty_like(K)
    n = K.shape[0]
    for i in range(n):
        pairwise[i, :] = preds[i] * K[i, :]
    # Symmetrize
    pairwise = (pairwise + pairwise.T) / 2.0
    return pairwise


def extract_spans_from_mst(pairwise_costs: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Build an MST from the surrogate pairwise costs (treated as edge weights)
    and return the resulting edges as the “spans”.
    """
    # Convert similarity (higher is better) to cost (lower is better)
    max_cost = np.max(pairwise_costs)
    cost_matrix = max_cost - pairwise_costs
    np.fill_diagonal(cost_matrix, 0.0)
    edges = prim_mst(cost_matrix)
    return edges


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Generate synthetic signatures from random byte strings
    random.seed(42)
    sigs = [signature_from_bytes(str(i).encode(), dim=8) for i in range(6)]

    # Synthetic target similarities (symmetric, zero diagonal)
    base = np.random.rand(6, 6)
    targets = ((base + base.T) / 2.0).tolist()
    for i in range(6):
        targets[i][i] = 0.0

    # Flatten upper triangle for training (RBF surrogate expects a vector per point)
    # Here we simply use the diagonal‑free entries as independent targets.
    flat_targets = [targets[i][j] for i in range(6) for j in range(6) if i != j]

    # Train α with NLMS refinement
    alpha = train_surrogate_with_nlms(sigs, flat_targets, epsilon=1.2, mu=0.3, epochs=3)

    # Obtain surrogate pairwise predictions
    pred_matrix = surrogate_predict(sigs, alpha, epsilon=1.2)

    # Build MST‑based spans
    spans = extract_spans_from_mst(pred_matrix)

    print("Surrogate coefficient vector (α):")
    print(alpha)
    print("\nPairwise prediction matrix:")
    print(pred_matrix)
    print("\nMST edges (spans):")
    for u, v, cost in spans:
        print(f"  {u} -- {v}  (cost={cost:.4f})")