# DARWIN HAMMER — match 1244, survivor 4
# gen: 6
# parent_a: hybrid_rectified_flow_hybrid_hybrid_hard_t_m184_s1.py (gen3)
# parent_b: hybrid_hybrid_gini_coeffici_hybrid_tropical_maxp_m1173_s3.py (gen5)
# born: 2026-05-29T23:34:43Z

"""
Hybrid Rectified‑Flow / Gini‑Tropical Algorithm
================================================

Parents
-------
* **Parent A** – *rectified_flow_hybrid_hybrid_hard_t_m184_s1.py*  
  Provides the straight‑line interpolant `interpolant(x0, x1, t)` and the
  target vector field `flow_target(x0, x1)`.

* **Parent B** – *hybrid_hybrid_gini_coeffici_hybrid_tropical_maxp_m1173_s3.py*  
  Supplies a Gini impurity measure, an RBF‑based similarity matrix and a
  tropical (max‑plus) belief propagation scheme.

Mathematical Bridge
-------------------
For a pair of data points `(x0, x1)` we first construct the rectified‑flow
interpolants `Z_t`.  These interpolants serve as *feature vectors* for the
Gini impurity `I = Gini(Z_t)`.  The flow target `v = x1 - x0` is interpreted
as a log‑probability seed `ℓ₀ = log‖v‖` for each node of a tiny graph whose
nodes are `{x0, Z_t, x1}`.

A Gaussian radial‑basis‑function (RBF) converts Euclidean distances between
graph nodes into a similarity matrix `S`.  In the tropical (max‑plus) algebra
the log‑similarities `log S` act as edge weights; belief propagation is then
a tropical matrix product

    ℓ_{k+1} = ℓ_k ⊗ log S   ⇔   (ℓ_{k+1})_i = max_j ( ℓ_k_j + log S_{j,i} ).

After a few propagation steps we obtain a belief score `B = mean(ℓ_final)`.

Finally a hybrid split score couples the three ingredients:

    score = α·I  +  β·B  +  γ·mean_row(S)

Thus the rectified‑flow geometry, impurity information, and tropical belief
are fused into a single quantitative criterion.

The module below implements this pipeline with three public functions that
demonstrate the hybrid operation.
"""

import math
import random
import sys
import pathlib
from typing import Sequence, Iterable, List, Tuple, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Rectified Flow utilities
# ----------------------------------------------------------------------


def interpolant(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Straight‑line interpolant Z_t = t * x1 + (1‑t) * x0.

    Parameters
    ----------
    x0, x1 : np.ndarray
        Arrays of shape (d,) or (B, d).  If one‑dimensional they are treated as a
        single batch element.
    t : np.ndarray
        Shape (B,) or scalar.  Broadcasted over the leading batch dimension.

    Returns
    -------
    np.ndarray
        Interpolated points of shape (B, d).
    """
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    t = np.asarray(t)

    # Ensure batch dimension
    if x0.ndim == 1:
        x0 = x0[None, :]
    if x1.ndim == 1:
        x1 = x1[None, :]
    if t.ndim == 0:
        t = np.full((x0.shape[0],), t, dtype=float)

    t = t.reshape(-1, 1)  # (B,1)
    return t * x1 + (1.0 - t) * x0


def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """
    Target vector field v_theta(Z_t, t) = (x1 - x0).

    Returns a constant vector for the whole trajectory.
    """
    return np.asarray(x1) - np.asarray(x0)


# ----------------------------------------------------------------------
# Parent B – Gini, RBF similarity, and tropical belief utilities
# ----------------------------------------------------------------------


def gini_coefficient(values: Iterable[float]) -> float:
    """
    Gini impurity of a non‑negative numeric collection.
    """
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def rbf_similarity_matrix(X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Gaussian RBF similarity S_{ij} = exp(-gamma * ||x_i - x_j||^2).

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Input points.
    gamma : float
        Bandwidth parameter (default 1.0).

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric similarity matrix with ones on the diagonal.
    """
    X = np.asarray(X, dtype=float)
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)  # (N,1)
    dists_sq = sq_norms + sq_norms.T - 2.0 * X @ X.T
    np.clip(dists_sq, a_min=0.0, a_max=None, out=dists_sq)  # numerical safety
    return np.exp(-gamma * dists_sq)


def tropical_propagate(log_belief: np.ndarray,
                       log_sim: np.ndarray,
                       steps: int = 5) -> np.ndarray:
    """
    Perform tropical (max‑plus) belief propagation.

    The update rule is
        b_{k+1}[i] = max_j ( b_k[j] + log_sim[j, i] ).

    Parameters
    ----------
    log_belief : np.ndarray, shape (N,)
        Initial log‑probabilities for each node.
    log_sim : np.ndarray, shape (N, N)
        Log‑similarities (edge weights) between nodes.
    steps : int
        Number of propagation iterations.

    Returns
    -------
    np.ndarray, shape (N,)
        Belief vector after `steps` tropical multiplications.
    """
    b = log_belief.copy()
    for _ in range(steps):
        # broadcasting: (N,1) + (N,N) -> (N,N), then max over rows
        b = np.max(b[:, None] + log_sim, axis=0)
    return b


# ----------------------------------------------------------------------
# Hybrid core functions (the new fused algorithm)
# ----------------------------------------------------------------------


def hybrid_feature_impurity(x0: np.ndarray,
                            x1: np.ndarray,
                            t_vals: np.ndarray) -> float:
    """
    Compute Gini impurity on the concatenated interpolant features.

    The interpolants Z_t are flattened and treated as a multiset of
    non‑negative values (absolute values are taken to satisfy the Gini
    pre‑condition).  The resulting impurity is a scalar `I`.
    """
    Z = interpolant(x0, x1, t_vals)                     # (B, d)
    flat = np.abs(Z).ravel()
    return gini_coefficient(flat)


def hybrid_belief_score(x0: np.ndarray,
                        x1: np.ndarray,
                        t_vals: np.ndarray,
                        gamma: float = 1.0,
                        steps: int = 5) -> float:
    """
    Build a tiny graph {x0, Z_t..., x1}, compute RBF similarities,
    initialise log‑belief with the norm of the flow target, and run
    tropical propagation.  Returns the average belief after propagation.
    """
    # Nodes: start, interpolants, end
    Z = interpolant(x0, x1, t_vals)                     # (B, d)
    nodes = np.vstack([x0[None, :], Z, x1[None, :]])   # (B+2, d)

    # Similarity matrix and its log
    S = rbf_similarity_matrix(nodes, gamma=gamma)      # (N,N)
    eps = np.finfo(float).eps
    log_S = np.log(np.clip(S, a_min=eps, a_max=None))

    # Initialise belief with magnitude of flow target (log‑norm)
    v = flow_target(x0, x1)                            # (d,)
    init_val = np.linalg.norm(v) + eps
    log_b0 = np.full(nodes.shape[0], np.log(init_val))

    # Tropical propagation
    log_b_final = tropical_propagate(log_b0, log_S, steps=steps)
    return float(np.mean(log_b_final))


def hybrid_split_score(x0: np.ndarray,
                       x1: np.ndarray,
                       t_vals: np.ndarray,
                       alpha: float = 1.0,
                       beta: float = 1.0,
                       gamma_w: float = 1.0,
                       rbf_gamma: float = 1.0,
                       steps: int = 5) -> float:
    """
    Unified hybrid score combining impurity, tropical belief and mean similarity.

    score = α·I  +  β·B  +  γ·mean_row(S)

    Returns a scalar that can be used for split evaluation or ranking.
    """
    # 1. Impurity from rectified‑flow interpolants
    I = hybrid_feature_impurity(x0, x1, t_vals)

    # 2. Tropical belief derived from RBF similarities
    B = hybrid_belief_score(x0, x1, t_vals, gamma=rbf_gamma, steps=steps)

    # 3. Mean similarity of the graph (excluding self‑loops)
    Z = interpolant(x0, x1, t_vals)
    nodes = np.vstack([x0[None, :], Z, x1[None, :]])
    S = rbf_similarity_matrix(nodes, gamma=rbf_gamma)
    # Zero out diagonal to avoid counting self‑similarity = 1
    np.fill_diagonal(S, 0.0)
    mean_sim = float(np.mean(S))

    return alpha * I + beta * B + gamma_w * mean_sim


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Simple reproducible demo
    rng = np.random.default_rng(seed=42)

    # Random 3‑dimensional points
    x0 = rng.normal(loc=0.0, scale=1.0, size=3)
    x1 = rng.normal(loc=2.0, scale=1.5, size=3)

    # Interpolation parameters
    t_vals = np.linspace(0.0, 1.0, num=5)  # five intermediate points

    # Compute components individually
    impurity = hybrid_feature_impurity(x0, x1, t_vals)
    belief = hybrid_belief_score(x0, x1, t_vals, gamma=0.5, steps=8)
    score = hybrid_split_score(x0, x1, t_vals,
                               alpha=0.7, beta=0.2, gamma_w=0.1,
                               rbf_gamma=0.5, steps=8)

    print("x0:", x0)
    print("x1:", x1)
    print("Interpolant t values:", t_vals)
    print(f"Gini impurity I = {impurity:.6f}")
    print(f"Tropical belief B = {belief:.6f}")
    print(f"Hybrid split score = {score:.6f}")

    # Verify that the functions run without raising
    assert isinstance(impurity, float)
    assert isinstance(belief, float)
    assert isinstance(score, float)