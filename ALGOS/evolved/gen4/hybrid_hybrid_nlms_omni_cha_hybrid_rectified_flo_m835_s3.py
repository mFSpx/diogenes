# DARWIN HAMMER — match 835, survivor 3
# gen: 4
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s2.py (gen1)
# parent_b: hybrid_rectified_flow_hybrid_hybrid_hard_t_m184_s1.py (gen3)
# born: 2026-05-29T23:31:18Z

"""Hybrid NLMS‑KAN Flow Graph Engine
===================================

This module fuses the two parent algorithms:

* **Parent A – NLMS (nlms.py)** – a normalized least‑mean‑squares adaptive filter that
  updates a weight vector ``w`` with ``w ← w + μ·e·x / (‖x‖²+ε)`` where
  ``e = target – w·x``.
* **Parent B – Rectified Flow + KAN (rectified_flow.py & KAN)** – a straight‑line
  interpolant ``Z_t = t·x1 + (1‑t)·x0`` that generates temporal features,
  which are fed into a Kolmogorov‑Arnold Network (KAN) built from B‑spline
  bases.

**Mathematical Bridge**

For a graph node ``i`` we first build the *impedance‑weighted neighbourhood
vector*


x_i^g = Σ_{j∈N(i)}  impedance_{ij} · φ_j                (1)


where ``φ_j`` is the static feature vector of neighbour ``j`` (e.g. a 2‑D
position).

A rectified‑flow interpolant between two temporal states ``x0_i`` and
``x1_i`` is then computed:


z_i(t) = t·x1_i + (1‑t)·x0_i                              (2)


The concatenated raw vector


u_i(t) = [ x_i^g ; z_i(t) ]                              (3)


is passed through a KAN non‑linear expansion ``ψ_i(t) = KAN(u_i(t); Θ)``,
where ``Θ`` are the spline knots and coefficients.  The resulting feature
vector is finally fed to the NLMS linear predictor


ŷ_i(t) = w·ψ_i(t)                                         (4)


and the error ``e_i = v_i – ŷ_i`` (with ``v_i`` the observed wavefront
velocity) drives the NLMS weight update.  In this way the adaptive filter
learns a linear combination of *non‑linear* KAN‑encoded graph‑temporal
features to model the propagation speed.

The code below implements the three core operations:
1. ``kan_transform`` – KAN spline expansion.
2. ``hybrid_predict`` – compute (4) for a given node and time.
3. ``hybrid_update`` – NLMS weight adaptation using the hybrid prediction.

A minimal smoke test builds a random graph, runs a single update step and
prints the updated weight vector.
"""

from __future__ import annotations

import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# NLMS core (Parent A)
# ----------------------------------------------------------------------


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear NLMS prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single NLMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Prediction error (target – prediction).
    """
    pred = nlms_predict(weights, x)
    error = target - pred
    norm = float(np.dot(x, x) + eps)
    new_weights = weights + (mu * error / norm) * x
    return new_weights, error


# ----------------------------------------------------------------------
# Rectified Flow utilities (Parent B – part A)
# ----------------------------------------------------------------------


def interpolant(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Straight‑line interpolant Z_t = t·x1 + (1‑t)·x0.

    Parameters
    ----------
    x0, x1 : np.ndarray, shape (d,)
        End‑point vectors.
    t : np.ndarray, shape (B,)
        Interpolation coefficients in [0, 1].

    Returns
    -------
    Z : np.ndarray, shape (B, d)
    """
    t = np.reshape(t, (-1, 1))  # (B,1)
    return t * x1 + (1.0 - t) * x0


def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Target vector field v = x1 – x0."""
    return x1 - x0


# ----------------------------------------------------------------------
# Simple KAN implementation (Parent B – part B)
# ----------------------------------------------------------------------


def _spline_piecewise_linear(x: np.ndarray, knots: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Evaluate a 1‑D piecewise‑linear spline defined by ``knots`` and ``coeffs``.
    ``knots`` must be sorted increasingly and have length = len(coeffs).

    Parameters
    ----------
    x : np.ndarray, shape (N,)
    knots : np.ndarray, shape (K,)
    coeffs : np.ndarray, shape (K,)

    Returns
    -------
    y : np.ndarray, shape (N,)
    """
    return np.interp(x, knots, coeffs)


def kan_transform(x: np.ndarray, params: Dict[str, List[np.ndarray]]) -> np.ndarray:
    """
    Apply a KAN‑style spline expansion to vector ``x``.

    ``params`` must contain:
        - "knots": list of 1‑D arrays, one per input dimension.
        - "coeffs": list of 1‑D arrays, same length as knots.

    The output is the concatenation of the spline‑evaluated scalars for each
    dimension, optionally followed by a bias term.

    Returns
    -------
    ψ : np.ndarray, shape (D+1,)
    """
    knots_list = params["knots"]
    coeffs_list = params["coeffs"]
    assert len(knots_list) == len(coeffs_list) == x.shape[0]

    transformed = np.empty_like(x, dtype=float)
    for idx, (kn, cf) in enumerate(zip(knots_list, coeffs_list)):
        transformed[idx] = _spline_piecewise_linear(
            np.array([x[idx]]), kn, cf
        )[0]
    # Append a constant bias (helps NLMS bias learning)
    return np.concatenate([transformed, np.array([1.0])])


# ----------------------------------------------------------------------
# Graph utilities (shared by both parents)
# ----------------------------------------------------------------------


def generate_synthetic_graph(
    num_nodes: int,
    avg_degree: int,
    seed: int | None = None,
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, np.ndarray]]:
    """
    Create an undirected random graph with impedance‑weighted edges and
    a random 2‑D feature vector for each node.

    Returns
    -------
    adjacency : dict{node: [(neighbor, impedance), ...]}
    features  : dict{node: np.ndarray shape (2,)}
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    adjacency: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_nodes)}
    features: Dict[int, np.ndarray] = {
        i: np.random.randn(2) for i in range(num_nodes)
    }

    # Simple Erdős‑Rényi style edge creation
    prob = min(1.0, avg_degree / (num_nodes - 1))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < prob:
                impedance = random.uniform(0.1, 5.0)
                adjacency[i].append((j, impedance))
                adjacency[j].append((i, impedance))
    return adjacency, features


def impedance_weighted_input(
    node: int,
    adjacency: Dict[int, List[Tuple[int, float]]],
    features: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Compute the graph‑based input vector (1) for ``node``:
        x_i^g = Σ_j impedance_ij * φ_j

    Returns
    -------
    x_g : np.ndarray, shape (feature_dim,)
    """
    neigh = adjacency.get(node, [])
    if not neigh:
        # No neighbours → zero vector
        return np.zeros_like(next(iter(features.values())))

    acc = np.zeros_like(next(iter(features.values())))
    for nbr, imp in neigh:
        acc += imp * features[nbr]
    return acc


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def hybrid_predict(
    weights: np.ndarray,
    node: int,
    adjacency: Dict[int, List[Tuple[int, float]]],
    features_t0: Dict[int, np.ndarray],
    features_t1: Dict[int, np.ndarray],
    t: float,
    kan_params: Dict[str, List[np.ndarray]],
) -> float:
    """
    Predict the wavefront velocity for ``node`` at interpolation time ``t``
    using the hybrid NLMS‑KAN pipeline.

    Steps
    -----
    1. Build graph‑based vector ``x_g`` via ``impedance_weighted_input``.
    2. Build flow interpolant ``z`` between the node's temporal states.
    3. Concatenate ``[x_g ; z]`` → raw vector ``u``.
    4. Apply KAN expansion → ``ψ = kan_transform(u, kan_params)``.
    5. Linear NLMS prediction → ``ŷ = w·ψ``.

    Returns
    -------
    pred : float
        Predicted velocity.
    """
    x_g = impedance_weighted_input(node, adjacency, features_t0)

    # Temporal interpolant for the *same* node (could also use neighbours)
    x0 = features_t0[node]
    x1 = features_t1[node]
    z = interpolant(x0, x1, np.array([t]))[0]  # shape (d,)

    u = np.concatenate([x_g, z])  # raw hybrid feature vector
    ψ = kan_transform(u, kan_params)
    return nlms_predict(weights, ψ)


def hybrid_update(
    weights: np.ndarray,
    node: int,
    adjacency: Dict[int, List[Tuple[int, float]]],
    features_t0: Dict[int, np.ndarray],
    features_t1: Dict[int, np.ndarray],
    target_velocity: float,
    t: float,
    kan_params: Dict[str, List[np.ndarray]],
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one hybrid NLMS update for ``node`` at time ``t``.

    Returns
    -------
    new_weights : np.ndarray
        Updated NLMS weight vector.
    error : float
        Prediction error (target – prediction).
    """
    x_g = impedance_weighted_input(node, adjacency, features_t0)
    x0 = features_t0[node]
    x1 = features_t1[node]
    z = interpolant(x0, x1, np.array([t]))[0]

    u = np.concatenate([x_g, z])
    ψ = kan_transform(u, kan_params)

    new_w, err = nlms_update(weights, ψ, target_velocity, mu=mu, eps=eps)
    return new_w, err


def initialise_kan_params(
    input_dim: int,
    num_knots: int = 5,
    seed: int | None = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Randomly initialise spline knots (sorted) and coefficients for each input
    dimension.  A constant bias term is appended later by ``kan_transform``.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    knots = []
    coeffs = []
    for _ in range(input_dim):
        # Uniform knots in a reasonable range
        k = np.sort(np.random.uniform(-3.0, 3.0, size=num_knots))
        c = np.random.randn(num_knots)  # coefficients can be any real number
        knots.append(k)
        coeffs.append(c)
    return {"knots": knots, "coeffs": coeffs}


# ----------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------


def _demo():
    # Parameters
    NUM_NODES = 20
    AVG_DEG = 4
    SEED = 42
    TIME_T = 0.3  # interpolation coefficient
    MU = 0.4

    # 1. Build synthetic graph and two temporal snapshots
    adj, feats0 = generate_synthetic_graph(NUM_NODES, AVG_DEG, seed=SEED)
    # Create a second snapshot by adding a small random displacement
    feats1 = {i: v + 0.1 * np.random.randn(*v.shape) for i, v in feats0.items()}

    # 2. Choose a random node and fabricate a target velocity
    node = random.choice(list(adj.keys()))
    # For demonstration we define the true velocity as the norm of the flow vector
    true_vel = float(np.linalg.norm(flow_target(feats0[node], feats1[node])))

    # 3. Initialise KAN parameters and NLMS weights
    # Input dimension = graph_feature_dim (2) + flow_dim (2) = 4
    INPUT_DIM = 4
    kan_params = initialise_kan_params(INPUT_DIM, num_knots=6, seed=SEED)

    # KAN adds one bias → total transformed dimension = INPUT_DIM + 1
    transformed_dim = INPUT_DIM + 1
    weights = np.zeros(transformed_dim)  # start from zero

    # 4. Run a single hybrid prediction & update
    pred_before = hybrid_predict(
        weights,
        node,
        adj,
        feats0,
        feats1,
        TIME_T,
        kan_params,
    )
    print(f"Initial prediction for node {node}: {pred_before:.4f} (target {true_vel:.4f})")

    weights, err = hybrid_update(
        weights,
        node,
        adj,
        feats0,
        feats1,
        true_vel,
        TIME_T,
        kan_params,
        mu=MU,
    )
    print(f"Error after update: {err:.4f}")

    pred_after = hybrid_predict(
        weights,
        node,
        adj,
        feats0,
        feats1,
        TIME_T,
        kan_params,
    )
    print(f"Post‑update prediction: {pred_after:.4f}")

    # sanity check: weights should have changed from zero
    assert not np.allclose(weights, 0.0), "Weights did not update."


if __name__ == "__main__":
    _demo()