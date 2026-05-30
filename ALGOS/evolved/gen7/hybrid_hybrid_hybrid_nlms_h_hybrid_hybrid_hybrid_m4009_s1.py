# DARWIN HAMMER — match 4009, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_hybrid_m1247_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_model__honeybee_store_m388_s2.py (gen3)
# born: 2026-05-29T23:53:08Z

"""Hybrid RBF‑Curvature Regret Planner

Parents:
- hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_hybrid_m1247_s2.py (RBF kernel,
  expected‑value propagation, regret‑weighted policy)
- hybrid_hybrid_hybrid_model__honeybee_store_m388_s2.py (VRAM allocation,
  Ollivier‑Ricci curvature, scalar budget store)

Mathematical bridge:
Both parents construct an n×n similarity matrix on a set of nodes.
Parent A builds an RBF kernel **K_RBF** from feature vectors,
Parent B builds a curvature matrix **K_curv** from scalar allocations.
We fuse them linearly:

    K_fused = λ·K_RBF + (1‑λ)·K_curv          (0 ≤ λ ≤ 1)

The fused matrix drives the expected‑value update (A) while its mean
curvature term feeds the scalar store dynamics (B).  The store value then
modulates the regret‑weighted policy, closing the loop.
"""

import os
import sys
import math
import random
import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Atomic decision with a base expected value."""
    id: str
    tokens: Tuple[str, ...]
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

# ----------------------------------------------------------------------
# Parent A – RBF kernel utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def rbf_kernel_matrix(features: Dict[str, List[float]],
                      epsilon: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    """Return K_RBF and the ordered node list."""
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes

# ----------------------------------------------------------------------
# Parent B – Curvature utilities
# ----------------------------------------------------------------------
def curvature_matrix(allocation: np.ndarray) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature for a fully‑connected graph.
    For nodes i≠j: κ_ij = 1 - |x_i - x_j|, where distance d_ij = 1.
    Diagonal entries are set to 0.
    """
    n = allocation.shape[0]
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                K[i, j] = 0.0
            else:
                K[i, j] = 1.0 - abs(allocation[i] - allocation[j])
    return K

def update_store(S: float,
                 dt: float,
                 inflow: float,
                 outflow: float,
                 curvature_avg: float,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 gamma: float = 1.0) -> float:
    """
    Scalar store dynamics.
    S_{t+1} = max(0, S_t + dt·(α·inflow + γ·curvature_avg - β·outflow))
    """
    delta = dt * (alpha * inflow + gamma * curvature_avg - beta * outflow)
    return max(0.0, S + delta)

# ----------------------------------------------------------------------
# Fusion layer – combine matrices and propagate expectations
# ----------------------------------------------------------------------
def fused_similarity_matrix(features: Dict[str, List[float]],
                            allocation: np.ndarray,
                            lambda_rbf: float = 0.5,
                            epsilon: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    """
    Build K_fused = λ·K_RBF + (1‑λ)·K_curv.
    Returns the fused matrix and the node ordering (same for both inputs).
    """
    K_rbf, nodes = rbf_kernel_matrix(features, epsilon)
    K_curv = curvature_matrix(allocation)
    K_fused = lambda_rbf * K_rbf + (1.0 - lambda_rbf) * K_curv
    return K_fused, nodes

def compute_expected_values(actions: List[MathAction],
                            similarity: np.ndarray) -> Dict[str, float]:
    """
    Propagate base expected values through the fused similarity matrix.
    Mirrors Parent A's compute_expected_values but uses the fused matrix.
    """
    n = len(actions)
    if similarity.shape != (n, n):
        raise ValueError("Similarity matrix size must match number of actions")
    expected = {}
    for i, act in enumerate(actions):
        ev = act.expected_value
        for j, other in enumerate(actions):
            if i != j:
                sim = similarity[i, j]
                ev += sim * (other.expected_value - act.expected_value)
        expected[act.id] = ev
    return expected

def regret_weighted_policy(updates: List[Tuple[str, float]],
                          actions: List[MathAction],
                          expected_vals: Dict[str, float],
                          store: float,
                          store_scale: float = 0.01) -> Dict[str, float]:
    """
    Combine regret updates with the current scalar store.
    The store acts as a scaling factor on the regret contribution.
    """
    policy: Dict[str, float] = {}
    for action_id, regret in updates:
        action = next((a for a in actions if a.id == action_id), None)
        if action is None:
            continue
        # Regret term weighted by deviation from base expected value
        delta = regret * (expected_vals[action_id] - action.expected_value)
        # Store scaling reduces/increases aggressiveness
        policy[action_id] = policy.get(action_id, 0.0) + delta * (1.0 + store_scale * store)
    return policy

# ----------------------------------------------------------------------
# High‑level hybrid step
# ----------------------------------------------------------------------
def hybrid_step(features: Dict[str, List[float]],
                actions: List[MathAction],
                store: float,
                updates: List[Tuple[str, float]],
                dt: float = 1.0,
                lambda_rbf: float = 0.6,
                epsilon: float = 1.0,
                alpha: float = 1.0,
                beta: float = 1.0,
                gamma: float = 0.5,
                store_scale: float = 0.01) -> Tuple[Dict[str, float],
                                                   Dict[str, float],
                                                   float]:
    """
    Perform one iteration of the hybrid algorithm:
      1. Build allocation vector from current base expected values.
      2. Fuse RBF and curvature matrices.
      3. Propagate expected values.
      4. Update scalar store using curvature average and inflow/outflow.
      5. Produce regret‑weighted policy using the updated store.
    Returns (expected_vals, policy, new_store).
    """
    # 1. Allocation vector (treated as VRAM‑like scalar per node)
    allocation = np.array([a.expected_value for a in actions], dtype=np.float64)

    # 2. Fuse similarity matrices
    K_fused, _ = fused_similarity_matrix(features, allocation,
                                         lambda_rbf=lambda_rbf,
                                         epsilon=epsilon)

    # 3. Expected‑value propagation
    expected_vals = compute_expected_values(actions, K_fused)

    # 4. Store dynamics
    inflow = sum(v for v in expected_vals.values() if v > 0)
    outflow = -sum(v for v in expected_vals.values() if v < 0)
    curvature_avg = np.mean(K_fused)  # fused matrix includes curvature contribution
    new_store = update_store(store, dt, inflow, outflow,
                             curvature_avg, alpha, beta, gamma)

    # 5. Regret‑weighted policy
    policy = regret_weighted_policy(updates, actions, expected_vals,
                                    new_store, store_scale)

    return expected_vals, policy, new_store

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic data
    features = {
        "a": [0.1, 0.2],
        "b": [0.4, 0.2],
        "c": [0.3, 0.8]
    }

    actions = [
        MathAction(id="a", tokens=("move",), expected_value=1.0),
        MathAction(id="b", tokens=("jump",), expected_value=0.5),
        MathAction(id="c", tokens=("shoot",), expected_value=-0.2)
    ]

    # Example regret updates (action_id, regret)
    updates = [("a", 0.3), ("b", -0.1), ("c", 0.2)]

    initial_store = 10.0

    ev, pol, store = hybrid_step(features, actions, initial_store, updates)

    print("Expected values:", ev)
    print("Policy:", pol)
    print("New store:", store)