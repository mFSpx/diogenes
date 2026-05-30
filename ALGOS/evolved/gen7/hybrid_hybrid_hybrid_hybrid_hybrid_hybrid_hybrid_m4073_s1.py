# DARWIN HAMMER — match 4073, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_liquid_m2175_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1826_s0.py (gen6)
# born: 2026-05-29T23:53:22Z

"""Hybrid Algorithm Fusion of hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_liquid_m2175_s2.py (A)
and hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1826_s0.py (B).

Mathematical bridge:
- Parent A provides a curvature matrix C derived from an adjacency matrix and
  Bayesian edge‑weights B = exp(C)·L (L = edge lengths).  C and B are purely
  matrix‑based scalings that can be interpreted as per‑edge “trust’’ or
  “conductivity’’ factors.
- Parent B updates a allocation vector w via a Normalised Least‑Mean‑Squares
  (NLMS) rule and evaluates similarity between vectors with a geometric
  (cosine‑like) measure.

The fusion treats the Bayesian edge‑weights B as a *diagonal pre‑conditioner*
for the NLMS adaptation: each component of w is updated with a step size
scaled by the corresponding edge‑weight.  The error signal fed to NLMS is the
difference between a target profile and the actual usage, where the error is
weighted by the similarity between the current allocation and the target
profile.  This yields a single coherent update that simultaneously respects
graph curvature (A) and adaptive vector learning (B)."""

import numpy as np
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Parent A – curvature and Bayesian edge weights
# ----------------------------------------------------------------------
def compute_curvature(adj_matrix: np.ndarray) -> np.ndarray:
    """Ollivier‑Ricci‑style curvature: C[i,j] = -1/(1+exp(-A[i,j])) for existing edges."""
    C = np.zeros_like(adj_matrix, dtype=float)
    mask = adj_matrix > 0
    C[mask] = -1.0 / (1.0 + np.exp(-adj_matrix[mask]))
    return C


def bayesian_edge_weights(curvature: np.ndarray, edge_lengths: np.ndarray) -> np.ndarray:
    """Bayesian edge weights B = exp(C) * L."""
    return np.exp(curvature) * edge_lengths


# ----------------------------------------------------------------------
# Parent B – NLMS adaptation, scheduling and geometric similarity
# ----------------------------------------------------------------------
MU = 0.1          # base step‑size
EPSILON = 1e-6    # stabiliser for norm


def w_base(day: int, groups: int) -> np.ndarray:
    """Weekday weight base – a deterministic seed vector."""
    return np.array([np.sin(2 * np.pi * (day + i) / groups) + 1 for i in range(groups)], dtype=float) / groups


def scheduling(w: np.ndarray, M_total: np.ndarray, M_available: np.ndarray) -> np.ndarray:
    """Actual memory usage given allocation w."""
    return np.minimum(w * M_total, M_available)


def geometric_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine‑like similarity in [0,1]."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b) + EPSILON
    return max(0.0, min(1.0, dot / norm))


def nlms_adaptation_weighted(w: np.ndarray, e: np.ndarray, x: np.ndarray,
                             weight_matrix: np.ndarray) -> np.ndarray:
    """
    Weighted NLMS update:
        w_new = w + MU * (B ∘ e) ∘ x / (||x||^2 + ε)

    where B is a diagonal matrix built from the Bayesian edge‑weights.
    """
    norm_x = np.dot(x, x) + EPSILON
    # Extract diagonal scaling factors
    scaling = np.diag(weight_matrix)
    delta = MU * (scaling * e) * x / norm_x
    return w + delta


# ----------------------------------------------------------------------
# Fusion – combined operation
# ----------------------------------------------------------------------
def hybrid_step(adj_matrix: np.ndarray,
                edge_lengths: np.ndarray,
                w: np.ndarray,
                target_profile: np.ndarray,
                M_total: np.ndarray,
                M_available: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Perform one hybrid iteration.

    Returns
    -------
    w_new : np.ndarray
        Updated allocation vector.
    diagnostics : dict
        Dictionary with intermediate quantities for inspection.
    """
    # 1. Graph‑based scalings (Parent A)
    C = compute_curvature(adj_matrix)
    B = bayesian_edge_weights(C, edge_lengths)

    # 2. Current usage and error (Parent B)
    usage = scheduling(w, M_total, M_available)
    error_vec = target_profile - usage

    # 3. Similarity‑driven modulation of the error
    sim = geometric_similarity(w, target_profile)
    mod_error = error_vec * sim  # shrink error when vectors are already similar

    # 4. Weighted NLMS adaptation using Bayesian edge‑weights
    w_new = nlms_adaptation_weighted(w, mod_error, w, B)

    diagnostics = {
        "curvature": C,
        "bayesian_weights": B,
        "usage": usage,
        "error": error_vec,
        "similarity": sim,
        "updated_w": w_new,
    }
    return w_new, diagnostics


# ----------------------------------------------------------------------
# Example utilities (policy tracking – retained from Parent A)
# ----------------------------------------------------------------------
_POLICY: dict[str, list[float]] = {}  # action_id -> [total_reward, count]


def reset_policy() -> None:
    """Clear the global reward statistics."""
    _POLICY.clear()


def update_policy(updates: list[tuple[str, float]]) -> None:
    """
    Accumulate rewards.

    Parameters
    ----------
    updates : list of (action_id, reward)
    """
    for action_id, reward in updates:
        stats = _POLICY.setdefault(action_id, [0.0, 0.0])
        stats[0] += reward
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph: 3 nodes, directed edges with arbitrary strengths
    adj = np.array([[0, 2.0, 0],
                    [0,   0, 1.5],
                    [0.5, 0,   0]], dtype=float)

    # Edge lengths (same shape as adj, zeros where no edge)
    lengths = np.where(adj > 0, 1.0, 0.0)

    # Allocation vector (3 groups)
    w = w_base(day=2, groups=3)

    # Target memory‑usage profile (arbitrary)
    target = np.array([0.7, 0.4, 0.6], dtype=float)

    # Total demand and available memory vectors
    M_total = np.array([1.0, 1.0, 1.0], dtype=float)
    M_available = np.array([0.8, 0.5, 0.9], dtype=float)

    # Run a few hybrid iterations
    for step in range(5):
        w, diag = hybrid_step(adj, lengths, w, target, M_total, M_available)
        print(f"Step {step+1}: w = {w.round(4)}  similarity={diag['similarity']:.4f}")

    # Demonstrate policy utilities
    update_policy([("alloc", 1.0), ("alloc", 0.5)])
    print("Policy stats:", _POLICY)