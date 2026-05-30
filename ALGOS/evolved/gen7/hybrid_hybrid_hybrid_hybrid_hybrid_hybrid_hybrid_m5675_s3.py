# DARWIN HAMMER — match 5675, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1201_s6.py (gen6)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s6.py (gen4)
# born: 2026-05-30T00:04:16Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from typing import Iterable, List, Tuple

# ----------------------------------------------------------------------
# Parent‑A core: B‑spline basis and a minimal KAN transform
# ----------------------------------------------------------------------
def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Cox‑de Boor recursion for uniform clamped B‑splines.

    Parameters
    ----------
    x : ndarray shape (N,)
        Evaluation points.
    grid : ndarray shape (G,)
        Interior knots (uniform spacing assumed).
    k : int, default 3
        Spline order (degree = k‑1). k=3 yields cubic splines.

    Returns
    -------
    B : ndarray shape (N, G + k)
        Basis matrix where B[i, j] = B_j,k(x_i).
    """
    n = x.shape[0]
    g = grid.shape[0]

    # Build clamped knot vector
    dx = grid[1] - grid[0] if g > 1 else 1.0
    extended = np.concatenate((
        np.full(k, grid[0] - dx * k),
        grid,
        np.full(k, grid[-1] + dx * k)
    ))

    # Zeroth‑order basis (piecewise constant)
    B = np.zeros((n, g + k), dtype=float)
    for i in range(n):
        for j in range(g + k):
            if extended[j] <= x[i] < extended[j + 1]:
                B[i, j] = 1.0
    # Recursion for higher orders
    for d in range(1, k):
        for i in range(n):
            for j in range(g + k - d):
                left_den = extended[j + d] - extended[j]
                right_den = extended[j + d + 1] - extended[j + 1]

                left = 0.0
                if left_den != 0:
                    left = ((x[i] - extended[j]) / left_den) * B[i, j]

                right = 0.0
                if right_den != 0:
                    right = ((extended[j + d + 1] - x[i]) / right_den) * B[i, j + 1]

                B[i, j] = left + right
    return B[:, : g + k - k]  # discard the extra columns introduced by recursion


def kan_transform(M: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Very light‑weight Kolmogorov‑Arnold‑Network style transform.

    For each column of `M` we evaluate a B‑spline basis and then linearly mix the
    basis functions with a randomly‑initialised weight matrix `W`.  The operation
    can be written as

        Ŧ = B(M) @ W

    where `B(M)` is the concatenated basis matrix for all columns.

    Parameters
    ----------
    M : ndarray shape (N, D)
        Input matrix.
    grid : ndarray
        Knot vector used for all columns.
    k : int
        Spline order.

    Returns
    -------
    Ŧ : ndarray shape (N, D)
        Transformed matrix with the same column count as `M`.
    """
    N, D = M.shape
    # Build basis for each column and stack horizontally
    B_all = []
    for d in range(D):
        col = M[:, d]
        B_col = bspline_basis(col, grid, k)  # shape (N, G')
        B_all.append(B_col)
    B_stacked = np.hstack(B_all)  # shape (N, D * G')
    # Random mixing matrix; fixed seed for reproducibility
    rng = np.random.default_rng(42)
    W = rng.standard_normal((B_stacked.shape[1], D))
    Ŧ = B_stacked @ W
    return Ŧ


# ----------------------------------------------------------------------
# Parent‑B core: bandit utilities and statistical helpers
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """Decay factor used for exploration scheduling."""
    if total_phases < 1 or current_phase < 1:
        raise ValueError("phases and phase must be positive")
    decay = 2 ** max(0, total_phases - current_phase)
    return min(1.0, 1.0 / decay)


def hoeffding_bound(R: float, delta: float, n: int) -> float:
    """Finite‑sample confidence radius."""
    if n <= 0:
        raise ValueError("sample size n must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def tropical_max_plus_gain(gains: np.ndarray) -> float:
    """Tropical (max‑plus) aggregation."""
    return float(np.max(gains))


# ----------------------------------------------------------------------
# Shared data structures (Parent‑B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Fusion primitives
# ----------------------------------------------------------------------
def fisher_information_weights(Ŧ: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute a Fisher‑information‑derived weight for each column of the transformed
    matrix `Ŧ`.  Assuming a Gaussian model for the column mean μ, the Fisher
    information for μ is `I = n / var`.  The function returns a normalized weight
    vector that sums to one.

    Parameters
    ----------
    Ŧ : ndarray shape (N, D)
        Transformed data.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    w : ndarray shape (D,)
        Normalised Fisher weights.
    """
    N, D = Ŧ.shape
    variances = np.var(Ŧ, axis=0, ddof=1) + eps
    infos = N / variances
    w = infos / np.sum(infos)
    return w


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def bandit_propensities(actions: List[MathAction], fisher_weights: np.ndarray,
                        explore_factor: float) -> np.ndarray:
    """
    Combine Fisher information with action features to produce a probability
    distribution over actions.

    The feature vector for an action a is taken as
        φ(a) = [expected_value, -cost, -risk]
    and the raw score is `s = fisher_weights · φ(a)`.  An exploration term
    (uniform mixture) controlled by `explore_factor` is mixed in.

    Parameters
    ----------
    actions : list[MathAction]
        Candidate actions.
    fisher_weights : ndarray shape (D,)
        Fisher weights from the transformed matrix (D must match the number of
        action features; we simply truncate or repeat as needed).
    explore_factor : float in [0,1]
        Weight of uniform exploration.

    Returns
    -------
    probs : ndarray shape (len(actions),)
        Probability of selecting each action.
    """
    # Build feature matrix (A x 3)
    feats = np.array([[a.expected_value, -a.cost, -a.risk] for a in actions], dtype=float)
    D_feat = feats.shape[1]

    # Align fisher_weights length to D_feat
    if fisher_weights.shape[0] < D_feat:
        # repeat last entries
        repeats = np.tile(fisher_weights[-1], D_feat - fisher_weights.shape[0])
        fisher_weights_extended = np.concatenate((fisher_weights, repeats))
    elif fisher_weights.shape[0] > D_feat:
        fisher_weights_extended = fisher_weights[:D_feat]
    else:
        fisher_weights_extended = fisher_weights

    scores = np.dot(feats, fisher_weights_extended)
    exploration = np.full(len(actions), explore_factor / len(actions))
    propensities = softmax(scores) * (1 - explore_factor) + exploration
    return propensities


def improved_bandit_propensities(actions: List[MathAction], fisher_weights: np.ndarray,
                                 explore_factor: float, regularizer: float = 1e-4) -> np.ndarray:
    """
    Improved version of bandit_propensities with L2 regularization.

    Parameters
    ----------
    actions : list[MathAction]
        Candidate actions.
    fisher_weights : ndarray shape (D,)
        Fisher weights from the transformed matrix (D must match the number of
        action features; we simply truncate or repeat as needed).
    explore_factor : float in [0,1]
        Weight of uniform exploration.
    regularizer : float
        L2 regularization strength.

    Returns
    -------
    probs : ndarray shape (len(actions),)
        Probability of selecting each action.
    """
    # Build feature matrix (A x 3)
    feats = np.array([[a.expected_value, -a.cost, -a.risk] for a in actions], dtype=float)
    D_feat = feats.shape[1]

    # Align fisher_weights length to D_feat
    if fisher_weights.shape[0] < D_feat:
        # repeat last entries
        repeats = np.tile(fisher_weights[-1], D_feat - fisher_weights.shape[0])
        fisher_weights_extended = np.concatenate((fisher_weights, repeats))
    elif fisher_weights.shape[0] > D_feat:
        fisher_weights_extended = fisher_weights[:D_feat]
    else:
        fisher_weights_extended = fisher_weights

    # Add L2 regularization
    fisher_weights_extended = fisher_weights_extended / (np.linalg.norm(fisher_weights_extended) + regularizer)

    scores = np.dot(feats, fisher_weights_extended)
    exploration = np.full(len(actions), explore_factor / len(actions))
    propensities = softmax(scores) * (1 - explore_factor) + exploration
    return propensities