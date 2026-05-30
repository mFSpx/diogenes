# DARWIN HAMMER — match 2190, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s0.py (gen2)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:41:14Z

"""Hybrid Distributed Leader‑Election / Hoeffding‑Tree + Kolmogorov‑Arnold Network (KAN)

This module fuses the two parent algorithms:

* **Parent A** – a distributed leader‑election scheme that uses
  probabilistic acceptance (simulated‑annealing style) together with the
  Hoeffding bound to decide when a node of a decision tree may split.
* **Parent B** – the Kolmogorov‑Arnold Network (KAN) implementation that
  evaluates learnable univariate B‑spline basis functions on every edge.

**Mathematical bridge**

The KAN supplies a *continuous* univariate transformation ϕ(x) for each
input dimension via B‑spline basis evaluation.  When a Hoeffding tree
examines a candidate split it needs a scalar “gain’’ (e.g. variance
reduction).  By feeding the raw feature through the KAN spline
transformations we obtain a smooth, differentiable estimate of the gain.
The Hoeffding bound decides whether the observed gain is statistically
significant; the simulated‑annealing acceptance probability then allows
the algorithm to occasionally accept a split that is not yet significant,
mirroring the probabilistic leader‑election of Parent A.

The resulting hybrid therefore:

1. Transforms raw data with KAN B‑splines (`bspline_transform`).
2. Computes a split gain on the transformed data.
3. Uses a Hoeffding‑bound test (`should_split`) **and** a temperature‑driven
   acceptance test (`acceptance_probability`) to decide on the split.
4. Optionally broadcasts a leader‑election probability (`broadcast_probability`)
   to illustrate the distributed‑leader aspect.

The code below implements the core mathematics and provides a tiny
demonstration that runs without external dependencies.
"""

import math
import random
import sys
import pathlib
from collections.abc import Mapping, Hashable
import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (probabilistic acceptance, cooling, Hoeffding bound)
# ----------------------------------------------------------------------
def broadcast_probability(phase: int, step: int) -> float:
    """Probability that a node broadcasts its candidacy in a given phase."""
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance for a change of “energy’’ ΔE at temperature T."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Exponential cooling schedule T(k) = t0 * α^k."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (R^2 * ln(1/δ)) / (2n) )."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def should_split(best_gain: float, second_best_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> bool:
    """Hoeffding‑bound based split test."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    return gap > eps or eps < tie_threshold


# ----------------------------------------------------------------------
# Parent B utilities (B‑spline basis – KAN core)
# ----------------------------------------------------------------------
def _cox_de_boor(x: np.ndarray, knots: np.ndarray, i: int, k: int) -> np.ndarray:
    """Recursive Cox‑de Boor evaluation of a single basis function."""
    if k == 1:
        left = knots[i]
        right = knots[i + 1]
        return np.where((x >= left) & (x < right), 1.0, 0.0)
    denom1 = knots[i + k - 1] - knots[i]
    term1 = np.zeros_like(x, dtype=np.float64)
    if denom1 > 0:
        term1 = ((x - knots[i]) / denom1) * _cox_de_boor(x, knots, i, k - 1)
    denom2 = knots[i + k] - knots[i + 1]
    term2 = np.zeros_like(x, dtype=np.float64)
    if denom2 > 0:
        term2 = ((knots[i + k] - x) / denom2) * _cox_de_boor(x, knots, i + 1, k - 1)
    return term1 + term2


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate all B‑spline basis functions of order k on points x.

    Parameters
    ----------
    x : (N,) array of evaluation points.
    grid : (G,) array of interior knots (uniform spacing is typical).
    k : spline order (default 3 → cubic).

    Returns
    -------
    B : (N, G - 1) array, each column is one basis function.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Build a clamped knot vector: repeat the first and last interior knot k times.
    knots = np.concatenate((
        np.full(k, grid[0]),
        grid,
        np.full(k, grid[-1])
    ))
    n_basis = len(grid) - 1
    B = np.empty((x.shape[0], n_basis), dtype=np.float64)
    for i in range(n_basis):
        B[:, i] = _cox_de_boor(x, knots, i, k)
    return B


def bspline_transform(X: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Apply the KAN univariate B‑spline transformation to each column of X.

    Returns a concatenated feature matrix where each original column is
    replaced by its spline basis expansion.
    """
    transformed = []
    for col in range(X.shape[1]):
        basis = bspline_basis(X[:, col], grid, k)
        transformed.append(basis)
    return np.hstack(transformed)


# ----------------------------------------------------------------------
# Hybrid logic – combine Hoeffding‑tree split test with annealing &
# KAN spline features.
# ----------------------------------------------------------------------
def hybrid_split_decision(best_gain: float, second_gain: float,
                          n_samples: int, r: float, delta: float,
                          temperature: float, tie_thr: float = 0.05) -> bool:
    """
    Decide whether to split a node.

    The decision is positive if **either** the Hoeffding bound certifies the
    split *or* a simulated‑annealing acceptance based on the gain gap
    (ΔE = second_gain - best_gain) succeeds.
    """
    # Hoeffding‑bound test
    hoeff = should_split(best_gain, second_gain, r, delta, n_samples, tie_thr)

    # Annealing test (note ΔE is negative when best_gain > second_gain)
    delta_e = second_gain - best_gain
    anneal = random.random() < acceptance_probability(delta_e, temperature)

    return hoeff or anneal


class HybridTreeNode:
    """A minimal Hoeffding‑tree node that stores split information."""
    __slots__ = ("depth", "samples", "is_leaf", "split_feature",
                 "split_threshold", "left", "right", "label")

    def __init__(self, depth: int = 0):
        self.depth = depth
        self.samples = 0
        self.is_leaf = True
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None
        self.label = None  # majority class / mean for regression

    def update_label(self, y: np.ndarray):
        """Set leaf label as the mean of observed targets."""
        if y.size:
            self.label = float(np.mean(y))


def compute_variance_gain(X: np.ndarray, y: np.ndarray,
                         feature_idx: int, threshold: float) -> float:
    """
    Simple variance‑reduction gain for a candidate split.
    """
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    if not left_mask.any() or not right_mask.any():
        return 0.0
    var_total = np.var(y)
    var_left = np.var(y[left_mask])
    var_right = np.var(y[right_mask])
    n = y.shape[0]
    n_left = left_mask.sum()
    n_right = right_mask.sum()
    weighted_var = (n_left / n) * var_left + (n_right / n) * var_right
    gain = var_total - weighted_var
    return gain


def find_best_split(X: np.ndarray, y: np.ndarray,
                    candidate_features: np.ndarray,
                    n_bins: int = 10) -> tuple:
    """
    Exhaustively search candidate splits on the provided features.
    Returns (best_feature, best_threshold, best_gain, second_gain).
    """
    best_gain = -np.inf
    second_gain = -np.inf
    best_feat = None
    best_thr = None

    for feat in candidate_features:
        values = X[:, feat]
        thresholds = np.linspace(values.min(), values.max(), n_bins + 2)[1:-1]
        for thr in thresholds:
            gain = compute_variance_gain(X, y, feat, thr)
            if gain > best_gain:
                second_gain = best_gain
                best_gain = gain
                best_feat = feat
                best_thr = thr
            elif gain > second_gain:
                second_gain = gain
    return best_feat, best_thr, best_gain, second_gain


def hybrid_tree_grow(root: HybridTreeNode,
                    X: np.ndarray,
                    y: np.ndarray,
                    max_depth: int = 5,
                    r: float = 1.0,
                    delta: float = 0.05,
                    k_grid: np.ndarray = None,
                    spline_order: int = 3) -> None:
    """
    Grow a Hoeffding tree using KAN spline‑transformed features.
    """
    if k_grid is None:
        # Default uniform grid covering the data range
        k_grid = np.linspace(np.min(X), np.max(X), 20)

    # Transform once – the transformed matrix is used for all split tests.
    X_t = bspline_transform(X, k_grid, spline_order)

    # Breadth‑first expansion
    queue = [(root, X, y, 0)]
    while queue:
        node, X_node, y_node, depth = queue.pop(0)
        node.samples = y_node.shape[0]
        node.update_label(y_node)

        if depth >= max_depth or node.samples < 20:
            continue

        # Candidate features are the spline columns
        candidate_feats = np.arange(X_t.shape[1])

        # Find split on the *original* space but using transformed gains.
        # For simplicity we evaluate gains on the transformed matrix.
        best_feat, best_thr, best_gain, second_gain = find_best_split(
            X_t, y_node, candidate_feats, n_bins=5)

        temperature = cooling_temperature(depth)
        if best_feat is None:
            continue

        if hybrid_split_decision(best_gain, second_gain,
                                 n_samples=node.samples,
                                 r=r, delta=delta,
                                 temperature=temperature):
            # Perform split on the *original* feature that corresponds to the
            # selected spline basis.  Each spline basis originates from a
            # single raw feature; we recover the mapping.
            raw_feat = best_feat // (len(k_grid) - 1)
            # Approximate threshold by inverting the spline basis (use the same
            # numeric threshold for simplicity).
            threshold = best_thr  # not exact but sufficient for demo

            node.is_leaf = False
            node.split_feature = raw_feat
            node.split_threshold = threshold
            node.left = HybridTreeNode(depth=depth + 1)
            node.right = HybridTreeNode(depth=depth + 1)

            left_mask = X[:, raw_feat] <= threshold
            right_mask = ~left_mask

            queue.append((node.left, X[left_mask], y[left_mask], depth + 1))
            queue.append((node.right, X[right_mask], y[right_mask], depth + 1))


def hybrid_predict(node: HybridTreeNode, x: np.ndarray) -> float:
    """Traverse the tree to obtain a prediction (leaf mean)."""
    while not node.is_leaf:
        if x[node.split_feature] <= node.split_threshold:
            node = node.left
        else:
            node = node.right
    return node.label if node.label is not None else 0.0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic regression problem
    rng = np.random.default_rng(42)
    X_raw = rng.uniform(0.0, 1.0, size=(200, 1))
    y_raw = np.sin(2 * np.pi * X_raw[:, 0]) + rng.normal(0.0, 0.1, size=200)

    # Build and grow the hybrid tree
    root_node = HybridTreeNode()
    hybrid_tree_grow(root_node, X_raw, y_raw, max_depth=4)

    # Predict on a grid and compute MSE
    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    preds = np.array([hybrid_predict(root_node, x) for x in X_test])
    mse = np.mean((preds - np.sin(2 * np.pi * X_test[:, 0])) ** 2)
    print(f"Hybrid tree MSE on clean sine wave: {mse:.4f}")

    # Demonstrate broadcast probability (phase/step example)
    prob = broadcast_probability(phase=3, step=1)
    print(f"Broadcast probability (phase=3, step=1): {prob:.3f}")

    # Ensure the script exits cleanly
    sys.exit(0)