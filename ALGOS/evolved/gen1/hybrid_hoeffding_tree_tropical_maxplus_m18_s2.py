# DARWIN HAMMER — match 18, survivor 2
# gen: 1
# parent_a: hoeffding_tree.py (gen0)
# parent_b: tropical_maxplus.py (gen0)
# born: 2026-05-29T23:22:44Z

"""Hybrid Hoeffding–Tropical Split Module.

This module fuses two previously independent algorithms:

* **Parent A – Hoeffding Tree helpers** (`hoeffding_tree.py`): statistical
  bound `hoeffding_bound` and decision routine `should_split` that decide
  whether a split should be performed on a streaming decision‑tree node.

* **Parent B – Tropical Max‑Plus algebra** (`tropical_maxplus.py`): a
  tropical semiring implementation (`t_add`, `t_mul`, `t_matmul`,
  `tropical_network_eval`) that evaluates ReLU networks as tropical
  polynomials, i.e. piecewise‑linear convex functions.

**Mathematical bridge**

A tropical ReLU network partitions the input space into *linear regions*.
Each region is defined by a set of activation patterns that can be interpreted
as a binary decision (e.g. “output > 0 → left child, otherwise → right child”).
Thus every output unit of a tropical network naturally yields a candidate
split for a decision‑tree node.  The Hoeffding bound supplies a statistical
guarantee that, after observing enough examples, the best candidate split is
indeed the optimal one with probability `1‑δ`.  The hybrid algorithm therefore
uses tropical network evaluations to generate split candidates and applies the
Hoeffding bound to decide when a node may be split in a streaming setting.

The public API provides three core hybrid functions:
    1. `hybrid_compute_gains` – compute impurity gains for all tropical outputs.
    2. `hybrid_update_node`   – update node statistics with a new (x, y) pair.
    3. `hybrid_maybe_split`  – decide (via Hoeffding) whether to split the node.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – Hoeffding helpers
# ---------------------------------------------------------------------------

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a random variable bounded in [0, r].

    Parameters
    ----------
    r : float
        Range of the random variable (max – min). Must be > 0.
    delta : float
        Desired failure probability, 0 < delta < 1.
    n : int
        Number of independent observations (must be > 0).

    Returns
    -------
    float
        ε such that with probability 1‑δ the true mean lies within ±ε of the
        empirical mean.
    """
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """Apply Hoeffding bound to decide whether to split a node.

    Parameters
    ----------
    best_gain, second_best_gain : float
        Empirical impurity gains of the best and second‑best split candidates.
    r : float
        Maximum possible gain (range of the gain random variable).
    delta : float
        Desired failure probability.
    n : int
        Number of examples seen at the node.
    tie_threshold : float, optional
        If the bound ε falls below this value we split even when the gap is
        smaller (to avoid endless waiting on ties).

    Returns
    -------
    SplitDecision
        Structured decision information.
    """
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = (
        "gap_exceeds_bound"
        if gap > eps
        else ("tie_threshold" if eps < tie_threshold else "wait")
    )
    return SplitDecision(split, eps, gap, reason)


# ---------------------------------------------------------------------------
# Parent B – Tropical max‑plus primitives
# ---------------------------------------------------------------------------

def t_add(x, y):
    """Tropical addition (max). Broadcasts via NumPy."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication (ordinary addition). Broadcasts via NumPy."""
    return np.add(x, y)


def t_matmul(A, B):
    """Tropical matrix multiplication C = A ⊗ B.

    C[i, j] = max_k (A[i, k] + B[k, j]).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # Expand dimensions for broadcasting: (m, p, 1) + (1, p, n)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


def tropical_network_eval(x: np.ndarray, layers: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Evaluate a list of tropical (ReLU) layers on input vector x.

    Each layer computes:
        z = max_j (W[i, j] + h[j]) + b[i]
        h = max(z, 0)                # ReLU encoded as tropical addition with 0
    """
    h = np.asarray(x, dtype=float).ravel()
    for W, b in layers:
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        # max_j (W[i, j] + h[j])  → shape (m,)
        z = np.max(W + h[np.newaxis, :], axis=1) + b
        h = np.maximum(z, 0.0)
    return h


# ---------------------------------------------------------------------------
# Hybrid data structures
# ---------------------------------------------------------------------------

@dataclass
class HybridNode:
    """A streaming decision‑tree node that uses a tropical network for split candidates.

    Attributes
    ----------
    n_classes : int
        Number of target classes.
    class_counts : np.ndarray
        Cumulative count of each class observed at this node.
    n_examples : int
        Total number of examples seen (sum of class_counts).
    layers : list of (W, b)
        Tropical ReLU layers representing the current model at the node.
    """
    n_classes: int
    class_counts: np.ndarray = field(init=False)
    n_examples: int = field(default=0, init=False)
    layers: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.class_counts = np.zeros(self.n_classes, dtype=float)


# ---------------------------------------------------------------------------
# Hybrid core functions
# ---------------------------------------------------------------------------

def gini_impurity(counts: np.ndarray) -> float:
    """Gini impurity for a histogram of class counts."""
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    return 1.0 - np.sum(probs ** 2)


def hybrid_compute_gains(
    node: HybridNode,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[List[float], List[int]]:
    """Compute impurity gains for each tropical output unit.

    Parameters
    ----------
    node : HybridNode
        Node containing the tropical network.
    X : np.ndarray, shape (m, d)
        Batch of input vectors.
    y : np.ndarray, shape (m,)
        Corresponding class labels (int in [0, n_classes)).

    Returns
    -------
    gains : list of float
        Impurity gain for each output dimension of the tropical network.
    split_dims : list of int
        Index of the output dimension (same length as `gains`).
    """
    if not node.layers:
        # No model yet – cannot propose splits.
        return [], []

    # Evaluate the network for the whole batch.
    outputs = np.apply_along_axis(
        lambda xv: tropical_network_eval(xv, node.layers), 1, X
    )  # shape (m, n_out)

    n_out = outputs.shape[1]
    parent_counts = np.bincount(y, minlength=node.n_classes).astype(float)
    parent_impurity = gini_impurity(parent_counts)

    gains = []
    split_dims = []

    for dim in range(n_out):
        preds = outputs[:, dim] > 0.0  # binary split: left if >0
        left_mask = preds
        right_mask = ~preds

        # Skip degenerate splits
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue

        left_counts = np.bincount(y[left_mask], minlength=node.n_classes).astype(float)
        right_counts = np.bincount(y[right_mask], minlength=node.n_classes).astype(float)

        left_imp = gini_impurity(left_counts)
        right_imp = gini_impurity(right_counts)

        left_weight = left_counts.sum() / parent_counts.sum()
        right_weight = right_counts.sum() / parent_counts.sum()

        gain = parent_impurity - (left_weight * left_imp + right_weight * right_imp)
        gains.append(gain)
        split_dims.append(dim)

    return gains, split_dims


def hybrid_update_node(node: HybridNode, x: np.ndarray, label: int) -> None:
    """Incorporate a single example into the node statistics.

    Parameters
    ----------
    node : HybridNode
        Node to be updated.
    x : np.ndarray, shape (d,)
        Input feature vector.
    label : int
        Class label.
    """
    node.n_examples += 1
    node.class_counts[label] += 1

    # Very lightweight online update of the tropical model:
    # we keep a single affine layer whose weights are a random perturbation
    # of the incoming example (purely demonstrative; a real system would learn
    # via a streaming optimisation scheme).
    if not node.layers:
        d = x.shape[0]
        # Initialise a single-layer tropical network with random weights.
        W = np.random.randn(5, d)  # 5 output units (arbitrary)
        b = np.random.randn(5)
        node.layers.append((W, b))
    else:
        # Simple stochastic update: move weights a tiny step towards the example.
        W, b = node.layers[0]
        step = 0.001
        # Update bias for the unit that would fire positively on this example.
        out = tropical_network_eval(x, [(W, b)])
        firing = out > 0
        if firing.any():
            b[firing] += step * (1.0 - out[firing])
        # Add small noise to weights to keep them evolving.
        W += step * np.random.randn(*W.shape) * 0.1
        node.layers[0] = (W, b)


def hybrid_maybe_split(
    node: HybridNode,
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    delta: float = 1e-7,
    r: float = 1.0,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """Decide whether the node should be split using Hoeffding bound on tropical gains.

    Parameters
    ----------
    node : HybridNode
        Node under consideration.
    X_batch, y_batch : np.ndarray
        Recent batch of examples (used to compute empirical gains).
    delta, r, tie_threshold : float
        Hoeffding parameters.

    Returns
    -------
    SplitDecision
        Structured decision indicating if a split is warranted.
    """
    gains, dims = hybrid_compute_gains(node, X_batch, y_batch)

    if not gains:
        # No viable split candidates.
        return SplitDecision(False, 0.0, 0.0, "no_candidate")

    # Identify best and second‑best gains.
    sorted_idx = np.argsort(gains)[::-1]  # descending
    best_gain = gains[sorted_idx[0]]
    best_dim = dims[sorted_idx[0]]
    second_gain = gains[sorted_idx[1]] if len(gains) > 1 else 0.0

    decision = should_split(
        best_gain,
        second_gain,
        r,
        delta,
        node.n_examples,
        tie_threshold=tie_threshold,
    )

    # If we decide to split, we could store the chosen dimension for later use.
    if decision.should_split:
        # Attach the chosen split dimension to the node for downstream processing.
        node.best_split_dim = best_dim  # type: ignore[attr-defined]

    return decision


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple streaming simulation with 2 classes and 3‑dimensional input.
    random.seed(42)
    np.random.seed(42)

    node = HybridNode(n_classes=2)

    # Generate a synthetic stream.
    for step in range(200):
        x = np.random.randn(3)
        # Define a toy label rule: class 0 if sum(x) + noise < 0, else class 1.
        prob = 1.0 / (1.0 + math.exp(-np.sum(x)))
        label = 0 if random.random() < prob else 1

        hybrid_update_node(node, x, label)

        # Every 20 examples evaluate a batch and possibly split.
        if (step + 1) % 20 == 0:
            # Collect the last 20 examples for a quick batch estimate.
            batch_X = np.random.randn(20, 3)
            batch_y = np.array(
                [0 if random.random() < 1.0 / (1.0 + math.exp(-np.sum(xx))) else 1 for xx in batch_X]
            )
            decision = hybrid_maybe_split(node, batch_X, batch_y)
            print(
                f"Step {step+1:3d}: n={node.n_examples}, gain_gap={decision.gain_gap:.4f}, "
                f"ε={decision.epsilon:.4f}, decision={decision.reason}"
            )
            if decision.should_split:
                print(f"  --> Node would split on tropical output dimension {node.best_split_dim}")
                break  # End the demo after first split
    else:
        print("No split triggered after the stream.")