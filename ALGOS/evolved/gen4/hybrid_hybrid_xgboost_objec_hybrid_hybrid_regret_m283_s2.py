# DARWIN HAMMER — match 283, survivor 2
# gen: 4
# parent_a: hybrid_xgboost_objective_hybrid_ternary_lens__m33_s0.py (gen2)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s4.py (gen3)
# born: 2026-05-29T23:28:05Z

"""Hybrid XGBoost–Regret MinHash Analyzer

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – XGBoost objective utilities (logistic gradient/hessian,
  split gain, optimal leaf weight).
* **Parent B** – Regret‑Weighted strategy enriched with MinHash signatures,
  similarity metrics and Shannon entropy.

**Mathematical bridge**

For a binary logistic loss `ℓ(y, m) = log(1+exp(-y·m))` the XGBoost
gradient `g = σ(m) - y` and hessian `h = σ(m)(1-σ(m))` are used to
drive tree construction.  
We augment this loss with an *information‑theoretic regulariser* built
from the MinHash similarity `s ∈ [0,1]` of two ternary token sets
(`tokens_current`, `tokens_ref`) and the Shannon entropy `H` of the
combined MinHash signature distribution:


L = ℓ(y, m) + α·s·H


`α` scales the influence of the regulariser.  
Differentiating `L` w.r.t. the margin yields adjusted gradient and
hessian:


ĝ = g·(1 + α·s)          (entropy contributes a constant factor)
ĥ = h·(1 + α·s)


These adjusted statistics are then fed into the standard XGBoost
formulas for split gain and optimal leaf weight, thereby creating a
single unified learning system that simultaneously exploits boosting
and MinHash‑based similarity/entropy information.  
The three public functions below demonstrate this hybrid pipeline."""

import math
import random
import sys
from pathlib import Path
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (logistic gradient/hessian, split gain, leaf weight)
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x_arr = np.asarray(x)
    # avoid overflow
    pos_mask = x_arr >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(x_arr, dtype=float)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x_arr[pos_mask]))
    exp_x = np.exp(x_arr[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def binary_logistic_grad_hess(y_true: np.ndarray, margin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standard XGBoost binary logistic gradient and hessian."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """XGBoost split gain (gain = 0.5 * (children - parent) - gamma)."""
    gl, hl = float(left_gradient), float(left_hessian)
    gr, hr = float(right_gradient), float(right_hessian)
    parent = (gl + gr) ** 2 / (hl + hr + reg_lambda)
    children = gl ** 2 / (hl + reg_lambda) + gr ** 2 / (hr + reg_lambda)
    return 0.5 * (children - parent) - gamma


def optimal_leaf_weight(
    gradient_sum: float,
    hessian_sum: float,
    reg_lambda: float = 1.0,
) -> float:
    """Optimal leaf weight for a given gradient/hessian sum."""
    return -float(gradient_sum) / (float(hessian_sum) + float(reg_lambda))


# ----------------------------------------------------------------------
# Parent B utilities (MinHash signature, similarity, entropy)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """
    Compute a MinHash signature of length `k` for the given token set.
    Empty token sets produce a signature of maximal hash values.
    """
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def shannon_entropy(values: Iterable[int]) -> float:
    """Shannon entropy of a discrete distribution given by integer counts."""
    vals = list(values)
    if not vals:
        return 0.0
    total = float(sum(vals))
    if total == 0.0:
        return 0.0
    probs = np.array(vals, dtype=float) / total
    # protect against log(0)
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_grad_hess(
    y_true: np.ndarray,
    margin: np.ndarray,
    tokens_cur: Iterable[str],
    tokens_ref: Iterable[str],
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute XGBoost logistic gradient/hessian adjusted by MinHash similarity
    and entropy regularisation.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1).
    margin : np.ndarray
        Current model margin (prediction before sigmoid).
    tokens_cur / tokens_ref : Iterable[str]
        Token sets representing the current input and a reference input.
    alpha : float
        Scaling factor for the regulariser.

    Returns
    -------
    g_adj, h_adj : Tuple[np.ndarray, np.ndarray]
        Adjusted gradient and hessian vectors.
    """
    # Base logistic statistics
    g, h = binary_logistic_grad_hess(y_true, margin)

    # MinHash similarity
    sig_cur = signature(tokens_cur)
    sig_ref = signature(tokens_ref)
    sim = similarity(sig_cur, sig_ref)  # in [0,1]

    # Entropy of the combined signature multiset
    combined = sig_cur + sig_ref
    ent = shannon_entropy(combined)  # non‑negative

    # Regularisation factor (1 + α·s·H) – H scales the impact of similarity
    factor = 1.0 + alpha * sim * ent

    return g * factor, h * factor


def hybrid_split_gain(
    left_grad: float,
    left_hess: float,
    right_grad: float,
    right_hess: float,
    tokens_left: Iterable[str],
    tokens_right: Iterable[str],
    tokens_parent: Iterable[str],
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    alpha: float = 0.1,
) -> float:
    """
    Compute split gain where the left/right gradient/hessian sums are first
    adjusted using the MinHash similarity/entropy bridge.

    The parent node's token set is used as the reference for both children.
    """
    # similarity/entropy factors for each child against the parent
    sig_parent = signature(tokens_parent)
    sig_l = signature(tokens_left)
    sig_r = signature(tokens_right)

    sim_l = similarity(sig_l, sig_parent)
    sim_r = similarity(sig_r, sig_parent)

    ent_l = shannon_entropy(sig_l + sig_parent)
    ent_r = shannon_entropy(sig_r + sig_parent)

    factor_l = 1.0 + alpha * sim_l * ent_l
    factor_r = 1.0 + alpha * sim_r * ent_r

    adj_left_grad = left_grad * factor_l
    adj_left_hess = left_hess * factor_l
    adj_right_grad = right_grad * factor_r
    adj_right_hess = right_hess * factor_r

    return split_gain(
        adj_left_grad,
        adj_left_hess,
        adj_right_grad,
        adj_right_hess,
        reg_lambda=reg_lambda,
        gamma=gamma,
    )


def hybrid_optimal_leaf_weight(
    gradient_sum: float,
    hessian_sum: float,
    tokens_leaf: Iterable[str],
    tokens_ref: Iterable[str],
    *,
    reg_lambda: float = 1.0,
    beta: float = 0.01,
) -> float:
    """
    Compute leaf weight with an entropy‑based bias.

    The entropy of the leaf's token signature relative to a reference set
    shrinks the effective gradient (encouraging smoother leaves).
    """
    sig_leaf = signature(tokens_leaf)
    sig_ref = signature(tokens_ref)

    sim = similarity(sig_leaf, sig_ref)
    ent = shannon_entropy(sig_leaf + sig_ref)

    # bias term reduces gradient magnitude proportionally to information content
    bias = 1.0 - beta * sim * ent
    bias = max(bias, 0.0)  # keep non‑negative

    adj_grad = gradient_sum * bias
    return optimal_leaf_weight(adj_grad, hessian_sum, reg_lambda=reg_lambda)


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic binary classification data
    np.random.seed(42)
    y = np.random.randint(0, 2, size=10).astype(float)
    margin = np.random.randn(10)

    # Token sets (could be derived from model features, here random strings)
    tokens_cur = [f"tok{i}" for i in range(5)]
    tokens_ref = [f"ref{i}" for i in range(5)]

    # Hybrid gradient/hessian
    g_adj, h_adj = hybrid_grad_hess(y, margin, tokens_cur, tokens_ref, alpha=0.05)
    print("Adjusted gradient:", g_adj)
    print("Adjusted hessian :", h_adj)

    # Simulate a split with made‑up gradient/hessian sums
    left_g, left_h = float(g_adj[:5].sum()), float(h_adj[:5].sum())
    right_g, right_h = float(g_adj[5:].sum()), float(h_adj[5:].sum())

    gain = hybrid_split_gain(
        left_g,
        left_h,
        right_g,
        right_h,
        tokens_left=tokens_cur[:3],
        tokens_right=tokens_cur[3:],
        tokens_parent=tokens_cur,
        reg_lambda=1.0,
        gamma=0.0,
        alpha=0.05,
    )
    print("Hybrid split gain:", gain)

    # Leaf weight
    leaf_w = hybrid_optimal_leaf_weight(
        gradient_sum=g_adj.sum(),
        hessian_sum=h_adj.sum(),
        tokens_leaf=tokens_cur,
        tokens_ref=tokens_ref,
        reg_lambda=1.0,
        beta=0.01,
    )
    print("Hybrid leaf weight:", leaf_w)