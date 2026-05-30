# DARWIN HAMMER — match 5675, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1201_s6.py (gen6)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s6.py (gen4)
# born: 2026-05-30T00:04:16Z

import math
import random
import sys
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from typing import Iterable, List, Tuple


# ----------------------------------------------------------------------
# Parent‑A core: B‑spline basis and a minimal KAN transform
# ----------------------------------------------------------------------
def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    n = x.shape[0]
    g = grid.shape[0]
    dx = grid[1] - grid[0] if g > 1 else 1.0
    extended = np.concatenate((
        np.full(k, grid[0] - dx * k),
        grid,
        np.full(k, grid[-1] + dx * k)
    ))
    B = np.zeros((n, g + k), dtype=float)
    for i in range(n):
        for j in range(g + k):
            if extended[j] <= x[i] < extended[j + 1]:
                B[i, j] = 1.0
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
    return B[:, : g + k - k]


def kan_transform(M: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    N, D = M.shape
    B_all = []
    for d in range(D):
        col = M[:, d]
        B_col = bspline_basis(col, grid, k)
        B_all.append(B_col)
    B_stacked = np.hstack(B_all)
    rng = np.random.default_rng(42)
    W = rng.standard_normal((B_stacked.shape[1], D))
    Ŧ = B_stacked @ W
    return Ŧ


# ----------------------------------------------------------------------
# Parent‑B core: bandit utilities and statistical helpers
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    if total_phases < 1 or current_phase < 1:
        raise ValueError("phases and phase must be positive")
    decay = 2 ** max(0, total_phases - current_phase)
    return min(1.0, 1.0 / decay)


def hoeffding_bound(R: float, delta: float, n: int) -> float:
    if n <= 0:
        raise ValueError("sample size n must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def tropical_max_plus_gain(gains: np.ndarray) -> float:
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
    N, D = Ŧ.shape
    variances = np.var(Ŧ, axis=0, ddof=1) + eps
    infos = N / variances
    w = infos / np.sum(infos)
    return w


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def bandit_propensities(actions: List[MathAction], fisher_weights: np.ndarray,
                        explore_factor: float) -> np.ndarray:
    feats = np.array([[a.expected_value, -a.cost, -a.risk] for a in actions], dtype=float)
    D_feat = feats.shape[1]
    if fisher_weights.shape[0] < D_feat:
        fisher_weights = np.pad(fisher_weights, (0, D_feat - fisher_weights.shape[0]), mode='constant', constant_values=fisher_weights[-1])
    elif fisher_weights.shape[0] > D_feat:
        fisher_weights = fisher_weights[:D_feat]
    scores = np.dot(fisher_weights, feats.T)
    uniform_prob = np.ones(len(actions)) / len(actions)
    probs = (1 - explore_factor) * softmax(scores) + explore_factor * uniform_prob
    return probs


# New fusion primitive: improved KAN transform with regularization
def improved_kan_transform(M: np.ndarray, grid: np.ndarray, k: int = 3, alpha: float = 0.1) -> np.ndarray:
    N, D = M.shape
    B_all = []
    for d in range(D):
        col = M[:, d]
        B_col = bspline_basis(col, grid, k)
        B_all.append(B_col)
    B_stacked = np.hstack(B_all)
    rng = np.random.default_rng(42)
    W = rng.standard_normal((B_stacked.shape[1], D))
    W_reg = W + alpha * np.eye(D)
    Ŧ = B_stacked @ W_reg
    return Ŧ


# New fusion primitive: improved Fisher information weights with robustness
def improved_fisher_information_weights(Ŧ: np.ndarray, eps: float = 1e-8, beta: float = 0.1) -> np.ndarray:
    N, D = Ŧ.shape
    variances = np.var(Ŧ, axis=0, ddof=1) + eps
    infos = N / (variances + beta * np.mean(variances))
    w = infos / np.sum(infos)
    return w


# Improved bandit propensities with robustness and regularization
def improved_bandit_propensities(actions: List[MathAction], fisher_weights: np.ndarray,
                                  explore_factor: float, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
    feats = np.array([[a.expected_value, -a.cost, -a.risk] for a in actions], dtype=float)
    D_feat = feats.shape[1]
    if fisher_weights.shape[0] < D_feat:
        fisher_weights = np.pad(fisher_weights, (0, D_feat - fisher_weights.shape[0]), mode='constant', constant_values=fisher_weights[-1])
    elif fisher_weights.shape[0] > D_feat:
        fisher_weights = fisher_weights[:D_feat]
    scores = np.dot(fisher_weights, feats.T)
    uniform_prob = np.ones(len(actions)) / len(actions)
    probs = (1 - explore_factor) * softmax(scores) + explore_factor * uniform_prob
    return probs