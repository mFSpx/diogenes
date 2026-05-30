# DARWIN HAMMER — match 18, survivor 1
# gen: 1
# parent_a: hoeffding_tree.py (gen0)
# parent_b: tropical_maxplus.py (gen0)
# born: 2026-05-29T23:22:44Z

"""
This module integrates the Hoeffding bound helpers for stream splits from hoeffding_tree.py
and the tropical max-plus algebra for LUCIDOTA from tropical_maxplus.py.
The mathematical bridge between these structures is found in the application of tropical polynomials
to model decision boundaries in ReLU networks, which in turn informs the decision to split in Hoeffding trees.
By converting ReLU layers to tropical form and evaluating them using tropical polynomial operations,
we can leverage the Hoeffding bound to guide the splitting process in a way that minimizes the impact of noise
in the data stream.

Author: [Your Name]
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

def t_matmul(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def t_polyval(coeffs, x):
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)

def relu_layer_to_tropical(W, b):
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    return W.copy(), b.copy()

def tropical_network_eval(x, layers):
    h = np.asarray(x, dtype=float).ravel()
    for W, b in layers:
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        z = np.max(W + h[np.newaxis, :], axis=1) + b
        h = np.maximum(z, 0.0)
    return h

def hybrid_split_decision(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05, W: np.ndarray = None, b: np.ndarray = None) -> SplitDecision:
    if W is None or b is None:
        return should_split(best_gain, second_best_gain, r, delta, n, tie_threshold)
    else:
        W_trop, b_trop = relu_layer_to_tropical(W, b)
        x = np.array([1.0])  # dummy input for tropical evaluation
        output = tropical_network_eval(x, [(W_trop, b_trop)])
        output_gain = np.max(output)
        eps = hoeffding_bound(r, delta, n)
        gap = output_gain - second_best_gain
        split = gap > eps or eps < tie_threshold
        reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
        return SplitDecision(split, eps, gap, reason)

def hybrid_tropical_eval(x: np.ndarray, layers: list, r: float, delta: float, n: int) -> np.ndarray:
    h = np.asarray(x, dtype=float).ravel()
    for i, (W, b) in enumerate(layers):
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        W_trop, b_trop = relu_layer_to_tropical(W, b)
        z = np.max(W + h[np.newaxis, :], axis=1) + b
        h = np.maximum(z, 0.0)
        eps = hoeffding_bound(r, delta, n)
        h = np.where(h > eps, h, 0.0)  # apply Hoeffding bound to output
    return h

def find_hybrid_breakpoints(W: np.ndarray, b: np.ndarray, x_range: tuple) -> np.ndarray:
    W_trop, b_trop = relu_layer_to_tropical(W, b)
    x_min, x_max = x_range
    breakpoints = []
    for i in range(W_trop.shape[0]):
        for j in range(i + 1, W_trop.shape[0]):
            if W_trop[i, 0] != W_trop[j, 0]:
                x = (b_trop[j] - b_trop[i]) / (W_trop[i, 0] - W_trop[j, 0])
                if x_min <= x <= x_max:
                    breakpoints.append(x)
    return np.unique(np.array(breakpoints))

if __name__ == "__main__":
    W = np.array([[1.0], [2.0]])
    b = np.array([0.0, 1.0])
    x = np.array([1.0])
    layers = [(W, b)]
    r = 0.5
    delta = 0.1
    n = 100
    tie_threshold = 0.05
    decision = hybrid_split_decision(1.0, 0.5, r, delta, n, tie_threshold, W, b)
    print(decision)
    output = hybrid_tropical_eval(x, layers, r, delta, n)
    print(output)
    x_range = (0.0, 10.0)
    breakpoints = find_hybrid_breakpoints(W, b, x_range)
    print(breakpoints)