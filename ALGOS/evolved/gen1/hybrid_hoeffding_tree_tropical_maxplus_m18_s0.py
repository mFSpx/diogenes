# DARWIN HAMMER — match 18, survivor 0
# gen: 1
# parent_a: hoeffding_tree.py (gen0)
# parent_b: tropical_maxplus.py (gen0)
# born: 2026-05-29T23:22:44Z

"""This module integrates the Hoeffding tree algorithm from hoeffding_tree.py and the Tropical max-plus algebra from tropical_maxplus.py.
The mathematical bridge between the two is the use of the Hoeffding bound to determine the splitting of nodes in the decision tree, 
while utilizing the Tropical max-plus algebra to evaluate the piecewise-linear convex functions that represent the decision boundaries of the tree.
This allows for the creation of a hybrid algorithm that combines the strengths of both approaches, providing a more robust and efficient decision tree learning algorithm."""

import numpy as np
import math
from dataclasses import dataclass
import random
import sys
import pathlib

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

def find_linear_regions(W, b, x_range):
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    x_min, x_max = x_range
    breakpoints = []
    for i in range(len(W)):
        for j in range(i+1, len(W)):
            if W[i, 0] != W[j, 0]:
                x = (b[j] - b[i]) / (W[i, 0] - W[j, 0])
                if x_min <= x <= x_max:
                    breakpoints.append(x)
    breakpoints = np.unique(breakpoints)
    return np.sort(breakpoints)

def hybrid_decision_tree(x, layers, r, delta, n):
    h = tropical_network_eval(x, layers)
    best_gain = np.max(h)
    second_best_gain = np.sort(h)[-2]
    decision = should_split(best_gain, second_best_gain, r, delta, n)
    if decision.should_split:
        return find_linear_regions(layers[-1][0], layers[-1][1], (np.min(x), np.max(x)))
    else:
        return None

def hybrid_decision_tree_eval(x, layers, r, delta, n):
    h = tropical_network_eval(x, layers)
    best_gain = np.max(h)
    second_best_gain = np.sort(h)[-2]
    decision = should_split(best_gain, second_best_gain, r, delta, n)
    if decision.should_split:
        return h
    else:
        return None

def main():
    x = np.array([1, 2, 3, 4, 5])
    layers = [(np.array([[1, 2], [3, 4]]), np.array([5, 6])), (np.array([[7, 8], [9, 10]]), np.array([11, 12]))]
    r = 0.5
    delta = 0.1
    n = 10
    print(hybrid_decision_tree(x, layers, r, delta, n))
    print(hybrid_decision_tree_eval(x, layers, r, delta, n))

if __name__ == "__main__":
    main()