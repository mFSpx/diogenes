# DARWIN HAMMER — match 5107, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_hybrid_m1866_s2.py (gen6)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s0.py (gen1)
# born: 2026-05-29T23:59:52Z

import numpy as np
import math
import random
import sys
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Any

# Sketch & RLCT utilities
def count_min_sketch(items: List[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table

def flatten_sketch(sketch: List[List[int]]) -> np.ndarray:
    return np.array([x for sublist in sketch for x in sublist])

def estimate_rlct(sketch: np.ndarray) -> float:
    return np.log(np.sum(sketch)) + 1e-6  # prevent division by zero

# Tropical max-plus algebra utilities
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

# Hoeffding tree utilities
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def should_split(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> bool:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    return gap > eps or eps < tie_threshold

# Hybrid operation
def hybrid_operation(sketch: List[List[int]], 
                     layers: List[Tuple[np.ndarray, np.ndarray]], 
                     best_gain: float, 
                     second_best_gain: float, 
                     r: float, 
                     delta: float, 
                     n: int) -> Tuple[np.ndarray, bool]:
    v = flatten_sketch(sketch)
    rlct = estimate_rlct(v)
    scaled_hoeffding_bound = hoeffding_bound(r, delta, n) * np.exp(rlct)  # improved scaling
    should_split_decision = should_split(best_gain, second_best_gain, r, delta, n)
    tropical_output = tropical_network_eval(v, layers)
    return tropical_output, should_split_decision

def tropical_network_eval(x, layers):
    h = np.asarray(x, dtype=float).ravel()
    for W, b in layers:
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        h = t_add(t_matmul(W, h.reshape(-1, 1)).ravel(), b)
    return h

# Public functions
def extract_features(items: List[Any]) -> np.ndarray:
    sketch = count_min_sketch(items)
    return flatten_sketch(sketch)

def router_update(sketch: List[List[int]], 
                  layers: List[Tuple[np.ndarray, np.ndarray]], 
                  best_gain: float, 
                  second_best_gain: float, 
                  r: float, 
                  delta: float, 
                  n: int) -> Tuple[np.ndarray, bool]:
    return hybrid_operation(sketch, layers, best_gain, second_best_gain, r, delta, n)

def leader_election(sketch: List[List[int]], 
                    layers: List[Tuple[np.ndarray, np.ndarray]], 
                    best_gain: float, 
                    second_best_gain: float, 
                    r: float, 
                    delta: float, 
                    n: int) -> bool:
    _, should_split_decision = hybrid_operation(sketch, layers, best_gain, second_best_gain, r, delta, n)
    return should_split_decision

if __name__ == "__main__":
    items = ["apple", "banana", "orange"]
    sketch = count_min_sketch(items)
    layers = [(np.array([[1, 2], [3, 4]]), np.array([5, 6]))]
    best_gain = 10.0
    second_best_gain = 5.0
    r = 1.0
    delta = 0.1
    n = 100
    tropical_output, should_split_decision = router_update(sketch, layers, best_gain, second_best_gain, r, delta, n)
    print(tropical_output)
    print(should_split_decision)