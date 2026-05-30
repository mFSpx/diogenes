# DARWIN HAMMER — match 2535, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1160_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hard_t_m845_s2.py (gen4)
# born: 2026-05-29T23:42:49Z

import math
import random
import sys
from pathlib import Path
from collections import Counter, deque
import numpy as np

def compute_shannon_entropy(evidence: list[str]) -> float:
    if not evidence:
        return 0.0
    counter = Counter(evidence)
    total = len(evidence)
    entropy = 0.0
    for cnt in counter.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy

def compute_edge_priors(edges: list[tuple[int, int]], evidence: list[str]) -> dict[tuple[int, int], float]:
    H = compute_shannon_entropy(evidence)
    if not edges:
        return {}
    weight = math.exp(-H)
    total_weight = weight * len(edges)
    prior = weight / total_weight
    return {e: prior for e in edges}

def compatibility_score(v: np.ndarray, m: np.ndarray) -> float:
    P = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    v2 = v[:2]
    m2 = m[:2]
    return float(v2.T @ P @ m2)

def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    target: float,
    base_mu: float,
    eps: float,
    compat: float,
    entropy: float,
) -> np.ndarray:
    mu_prime = base_mu * math.exp(-entropy)
    y = float(w @ x)
    error = target - y
    power = float(x @ x) + eps
    increment = mu_prime * compat * error * x / power
    return w + increment

def hybrid_step(
    evidence: list[str],
    edges: list[tuple[int, int]],
    x: np.ndarray,
    target: float,
    w: np.ndarray,
    base_mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    H = compute_shannon_entropy(evidence)
    priors = compute_edge_priors(edges, evidence)
    m = np.random.rand(*w.shape)
    s = compatibility_score(w, m)
    w_new = nlms_update(w, x, target, base_mu, eps, s, H)
    return w_new, priors

def improved_hybrid_step(
    evidence: list[str],
    edges: list[tuple[int, int]],
    x: np.ndarray,
    target: float,
    w: np.ndarray,
    base_mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    H = compute_shannon_entropy(evidence)
    priors = compute_edge_priors(edges, evidence)
    m = np.random.rand(*w.shape)
    s = compatibility_score(w, m)
    mu_prime = base_mu * math.exp(-H)
    y = float(w @ x)
    error = target - y
    power = float(x @ x) + eps
    increment = mu_prime * s * error * x / power
    w_new = w + increment
    return w_new, priors

def improved_hybrid_step_with_regularization(
    evidence: list[str],
    edges: list[tuple[int, int]],
    x: np.ndarray,
    target: float,
    w: np.ndarray,
    base_mu: float = 0.5,
    eps: float = 1e-9,
    reg: float = 0.01,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    H = compute_shannon_entropy(evidence)
    priors = compute_edge_priors(edges, evidence)
    m = np.random.rand(*w.shape)
    s = compatibility_score(w, m)
    mu_prime = base_mu * math.exp(-H)
    y = float(w @ x)
    error = target - y
    power = float(x @ x) + eps
    increment = mu_prime * s * error * x / power - reg * w
    w_new = w + increment
    return w_new, priors

if __name__ == "__main__":
    evidence = ["alpha", "beta", "alpha", "gamma", "beta", "beta", "delta"]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    rng = np.random.default_rng(42)
    x = rng.random(10)
    target = 0.75
    w = rng.random(10)
    w_updated, edge_priors = improved_hybrid_step_with_regularization(evidence, edges, x, target, w)
    print("Entropy H:", compute_shannon_entropy(evidence))
    print("Edge priors:", edge_priors)
    print("Old weights (first 5):", w[:5])
    print("Updated weights (first 5):", w_updated[:5])
    print("Weight change norm:", np.linalg.norm(w_updated - w))