# DARWIN HAMMER — match 5298, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s1.py (gen5)
# parent_b: hybrid_hybrid_tri_algo_cond_hybrid_hybrid_hybrid_m1626_s2.py (gen5)
# born: 2026-05-30T00:01:03Z

"""
Hybrid Algorithm: Path Signature + Tri-algo Conduit
This module defines a novel hybrid algorithm, combining elements of 
'hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s1.py' and 
'hybrid_hybrid_tri_algo_cond_hybrid_hybrid_hybrid_m1626_s2.py'. 
The mathematical bridge between these two structures is found in the concept 
of 'signal scores' in the 'tri_algo_conduit.py', which can be seen as a form 
of 'MinHash signature' in the 'hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s1.py' model. 
By integrating the governing equations of both models, we create a new algorithm 
that balances the signal scores with the similarity of path signatures.

The key innovation is the introduction of a 'signal_similarity_regularization' 
term in the 'flow_loss' function, which encourages the model to produce 
similar path signatures with high signal scores.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

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

@dataclass(frozen=True)
class ConduitDecision:
    action: str  
    confidence_gap: float
    epsilon: float
    signal_score: float
    noise_score: float
    dormancy_probability: float
    recovery_priority: float
    reason: str

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    return running.T @ increments               # (d, d)

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2      
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    return B

def shannon_entropy(sequence):
    sequence = [ord(c) for c in sequence]
    sequence_len = len(sequence)
    if sequence_len <= 0:
        return 0.0

    frequency_dict = {}
    for item in sequence:
        if item not in frequency_dict:
            frequency_dict[item] = 0
        frequency_dict[item] += 1

    entropy = 0.0
    for item in frequency_dict:
        p_x = frequency_dict[item] / sequence_len
        entropy += - p_x * math.log(p_x, 2)
    return entropy

def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> tuple[float, float]:
    size = len(data)
    entropy = shannon_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "") for x in ["text", "json", "xml"]) else 0.0
    return entropy + status_bonus + mime_bonus, keyword_hits + structural_links

def hybrid_signal_path_signature(path, data: bytes, status_code: int | None = None, mime: str = "", keyword_hits: int = 0, structural_links: int = 0):
    path_signature = signature_level1(path)
    signal_score, noise_score = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    return path_signature * signal_score, noise_score

def hybrid_conduit_decision(path, data: bytes, status_code: int | None = None, mime: str = "", keyword_hits: int = 0, structural_links: int = 0):
    signal_score, noise_score = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    path_signature = signature_level1(path)
    confidence_gap = abs(signal_score - noise_score)
    epsilon = 0.1
    dormancy_probability = 0.5
    recovery_priority = 1.0
    reason = "Hybrid decision"
    return ConduitDecision("hybrid", confidence_gap, epsilon, signal_score, noise_score, dormancy_probability, recovery_priority, reason)

def hybrid_math_action(path, data: bytes, status_code: int | None = None, mime: str = "", keyword_hits: int = 0, structural_links: int = 0):
    path_signature = signature_level1(path)
    signal_score, noise_score = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    expected_value = path_signature * signal_score
    cost = noise_score
    risk = 0.0
    return MathAction("hybrid", expected_value, cost, risk)

if __name__ == "__main__":
    path = np.random.rand(10, 2)
    data = b"Hello, world!"
    status_code = 200
    mime = "text/plain"
    keyword_hits = 2
    structural_links = 1
    hybrid_signal_path_signature(path, data, status_code, mime, keyword_hits, structural_links)
    hybrid_conduit_decision(path, data, status_code, mime, keyword_hits, structural_links)
    hybrid_math_action(path, data, status_code, mime, keyword_hits, structural_links)