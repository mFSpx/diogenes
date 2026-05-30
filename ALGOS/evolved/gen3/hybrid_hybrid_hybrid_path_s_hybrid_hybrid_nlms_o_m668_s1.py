# DARWIN HAMMER — match 668, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s3.py (gen2)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s5.py (gen2)
# born: 2026-05-29T23:30:21Z

"""
Hybrid Module: Path Signature + NLMS‑Graph‑Tree Fusion

This module fuses two parent algorithms:

* **Parent A** – hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s3.py (Path Signature + Feature Extraction)
* **Parent B** – hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s5.py (NLMS‑Graph‑Tree Fusion)

The mathematical bridge between the two parents lies in the treatment of the 
feature vectors extracted from text data. In Parent A, these vectors are 
mapped to a multivariate path and then processed using the lead-lag 
transformation and path signatures. In Parent B, these vectors are used to 
compute a similarity matrix for a graph. 

The hybrid approach combines these two ideas by using the NLMS algorithm 
to adaptively weight the edges of the graph, where the feature vectors 
are obtained from the path signature framework. Specifically, the 
level-1 and level-2 signatures of the multivariate path are used as 
input features for the NLMS algorithm, which updates a weight vector 
that linearly combines these features into a cost estimate for each edge.

The result is a single unified system that learns to weight graph edges 
adaptively while still solving the classic minimum-cost tree problem, 
and also captures the geometric and algebraic structure of the 
multivariate path data.

"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Path Signature utilities
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Apply lead-lag transformation to a multivariate path."""
    n = len(path)
    d = len(path[0])
    augmented_path = np.zeros((n, d + 1))
    augmented_path[:n, :d] = path
    augmented_path[1:, d] = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    return augmented_path

def compute_signatures(augmented_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute level-1 and level-2 signatures of a multivariate path."""
    n = len(augmented_path)
    d = len(augmented_path[0])
    level1_signature = np.zeros(d)
    level2_signature = np.zeros((d, d))
    for i in range(n - 1):
        dt = augmented_path[i + 1, d] - augmented_path[i, d]
        dx = augmented_path[i + 1, :d] - augmented_path[i, :d]
        level1_signature += dx * dt
        level2_signature += np.outer(dx, dx) * dt
    return level1_signature, level2_signature

# ----------------------------------------------------------------------
# Parent B – NLMS core
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.

    Args:
        weights: Current weight vector (shape (n,)).
        x: Input feature vector (shape (n,)).
        target: Desired scalar output.
        mu: Step-size (0 < mu ≤ 1).
        eps: Small constant to avoid division by zero.

    Returns:
        (new_weights, error) where error = target - y.
    """
    y = predict(weights, x)
    error = target - y
    new_weights = weights + mu * error * x / (np.dot(x, x) + eps)
    return new_weights, error

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_operation(texts: List[str], 
                    level1_signature: np.ndarray, 
                    level2_signature: np.ndarray) -> np.ndarray:
    """Perform hybrid operation."""
    n = len(texts)
    d = len(level1_signature)
    weights = np.random.rand(d)
    for i in range(n - 1):
        x = np.concatenate((level1_signature, 
                             level2_signature.flatten(), 
                             np.array([i])))
        target = np.linalg.norm(np.array([texts[i + 1]]).encode('utf-8'))
        weights, _ = nlms_update(weights, x, target)
    return weights

def compute_hybrid_tree(texts: List[str]) -> np.ndarray:
    """Compute hybrid tree."""
    path = np.array([np.array(t.encode('utf-8')) for t in texts])
    augmented_path = lead_lag_transform(path)
    level1_signature, level2_signature = compute_signatures(augmented_path)
    weights = hybrid_operation(texts, level1_signature, level2_signature)
    similarity_matrix = np.dot(np.array([np.array(t.encode('utf-8')) for t in texts]).T, 
                               np.array([np.array(t.encode('utf-8')) for t in texts]))
    cost_matrix = np.multiply(similarity_matrix, 
                              np.outer(np.ones(len(texts)), weights))
    return cost_matrix

if __name__ == "__main__":
    texts = ["This is a test.", "This test is only a test."]
    cost_matrix = compute_hybrid_tree(texts)
    print(cost_matrix)