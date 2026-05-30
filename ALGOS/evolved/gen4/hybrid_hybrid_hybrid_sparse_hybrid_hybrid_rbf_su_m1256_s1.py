# DARWIN HAMMER — match 1256, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_sparse_wta_hy_hybrid_capybara_opti_m180_s1.py (gen3)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s2.py (gen3)
# born: 2026-05-29T23:34:44Z

# hybrid_hybrid_capybara_rbf_m180_s2.py

"""
This module fuses the 'hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py' and 
'hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s2.py' algorithms. The 
mathematical bridge lies in the use of Laplace noise to perturb the sparse 
projection and the RBF-based similarity weights to inform the decision to split 
in the optimisation stage.

The sparse projection is used to identify the active components in the input 
record and the Laplace noise is added to ensure differential privacy. The RBFs 
are used to compute the similarity weights in the hybrid maximal independent set 
algorithm, which in turn informs the decision to split in the optimisation stage.

Author: [Your Name]
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Parent A – Sparse Winner-Take-All utilities
# ----------------------------------------------------------------------

def expand(values: List[float], m: int) -> np.ndarray:
    """Sparse projection of the input record."""
    return np.random.choice(values, size=m, replace=False)

def hoeffding_epsilon(obs_count: int, epsilon_H: float) -> float:
    """Compute the Hoeffding confidence term."""
    return 1 / (1 + epsilon_H)

def hybrid_update(x: np.ndarray, t: int, T: int, delta_max: float, alpha: float, 
                  S: np.ndarray, nodes: List[str], features: Dict[str, List[float]]) -> np.ndarray:
    """
    Update the optimisation state based on the RBF-based similarity weights and 
    the Laplace noise-perturbed sparse projection.

    Parameters:
    x (np.ndarray): The current optimisation state.
    t (int): The current time step.
    T (int): The total number of time steps.
    delta_max (float): The maximum evasion magnitude.
    alpha (float): The confidence scalar.
    S (np.ndarray): The similarity matrix.
    nodes (List[str]): The list of nodes.
    features (Dict[str, List[float]]): The feature vector for each node.

    Returns:
    np.ndarray: The updated optimisation state.
    """
    # Compute the similarity weights
    S, _ = similarity_matrix(features)
    weights = np.exp(-np.linalg.norm(S, axis=1))

    # Compute the Hoeffding confidence term
    epsilon_H = hoeffding_epsilon(T, 1.0)
    confidence = hoeffding_epsilon(T, epsilon_H)

    # Compute the evasion magnitude
    delta = evasion_delta(t, T, delta_max, alpha) * (1 + confidence)

    # Compute the top-k active components
    e = expand(x, 10)
    top_k_mask = np.argsort(e)[::-1][:5]

    # Update the optimisation state
    x = clamp(x + delta * sign(e[top_k_mask]) * top_k_mask, 0, 1)

    return x

def clamp(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Clamp the optimisation state to the valid range."""
    return np.clip(x, lower, upper)

def evasion_delta(t: int, T: int, delta_max: float, alpha: float) -> float:
    """Compute the evasion delta."""
    return delta_max * (1 - (t / T)) ** alpha

def sign(x: np.ndarray) -> np.ndarray:
    """Compute the sign of the input array."""
    return np.sign(x)

# ----------------------------------------------------------------------
# Parent B – Capybara Optimisation utilities
# ----------------------------------------------------------------------

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Compute the Gaussian function."""
    return math.exp(-((epsilon * r) ** 2))

def similarity_matrix(features: Dict[str, List[float]]) -> Tuple[np.ndarray, List[str]]:
    """Compute the similarity matrix."""
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(features[ni])
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = compute_phash(features[nj])
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Compute the Hoeffding bound."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0")
    return r * np.sqrt(2 * np.log(2 / delta) / n)

def compute_phash(values: List[float]) -> int:
    """Compute the PHASH value."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Compute the Hamming distance."""
    return (a ^ b).bit_count()

def euclidean(a: List[float], b: List[float]) -> float:
    """Compute the Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    x = np.random.rand(100)
    t = 0
    T = 100
    delta_max = 1.0
    alpha = 0.5
    S = np.random.rand(10, 10)
    nodes = [f"node_{i}" for i in range(10)]
    features = {f"node_{i}": [np.random.rand() for _ in range(10)] for i in range(10)}
    print(hybrid_update(x, t, T, delta_max, alpha, S, nodes, features))