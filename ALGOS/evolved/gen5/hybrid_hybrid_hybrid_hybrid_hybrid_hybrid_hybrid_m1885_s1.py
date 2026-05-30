# DARWIN HAMMER — match 1885, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1012_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_tri_algo_cond_m500_s0.py (gen4)
# born: 2026-05-29T23:39:24Z

"""
Hybrid Algorithm: Perceptual Hashing with Sheaf-Associative Memory and Signal-Honesty Regularization

Parents:
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1012_s2.py (Perceptual Hashing with Bayesian Inference)
- hybrid_hybrid_hybrid_hybrid_hybrid_tri_algo_cond_m500_s0.py (Sheaf-Associative Memory with Signal-Honesty Regularization)

Mathematical Bridge:
The bridge is the use of the perceptual hash as a feature vector in the sheaf-associative memory.
Concretely, we use the hamming distance between perceptual hashes as a measure of similarity
in the sheaf consistency term.

The total hybrid energy is

    E_hybrid(x) = (1 – σ)·E_mem(x) + σ·E_sheaf(x),

where σ = signal_score ∈ [0,1] (higher signal → stronger sheaf regularisation,
lower signal → memory‑driven dynamics).
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import List, Iterable, Tuple

def compute_phash(values: List[float]) -> int:
    """
    Compute a 64‑bit perceptual hash from a list of floats.

    The median of the values is used as a threshold; each of the first
    64 values contributes one bit (1 if the value is >= median, else 0).
    If fewer than 64 values are supplied the remaining bits are set to 0.
    """
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for i in range(64):
        v = values[i] if i < len(values) else 0.0
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Return the Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Bayesian marginal probability."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Bayesian posterior update."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def signal_scores(data: bytes, sample: int = 8192) -> float:
    """Estimate Shannon entropy of the first *sample* bytes of *data*."""
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(list(chunk)) / 8.0  # bits → bytes


def shannon_entropy(chunk):
    """Classic Shannon entropy (base‑2) for a list of byte values."""
    entropy = 0.0
    for x in set(chunk):
        p_x = chunk.count(x) / len(chunk)
        entropy -= p_x * math.log(p_x, 2)
    return entropy


def hybrid_energy(x: np.ndarray, W: np.ndarray, b: np.ndarray, signal_score: float, R: np.ndarray) -> float:
    """
    Evaluate the hybrid energy.

    Parameters:
    x (np.ndarray): Sheaf-section vector
    W (np.ndarray): Associative memory weight matrix
    b (np.ndarray): Associative memory bias vector
    signal_score (float): Signal score ∈ [0,1]
    R (np.ndarray): Sheaf structure matrix

    Returns:
    float: Hybrid energy
    """
    E_mem = -0.5 * np.dot(x.T, np.dot(W, x)) + np.dot(b.T, x)
    E_sheaf = np.sum(np.linalg.norm(np.dot(R, x), axis=1) ** 2)
    return (1 - signal_score) * E_mem + signal_score * E_sheaf


def hybrid_update_rule(x: np.ndarray, W: np.ndarray, b: np.ndarray, signal_score: float, R: np.ndarray) -> np.ndarray:
    """
    Update the sheaf-section vector using gradient descent.

    Parameters:
    x (np.ndarray): Sheaf-section vector
    W (np.ndarray): Associative memory weight matrix
    b (np.ndarray): Associative memory bias vector
    signal_score (float): Signal score ∈ [0,1]
    R (np.ndarray): Sheaf structure matrix

    Returns:
    np.ndarray: Updated sheaf-section vector
    """
    grad_E_mem = -np.dot(W, x) + b
    grad_E_sheaf = 2 * np.dot(R.T, np.dot(R, x))
    grad_E_hybrid = (1 - signal_score) * grad_E_mem + signal_score * grad_E_sheaf
    return x - 0.1 * grad_E_hybrid


def perceptual_sheaf_associative_memory(phash_values: List[List[float]], signal_scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the hybrid algorithm.

    Parameters:
    phash_values (List[List[float]]): List of perceptual hash values
    signal_scores (List[float]): List of signal scores

    Returns:
    Tuple[np.ndarray, np.ndarray]: Sheaf-section vector and associative memory weight matrix
    """
    phashes = [compute_phash(values) for values in phash_values]
    hamming_distances = np.array([[hamming_distance(phash1, phash2) for phash2 in phashes] for phash1 in phashes])
    W = -hamming_distances
    b = np.zeros(len(phashes))
    R = np.eye(len(phashes))
    x = np.random.rand(len(phashes))
    for i in range(100):
        signal_score = signal_scores[i % len(signal_scores)]
        x = hybrid_update_rule(x, W, b, signal_score, R)
    return x, W


if __name__ == "__main__":
    phash_values = [[random.random() for _ in range(64)] for _ in range(10)]
    signal_scores = [signal_scores(b"random data") for _ in range(10)]
    x, W = perceptual_sheaf_associative_memory(phash_values, signal_scores)
    print(x)
    print(W)