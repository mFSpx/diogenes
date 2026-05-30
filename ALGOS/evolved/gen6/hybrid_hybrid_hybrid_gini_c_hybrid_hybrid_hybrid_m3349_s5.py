# DARWIN HAMMER — match 3349, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_gini_coeffici_hybrid_hybrid_worksh_m2277_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s0.py (gen3)
# born: 2026-05-29T23:49:32Z

import math
import random
import sys
import pathlib
import numpy as np
from datetime import datetime as dt
from typing import Iterable

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = object
Graph = dict
FeatureVec = list

# ----------------------------------------------------------------------
# Parent A – Gini and similarity utilities
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Return the Gini coefficient of a non-negative value collection."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def compute_phash(values: list[float]) -> int:
    """Simple perceptual hash: 1 bit per value (up to 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values:
        bits <<= 1
        bits |= int(v >= avg)
    return bits


def similarity_matrix(groups: list[str], feature_vecs: dict[str, list[float]]) -> np.ndarray:
    """Return a similarity matrix for the groups using perceptual hashes."""
    n = len(groups)
    similarity = np.zeros((n, n))
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            similarity[i, j] = compute_phash(feature_vecs[group1]) == compute_phash(feature_vecs[group2])
    return similarity


# ----------------------------------------------------------------------
# Parent B – weekday weight vector and GPU memory allocation
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: list[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row-stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow / 7.0)
    return (1.0 + np.sin(base_angles + phase)) / n


def gpu_memory_allocation(groups: list[str], memory_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the GPU memory allocation for the groups based on the weights."""
    n = len(groups)
    allocation = np.zeros((n,))
    for i in range(n):
        allocation[i] = np.sum(memory_matrix[i] * weights)
    return allocation


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_weight_vector(groups: list[str], feature_vecs: dict[str, list[float]], dow: int) -> np.ndarray:
    """
    Hybrid weight vector for *groups* based on Gini-coefficient modulated 
    sinusoidal rotation and structural similarity.
    """
    similarity = similarity_matrix(groups, feature_vecs)
    gini = gini_coefficient([np.mean(similarity[i]) for i in range(len(groups))])
    base_angles = np.arange(len(groups)) * (2.0 * math.pi) / len(groups)
    phase = (2.0 * math.pi) * (dow / 7.0)
    modulated_angles = base_angles + gini * np.pi / 2
    weights = (1.0 + np.sin(modulated_angles + phase)) / len(groups)
    return weights / np.sum(weights)


def hybrid_gpu_memory_allocation(groups: list[str], feature_vecs: dict[str, list[float]], memory_matrix: np.ndarray, dow: int) -> np.ndarray:
    """Return the hybrid GPU memory allocation for the groups."""
    weights = hybrid_weight_vector(groups, feature_vecs, dow)
    return gpu_memory_allocation(groups, memory_matrix, weights)


if __name__ == "__main__":
    groups = ["A", "B", "C"]
    feature_vecs = {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0], "C": [0.0, 0.0, 1.0]}
    memory_matrix = np.array([[1000, 500, 500], [500, 1000, 500], [500, 500, 1000]])
    dow = dt.now().weekday()
    weights = hybrid_weight_vector(groups, feature_vecs, dow)
    allocation = hybrid_gpu_memory_allocation(groups, feature_vecs, memory_matrix, dow)
    print("Hybrid Weight Vector:", weights)
    print("Hybrid GPU Memory Allocation:", allocation)