# DARWIN HAMMER — match 2843, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hdc_se_m843_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fracti_omni_chaotic_sprint_m2520_s3.py (gen3)
# born: 2026-05-29T23:46:16Z

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Basic Types
# ----------------------------------------------------------------------
Node = Hashable
FeatureVec = Sequence[float]
Vector = List[int]  # bipolar hypervector (‑1 / +1)

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def compute_phash(values: List[float]) -> int:
    """Simple permutation hash."""
    return int(hashlib.sha256(str(values).encode()).hexdigest(), 16)

# ----------------------------------------------------------------------
# Core primitives
# ----------------------------------------------------------------------
def random_hv(d: int = 10000, kind: str = "complex", seed: int = None) -> np.ndarray:
    """Generate a random hyper‑vector of dimension *d*."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    elif kind == "bipolar":
        return np.where(rng.uniform(0.0, 1.0, size=d) < 0.5, -1, 1)
    elif kind == "real":
        return rng.uniform(0.0, 1.0, size=d)

def hoeffding_bound(n: int, confidence: float = 0.95, epsilon: float = 0.1) -> float:
    """Hoeffding bound for a given confidence and epsilon."""
    return math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * n))

def gini_coefficient(frequencies: List[float]) -> float:
    """Gini coefficient of a frequency distribution."""
    n = len(frequencies)
    mean = sum(frequencies) / n
    variance = sum((x - mean) ** 2 for x in frequencies) / n
    return variance / (mean ** 2)

def fractional_exponent(kernel: float, alpha: float) -> float:
    """Apply a fractional exponent to the RBF kernel."""
    if kernel == 0:
        return 0
    return kernel ** alpha

# ----------------------------------------------------------------------
# Sparse-WTA Expansion
# ----------------------------------------------------------------------
def sparse_wta_expansion(feature_vec: FeatureVec, dim: int = 10000, k: int = 100) -> Vector:
    """Sparse-WTA expansion of a feature vector."""
    indices = np.argsort(feature_vec)[-k:]
    hv = np.zeros(dim)
    hv[indices] = np.sign(feature_vec[indices])
    return hv.tolist()

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def hybrid_rbf_hd_similarity(a: FeatureVec, b: FeatureVec, alpha: float = 1.0, dim: int = 10000, k: int = 100) -> float:
    """Compute the hybrid RBF-HD similarity between two feature vectors."""
    hv_a = sparse_wta_expansion(a, dim, k)
    hv_b = sparse_wta_expansion(b, dim, k)
    eucl_dist = np.linalg.norm(np.array(hv_a) - np.array(hv_b))
    rbf_kernel = gaussian(eucl_dist)
    fractional_kernel = fractional_exponent(rbf_kernel, alpha)
    return fractional_kernel

def hybrid_hv_similarity(hv1: Vector, hv2: Vector, alpha: float = 1.0) -> float:
    """Compute the hybrid similarity between two hypervectors."""
    cosine_sim = np.dot(hv1, hv2) / (np.linalg.norm(hv1) * np.linalg.norm(hv2))
    eucl_dist = np.linalg.norm(np.array(hv1) - np.array(hv2))
    rbf_kernel = gaussian(eucl_dist)
    fractional_kernel = fractional_exponent(rbf_kernel, alpha)
    return (cosine_sim + fractional_kernel) / 2.0

def hybrid_gini_hoeffding(frequencies: List[float], confidence: float = 0.95, epsilon: float = 0.1) -> Tuple[float, float]:
    """Compute the Gini coefficient and Hoeffding bound of a frequency distribution."""
    gini = gini_coefficient(frequencies)
    hoeffding = hoeffding_bound(len(frequencies), confidence, epsilon)
    return gini, hoeffding

if __name__ == "__main__":
    # Smoke test
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    print(hybrid_rbf_hd_similarity(a, b))

    hv1 = random_hv(10, "bipolar").tolist()
    hv2 = random_hv(10, "bipolar").tolist()
    print(hybrid_hv_similarity(hv1, hv2))

    frequencies = [0.1, 0.2, 0.3, 0.4]
    gini, hoeffding = hybrid_gini_hoeffding(frequencies)
    print(gini, hoeffding)