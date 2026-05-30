# DARWIN HAMMER — match 3376, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hybrid_decisi_m1704_s1.py (gen4)
# parent_b: hybrid_hybrid_decreasing_pr_hybrid_hybrid_worksh_m2293_s0.py (gen4)
# born: 2026-05-29T23:49:35Z

"""Hybrid Fusion Module
Combines:
- Parent A (hybrid_hybrid_perceptual_de_hybrid_hybrid_decisi_m1704_s1.py): radial‑basis function surrogate model with perceptual hashing for representative point selection.
- Parent B (hybrid_hybrid_decreasing_pr_hybrid_hybrid_worksh_m2293_s0.py): Schoolfield temperature‑dependent developmental rate ρ(T) and weekday‑dependent weight vector w_i(d).

Mathematical Bridge
The bridge is built by letting the temperature‑dependent rate ρ(T) scale the Gaussian kernel width ε of the RBF model, while the weekday‑dependent weight vector w_i(d) supplies per‑cluster importance factors derived from perceptual‑hash clusters. The final prediction is a weighted RBF sum:
    ŷ(q) = Σ_i w_{c(i)}·G(‖q‑x_i‖, ε·ρ(T))·y_i / Σ_i w_{c(i)}·G(‖q‑x_i‖, ε·ρ(T))
where c(i) is the hash‑cluster of point i.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Dict, Tuple
import numpy as np
import datetime as dt

Vector = Sequence[float]

# ---------- Parent A building blocks ----------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Perceptual hash: 1‑bit per value compared to the mean."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:          # limit to 64 bits
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return (a ^ b).bit_count()

def cluster_by_phash(hashes: Dict[int, int], max_distance: int = 4) -> List[List[int]]:
    """Group indices whose hashes are within max_distance Hamming distance."""
    clusters: List[List[int]] = []
    for idx, h in hashes.items():
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance:
                c.append(idx)
                break
        else:
            clusters.append([idx])
    return clusters

# ---------- Parent B building blocks ----------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol‑1 K‑1

def developmental_rate(params: SchoolfieldParams, T: float) -> float:
    """Schoolfield temperature‑dependent developmental rate ρ(T)."""
    T_low, T_high = params.t_low, params.t_high
    if T < T_low:
        delta_h, R = params.delta_h_low, params.r_cal
    else:
        delta_h, R = params.delta_h_high, params.r_cal
    rho = params.rho_25 * np.exp((delta_h / R) * (1.0 / T_low - 1.0 / T))
    return float(np.clip(rho, 0.0, 1.0))

def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Produce a normalized weight vector for a set of groups (clusters)
    based on the day‑of‑week (dow ∈ {0,…,6}, Monday=0).
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    # Simple sinusoidal modulation that stays positive
    base = np.arange(n)
    raw = np.cos(2 * math.pi * (base + dow) / 7.0) + 1.0   # range [0,2]
    return raw / raw.sum()

# ---------- Fusion core ----------
class FusionRBFModel:
    """
    Hybrid model that:
    1. Clusters data points via perceptual hashing.
    2. Assigns weekday‑dependent cluster weights.
    3. Scales the Gaussian kernel width by the temperature‑dependent rate ρ(T).
    """
    def __init__(
        self,
        data_features: List[Vector],
        data_values: List[float],
        epsilon: float = 1.0,
        school_params: SchoolfieldParams | None = None,
    ):
        if len(data_features) != len(data_values):
            raise ValueError("features and values must have same length")
        self.X = np.array(data_features, dtype=float)
        self.y = np.array(data_values, dtype=float)
        self.epsilon = epsilon
        self.params = school_params or SchoolfieldParams()
        # --- hashing & clustering ---
        self.hashes = {i: compute_phash(list(self.X[i])) for i in range(len(self.X))}
        self.clusters = cluster_by_phash(self.hashes)               # List[List[int]]
        # map point index → cluster id
        self.point2cluster = {}
        for cid, members in enumerate(self.clusters):
            for idx in members:
                self.point2cluster[idx] = cid
        # store cluster identifiers for weight vector creation
        self.cluster_ids = [f"c{cid}" for cid in range(len(self.clusters))]

    def _kernel_matrix(self, query: Vector, T: float) -> np.ndarray:
        """Compute Gaussian kernel values between query and all data points."""
        rho = developmental_rate(self.params, T)               # ρ(T) ∈ [0,1]
        scaled_eps = self.epsilon * (0.5 + 0.5 * rho)           # keep ε > 0
        diffs = np.linalg.norm(self.X - query, axis=1)
        vec_gauss = np.vectorize(lambda r: gaussian(r, scaled_eps))
        return vec_gauss(diffs)

    def predict(self, query: Vector, T: float, date: dt.date) -> float:
        """
        Return the hybrid RBF prediction for a query vector.
        Temperature T (K) modulates kernel width,
        while the weekday of `date` modulates cluster weights.
        """
        if len(query) != self.X.shape[1]:
            raise ValueError("query dimensionality mismatch")
        # 1️⃣ kernel values
        k = self._kernel_matrix(np.array(query, dtype=float), T)   # shape (n,)
        # 2️⃣ weekday‑dependent cluster weights
        dow = date.weekday()                                        # Monday=0
        w_cluster = weekday_weight_vector(self.cluster_ids, dow)   # shape (n_clusters,)
        # map each point to its cluster weight
        w_point = np.array([w_cluster[self.point2cluster[i]] for i in range(len(self.X))])
        # 3️⃣ weighted RBF aggregation
        numerator = np.sum(w_point * k * self.y)
        denominator = np.sum(w_point * k) + 1e-12                    # avoid division by zero
        return float(numerator / denominator)

# ---------- Helper functions demonstrating hybrid operation ----------
def compute_phash_and_clusters(features: List[Vector]) -> Tuple[Dict[int, int], List[List[int]]]:
    """Hash each feature vector and return both the hash dict and resulting clusters."""
    hashes = {i: compute_phash(list(vec)) for i, vec in enumerate(features)}
    clusters = cluster_by_phash(hashes)
    return hashes, clusters

def weekday_scaled_developmental_rate(T: float, date: dt.date, params: SchoolfieldParams) -> float:
    """
    Combine temperature rate ρ(T) with a weekday factor.
    The weekday factor is the average of the weekday weight vector.
    """
    rho = developmental_rate(params, T)
    dow = date.weekday()
    # use a single‑group weight vector to obtain a scalar factor
    w = weekday_weight_vector(["dummy"], dow)[0]   # always 1.0 for single group
    return rho * w

def fused_rbf_predict(
    query: Vector,
    data_features: List[Vector],
    data_values: List[float],
    T: float,
    date: dt.date,
    epsilon: float = 1.0,
) -> float:
    """
    One‑shot convenience wrapper that builds a FusionRBFModel
    and returns the prediction for the supplied query.
    """
    model = FusionRBFModel(data_features, data_values, epsilon=epsilon)
    return model.predict(query, T, date)

# ---------- Smoke test ----------
if __name__ == "__main__":
    # synthetic data
    dim = 5
    n_samples = 12
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, dim)).tolist()
    y = rng.random(n_samples).tolist()

    # random query
    q = rng.random(dim).tolist()

    # environmental conditions
    temperature_K = 295.0                     # ~22 °C
    today = dt.date.today()

    # run fusion prediction
    pred = fused_rbf_predict(q, X, y, temperature_K, today, epsilon=0.8)
    print(f"Hybrid RBF prediction: {pred:.6f}")