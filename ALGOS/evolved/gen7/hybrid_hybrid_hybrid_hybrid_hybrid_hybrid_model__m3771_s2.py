# DARWIN HAMMER — match 3771, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m1236_s0.py (gen6)
# parent_b: hybrid_hybrid_model_pool_hy_hybrid_hybrid_minimu_m1971_s3.py (gen4)
# born: 2026-05-29T23:51:39Z

import numpy as np
import math
import random
from collections import defaultdict
from typing import Tuple, Iterable, List, Dict


def _normalize_prob(p: float) -> float:
    """Clamp a probability to the [0, 1] interval."""
    return max(0.0, min(1.0, p))


class HybridSystem:
    """
    A deeper integration of Voronoi partitioning, B‑spline basis construction,
    hyper‑dimensional similarity, and Bayesian decision making.
    """

    def __init__(self, hd_dim: int = 10_000, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.hd_dim = hd_dim
        self.hd_vectors: Dict[str, np.ndarray] = {}
        self.pheromones: Dict[Tuple[int, int], float] = defaultdict(float)

    # --------------------------------------------------------------------- #
    # 1. Lead‑lag transform (path augmentation)
    # --------------------------------------------------------------------- #
    def lead_lag_transform(self, path: Iterable[Iterable[float]]) -> np.ndarray:
        """
        Convert a sequence of d‑dimensional points into a lead‑lag representation.
        Handles 1‑D input gracefully.
        """
        path_arr = np.atleast_2d(np.asarray(list(path), dtype=float))
        if path_arr.ndim != 2:
            raise ValueError("path must be a 2‑D array of shape (T, d)")
        T, d = path_arr.shape
        out = np.empty((2 * T - 1, 2 * d), dtype=float)

        out[0::2] = np.repeat(path_arr, repeats=2, axis=1)[0:T]
        out[1::2] = np.concatenate([path_arr[1:], path_arr[:-1]], axis=1)[: T - 1]

        out[-1] = np.concatenate([path_arr[-1], path_arr[-1]])
        return out

    # --------------------------------------------------------------------- #
    # 2. B‑spline basis using Cox‑de Boor recursion (any order k >= 1)
    # --------------------------------------------------------------------- #
    def bspline_basis(self, x: np.ndarray, knots: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Compute the B‑spline basis matrix B where B[i, j] = N_{j,k}(x_i).
        Parameters
        ----------
        x : (N,) array_like
            Evaluation points.
        knots : (m,) array_like
            Non‑decreasing knot vector.
        k : int
            Spline order (degree = k‑1). Must be >= 1.
        Returns
        -------
        B : (N, m‑k‑1) ndarray
            Basis matrix.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        knots = np.asarray(knots, dtype=float)

        if k < 1:
            raise ValueError("Spline order k must be >= 1")
        if len(knots) < k + 2:
            raise ValueError("Knot vector too short for the requested order")

        m = len(knots) - 1  # last index
        n_basis = m - k  # number of basis functions

        # Initialize zero‑order (piecewise constant) basis
        N = np.zeros((len(x), n_basis + k))
        for i in range(n_basis + k):
            left = knots[i]
            right = knots[i + 1]
            mask = (x >= left) & (x < right) if right != left else np.zeros_like(x, dtype=bool)
            N[:, i] = mask.astype(float)

        # Recursion for higher orders
        for d in range(1, k):
            for i in range(n_basis + k - d):
                left_den = knots[i + d] - knots[i]
                right_den = knots[i + d + 1] - knots[i + 1]

                left_num = x - knots[i]
                right_num = knots[i + d + 1] - x

                left_term = np.where(left_den > 0,
                                     left_num / left_den * N[:, i],
                                     0.0)
                right_term = np.where(right_den > 0,
                                      right_num / right_den * N[:, i + 1],
                                      0.0)
                N[:, i] = left_term + right_term

        return N[:, :n_basis]

    # --------------------------------------------------------------------- #
    # 3. Hyper‑dimensional vector utilities
    # --------------------------------------------------------------------- #
    def _hd_vector(self, key: str) -> np.ndarray:
        """Retrieve or lazily create a random binary hyper‑dimensional vector."""
        if key not in self.hd_vectors:
            self.hd_vectors[key] = self.rng.integers(0, 2, size=self.hd_dim, dtype=np.int8)
        return self.hd_vectors[key]

    @staticmethod
    def _hd_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for binary (+1/‑1) hyper‑dimensional vectors."""
        a = a * 2 - 1
        b = b * 2 - 1
        dot = np.dot(a, b)
        norm = math.sqrt(np.dot(a, a) * np.dot(b, b))
        return dot / norm if norm != 0 else 0.0

    def hyperdim_similarity(self, key_a: str, key_b: str) -> float:
        """Public similarity between two named hyper‑dimensional symbols."""
        vec_a = self._hd_vector(key_a)
        vec_b = self._hd_vector(key_b)
        return self._hd_cosine_similarity(vec_a, vec_b)

    # --------------------------------------------------------------------- #
    # 4. Voronoi partitioning on a discrete grid (naïve O(N·M) implementation)
    # --------------------------------------------------------------------- #
    @staticmethod
    def voronoi_partition(points: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """
        Assign each grid point to the nearest seed point (Euclidean distance).
        Parameters
        ----------
        points : (P, d) array_like
            Seed points.
        grid : (G, d) array_like
            Query points (e.g., a lattice of feature vectors).
        Returns
        -------
        labels : (G,) ndarray of ints
            Index of the nearest seed for each grid point.
        """
        points = np.asarray(points, dtype=float)
        grid = np.asarray(grid, dtype=float)

        if points.ndim != 2 or grid.ndim != 2:
            raise ValueError("Both points and grid must be 2‑D arrays")

        diff = grid[:, None, :] - points[None, :, :]  # (G, P, d)
        dists = np.linalg.norm(diff, axis=2)          # (G, P)
        labels = np.argmin(dists, axis=1)
        return labels

    # --------------------------------------------------------------------- #
    # 5. Information‑theoretic utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def shannon_entropy(values: np.ndarray, bins: int = 10) -> float:
        """
        Estimate Shannon entropy by histogram binning.
        """
        hist, _ = np.histogram(values, bins=bins, density=True)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # avoid log(0)
        return -np.sum(prob * np.log2(prob))

    @staticmethod
    def curvature_score(features: np.ndarray) -> float:
        """
        Compute a curvature‑like statistic: normalized standard deviation.
        """
        std = np.std(features)
        mean = np.mean(features)
        return _normalize_prob(std / (abs(mean) + 1e-9))

    # --------------------------------------------------------------------- #
    # 6. Bayesian decision with hyper‑dimensional false‑positive estimate
    # --------------------------------------------------------------------- #
    def bayes_update(self, prior: float, likelihood: float, false_pos: float) -> float:
        """
        Return posterior P(H|E) using Bayes rule with an explicit false‑positive term.
        """
        prior = _normalize_prob(prior)
        likelihood = _normalize_prob(likelihood)
        false_pos = _normalize_prob(false_pos)

        numerator = prior * likelihood
        denominator = numerator + (1 - prior) * false_pos
        return numerator / denominator if denominator != 0 else 0.0

    def load_model_with_curvature_and_entropy(self, model_key: str,
                                               curvature: float,
                                               entropy: float) -> float:
        """
        Combine curvature (as prior) and entropy (as likelihood) with a
        hyper‑dimensional similarity based false‑positive estimate.
        """
        # Map curvature to a prior probability
        prior = curvature

        # Map entropy (higher entropy -> less certain) to a likelihood
        # We invert entropy because high entropy should lower confidence.
        max_entropy = math.log2(10)  # assuming 10 bins as default
        likelihood = 1.0 - _normalize_prob(entropy / max_entropy)

        # False‑positive estimate from similarity between model and a generic token
        fp_similarity = self.hyperdim_similarity(model_key, "generic_token")
        false_pos = _normalize_prob(0.5 * (1 - fp_similarity))  # lower similarity → higher false‑pos

        return self.bayes_update(prior, likelihood, false_pos)

    # --------------------------------------------------------------------- #
    # 7. High‑level summary
    # --------------------------------------------------------------------- #
    def hybrid_summary(self, features: Iterable[float], model_key: str) -> float:
        """
        Compute a single scalar representing the model‑load decision.
        """
        features_arr = np.asarray(list(features), dtype=float)
        if features_arr.size == 0:
            raise ValueError("features must contain at least one element")

        curvature = self.curvature_score(features_arr)
        entropy = self.shannon_entropy(features_arr)

        return self.load_model_with_curvature_and_entropy(model_key, curvature, entropy)


if __name__ == "__main__":
    hybrid = HybridSystem(seed=42)

    # Example usage
    feats = np.random.rand(50)
    model_name = "example_model"

    decision_score = hybrid.hybrid_summary(feats, model_name)
    print(f"Model load decision score: {decision_score:.4f}")

    # Demonstrate Voronoi partitioning
    seeds = np.random.rand(5, 2)
    grid = np.random.rand(200, 2)
    labels = HybridSystem.voronoi_partition(seeds, grid)
    print(f"Voronoi labels for first 10 grid points: {labels[:10]}")