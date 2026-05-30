# DARWIN HAMMER — match 4395, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m1236_s1.py (gen6)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_label__m4_s1.py (gen4)
# born: 2026-05-29T23:55:29Z

import numpy as np
import math
from typing import Tuple, List


def _validate_path(path: np.ndarray) -> Tuple[int, int]:
    """Validate that path is a 2‑D array with shape (T, d)."""
    if path.ndim != 2:
        raise ValueError("path must be a 2‑dimensional array (T, d)")
    T, d = path.shape
    if T < 1:
        raise ValueError("path must contain at least one point")
    return T, d


class HybridSystem:
    """Core utilities for the hybrid algorithm."""

    # --------------------------------------------------------------------- #
    #  Lead‑lag transformation (vectorised, safe)
    # --------------------------------------------------------------------- #
    @staticmethod
    def lead_lag_transform(path: np.ndarray) -> np.ndarray:
        """
        Duplicate each point and interleave it with the next point.
        For a path of shape (T, d) the output shape is (2·T‑1, 2·d).
        """
        T, d = _validate_path(path)
        # duplicate each point
        dup = np.repeat(path, 2, axis=0)                     # (2·T, d)
        # shift the duplicated array by one row to create the lag part
        lag = np.vstack([dup[1:], dup[-1:]])                  # (2·T, d)
        # interleave original and lagged vectors
        out = np.empty((2 * T - 1, 2 * d), dtype=path.dtype)
        out[0::2] = np.concatenate([path, path[-1:]], axis=0)  # repeat last for odd length
        out[1::2] = lag[:-1]                                   # drop the final extra row
        return out

    # --------------------------------------------------------------------- #
    #  B‑spline basis (Cox‑de Boor recursion, fully vectorised)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _cox_de_boor(x: np.ndarray, knots: np.ndarray, i: int, k: int) -> np.ndarray:
        """Recursive definition of B‑spline basis function N_{i,k}."""
        if k == 0:
            return np.where((knots[i] <= x) & (x < knots[i + 1]), 1.0, 0.0)
        left = np.zeros_like(x, dtype=float)
        right = np.zeros_like(x, dtype=float)

        denom_left = knots[i + k] - knots[i]
        if denom_left > 0:
            left = ((x - knots[i]) / denom_left) * HybridSystem._cox_de_boor(x, knots, i, k - 1)

        denom_right = knots[i + k + 1] - knots[i + 1]
        if denom_right > 0:
            right = ((knots[i + k + 1] - x) / denom_right) * HybridSystem._cox_de_boor(x, knots, i + 1, k - 1)

        return left + right

    @classmethod
    def bspline_basis(cls, x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
        """
        Evaluate all B‑spline basis functions of a given degree on points x.
        Returns a matrix B of shape (len(x), n_basis) where
        n_basis = len(knots) - degree - 1.
        """
        x = np.asarray(x, dtype=float)
        knots = np.asarray(knots, dtype=float)

        if knots.ndim != 1:
            raise ValueError("knots must be a 1‑D array")
        if degree < 0:
            raise ValueError("degree must be non‑negative")
        n_basis = len(knots) - degree - 1
        if n_basis <= 0:
            raise ValueError("Invalid knot vector for the requested degree")

        B = np.empty((x.size, n_basis), dtype=float)
        for i in range(n_basis):
            B[:, i] = cls._cox_de_boor(x, knots, i, degree)
        return B

    # --------------------------------------------------------------------- #
    #  Gaussian beam and Fisher information (vectorised)
    # --------------------------------------------------------------------- #
    @staticmethod
    def gaussian_beam(theta: np.ndarray, center: float, width: float) -> np.ndarray:
        if width <= 0:
            raise ValueError("width must be positive")
        theta = np.asarray(theta, dtype=float)
        z = (theta - center) / width
        return np.exp(-0.5 * z ** 2)

    @staticmethod
    def fisher_score(theta: np.ndarray, center: float, width: float, eps: float = 1e-12) -> np.ndarray:
        """
        Fisher information for a Gaussian beam.
        Returns a vector of the same shape as theta.
        """
        intensity = np.maximum(HybridSystem.gaussian_beam(theta, center, width), eps)
        derivative = intensity * (-(theta - center) / (width ** 2))
        return (derivative ** 2) / intensity

    # --------------------------------------------------------------------- #
    #  Voronoi partitioning (nearest‑seed assignment)
    # --------------------------------------------------------------------- #
    @staticmethod
    def voronoi_regions(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
        """
        Assign each point to the index of its nearest seed.
        points : (N, d)
        seeds  : (M, d)
        Returns an integer array of length N.
        """
        points = np.asarray(points, dtype=float)
        seeds = np.asarray(seeds, dtype=float)

        if points.ndim != 2 or seeds.ndim != 2:
            raise ValueError("points and seeds must be 2‑D arrays")
        if points.shape[1] != seeds.shape[1]:
            raise ValueError("dimensionality of points and seeds must match")

        # Compute squared Euclidean distance matrix efficiently
        diff = points[:, None, :] - seeds[None, :, :]          # (N, M, d)
        dist2 = np.einsum('nmd,nmd->nm', diff, diff)          # (N, M)
        return np.argmin(dist2, axis=1)

    # --------------------------------------------------------------------- #
    #  Simplified SSIM (grayscale only, window‑less)
    # --------------------------------------------------------------------- #
    @staticmethod
    def ssim(img1: np.ndarray, img2: np.ndarray, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> float:
        """
        Compute a global SSIM index for two 2‑D grayscale images.
        The implementation follows the original paper but without a sliding window.
        """
        if img1.shape != img2.shape:
            raise ValueError("img1 and img2 must have the same shape")
        if img1.ndim != 2:
            raise ValueError("SSIM implementation expects 2‑D grayscale images")

        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        return float(numerator / denominator)

    # --------------------------------------------------------------------- #
    #  Recovery priority (deeper integration of Fisher & SSIM)
    # --------------------------------------------------------------------- #
    @staticmethod
    def recovery_priority(fisher: np.ndarray, ssim_val: float, basis_sum: float) -> float:
        """
        Combine Fisher information, structural similarity and basis mass.
        The formula is deliberately non‑linear to emphasise regions that are
        both informative (high Fisher) and structurally consistent (high SSIM).
        """
        fisher_mean = fisher.mean()
        return float((fisher_mean ** 0.5) * ssim_val * math.log1p(basis_sum))


# ------------------------------------------------------------------------- #
#  Public API – thin wrappers that keep the original signatures
# ------------------------------------------------------------------------- #
def hybrid_operation(path: np.ndarray,
                    theta: np.ndarray,
                    center: float,
                    width: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sys = HybridSystem()
    transformed = sys.lead_lag_transform(path)
    # use a uniform knot vector with clamped ends
    knots = np.linspace(0, 1, len(path) + sys.bspline_basis.__func__.__code__.co_argcount)
    basis = sys.bspline_basis(np.linspace(0, 1, len(path)), knots)
    fisher = sys.fisher_score(theta, center, width)
    return transformed, basis, fisher


def hybrid_fusion(path1: np.ndarray,
                 path2: np.ndarray,
                 theta: np.ndarray,
                 center: float,
                 width: float) -> Tuple[np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
    sys = HybridSystem()
    t1 = sys.lead_lag_transform(path1)
    t2 = sys.lead_lag_transform(path2)

    knots1 = np.linspace(0, 1, len(path1) + 4)   # degree 3 → 4 extra knots
    knots2 = np.linspace(0, 1, len(path2) + 4)

    b1 = sys.bspline_basis(np.linspace(0, 1, len(path1)), knots1)
    b2 = sys.bspline_basis(np.linspace(0, 1, len(path2)), knots2)

    f1 = sys.fisher_score(theta, center, width)
    f2 = sys.fisher_score(theta, center, width)

    return t1, t2, b1, b2, f1, f2


def hybrid_recovery(path: np.ndarray,
                    theta: np.ndarray,
                    center: float,
                    width: float,
                    reference_image: np.ndarray) -> Tuple[np.ndarray,
                                                          np.ndarray,
                                                          np.ndarray,
                                                          float]:
    sys = HybridSystem()
    transformed = sys.lead_lag_transform(path)

    knots = np.linspace(0, 1, len(path) + 4)
    basis = sys.bspline_basis(np.linspace(0, 1, len(path)), knots)

    fisher = sys.fisher_score(theta, center, width)

    # generate a synthetic image from the transformed path for SSIM comparison
    # (simple rasterisation onto a 64×64 grid)
    img = np.zeros((64, 64), dtype=float)
    coords = np.clip((transformed[:, :2] * 63).astype(int), 0, 63)
    img[coords[:, 0], coords[:, 1]] = 1.0

    ssim_val = sys.ssim(img, reference_image)

    priority = sys.recovery_priority(fisher, ssim_val, basis.sum())
    return transformed, basis, fisher, priority


if __name__ == "__main__":
    # Demo with random data
    rng = np.random.default_rng(42)

    path = rng.random((10, 2))
    theta = rng.random(10)
    center = 0.5
    width = 0.1

    # reference image for SSIM – a random binary pattern
    ref_img = rng.integers(0, 2, size=(64, 64)).astype(float)

    t, b, f = hybrid_operation(path, theta, center, width)
    t1, t2, b1, b2, f1, f2 = hybrid_fusion(path, path, theta, center, width)
    t_rec, b_rec, f_rec, priority = hybrid_recovery(path, theta, center, width, ref_img)

    print("Hybrid operation output shapes:", t.shape, b.shape, f.shape)
    print("Hybrid fusion output shapes:", t1.shape, t2.shape, b1.shape, b2.shape, f1.shape, f2.shape)
    print("Recovery priority:", priority)