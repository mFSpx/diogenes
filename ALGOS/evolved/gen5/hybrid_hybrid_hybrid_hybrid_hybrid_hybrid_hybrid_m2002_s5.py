# DARWIN HAMMER — match 2002, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_possum_filter_m581_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s4.py (gen3)
# born: 2026-05-29T23:40:27Z

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


Vector = Sequence[float]


def _to_numpy(v: Vector) -> np.ndarray:
    """Convert any sequence of numbers to a 1‑D float ndarray."""
    return np.asarray(v, dtype=float).ravel()


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    a_arr, b_arr = _to_numpy(a), _to_numpy(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError("vectors must have same dimension")
    return float(np.linalg.norm(a_arr - b_arr))


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat, lon) pairs."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def ssim(v1: Vector, v2: Vector) -> float:
    """
    A lightweight structural similarity proxy for 1‑D vectors.
    Uses cosine similarity scaled to [0, 1] (1 = identical, 0 = orthogonal).
    """
    a, b = _to_numpy(v1), _to_numpy(v2)
    if a.shape != b.shape:
        raise ValueError("vectors must have same dimension")
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = dot / (norm_a * norm_b)
    # map from [-1, 1] to [0, 1]
    return (cosine + 1.0) / 2.0


def shannon_entropy(text: str) -> float:
    """Shannon entropy of token frequencies in *text*."""
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = float(sum(counts.values()))
    probs = np.fromiter((c / total for c in counts.values()), dtype=float)
    return float(-np.sum(probs * np.log(probs)))


def _normalized_entropy(text: str) -> float:
    """
    Entropy normalised by the maximum possible entropy for the observed
    vocabulary size. Returns 0 when the vocabulary size is 1 (no uncertainty).
    """
    tokens = re.findall(r"\w+", text.lower())
    vocab_size = len(set(tokens))
    if vocab_size <= 1:
        return 0.0
    max_entropy = math.log(vocab_size)
    return shannon_entropy(text) / max_entropy


def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """
    Solve Ax = b using a robust LU decomposition (NumPy) with a fallback
    to a simple Gaussian elimination for pure‑Python environments.
    """
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    try:
        x = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        # Singular or ill‑conditioned; use least‑squares as a graceful fallback.
        x, *_ = np.linalg.lstsq(A, B, rcond=None)
    return x.tolist()


@dataclass(frozen=True)
class SpatialAwareSurrogate:
    """
    Radial‑basis surrogate enriched with spatial‑aware weighting.
    *centers* – (n_centers, dim) array of RBF centers.
    *weights* – (n_centers,) array of learned coefficients.
    *epsilon* – shape parameter of the Gaussian kernel.
    """
    centers: np.ndarray
    weights: np.ndarray
    epsilon: float = 1.0

    def __post_init__(self) -> None:
        if self.centers.ndim != 2:
            raise ValueError("centers must be a 2‑D array (n_centers, dimension)")
        if self.weights.ndim != 1:
            raise ValueError("weights must be a 1‑D array")
        if self.centers.shape[0] != self.weights.shape[0]:
            raise ValueError("number of centers must match number of weights")

    @staticmethod
    def fit(
        X: Iterable[Vector],
        Y: Iterable[float],
        epsilon: float = 1.0,
        reg: float = 1e-8,
    ) -> "SpatialAwareSurrogate":
        """
        Fit a surrogate to data (X, Y) by solving the regularised linear system
        (ΦᵀΦ + reg·I)w = ΦᵀY, where Φ_{ij}=gaussian(||x_i‑c_j||, ε).
        """
        X_arr = np.asarray(list(X), dtype=float)
        Y_arr = np.asarray(list(Y), dtype=float).ravel()
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2‑D iterable of vectors")
        n_centers = X_arr.shape[0]
        # Use the training points themselves as RBF centers (a common heuristic)
        centers = X_arr.copy()
        # Build the design matrix Φ
        dists = np.linalg.norm(
            X_arr[:, None, :] - centers[None, :, :], axis=2
        )  # shape (n_samples, n_centers)
        Phi = np.exp(-((epsilon * dists) ** 2))
        # Regularised normal equations
        A = Phi.T @ Phi + reg * np.eye(n_centers)
        b = Phi.T @ Y_arr
        weights = np.linalg.solve(A, b)
        return SpatialAwareSurrogate(centers=centers, weights=weights, epsilon=epsilon)

    def evaluate(self, x: Vector) -> float:
        """RBF evaluation at a single point."""
        x_arr = _to_numpy(x)
        if x_arr.shape[0] != self.centers.shape[1]:
            raise ValueError(
                f"input dimension {x_arr.shape[0]} does not match center dimension {self.centers.shape[1]}"
            )
        dists = np.linalg.norm(self.centers - x_arr, axis=1)
        kernels = np.exp(-((self.epsilon * dists) ** 2))
        return float(kernels @ self.weights)


def hybrid_score(v1: Vector, v2: Vector, text: str, lambda_: float = 0.5) -> float:
    """
    Combine structural similarity and textual uncertainty.
    The score lies in [0, 1] for λ∈[0,1].
    """
    ssim_val = ssim(v1, v2)
    norm_entropy = _normalized_entropy(text)
    # Guard against λ outside [0,1] – clamp for stability.
    lam = max(0.0, min(1.0, lambda_))
    return ssim_val * (1.0 - lam * norm_entropy)


def hybrid_surrogate_model(
    x: Vector,
    v1: Vector,
    v2: Vector,
    text: str,
    surrogate: SpatialAwareSurrogate,
    lambda_: float = 0.5,
) -> float:
    """
    Deeply integrated model:
    1. Compute a hybrid SSIM‑entropy score.
    2. Evaluate the spatial‑aware RBF surrogate.
    3. Modulate the surrogate output by the hybrid score.
    """
    score = hybrid_score(v1, v2, text, lambda_)
    rbf_val = surrogate.evaluate(x)
    return rbf_val * score


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy training data for illustration
    X_train = [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ]
    Y_train = [0.8, 0.9, 1.0]

    # Fit the surrogate
    surrogate = SpatialAwareSurrogate.fit(X_train, Y_train, epsilon=0.8)

    # Query point
    x_query = [1.5, 2.5, 3.5]
    v1 = [4.0, 5.0, 6.0]
    v2 = [7.0, 8.0, 9.0]
    text = "This is a test text with some repeated repeated words"

    result = hybrid_surrogate_model(
        x_query, v1, v2, text, surrogate, lambda_=0.3
    )
    print(f"Hybrid surrogate output: {result:.6f}")