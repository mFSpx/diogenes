# DARWIN HAMMER — match 732, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m379_s0.py (gen4)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s3.py (gen3)
# born: 2026-05-29T23:30:44Z

import numpy as np
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, List


# ----------------------------------------------------------------------
# Tropical (max‑plus) algebra utilities
# ----------------------------------------------------------------------
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (ordinary addition)."""
    return np.add(x, y)


def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication:
        (A ⊗ B)[i, j] = max_k (A[i, k] + B[k, j])
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape[1] != B.shape[0]:
        raise ValueError("inner dimensions must agree for tropical matmul")
    # Broadcast to compute all pairwise sums, then max over k
    # shape -> (i, k, j)
    sums = A[:, :, np.newaxis] + B[np.newaxis, :, :]
    return np.max(sums, axis=1)


def t_polyval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Evaluate a tropical (max‑plus) polynomial
        p(x) = max_i (c_i + i * x)
    where i is the exponent (0‑based).
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)

    # exponents as a column vector, broadcast over x's shape
    exponents = np.arange(coeffs.size, dtype=float).reshape((-1,) + (1,) * x.ndim)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents * x
    return np.max(terms, axis=0)


# ----------------------------------------------------------------------
# Radial Basis Function (RBF) surrogate utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance."""
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    return float(np.linalg.norm(a - b))


def rbf_kernel(x: np.ndarray, c: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF kernel evaluated for all centers c."""
    diff = x - c  # shape (n_centers, dim)
    sq_norm = np.sum(diff ** 2, axis=1)
    return np.exp(-sq_norm / (2 * sigma ** 2))


class TropicalRBFModel:
    """
    Surrogate that maps a state vector to a set of tropical polynomial
    coefficients using a Gaussian RBF expansion.
    """

    def __init__(self,
                 centers: np.ndarray,
                 sigma: float = 0.1,
                 regularizer: float = 1e-6):
        """
        Parameters
        ----------
        centers : (n_centers, dim) array
            Training points that act as RBF centres.
        sigma : float
            Width of the Gaussian kernel.
        regularizer : float
            Small value added to the diagonal for numerical stability.
        """
        self.centers = np.asarray(centers, dtype=float)
        self.sigma = float(sigma)
        self.regularizer = float(regularizer)
        self.weights: np.ndarray | None = None  # to be learned

    def fit(self, targets: np.ndarray) -> None:
        """
        Fit the surrogate to map each centre to a target coefficient vector.
        Linear least‑squares solution of Φ w = targets, where Φ is the RBF matrix.
        """
        targets = np.asarray(targets, dtype=float)
        if targets.shape[0] != self.centers.shape[0]:
            raise ValueError("targets must have one row per centre")

        # Build the RBF design matrix Φ (n_centers × n_centers)
        Phi = rbf_kernel(self.centers[:, None, :],
                         self.centers[None, :, :],
                         self.sigma)
        # Regularised normal equations
        A = Phi.T @ Phi + self.regularizer * np.eye(Phi.shape[1])
        b = Phi.T @ targets
        self.weights = np.linalg.solve(A, b)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict tropical coefficients for a new state vector x.
        Returns a 1‑D array of coefficients.
        """
        if self.weights is None:
            raise RuntimeError("model must be fitted before prediction")
        x = np.asarray(x, dtype=float).reshape(1, -1)  # (1, dim)
        phi = rbf_kernel(x, self.centers, self.sigma)  # (1, n_centers)
        return (phi @ self.weights).ravel()


# ----------------------------------------------------------------------
# Similarity computation in the fused space
# ----------------------------------------------------------------------
def tropical_similarity(v1: np.ndarray,
                        v2: np.ndarray,
                        model: TropicalRBFModel,
                        epsilon: float = 1.0) -> float:
    """
    Compute a similarity score between two state vectors.
    1. Use the surrogate to obtain tropical coefficients for each vector.
    2. Evaluate the tropical polynomial at the original vectors.
    3. Compare the two evaluations with a Gaussian kernel on their Euclidean distance.
    """
    coeffs1 = model.predict(v1)
    coeffs2 = model.predict(v2)

    eval1 = t_polyval(coeffs1, v1)
    eval2 = t_polyval(coeffs2, v2)

    dist = euclidean(eval1, eval2)
    return gaussian(dist, epsilon)


def compute_average_tropical_ssim(state_vectors: Iterable[Iterable[float]],
                                  model: TropicalRBFModel,
                                  epsilon: float = 1.0) -> float:
    """
    Pairwise average similarity (SSIM‑like) over a collection of state vectors.
    """
    vectors = [np.asarray(v, dtype=float) for v in state_vectors]
    if len(vectors) < 2:
        raise ValueError("need at least two vectors to compute similarity")

    sims: List[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(tropical_similarity(vectors[i], vectors[j], model, epsilon))
    return float(np.mean(sims))


# ----------------------------------------------------------------------
# Hoeffding‑bound based split decision utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0:
        raise ValueError("r must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt((r ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def should_split(best_gain: float,
                 second_best_gain: float,
                 r: float,
                 delta: float,
                 n: int,
                 tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    if gap > eps:
        reason = "gap_exceeds_bound"
    elif eps < tie_threshold:
        reason = "tight_bound"
    else:
        reason = "insufficient_gap"
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic training data for the surrogate
    train_vectors = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])
    # For illustration we set target tropical coefficients equal to the vectors themselves
    # (any mapping could be used – this keeps the example self‑contained)
    target_coeffs = train_vectors.copy()

    # Build and fit the surrogate
    rbf_model = TropicalRBFModel(centers=train_vectors, sigma=0.5)
    rbf_model.fit(target_coeffs)

    # Compute the fused similarity metric
    avg_sim = compute_average_tropical_ssim(train_vectors, rbf_model, epsilon=1.0)
    print(f"Average tropical‑RBF similarity: {avg_sim:.6f}")

    # Hoeffding‑bound split decision demo
    decision = should_split(best_gain=10.0,
                            second_best_gain=5.0,
                            r=1.0,
                            delta=0.1,
                            n=100)
    print(f"Should split? {decision.should_split} (reason: {decision.reason})")