# DARWIN HAMMER — match 1563, survivor 3
# gen: 6
# parent_a: hybrid_rlct_grokking_hybrid_nlms_omni_cha_m118_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m702_s0.py (gen5)
# born: 2026-05-29T23:37:37Z

"""Hybrid Algorithm: rlct_nlms_omni_chaotic_sprint + rbf_surrogate_morphology
This module fuses the core topologies of two parent algorithms:

1. **Parent A (rlct_nlms_omni_chaotic_sprint)** – provides the Real Log Canonical
   Threshold (RLCT) estimator and a Normalized Least‑Mean‑Squares (NLMS) adaptive
   filter.  The RLCT, derived from the loss trajectory, is used to modulate the NLMS
   step‑size μ, giving a geometry‑aware adaptation rate.

2. **Parent B (rbf_surrogate_morphology)** – defines geometric endpoint descriptions,
   a Gaussian radial basis function (RBF) surrogate, and a linear solver that learns
   a mapping from morphological feature vectors to a similarity score in [0, 1].

**Mathematical Bridge**
The bridge is the *feature‑space* shared by both algorithms.  The NLMS filter works
on an input vector **x** ∈ ℝᵈ, while the RBF surrogate operates on a morphological
feature vector **g** ∈ ℝᵈ (length, width, height, mass).  By treating the NLMS weight
vector **w** as a proxy for a morphological descriptor, we can feed **w** (or a
derived version) into the RBF surrogate.  The surrogate’s output α∈[0,1] then
blends the NLMS update with the surrogate’s learned knowledge:

  **w⁺ = (1‑α)·w_NLMS + α·w_RBF**

where **w_NLMS** is the NLMS‑updated weight vector (with RLCT‑adapted μ) and
**w_RBF** are the surrogate coefficients projected onto the same dimension.
This creates a unified adaptive system that simultaneously respects statistical
learning theory (RLCT) and geometric similarity (RBF surrogate)."""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (RLCT & NLMS)
# ----------------------------------------------------------------------
def bayesian_information_criterion(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """Standard BIC = -2*logL + n_params*log(n)."""
    return -2.0 * log_likelihood + n_params * math.log(n_samples)


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(np.dot(weights, x))


def nlms_update(weights: np.ndarray, x: np.ndarray, target: float,
                mu: float = 0.5, eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """Classic NLMS weight update."""
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")
    y = nlms_predict(weights, x)
    error = target - y
    power = float(np.dot(x, x)) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error


def estimate_rlct_from_losses(losses: Sequence[float]) -> float:
    """
    Very simple RLCT estimator.
    Fits log(loss) = - (rlct/2) * log(iter) + C  →  rlct = -2 * slope.
    Returns a non‑negative estimate.
    """
    if len(losses) < 2:
        return 0.0
    it = np.arange(1, len(losses) + 1)
    log_it = np.log(it)
    log_loss = np.log(np.maximum(losses, 1e-12))
    # linear regression (least squares)
    A = np.vstack([log_it, np.ones_like(log_it)]).T
    slope, _ = np.linalg.lstsq(A, log_loss, rcond=None)[0]
    rlct = max(0.0, -2.0 * slope)
    return rlct


def nlms_update_rlct(weights: np.ndarray, x: np.ndarray, target: float,
                    loss_history: Sequence[float],
                    mu_base: float = 0.5, eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    NLMS update where the step size μ is modulated by an RLCT estimate.
    Larger RLCT (more degenerate landscape) yields a smaller effective μ.
    """
    rlct = estimate_rlct_from_losses(loss_history)
    mu = mu_base / (1.0 + rlct)  # simple damping
    return nlms_update(weights, x, target, mu=mu, eps=eps)


# ----------------------------------------------------------------------
# Parent B utilities (Geometric Morphology & RBF Surrogate)
# ----------------------------------------------------------------------
Vector = Sequence[float]


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass
class Endpoint:
    length: float
    width: float
    height: float
    mass: float


class Morphology:
    """Geometric description of an endpoint."""
    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint

    def get_geometric_properties(self) -> Vector:
        return (self.endpoint.length,
                self.endpoint.width,
                self.endpoint.height,
                self.endpoint.mass)


class RBF_Surrogate:
    """
    Learns a linear combination of Gaussian RBFs over a set of training
    geometric vectors.  The solution is obtained by solving A·c = b,
    where A_ij = φ(||g_i‑g_j||) and c are the surrogate coefficients.
    """
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.coeffs: np.ndarray = np.array([])   # learned coefficients
        self.train_vectors: List[Vector] = []    # geometric training set

    def _phi(self, a: Vector, b: Vector) -> float:
        return gaussian(euclidean(a, b), self.epsilon)

    def fit(self, vectors: List[Vector], targets: List[float]) -> None:
        """Train the surrogate on (vector, target) pairs."""
        n = len(vectors)
        if n == 0:
            raise ValueError("training set empty")
        A = np.empty((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                A[i, j] = self._phi(vectors[i], vectors[j])
        self.coeffs = np.linalg.solve(A, np.array(targets, dtype=float))
        self.train_vectors = vectors.copy()

    def predict(self, vector: Vector) -> float:
        """Predict a similarity score for a new geometric vector."""
        if self.coeffs.size == 0:
            raise RuntimeError("surrogate not fitted")
        phi = np.array([self._phi(vector, gv) for gv in self.train_vectors])
        return float(np.dot(self.coeffs, phi))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_nlms_rbf_step(weights_nlms: np.ndarray,
                        x_signal: np.ndarray,
                        target_signal: float,
                        loss_history: Sequence[float],
                        surrogate: RBF_Surrogate,
                        morph_vectors: List[Vector]) -> Tuple[np.ndarray, float, float]:
    """
    Perform one hybrid adaptation step:
      1. NLMS weight update with RLCT‑adjusted μ.
      2. Use the updated NLMS weights (projected to the geometry space) as a
         query for the RBF surrogate, obtaining a blending factor α∈[0,1].
      3. Blend the NLMS weights with the surrogate coefficients (truncated/padded
         to the NLMS dimension) using α.
    Returns the blended weight vector, the NLMS error, and the blending factor.
    """
    # 1️⃣ NLMS update
    w_nlms, error = nlms_update_rlct(weights_nlms, x_signal, target_signal, loss_history)

    # 2️⃣ Map NLMS weights to a geometric vector (simple scaling)
    #    We assume the NLMS dimension equals the morphological feature dimension.
    geom_query = tuple(float(v) for v in w_nlms)  # length‑width‑height‑mass proxy

    # 3️⃣ Surrogate provides blending factor α
    alpha = surrogate.predict(geom_query)
    alpha = max(0.0, min(1.0, alpha))  # clamp for safety

    # 4️⃣ Blend
    #    If dimensions differ, truncate or pad the surrogate coefficients.
    coeffs = surrogate.coeffs
    if coeffs.shape[0] < w_nlms.shape[0]:
        # pad with zeros
        pad = np.zeros(w_nlms.shape[0] - coeffs.shape[0])
        coeffs_padded = np.concatenate([coeffs, pad])
    else:
        coeffs_padded = coeffs[: w_nlms.shape[0]]
    blended = (1.0 - alpha) * w_nlms + alpha * coeffs_padded
    return blended, error, alpha


def train_hybrid_system(signal_dim: int,
                        training_signals: List[Tuple[np.ndarray, float]],
                        morphologies: List[Morphology],
                        surrogate_eps: float = 1.0) -> Tuple[np.ndarray, RBF_Surrogate, List[Vector]]:
    """
    Trains both the NLMS filter (via a single pass) and the RBF surrogate.
    Returns initial NLMS weights, a fitted surrogate, and the list of geometric vectors.
    """
    # Initialise NLMS weights to small random values
    rng = np.random.default_rng()
    w0 = rng.normal(scale=0.01, size=signal_dim).astype(float)

    # Simple one‑epoch NLMS pass to obtain a reasonable starting point
    loss_hist: List[float] = []
    for x, y in training_signals:
        w0, err = nlms_update(w0, x, y)
        loss_hist.append(err ** 2)

    # Prepare surrogate training data
    geom_vectors = [m.get_geometric_properties() for m in morphologies]
    # For illustration, map each geometry to a synthetic similarity target
    # based on its norm (scaled to [0,1]).
    norms = np.linalg.norm(np.array(geom_vectors), axis=1)
    sim_targets = (norms - norms.min()) / (norms.max() - norms.min() + 1e-12)

    surrogate = RBF_Surrogate(epsilon=surrogate_eps)
    surrogate.fit(geom_vectors, list(sim_targets))
    return w0, surrogate, geom_vectors


def evaluate_hybrid(weights: np.ndarray,
                    test_signal: np.ndarray,
                    test_target: float,
                    loss_history: List[float],
                    surrogate: RBF_Surrogate,
                    morph_vectors: List[Vector]) -> Tuple[float, float]:
    """
    Runs a single hybrid step on test data and returns the absolute error
    after blending and the blending factor α.
    """
    blended, err, alpha = hybrid_nlms_rbf_step(weights,
                                               test_signal,
                                               test_target,
                                               loss_history,
                                               surrogate,
                                               morph_vectors)
    prediction = nlms_predict(blended, test_signal)
    final_error = abs(test_target - prediction)
    return final_error, alpha


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic problem dimension
    D = 4

    # Generate synthetic training signals (random Gaussian inputs)
    rng = np.random.default_rng(42)
    train_data = []
    for _ in range(30):
        x = rng.normal(size=D)
        # Underlying linear model with unknown true weights
        true_w = np.array([0.5, -0.3, 0.8, -0.1])
        y = float(np.dot(true_w, x) + rng.normal(scale=0.05))
        train_data.append((x, y))

    # Generate synthetic morphological endpoints
    endpoints = []
    for _ in range(6):
        ep = Endpoint(
            length=rng.uniform(0.5, 2.0),
            width=rng.uniform(0.5, 2.0),
            height=rng.uniform(0.5, 2.0),
            mass=rng.uniform(1.0, 5.0)
        )
        endpoints.append(Morphology(ep))

    # Train hybrid system
    init_weights, rbf_surrogate, geom_vecs = train_hybrid_system(
        signal_dim=D,
        training_signals=train_data,
        morphologies=endpoints,
        surrogate_eps=1.2
    )

    # Prepare a test example
    test_x = rng.normal(size=D)
    test_y = float(np.dot(np.array([0.5, -0.3, 0.8, -0.1]), test_x) + rng.normal(scale=0.05))

    # Dummy loss history (use training losses)
    dummy_loss_hist = [abs(y - nlms_predict(init_weights, x)) ** 2 for x, y in train_data]

    # Evaluate hybrid step
    err, alpha = evaluate_hybrid(init_weights,
                                 test_x,
                                 test_y,
                                 dummy_loss_hist,
                                 rbf_surrogate,
                                 geom_vecs)

    print(f"Hybrid prediction error: {err:.6f}")
    print(f"Blending factor α from RBF surrogate: {alpha:.4f}")