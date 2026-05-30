# DARWIN HAMMER — match 5392, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m2072_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m886_s0.py (gen5)
# born: 2026-05-30T00:01:35Z

"""Hybrid RBF‑TTT Fusion
Parent A: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m2072_s0.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m886_s0.py

Mathematical bridge:
The TTT‑Linear weight matrix **W** maps an input vector *x* into a latent feature
vector *z = W·x*.  This latent vector is supplied as the argument of the
Radial‑Basis‑Function surrogate model inherited from Parent A.  The RBF
predicts a scalar *ŷ = φ(z)* which is interpreted as a target for the same
linear mapping.  Consequently the prediction error *(W·x − ŷ)* drives a
gradient‑descent update of **W**, while the surrogate itself can be re‑fit
with newly gathered (z, ŷ) pairs.  The two subsystems are thus tightly
coupled: the linear transform supplies features for the surrogate, and the
surrogate’s scalar output closes the loop by correcting the transform.

The module provides a minimal yet functional hybrid that demonstrates this
interaction.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent A – RBF surrogate utilities
# ----------------------------------------------------------------------
def _gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))

def _euclidean(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

class RBFSurrogate:
    """Standard RBF interpolant:  ŷ(x) = Σ_i w_i·exp(-ε²‖x‑c_i‖²)"""
    def __init__(self, centers: list[tuple[float, ...]], weights: np.ndarray, epsilon: float = 1.0):
        self.centers = centers            # List of tuples, each centre in ℝ^d
        self.weights = weights            # 1‑D array, shape (n_centers,)
        self.epsilon = epsilon

    def predict(self, x: list[float]) -> float:
        """Evaluate the surrogate at point *x*."""
        return float(
            sum(
                w * _gaussian(_euclidean(x, c), self.epsilon)
                for w, c in zip(self.weights, self.centers)
            )
        )

def fit_rbf(points: list[list[float]],
            values: list[float],
            epsilon: float = 1.0,
            ridge: float = 1e-9) -> RBFSurrogate:
    """Fit an RBF surrogate by solving (Φ+λI)w = y."""
    n = len(points)
    if n == 0:
        raise ValueError("No points to fit.")
    phi = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            phi[i, j] = _gaussian(_euclidean(points[i], points[j]), epsilon)
    phi += ridge * np.eye(n)
    y = np.asarray(values, dtype=float)
    weights = np.linalg.solve(phi, y)
    centers = [tuple(p) for p in points]
    return RBFSurrogate(centers, weights, epsilon)

# ----------------------------------------------------------------------
# Parent B – TTT‑Linear utilities (simple linear mapping with GD)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize a weight matrix W ∈ ℝ^{d_out×d_in}."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in), dtype=float) * scale

def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray) -> float:
    """Quadratic loss ‖W·x − target‖²."""
    diff = W @ x - target
    return float(np.sum(diff ** 2))

def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gradient of the loss w.r.t. W."""
    diff = W @ x - target
    # grad = 2·diff·xᵀ  (outer product)
    return 2.0 * np.outer(diff, x)

def ttt_step(W: np.ndarray, x: np.ndarray, target: np.ndarray, eta: float) -> np.ndarray:
    """One SGD step on W."""
    grad = ttt_grad(W, x, target)
    return W - eta * grad

# ----------------------------------------------------------------------
# Helper – simple perceptual hash (used for sketch‑like indexing)
# ----------------------------------------------------------------------
def compute_phash(values: list[float]) -> int:
    """64‑bit perceptual hash of a float list."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def init_hybrid_model(points: list[list[float]],
                      values: list[float],
                      d_input: int,
                      epsilon: float = 1.0,
                      scale: float = 0.01,
                      seed: int = 0) -> dict:
    """
    Build a hybrid model consisting of:
      * an RBF surrogate fitted on (points, values)
      * a TTT‑Linear weight matrix W mapping d_input‑dimensional inputs
        to the RBF feature space (dimension = len(points))

    Returns a dict with keys:
        'rbf'   -> RBFSurrogate instance
        'W'     -> np.ndarray of shape (n_centers, d_input)
        'epsilon' -> RBF shape parameter (stored for possible re‑fit)
    """
    rbf = fit_rbf(points, values, epsilon=epsilon)
    n_centers = len(points)
    W = init_ttt(d_in=d_input, d_out=n_centers, scale=scale, seed=seed)
    return {'rbf': rbf, 'W': W, 'epsilon': epsilon}

def hybrid_predict(x: np.ndarray, model: dict) -> float:
    """
    Perform a forward pass:
      z = W·x                (latent feature vector)
      ŷ = RBF(z)            (scalar prediction)

    Parameters
    ----------
    x : np.ndarray, shape (d_input,)
    model : dict as returned by ``init_hybrid_model``

    Returns
    -------
    float – predicted scalar.
    """
    if x.ndim != 1:
        raise ValueError("Input x must be a 1‑D array.")
    z = model['W'] @ x                      # shape (n_centers,)
    return model['rbf'].predict(z.tolist())

def hybrid_train_step(model: dict,
                      x: np.ndarray,
                      true_scalar: float,
                      eta: float = 1e-3,
                      ridge: float = 1e-9) -> None:
    """
    One training iteration that couples the two subsystems:

    1. Compute latent vector z = W·x.
    2. Obtain surrogate prediction ŷ = φ(z).
    3. Treat ŷ as the *target* for the linear map and perform a gradient step
       on W to reduce the error (W·x − ŷ).
    4. Append (z, true_scalar) to the RBF dataset and re‑fit the surrogate
       incrementally (simple re‑fit on the whole set for clarity).

    The function mutates ``model`` in‑place.
    """
    # ----- 1. latent representation -----
    W = model['W']
    z = W @ x                                 # (n_centers,)

    # ----- 2. surrogate prediction -----
    y_hat = model['rbf'].predict(z.tolist())

    # ----- 3. update W using surrogate output as target -----
    target_vec = np.full_like(z, y_hat)       # broadcast scalar to same shape
    W_new = ttt_step(W, x, target_vec, eta)
    model['W'] = W_new

    # ----- 4. optionally improve the surrogate -----
    # Store new training pair (z, true_scalar)
    # For a lightweight demo we keep the full history inside the model.
    if 'history' not in model:
        model['history'] = {'points': [], 'values': []}
    model['history']['points'].append(z.tolist())
    model['history']['values'].append(true_scalar)

    # Re‑fit the surrogate on the accumulated data.
    # This is O(N³) but acceptable for the tiny demo sizes.
    points = model['history']['points']
    values = model['history']['values']
    model['rbf'] = fit_rbf(points, values, epsilon=model['epsilon'], ridge=ridge)

def hybrid_hash_summary(model: dict) -> int:
    """
    Produce a 64‑bit hash summarising the current surrogate weights.
    Useful as a cheap fingerprint for sketch‑like monitoring.
    """
    return compute_phash(model['rbf'].weights.tolist())

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data: 5‑dimensional inputs, 8 RBF centres.
    d_input = 5
    n_centres = 8
    random.seed(42)
    np.random.seed(42)

    # Random centres in ℝ^{d_input}
    centres = [np.random.randn(d_input).tolist() for _ in range(n_centres)]
    # Random scalar values associated with centres
    centre_vals = np.random.randn(n_centres).tolist()

    # Initialise hybrid model
    model = init_hybrid_model(centres, centre_vals, d_input, epsilon=0.8, scale=0.05, seed=123)

    # Random query vector
    x_query = np.random.randn(d_input)

    # Forward prediction before any training
    pred_before = hybrid_predict(x_query, model)
    print(f"Prediction before training: {pred_before:.4f}")

    # Simulated ground‑truth scalar (e.g., from an external oracle)
    true_scalar = float(np.sin(x_query.sum()))  # arbitrary smooth function

    # Perform a few training steps
    for i in range(5):
        hybrid_train_step(model, x_query, true_scalar, eta=1e-2)
        pred = hybrid_predict(x_query, model)
        loss = (pred - true_scalar) ** 2
        print(f"Step {i+1:02d}: pred={pred:.4f}, loss={loss:.6f}")

    # Hash fingerprint
    fingerprint = hybrid_hash_summary(model)
    print(f"Model fingerprint (64‑bit): {fingerprint:#018x}")