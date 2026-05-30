# DARWIN HAMMER — match 3390, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s2.py (gen4)
# parent_b: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hybrid_m1904_s2.py (gen5)
# born: 2026-05-29T23:49:51Z

"""HybridPathKANClifford
This module fuses the two parent algorithms:

* **Parent A** – “HybridPathKANFoldChange” contributes the lead‑lag transform,
  a B‑spline expansion of the transformed path, a fold‑change detector and an
  Euler‑integrated weight‑matrix update whose learning‑rate is modulated by the
  fold‑change factor.

* **Parent B** – “Hybrid Path Signature – Clifford Geometric Product & B‑Spline
  Basis” contributes a Clifford‑algebraic geometric product that can combine
  multivector representations of the path signature.

**Mathematical bridge** – The B‑spline basis produced from the lead‑lag
transformed path is used to build a multivector (scalar + vector + bivector)
that approximates the level‑0/1/2 signature.  The weight matrix **W** maps the
flattened multivector to a scalar prediction.  During training the gradient of a
squared‑error loss with respect to **W** is integrated with a single Euler step.
The effective learning‑rate is multiplied by a fold‑change factor
`γ = 1 + κ·Δ`, where `Δ` is the relative change of the loss between two
consecutive steps.  The geometric product can be used to combine two such
multivectors (e.g. for higher‑order interactions) before the linear mapping.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Tuple, Dict, Any

# ----------------------------------------------------------------------
# Core building blocks (lead‑lag + B‑spline)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) copies of a path.
    Input shape (T, d). Output shape (2*T‑1, 2*d).
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     np.zeros(d)])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], np.zeros(d)])
    return out

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Simple piecewise‑constant B‑spline basis of degree k‑1.
    Returns an (len(x), len(grid)+k‑2) matrix where each column is the indicator
    of x belonging to a knot interval.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    # extended knot vector with multiplicity k‑1 at the ends
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])
    n_basis = len(grid) + k - 2
    B = np.zeros((len(x), n_basis), dtype=np.float64)
    for i in range(n_basis):
        left = t[i]
        right = t[i + 1]
        # last interval is closed on the right to capture the endpoint
        if i == n_basis - 1:
            B[:, i] = (x >= left) & (x <= right)
        else:
            B[:, i] = (x >= left) & (x < right)
    return B

# ----------------------------------------------------------------------
# Clifford multivector utilities
# ----------------------------------------------------------------------
def construct_multivector(basis_vals: np.ndarray) -> Dict[str, Any]:
    """
    Build a multivector (scalar, vector, bivector) from B‑spline coefficients.
    - scalar part : sum of all coefficients (level‑0 signature)
    - vector part : weighted sum of coefficients per dimension (level‑1)
    - bivector part : antisymmetric outer product approximating level‑2
    """
    # Assume basis_vals shape (N, M) where N = time samples, M = number of basis.
    # We compress across time by simple averaging.
    coeff = np.mean(basis_vals, axis=0)  # (M,)
    d = int(math.sqrt(len(coeff)))  # crude guess for dimensionality
    # For robustness, fall back to 1‑D vector if guess fails.
    if d * d != len(coeff):
        d = len(coeff)
    scalar = np.sum(coeff)
    vector = coeff[:d] if d <= len(coeff) else np.pad(coeff, (0, d - len(coeff)))
    # Build a bivector as the antisymmetric matrix of outer products of the vector.
    bivector = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(i + 1, d):
            bivector[i, j] = vector[i] * vector[j] - vector[j] * vector[i]
    return {"s": scalar, "v": vector, "b": bivector}

def geometric_product(mv1: Dict[str, Any], mv2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Geometric product for multivectors limited to grades 0,1,2.
    Returns a new multivector with the same grade structure.
    """
    s1, v1, b1 = mv1["s"], mv1["v"], mv1["b"]
    s2, v2, b2 = mv2["s"], mv2["v"], mv2["b"]
    # scalar part
    s = s1 * s2 + np.dot(v1, v2) + np.trace(np.dot(b1, b2))
    # vector part
    v = s1 * v2 + s2 * v1 + np.dot(b1, v2) + np.dot(b2, v1)
    # bivector part (antisymmetric)
    b = s1 * b2 + s2 * b1 + np.outer(v1, v2) - np.outer(v2, v1)
    # enforce antisymmetry
    b = 0.5 * (b - b.T)
    return {"s": s, "v": v, "b": b}

def flatten_multivector(mv: Dict[str, Any]) -> np.ndarray:
    """
    Flatten a multivector into a 1‑D vector: [scalar, vector..., bivector upper‑tri].
    """
    s = np.array([mv["s"]], dtype=float)
    v = np.asarray(mv["v"], dtype=float).ravel()
    b = mv["b"]
    d = b.shape[0]
    # extract upper‑triangular part without diagonal
    idx = np.triu_indices(d, k=1)
    b_flat = b[idx]
    return np.concatenate([s, v, b_flat])

# ----------------------------------------------------------------------
# Fold‑change aware Euler update
# ----------------------------------------------------------------------
def fold_change_factor(prev_loss: float, curr_loss: float, kappa: float = 0.1) -> float:
    """
    Compute modulation factor γ = 1 + κ·Δ where Δ = (prev‑loss − curr‑loss)/prev‑loss.
    If prev_loss is zero, γ defaults to 1.0.
    """
    if prev_loss == 0.0:
        return 1.0
    delta = (prev_loss - curr_loss) / prev_loss
    return 1.0 + kappa * delta

def euler_weight_update(W: np.ndarray, grad: np.ndarray, lr: float, gamma: float) -> np.ndarray:
    """
    Perform a single Euler integration step:
        W_new = W - lr * γ * grad
    """
    return W - lr * gamma * grad

# ----------------------------------------------------------------------
# Hybrid training step
# ----------------------------------------------------------------------
def hybrid_step(
    path: np.ndarray,
    W: np.ndarray,
    target: float,
    prev_loss: float,
    lr: float = 0.01,
    kappa: float = 0.1,
    grid: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """
    Executes one training iteration:
      1. Lead‑lag transform the raw path.
      2. Expand each dimension with a B‑spline basis.
      3. Build a multivector from the basis matrix.
      4. Optionally combine the multivector with itself via the geometric product
         (demonstrates the Clifford bridge).
      5. Flatten, linearly map with W, compute squared‑error loss.
      6. Compute gradient w.r.t. W, modulate the learning‑rate with the
         fold‑change factor, and update W with an Euler step.
    Returns the updated weight matrix and the current loss.
    """
    # 1. Lead‑lag
    ll = lead_lag_transform(path)                     # (2T‑1, 2d)

    # 2. B‑spline basis per dimension (use a uniform grid on [0,1])
    if grid is None:
        grid = np.linspace(0.0, 1.0, num=5)            # simple 5‑knot grid
    # Normalise the lead‑lag time axis to [0,1] for basis evaluation
    t_norm = np.linspace(0.0, 1.0, ll.shape[0])
    B = bspline_basis(t_norm, grid)                  # (2T‑1, M)

    # 3. Construct multivector from the basis-weighted features
    #    Multiply the basis matrix by the lead‑lag values (outer product)
    #    to obtain coefficient matrix of shape (M, 2d)
    coeffs = B.T @ ll                                   # (M, 2d)
    mv = construct_multivector(coeffs)                 # multivector dict

    # 4. Demonstrate Clifford product (here we square the multivector)
    mv = geometric_product(mv, mv)

    # 5. Linear prediction
    phi = flatten_multivector(mv)                       # (p,)
    y_pred = float(W @ phi)                             # scalar output
    loss = 0.5 * (y_pred - target) ** 2

    # 6. Gradient w.r.t. W (∂L/∂W = (y_pred‑target) * φ)
    grad_W = (y_pred - target) * phi

    gamma = fold_change_factor(prev_loss, loss, kappa)
    W_new = euler_weight_update(W, grad_W, lr, gamma)

    return W_new, loss

# ----------------------------------------------------------------------
# Utility to initialise weight matrix
# ----------------------------------------------------------------------
def initialise_weights(feature_dim: int, output_dim: int = 1, scale: float = 0.01) -> np.ndarray:
    """
    Randomly initialise a weight matrix of shape (output_dim, feature_dim).
    """
    rng = np.random.default_rng()
    return scale * rng.standard_normal((output_dim, feature_dim))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a random 2‑D path of length 10
    T, d = 10, 2
    rng = np.random.default_rng(42)
    path = rng.standard_normal((T, d))

    # Initialise weights based on an expected feature dimension.
    # Feature dimension = 1 (scalar) + d (vector) + d*(d‑1)//2 (bivector)
    d_feat = 1 + d + d * (d - 1) // 2
    W = initialise_weights(d_feat)

    # Dummy target – use the sum of coordinates as a simple regression target
    target = float(np.sum(path))

    prev_loss = float('inf')
    for epoch in range(5):
        W, loss = hybrid_step(
            path=path,
            W=W,
            target=target,
            prev_loss=prev_loss,
            lr=0.05,
            kappa=0.2,
        )
        print(f"Epoch {epoch+1}: loss = {loss:.6f}")
        prev_loss = loss
    print("Smoke test completed without errors.")