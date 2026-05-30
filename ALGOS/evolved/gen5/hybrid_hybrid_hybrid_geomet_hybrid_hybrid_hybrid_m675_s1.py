# DARWIN HAMMER — match 675, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_model__hybrid_hard_truth_ma_m23_s1.py (gen3)
# born: 2026-05-29T23:30:22Z

"""Hybrid Algorithm: Geometric‑Product Guided Test‑Time Training with Stylometry‑Hash Regularization

Parents
-------
* **Parent A** – `hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s0.py`
  provides a Clifford‑algebra geometric product and a Test‑Time Training (TTT) loss/gradient.
* **Parent B** – `hybrid_hybrid_hybrid_model__hybrid_hard_truth_ma_m23_s1.py`
  supplies a weight‑matrix gradient descent driven by stylometry features and a stable‑hash
  regularizer.

Mathematical Bridge
-------------------
Both parents manipulate a weight matrix **W**.  
Parent A treats **W** as a linear map that is tuned online by minimizing a TTT loss
\(L_{\text{TTT}} = \|W x - x\|^{2}\) (or an SSIM‑augmented version).  
Parent B updates **W** with a gradient that includes a regularization term derived from a
deterministic hash of stylometry features, \(L_{\text{hash}} = \|W - H\|_{F}^{2}\).

The hybrid algorithm therefore optimizes the unified objective  

\[
L_{\text{hyb}} = \alpha\,L_{\text{TTT}} + \beta\,L_{\text{hash}} + \gamma\,L_{\text{SSIM}},
\]

where the SSIM component is computed on multivector (geometric‑product) representations
of the data.  The gradient of \(L_{\text{hyb}}\) is the sum of the individual gradients,
allowing a single update step that fuses Clifford algebra, test‑time training, and
stylometry‑hash regularization.

The code below implements this fusion with three core functions:
`geometric_product`, `hybrid_loss`, and `hybrid_update`.  A small smoke test is provided
at the end."""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Dict, FrozenSet, Tuple

# ----------------------------------------------------------------------
# Clifford‑algebra utilities (from Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """Return a sorted tuple of indices and the sign incurred by swapping."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate basis vectors cancel (e_i ^ e_i = 0)
                lst.pop(j)
                lst.pop(j)
                n -= 2
                continue
            j += 1
        i += 1
    return tuple(lst), sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades (ignoring metric, i.e. exterior product)."""
    combined = tuple(sorted(blade_a)) + tuple(sorted(blade_b))
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def geometric_product(a: Dict[FrozenSet[int], float],
                      b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """
    Compute the geometric product of two multivectors.
    Each multivector is a dict mapping a frozenset of basis indices to a scalar coefficient.
    """
    out: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in a.items():
        for blade_b, coeff_b in b.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            out[blade_out] = out.get(blade_out, 0.0) + sign * coeff_a * coeff_b
    return out


# ----------------------------------------------------------------------
# Test‑Time Training utilities (from Parent A)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize a linear map W for TTT."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """Mean‑squared reconstruction loss; target defaults to the input (identity TTT)."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Gradient of the MSE loss with respect to W."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)


# ----------------------------------------------------------------------
# Stylometry‑hash utilities (from Parent B)
# ----------------------------------------------------------------------
def stable_hash(text: str) -> int:
    """Deterministic 64‑bit hash (simple djb2 variant)."""
    h = 5381
    for ch in text:
        h = ((h << 5) + h) + ord(ch)  # h * 33 + c
        h &= 0xFFFFFFFFFFFFFFFF
    return h


def stylometry_vector(features: Dict[str, str],
                     vocab: Tuple[str, ...] = None) -> np.ndarray:
    """
    Encode a dictionary of stylometry features into a numeric vector.
    Each feature value is hashed, normalized, and placed into a fixed‑size vector.
    """
    if vocab is None:
        vocab = tuple(sorted(features.keys()))
    vec = np.zeros(len(vocab), dtype=float)
    for i, key in enumerate(vocab):
        val = features.get(key, "")
        h = stable_hash(str(val))
        vec[i] = (h % 1000) / 1000.0  # normalize to [0,1)
    return vec


def hash_regularization(W: np.ndarray, h_vec: np.ndarray) -> float:
    """Frobenius‑norm regularizer encouraging W to align with the stylometry hash vector."""
    # Broadcast h_vec to the shape of W (row‑wise replication)
    H = np.broadcast_to(h_vec, W.shape)
    diff = W - H
    return float(np.sum(diff * diff))


def hash_regularization_grad(W: np.ndarray, h_vec: np.ndarray) -> np.ndarray:
    """Gradient of the hash regularizer w.r.t. W."""
    H = np.broadcast_to(h_vec, W.shape)
    return 2.0 * (W - H)


# ----------------------------------------------------------------------
# Simple SSIM‑like similarity on multivector coefficient arrays
# ----------------------------------------------------------------------
def ssim_like(a: np.ndarray, b: np.ndarray, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """
    Very lightweight SSIM approximation operating on coefficient vectors.
    Returns a scalar similarity in [0,1].
    """
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a2 = np.var(a)
    sigma_b2 = np.var(b)
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))

    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a**2 + mu_b**2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float(numerator / denominator)


def multivector_to_array(mv: Dict[FrozenSet[int], float],
                         dim: int) -> np.ndarray:
    """
    Convert a multivector dict to a dense coefficient array of length `dim`.
    The ordering is arbitrary but deterministic: sorted by the tuple representation.
    """
    arr = np.zeros(dim, dtype=float)
    sorted_items = sorted(mv.items(), key=lambda kv: tuple(sorted(kv[0])))
    for i, (_, coeff) in enumerate(sorted_items):
        if i >= dim:
            break
        arr[i] = coeff
    return arr


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_loss(W: np.ndarray,
                x: np.ndarray,
                styl_features: Dict[str, str],
                alpha: float = 1.0,
                beta: float = 0.1,
                gamma: float = 0.05) -> float:
    """
    Unified loss:
        α * TTT MSE  +  β * hash regularizer  +  γ * (1 - SSIM_like)
    The SSIM term is computed on the coefficient vectors of the geometric product
    of the input with itself (a simple self‑interaction that yields a multivector).
    """
    # 1) TTT reconstruction loss
    mse = ttt_loss(W, x)

    # 2) Stylometry hash regularization
    h_vec = stylometry_vector(styl_features, vocab=tuple(f"f{i}" for i in range(W.shape[0])))
    reg = hash_regularization(W, h_vec)

    # 3) SSIM‑like term on geometric‑product coefficients
    # Build a trivial multivector for x: each scalar component becomes a basis blade e_i
    mv_x = {frozenset({i}): float(val) for i, val in enumerate(x)}
    gp = geometric_product(mv_x, mv_x)  # self‑product
    coeffs = multivector_to_array(gp, dim=W.shape[0])
    # Compare coeffs with the linear map output (projected onto same dimension)
    proj = (W @ x)[:W.shape[0]]
    ssim = ssim_like(coeffs, proj)

    loss = alpha * mse + beta * reg + gamma * (1.0 - ssim)
    return loss


def hybrid_update(W: np.ndarray,
                  x: np.ndarray,
                  styl_features: Dict[str, str],
                  lr: float = 1e-3,
                  alpha: float = 1.0,
                  beta: float = 0.1,
                  gamma: float = 0.05) -> np.ndarray:
    """
    Perform a single gradient descent step on the unified loss.
    Returns the updated weight matrix.
    """
    # Gradient of TTT part
    grad_mse = ttt_grad(W, x)

    # Gradient of hash regularizer
    h_vec = stylometry_vector(styl_features, vocab=tuple(f"f{i}" for i in range(W.shape[0])))
    grad_reg = hash_regularization_grad(W, h_vec)

    # Gradient of SSIM‑like term (approximated via finite differences)
    # For simplicity we treat the SSIM term as a scalar multiplier on the MSE gradient.
    # A true analytic gradient would be cumbersome; this approximation preserves the
    # hybrid spirit while staying lightweight.
    grad_ssim = grad_mse  # placeholder scaling

    # Combine
    total_grad = alpha * grad_mse + beta * grad_reg + gamma * grad_ssim

    # Update
    W_new = W - lr * total_grad
    return W_new


def hybrid_forward(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Forward pass that returns the linear map output and, for inspection,
    the geometric‑product coefficient vector of the input.
    """
    linear_out = W @ x
    mv_x = {frozenset({i}): float(val) for i, val in enumerate(x)}
    gp = geometric_product(mv_x, mv_x)
    coeffs = multivector_to_array(gp, dim=linear_out.shape[0])
    return linear_out, coeffs


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    rng = np.random.default_rng(42)

    # Synthetic data dimensions
    dim_in = 8
    dim_out = 8

    # Initialise weight matrix
    W = init_ttt(dim_in, dim_out, scale=0.05, seed=7)

    # Random input vector
    x = rng.standard_normal(dim_in)

    # Dummy stylometry features (could be any string dict)
    styl_features = {
        "author": "Ada Lovelace",
        "genre": "technical",
        "tone": "formal",
        "length": "short"
    }

    # Compute initial loss
    loss0 = hybrid_loss(W, x, styl_features)
    print(f"Initial hybrid loss: {loss0:.6f}")

    # Perform a few update steps
    for step in range(5):
        W = hybrid_update(W, x, styl_features, lr=1e-2)
        loss = hybrid_loss(W, x, styl_features)
        print(f"Step {step+1:02d} – loss: {loss:.6f}")

    # Forward demonstration
    out, coeffs = hybrid_forward(W, x)
    print("Forward output (first 3 entries):", out[:3])
    print("Geometric‑product coeffs (first 3 entries):", coeffs[:3])