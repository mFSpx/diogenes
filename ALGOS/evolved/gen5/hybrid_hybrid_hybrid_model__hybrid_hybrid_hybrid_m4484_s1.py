# DARWIN HAMMER — match 4484, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_lens__m159_s0.py (gen4)
# born: 2026-05-29T23:56:09Z

"""
Hybrid module merging:

- Parent A: TTT‑Linear model (weight matrix W, reconstruction loss, gradient).
- Parent B: Energy‑based Dense Associative Memory with Shannon entropy.

Mathematical bridge:
The linear transformation q = W·x produced by the TTT‑Linear model is treated as a
query vector for the energy function of the associative memory.  The hybrid loss
combines the reconstruction loss of the TTT model with the energy term (and
optionally the entropy term) evaluated on q.  Gradients flow from the energy
back through q to update W, yielding a unified optimisation that simultaneously
compresses the input (TTT) and respects the energy‑based memory landscape (B).
"""

import numpy as np
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Utility functions (softmax, log‑sum‑exp) – shared by both parents
# ----------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _lse(z: np.ndarray) -> float:
    """Log‑sum‑exp, numerically stable."""
    m = np.max(z)
    return float(m + np.log(np.exp(z - m).sum()))


# ----------------------------------------------------------------------
# Parent A – TTT‑Linear core
# ----------------------------------------------------------------------
def init_weights(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize weight matrix W of shape (d_out, d_in)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """Reconstruction loss ‖W x − target‖² (target defaults to x)."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Gradient of the reconstruction loss w.r.t. W."""
    pred = W @ x
    t = x if target is None else target
    grad = 2.0 * np.outer(pred - t, x)
    return grad


# ----------------------------------------------------------------------
# Parent B – Energy & entropy core
# ----------------------------------------------------------------------
def energy(query: np.ndarray, M: np.ndarray, beta: float = 1.0) -> float:
    """Energy = - (1/β)·log ∑ exp(β·M q) + ½‖q‖²."""
    scores = beta * (M @ query)
    lse_term = _lse(scores) / beta
    quadratic = 0.5 * (query @ query)
    return -lse_term + quadratic


def shannon_entropy(query: np.ndarray) -> float:
    """Shannon entropy of the squared‑amplitude distribution of a vector."""
    p = np.abs(query) ** 2
    p_sum = p.sum()
    if p_sum == 0:
        return 0.0
    p = p / p_sum
    # avoid log(0) by masking
    mask = p > 0
    return -float(np.sum(p[mask] * np.log2(p[mask])))


def energy_grad_query(query: np.ndarray, M: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Gradient of the energy function w.r.t. the query vector q.

    ∂E/∂q = -Mᵀ·softmax(β·M·q) + q
    """
    scores = beta * (M @ query)
    probs = _softmax(scores)               # softmax(β·M·q)
    return -M.T @ probs + query


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_forward(W: np.ndarray, x: np.ndarray, M: np.ndarray, beta: float = 1.0):
    """
    Forward pass producing:
    - reconstruction loss,
    - energy,
    - entropy,
    using q = W·x as the query vector.
    Returns a tuple (recon_loss, energy_val, entropy_val).
    """
    q = W @ x
    recon = ttt_loss(W, x)                 # target defaults to x
    eng = energy(q, M, beta)
    ent = shannon_entropy(q)
    return recon, eng, ent


def hybrid_loss(W: np.ndarray, x: np.ndarray, M: np.ndarray,
                beta: float = 1.0, lam_energy: float = 0.1,
                lam_entropy: float = 0.0) -> float:
    """
    Composite loss:
        L = recon_loss + λ₁·energy(q) + λ₂·entropy(q)
    where q = W·x.
    """
    recon, eng, ent = hybrid_forward(W, x, M, beta)
    return recon + lam_energy * eng + lam_entropy * ent


def hybrid_grad(W: np.ndarray, x: np.ndarray, M: np.ndarray,
                beta: float = 1.0, lam_energy: float = 0.1,
                lam_entropy: float = 0.0) -> np.ndarray:
    """
    Gradient of the composite loss w.r.t. W.

    dL/dW = d(recon)/dW + λ₁·(∂E/∂q)·xᵀ + λ₂·(∂H/∂q)·xᵀ

    The entropy gradient ∂H/∂q = -2·q·log₂(p) - 2·q, where p = |q|² / ‖q‖².
    (Derived from H = -∑p log₂ p, p_i = q_i² / Σ q².)
    """
    # Reconstruction part
    grad = ttt_grad(W, x)

    # Query vector
    q = W @ x

    # Energy contribution
    if lam_energy != 0.0:
        dE_dq = energy_grad_query(q, M, beta)          # shape (d_out,)
        grad += lam_energy * np.outer(dE_dq, x)

    # Entropy contribution
    if lam_entropy != 0.0:
        # Compute probabilities p_i = q_i^2 / sum(q^2)
        sq = q ** 2
        sum_sq = sq.sum()
        if sum_sq == 0:
            dH_dq = np.zeros_like(q)
        else:
            p = sq / sum_sq
            # derivative of -∑ p log₂ p w.r.t q_i
            # dH/dq_i = - (2 q_i / sum_sq) * (log₂ p_i + 1/ln 2)
            # because d p_i / d q_i = 2 q_i / sum_sq - 2 q_i * sq_i / sum_sq²
            # Using a compact form:
            log_p = np.log2(p, where=p > 0, out=np.zeros_like(p))
            const = 1.0 / math.log(2)  # conversion from ln to log2
            dH_dq = -2.0 * q * (log_p + const) / sum_sq
        grad += lam_entropy * np.outer(dH_dq, x)

    return grad


def hybrid_step(W: np.ndarray, x: np.ndarray, M: np.ndarray,
                lr: float = 1e-3, beta: float = 1.0,
                lam_energy: float = 0.1, lam_entropy: float = 0.0) -> np.ndarray:
    """
    Perform a single gradient‑descent update on W using the hybrid loss.
    Returns the updated weight matrix.
    """
    g = hybrid_grad(W, x, M, beta, lam_energy, lam_entropy)
    return W - lr * g


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # dimensions
    d_in = 8
    d_out = 8
    seed = 42

    # initialise components
    W = init_weights(d_in, d_out, scale=0.01, seed=seed)
    x = np.random.default_rng(seed + 1).normal(size=d_in)
    M = np.random.default_rng(seed + 2).normal(size=(d_out, d_out))

    # compute forward quantities
    recon, eng, ent = hybrid_forward(W, x, M, beta=0.5)
    print(f"Reconstruction loss: {recon:.6f}")
    print(f"Energy:               {eng:.6f}")
    print(f"Entropy:              {ent:.6f}")

    # compute loss and gradient
    loss = hybrid_loss(W, x, M, beta=0.5, lam_energy=0.2, lam_entropy=0.05)
    grad = hybrid_grad(W, x, M, beta=0.5, lam_energy=0.2, lam_entropy=0.05)
    print(f"Hybrid loss: {loss:.6f}")
    print(f"Gradient norm: {np.linalg.norm(grad):.6f}")

    # one optimisation step
    W_new = hybrid_step(W, x, M, lr=1e-3, beta=0.5, lam_energy=0.2, lam_entropy=0.05)
    print("Step completed, weight change norm:",
          np.linalg.norm(W_new - W))