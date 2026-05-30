# DARWIN HAMMER — match 461, survivor 2
# gen: 5
# parent_a: hybrid_geometric_product_hybrid_model_vram_sc_m22_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m207_s1.py (gen4)
# born: 2026-05-29T23:29:03Z

"""Hybrid Geometric‑Decision‑Capybara Optimizer

Parents:
- `hybrid_geometric_product_hybrid_model_vram_sc_m22_s2.py` (GA rotor + TTT linear model + VRAM scheduler)
- `hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m207_s1.py` (Shannon‑entropy decision hygiene + RBF surrogate + Capybara Optimization)

Mathematical bridge:
Both parents expose a *scalar modulation* of a bilinear learning step.
Parent A updates a weight matrix `W` and a rotor `R` by gradient‑like
steps that are scaled by a learning‑rate scheduler.
Parent B computes a scalar Shannon entropy `H` from decision‑feature counts
and a surrogate prediction `ŝ` from a radial‑basis‑function (RBF) model.
We fuse them by letting the effective learning rates be

η_w' = η_w * (1 + H) * σ(ŝ)
η_r' = η_r * (1 + H) * σ(ŝ)

where `σ` is a sigmoid that maps the RBF output to a positive scaling factor.
Thus the decision‑entropy and the surrogate model jointly steer the GA‑TTT
updates, while the original VRAM scheduler still caps the overall step size.

The module implements:
- entropy calculation,
- a lightweight RBF surrogate,
- rotor utilities (matrix‑based GA rotor),
- a hybrid update that blends TTT linear learning with entropy‑ and
  surrogate‑scaled rates,
- a simple VRAM‑budget scheduler,
- a sequence processor that demonstrates the unified system.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
import numpy as np

# ---------------------------------------------------------------------------
# 1. Decision‑hygiene utilities (Shannon entropy)
# ---------------------------------------------------------------------------

def shannon_entropy(counts: np.ndarray) -> float:
    """Return the Shannon entropy of a non‑negative integer count vector.

    The vector is first normalised to a probability distribution.
    Zero entries are ignored in the sum.
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # avoid log(0) by masking
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log2(probs[mask])))


# ---------------------------------------------------------------------------
# 2. Radial‑basis surrogate model
# ---------------------------------------------------------------------------

def rbf_surrogate(x: np.ndarray,
                  centers: np.ndarray,
                  gamma: float,
                  weights: np.ndarray) -> float:
    """Predict a scalar using an RBF surrogate.

    `centers` shape = (n_centers, dim), `weights` shape = (n_centers,).
    The kernel is `exp(-gamma * ||x - c||^2)`.
    """
    diffs = centers - x  # (n_centers, dim)
    sq_norms = np.einsum('ij,ij->i', diffs, diffs)  # (n_centers,)
    kernels = np.exp(-gamma * sq_norms)
    return float(np.dot(weights, kernels))


# ---------------------------------------------------------------------------
# 3. Clifford‑geometric rotor utilities (matrix representation)
# ---------------------------------------------------------------------------

def generate_random_rotor(dim: int) -> np.ndarray:
    """Generate a random orthogonal matrix (rotor) of size (dim, dim).

    The matrix is drawn from the Haar distribution via QR decomposition.
    """
    a = np.random.randn(dim, dim)
    q, r = np.linalg.qr(a)
    # Ensure a proper rotation (determinant = +1)
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def apply_rotor(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rotate a Euclidean vector `x` with rotor matrix `R` (R @ x)."""
    return R @ x


# ---------------------------------------------------------------------------
# 4. VRAM‑budget scheduler (simple step‑wise cap)
# ---------------------------------------------------------------------------

def vram_scheduler(step: int, budget: int, base_lr: float) -> float:
    """Return a scaling factor for the learning rate based on a VRAM budget.

    Every `budget` steps the factor is halved, mimicking a reduced
    learning‑rate when memory pressure grows.
    """
    factor = 0.5 ** (step // budget)
    return base_lr * factor


# ---------------------------------------------------------------------------
# 5. Hybrid update that merges TTT, rotor, entropy and RBF scaling
# ---------------------------------------------------------------------------

def hybrid_step(W: np.ndarray,
                R: np.ndarray,
                x: np.ndarray,
                eta_w: float,
                eta_r: float,
                entropy: float,
                rbf_pred: float,
                vram_factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Perform one hybrid learning step.

    Returns updated (W, R).
    """
    # ----- TTT linear forward pass (with rotor) -----
    x_rot = apply_rotor(R, x)          # rotate input
    y = W @ x_rot                      # linear map
    err = y - x_rot                    # error (target is the original rotated vector)

    # ----- Modulation scalars -----
    # sigmoid maps RBF output (which may be any real) to (0,1)
    sigmoid = 1.0 / (1.0 + math.exp(-rbf_pred))
    scale = (1.0 + entropy) * sigmoid * vram_factor

    eta_w_eff = eta_w * scale
    eta_r_eff = eta_r * scale

    # ----- Weight matrix gradient step (SGD) -----
    # loss L = 0.5 * ||err||^2  => dL/dW = err @ x_rot.T
    W_new = W - eta_w_eff * np.outer(err, x_rot)

    # ----- Rotor update via infinitesimal bivector generator -----
    # Generator G = x_rot ⊗ err - err ⊗ x_rot  (skew‑symmetric)
    G = np.outer(x_rot, err) - np.outer(err, x_rot)   # shape (dim, dim)
    # First‑order Rodrigues update: R_new ≈ R - eta_r_eff * G @ R
    R_new = R - eta_r_eff * (G @ R)

    # Re‑orthogonalise to keep R a proper rotor (optional but stabilises)
    # Using QR on R_new to project back onto O(dim)
    q, _ = np.linalg.qr(R_new)
    # Ensure determinant +1
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    R_new = q

    return W_new, R_new


# ---------------------------------------------------------------------------
# 6. Sequence processor that demonstrates the full hybrid system
# ---------------------------------------------------------------------------

def hybrid_process(x_seq: np.ndarray,
                   W: np.ndarray,
                   R: np.ndarray,
                   eta_w: float,
                   eta_r: float,
                   entropy_counts: np.ndarray,
                   rbf_centers: np.ndarray,
                   rbf_gamma: float,
                   rbf_weights: np.ndarray,
                   vram_budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Process a sequence of input vectors with the hybrid optimizer.

    Parameters
    ----------
    x_seq : (n_steps, dim) array of inputs
    W, R : initial weight matrix and rotor
    eta_w, eta_r : base learning rates
    entropy_counts : (n_features,) integer vector for entropy computation
    rbf_* : parameters of the RBF surrogate
    vram_budget : integer controlling the VRAM scheduler

    Returns
    -------
    (W_final, R_final)
    """
    dim = W.shape[0]
    # Pre‑compute the scalar entropy once (could be updated online)
    H = shannon_entropy(entropy_counts.astype(float))

    for step, x in enumerate(x_seq):
        # surrogate prediction based on current input
        ŝ = rbf_surrogate(x, rbf_centers, rbf_gamma, rbf_weights)

        # VRAM‑scaled learning rate factor
        lr_factor = vram_scheduler(step, vram_budget, 1.0)

        # Hybrid update
        W, R = hybrid_step(W, R, x, eta_w, eta_r, H, ŝ, lr_factor)

    return W, R


# ---------------------------------------------------------------------------
# 7. Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Dimensions
    dim = 5
    n_steps = 20
    n_features = 8
    n_centers = 10

    # Random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialise weight matrix and rotor
    W0 = np.random.randn(dim, dim) * 0.1
    R0 = generate_random_rotor(dim)

    # Random input sequence
    X_seq = np.random.randn(n_steps, dim)

    # Decision‑hygiene feature counts (simulated)
    feature_counts = np.random.randint(0, 20, size=n_features)

    # RBF surrogate parameters
    centers = np.random.randn(n_centers, dim)
    gamma = 0.5
    weights = np.random.randn(n_centers) * 0.1

    # Learning rates and VRAM budget
    eta_w_base = 0.01
    eta_r_base = 0.005
    vram_budget = 7  # halve LR every 7 steps

    # Run the hybrid optimizer
    W_final, R_final = hybrid_process(
        X_seq,
        W0,
        R0,
        eta_w_base,
        eta_r_base,
        feature_counts,
        centers,
        gamma,
        weights,
        vram_budget,
    )

    # Simple sanity checks (no exceptions, shapes preserved)
    assert W_final.shape == (dim, dim)
    assert R_final.shape == (dim, dim)
    # Verify R_final is (approximately) orthogonal
    ortho_error = np.linalg.norm(R_final.T @ R_final - np.eye(dim))
    print(f"Final orthogonality error of rotor: {ortho_error:.2e}")
    print("Hybrid optimizer completed successfully.")