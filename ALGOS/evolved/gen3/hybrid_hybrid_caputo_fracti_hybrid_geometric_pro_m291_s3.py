# DARWIN HAMMER — match 291, survivor 3
# gen: 3
# parent_a: hybrid_caputo_fractional_minimum_cost_tree_m35_s2.py (gen1)
# parent_b: hybrid_geometric_product_hybrid_model_vram_sc_m22_s2.py (gen2)
# born: 2026-05-29T23:28:16Z

"""Hybrid Caputo‑Geometric Minimum‑Cost Tree (HCG‑MCT)

This module fuses the two parent algorithms:

* **Parent A** – `hybrid_caputo_fractional_minimum_cost_tree_m35_s2.py`  
  Provides a Caputo fractional derivative kernel φ(t;α)≈t^{‑α} (implemented via a
  Lanczos‑approximated Gamma function) that yields long‑range memory weights for
  graph edges.

* **Parent B** – `hybrid_geometric_product_hybrid_model_vram_sc_m22_s2.py`  
  Supplies Clifford‑geometric‑product utilities, in particular a rotor R that
  rotates vectors by a bilinear map `R * x * ~R`.  The rotor is updated by a
  skew‑symmetric bivector derived from the error of a linear model `W @ x`.

**Mathematical bridge** – Both parents operate on weighted bilinear maps.
The fractional kernel supplies scalar edge‑weights `w_ij(t,α)` that can be
embedded into a multivector as a scalar part.  The geometric product then
applies a rotor R (an orthogonal matrix) to the vector state before the linear
model `W`.  The hybrid cost of a tree therefore mixes *edge length* `ℓ_ij`
with *fractional memory weight* `w_ij`, while the learning dynamics update
both `W` and `R` using the same loss.

The code below implements the combined system:
* fractional‑memory edge weights (`caputo_weights`),
* rotor application (`apply_rotor`),
* hybrid tree cost (`hybrid_tree_cost`),
* a single‑step hybrid TTT‑GA update (`hybrid_ttt_ga_step`),
* a sequence‑level driver with a toy VRAM scheduler (`hybrid_ttt_ga_vram`).

All functions are pure NumPy and use only the allowed standard‑library imports.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Lanczos approximation – needed for the Caputo kernel (Parent A)
# ---------------------------------------------------------------------------

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])


def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of Γ(z) for z>0."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, len(_LANCZOS_C)):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


def caputo_kernel(delta_t: np.ndarray, alpha: float) -> np.ndarray:
    """
    Power‑law decay kernel φ(Δt;α) = Δt^{‑α} / Γ(1‑α).
    The Gamma function is evaluated with the Lanczos approximation.
    """
    # avoid division by zero – set zero‑lag to a very small positive number
    safe_dt = np.where(delta_t <= 0, np.finfo(float).eps, delta_t)
    phi = safe_dt ** (-alpha) / gamma_lanczos(1 - alpha)
    return phi


def caputo_weights(t: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute normalized Caputo weights for a set of timestamps `t`.
    Returns a symmetric matrix `W` where W[i,j] = w_ij (zero on the diagonal).
    """
    n = t.size
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i):
            delta = t[i] - t[j]
            if delta > 0:
                w = caputo_kernel(np.array([delta]), alpha)[0]
                W[i, j] = w
                W[j, i] = w
    total = W.sum()
    if total > 0:
        W /= total
    return W


# ---------------------------------------------------------------------------
# Clifford geometric product utilities – rotor handling (Parent B)
# ---------------------------------------------------------------------------

def _skew_symmetric_from_bivector(biv: np.ndarray) -> np.ndarray:
    """
    Given a bivector matrix `biv` (not necessarily skew‑symmetric),
    return its skew‑symmetric part which will serve as a generator of an
    infinitesimal rotation.
    """
    return 0.5 * (biv - biv.T)


def rotor_from_generator(G: np.ndarray, eta: float = 1.0) -> np.ndarray:
    """
    Build an orthogonal rotor matrix from a skew‑symmetric generator `G`
    using a first‑order exponential map: R ≈ I + η·G.
    The result is re‑orthogonalised via QR to guarantee numerical stability.
    """
    R_approx = np.eye(G.shape[0]) + eta * G
    # QR decomposition returns an orthogonal Q and an upper‑triangular R;
    # we keep Q as the orthogonal rotor.
    Q, _ = np.linalg.qr(R_approx)
    return Q


def apply_rotor(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Rotate a Euclidean vector `x` with rotor `R` (matrix representation of
    a Clifford rotor).  This is the sandwich product `R * x * Rᵀ`
    reduced to a linear map because `R` is orthogonal.
    """
    return R @ x


# ---------------------------------------------------------------------------
# Hybrid tree cost – mixes edge lengths with Caputo fractional weights
# ---------------------------------------------------------------------------

def hybrid_tree_cost(edge_list: list[tuple[int, int]],
                     edge_lengths: dict[tuple[int, int], float],
                     node_times: np.ndarray,
                     alpha: float) -> float:
    """
    Compute the hybrid minimum‑cost tree score.

    For each edge (i,j) we combine the physical length ℓ_ij with the
    fractional memory weight w_ij(t,α).  The total cost is

        C = Σ_{i>j} ℓ_ij · w_ij .

    Parameters
    ----------
    edge_list : list of (i, j) index pairs defining the tree.
    edge_lengths : dict mapping (i, j) → length (float).
    node_times : 1‑D array of timestamps for each node.
    alpha : fractional order 0<α<1.

    Returns
    -------
    float – the hybrid cost.
    """
    W = caputo_weights(node_times, alpha)          # symmetric weight matrix
    cost = 0.0
    for i, j in edge_list:
        w = W[i, j]
        ℓ = edge_lengths.get((i, j), edge_lengths.get((j, i), 0.0))
        cost += ℓ * w
    return cost


# ---------------------------------------------------------------------------
# Hybrid TTT‑GA step – updates linear model W and rotor R together
# ---------------------------------------------------------------------------

def hybrid_ttt_ga_step(W: np.ndarray,
                       R: np.ndarray,
                       x: np.ndarray,
                       eta_w: float,
                       eta_r: float,
                       vram_full: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Perform a single learning step:

    1. Rotate the input vector `x` with the current rotor `R`.
    2. Predict `ŷ = W @ x_rot`.
    3. Compute error `e = ŷ - x_rot` and quadratic loss `L = 0.5·||e||²`.
    4. Gradient descent on `W`:   W ← W - η_w·e·x_rotᵀ.
    5. Build a bivector generator `B = x ∧ e = x·eᵀ - e·xᵀ` (skew‑symmetric).
    6. Update rotor via `R ← orthogonalise(R·(I - η_r·B))`.

    The VRAM scheduler (toy) halves the learning rates when `vram_full`
    is True, mimicking a reduced‑budget regime.

    Returns
    -------
    (W_new, R_new, loss)
    """
    if vram_full:
        eta_w *= 0.5
        eta_r *= 0.5

    # 1. rotate
    x_rot = apply_rotor(R, x)

    # 2. predict
    y_hat = W @ x_rot

    # 3. error & loss
    e = y_hat - x_rot
    loss = 0.5 * np.dot(e, e)

    # 4. update linear weights
    W_new = W - eta_w * np.outer(e, x_rot)

    # 5. bivector generator (skew‑symmetric)
    B = np.outer(x, e) - np.outer(e, x)   # x∧e
    G = _skew_symmetric_from_bivector(B)

    # 6. rotor update and re‑orthogonalisation
    R_tilde = R @ (np.eye(R.shape[0]) - eta_r * G)
    Q, _ = np.linalg.qr(R_tilde)          # ensure orthogonality
    R_new = Q

    return W_new, R_new, loss


# ---------------------------------------------------------------------------
# Sequence driver with a simplistic VRAM scheduler (Parent B component)
# ---------------------------------------------------------------------------

def hybrid_ttt_ga_vram(x_seq: list[np.ndarray],
                       t_seq: list[float],
                       alpha: float,
                       edge_list: list[tuple[int, int]],
                       edge_lengths: dict[tuple[int, int], float],
                       dim: int,
                       eta_w: float = 0.01,
                       eta_r: float = 0.005,
                       vram_budget: int = 1000) -> dict:
    """
    Process a sequence of inputs using the hybrid Caputo‑Geometric model.

    The VRAM scheduler monitors the number of processed steps; once the
    processed count exceeds `vram_budget` the scheduler signals a reduced‑
    learning‑rate regime (the `vram_full` flag).

    Returns a dictionary with final parameters and the history of losses.
    """
    # initialise linear map and rotor
    W = np.eye(dim) * 0.1
    R = np.eye(dim)

    losses = []
    node_times = np.array(t_seq, dtype=float)

    for step, (x, t) in enumerate(zip(x_seq, t_seq)):
        vram_full = step >= vram_budget
        W, R, loss = hybrid_ttt_ga_step(W, R, x, eta_w, eta_r, vram_full)
        losses.append(loss)

        # optional: recompute a hybrid tree cost every 50 steps for diagnostics
        if (step + 1) % 50 == 0:
            cost = hybrid_tree_cost(edge_list, edge_lengths, node_times[: step + 1], alpha)
            # store or print diagnostics – here we just append to losses list
            losses.append(cost)

    result = {
        "W_final": W,
        "R_final": R,
        "loss_history": np.array(losses),
    }
    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # simple synthetic test with 5‑dimensional vectors
    dim = 5
    rng = np.random.default_rng(42)

    # generate a random sequence of 200 vectors and timestamps
    seq_len = 200
    x_sequence = [rng.normal(size=dim) for _ in range(seq_len)]
    t_sequence = [float(i) for i in range(seq_len)]   # unit spacing

    # build a toy tree (a chain) and assign unit lengths
    edges = [(i, i + 1) for i in range(dim - 1)]
    lengths = {(i, i + 1): 1.0 for i in range(dim - 1)}

    alpha = 0.35

    out = hybrid_ttt_ga_vram(
        x_seq=x_sequence,
        t_seq=t_sequence,
        alpha=alpha,
        edge_list=edges,
        edge_lengths=lengths,
        dim=dim,
        eta_w=0.02,
        eta_r=0.01,
        vram_budget=150
    )

    print("Final linear map (W) norm:", np.linalg.norm(out["W_final"]))
    print("Final rotor orthogonality error:",
          np.linalg.norm(out["R_final"] @ out["R_final"].T - np.eye(dim)))
    print("Loss history length:", out["loss_history"].size)
    print("First 5 losses:", out["loss_history"][:5])