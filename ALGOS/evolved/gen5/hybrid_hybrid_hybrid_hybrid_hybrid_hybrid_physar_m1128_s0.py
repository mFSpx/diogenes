# DARWIN HAMMER — match 1128, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s4.py (gen3)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_gliner_zero_s_m66_s1.py (gen4)
# born: 2026-05-29T23:32:57Z

"""Hybrid Algorithm: SSIM‑Guided Conductance Test‑Time Training (Hybrid‑SSIM‑CTT)

Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s4.py`
  provides a Structural Similarity (SSIM) based loss and a test‑time
  training (TTT) scheme that updates a weight matrix **W** by gradient
  descent on a combined reconstruction‑+‑SSIM loss.

* **Parent B** – `hybrid_hybrid_physarum_netw_hybrid_gliner_zero_s_m66_s1.py`
  supplies a flux‑based conductance dynamics on a graph:
      q = g / ℓ · (p_a – p_b)
      g←g+Δt·(γ·|q| – δ·g)
  together with a label‑extraction routine that scores spans with the
  computed flux.

Mathematical Bridge
-------------------
We interpret the TTT weight matrix **W** as a *conductance matrix* of a
complete (or sparse) graph whose nodes correspond to feature dimensions.
Edge lengths are stored in a separate matrix **L** (positive scalars) and
node pressures **p** are derived from the current input sample **x**
(e.g. p = x normalised).  The reconstruction gradient
∇_W L_rec = 2 (W·x – x) xᵀ provides a standard descent direction,
while the flux‑based update from Parent B supplies an *adaptive
conductance* term that pushes edges with large fluxes to higher
conductance.  The hybrid update for each entry (i,j) is therefore

    g_ij ← max(0,
               g_ij
               + Δt·(γ·|q_ij| – δ·g_ij)          # Physarum conductance dynamics
               – η·∇_W L_hybrid_ij)             # TTT gradient descent

where q_ij = g_ij / ℓ_ij · (p_i – p_j) and L_hybrid = α·L_rec + β·L_ssim.
Thus the two parent topologies are mathematically fused into a single
parameter‑update rule that simultaneously optimises a reconstruction /
SSIM objective and respects flux‑driven conductance adaptation.

The module below implements:
1. `ssim` – 1‑D SSIM.
2. `hybrid_loss` – α·L_rec + β·L_ssim.
3. `flux_matrix` – pairwise fluxes from conductances, edge lengths and pressures.
4. `hybrid_step` – performs the combined update of **W**.
5. `label_extraction` and `score_spans_with_flux` – a toy pipeline that
   extracts label spans from text and scores them using the current flux
   matrix, demonstrating the interaction between the graph dynamics and
   a downstream NLP‑like task.

All dependencies are limited to the Python standard library and NumPy.
"""

import sys
import math
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – SSIM utilities
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """
    Compute the Structural Similarity (SSIM) index for two 1‑D signals.
    The implementation follows the classic formula with
    C1 = (k1·DR)², C2 = (k2·DR)².
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

    return float(num / den)


def reconstruction_loss(W: np.ndarray, x: np.ndarray) -> float:
    """
    L_rec = ||W·x – x||²
    """
    residual = W @ x - x
    return float(np.sum(residual ** 2))


def hybrid_loss(W: np.ndarray, x: np.ndarray,
                alpha: float = 1.0, beta: float = 1.0) -> float:
    """
    L_hybrid = α·L_rec + β·(1 – SSIM(W·x, x))
    """
    rec = reconstruction_loss(W, x)
    s = ssim(W @ x, x)
    return alpha * rec + beta * (1.0 - s)


def grad_reconstruction(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Analytic gradient of L_rec w.r.t. W:
        ∂L_rec/∂W = 2 (W·x – x) xᵀ
    """
    residual = W @ x - x
    return 2.0 * np.outer(residual, x)


def grad_ssim_approx(W: np.ndarray, x: np.ndarray,
                     beta: float = 1.0) -> np.ndarray:
    """
    Approximate ∂(1‑SSIM)/∂W by scaling the reconstruction gradient
    with the SSIM loss value. This follows the heuristic used in the
    original hybrid parent.
    """
    s = ssim(W @ x, x)
    loss_factor = 1.0 - s
    return loss_factor * grad_reconstruction(W, x) * beta


# ----------------------------------------------------------------------
# Parent B – Flux / Conductance dynamics
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """
    q = g / ℓ · (p_a – p_b)
    """
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0,
                       gain: float = 1.0,
                       decay: float = 0.05) -> float:
    """
    g ← max(0, g + dt·(gain·|q| – decay·g))
    """
    if dt < 0 or decay < 0:
        raise ValueError("dt and decay must be non‑negative")
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def flux_matrix(G: np.ndarray, L: np.ndarray,
                pressures: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise flux matrix Q where
        Q_ij = flux(g_ij, ℓ_ij, p_i, p_j)
    The matrices G, L are symmetric with zeros on the diagonal.
    """
    n = G.shape[0]
    Q = np.zeros_like(G)
    for i in range(n):
        for j in range(i + 1, n):
            q = flux(G[i, j], L[i, j], pressures[i], pressures[j])
            Q[i, j] = q
            Q[j, i] = -q   # antisymmetric flux
    return Q


# ----------------------------------------------------------------------
# Hybrid core: unified update of W (conductances) using both losses
# ----------------------------------------------------------------------
def hybrid_step(W: np.ndarray,
                x: np.ndarray,
                edge_lengths: np.ndarray,
                pressures: np.ndarray,
                alpha: float = 1.0,
                beta: float = 1.0,
                dt: float = 1.0,
                gain: float = 1.0,
                decay: float = 0.05,
                lr: float = 0.01) -> Tuple[np.ndarray, float]:
    """
    Perform one hybrid update:
      1. Compute hybrid loss and its gradient (reconstruction + SSIM).
      2. Compute fluxes from the current conductance matrix (W) and
         pressures derived from the input sample.
      3. Update each conductance entry with the Physarum rule and the
         gradient‑descent step.

    Returns the updated matrix and the scalar loss value.
    """
    # 1. loss and gradient
    loss = hybrid_loss(W, x, alpha, beta)
    grad_rec = grad_reconstruction(W, x)
    grad_ssim = grad_ssim_approx(W, x, beta)
    total_grad = alpha * grad_rec + grad_ssim   # shape (n, n)

    # 2. fluxes (physarum dynamics)
    Q = flux_matrix(W, edge_lengths, pressures)

    # 3. conductance update
    G_new = np.copy(W)
    n = W.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            q_ij = Q[i, j]
            g_ij = W[i, j]
            # Physarum conductance term
            g_updated = update_conductance(g_ij, q_ij, dt, gain, decay)
            # Gradient‑descent term (note the minus sign for descent)
            g_updated -= lr * total_grad[i, j]
            # Enforce symmetry and non‑negativity
            g_updated = max(0.0, g_updated)
            G_new[i, j] = G_new[j, i] = g_updated

    # Diagonal entries are kept at zero (no self‑loops)
    np.fill_diagonal(G_new, 0.0)

    return G_new, loss


# ----------------------------------------------------------------------
# Demonstration utilities (label extraction + flux scoring)
# ----------------------------------------------------------------------
def label_extraction(text: str, labels: List[str]) -> List[Tuple[int, int, str]]:
    """
    Very simple label finder that returns spans (start, end, label)
    for exact word matches after normalising spaces and hyphens.
    """
    spans = []
    for label in labels:
        pattern = re.escape(label.replace(" / ", " ").replace("-", " "))
        regex = re.compile(rf"(?<!\w){pattern}(?!\w)", flags=re.IGNORECASE)
        for m in regex.finditer(text):
            spans.append((m.start(), m.end(), label))
    return spans


def score_spans_with_flux(spans: List[Tuple[int, int, str]],
                          flux_mat: np.ndarray,
                          node_map: dict) -> List[Tuple[int, int, str, float]]:
    """
    Map each span to a pair of nodes (start → end) using `node_map`,
    then assign a score equal to the absolute flux magnitude between the
    corresponding nodes.  This illustrates how the graph dynamics can be
    reused for a downstream task.
    """
    scored = []
    for start, end, label in spans:
        i = node_map.get(start, None)
        j = node_map.get(end, None)
        if i is None or j is None:
            score = 0.0
        else:
            score = abs(flux_mat[i, j])
        scored.append((start, end, label, score))
    return scored


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Problem size
    dim = 8

    # Random seed for reproducibility
    rng = np.random.default_rng(42)

    # Initialise weight / conductance matrix (symmetric, zero diagonal)
    W = rng.random((dim, dim))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)

    # Random edge lengths (positive)
    L = rng.random((dim, dim)) + 0.1
    L = (L + L.T) / 2.0
    np.fill_diagonal(L, 0.0)

    # Input sample (1‑D signal)
    x = rng.integers(0, 256, size=dim).astype(np.float64)

    # Pressures derived from the input (simple normalisation)
    pressures = (x - x.min()) / (x.max() - x.min() + 1e-12)

    # One hybrid update
    W, loss = hybrid_step(W, x, L, pressures,
                          alpha=0.8, beta=0.2,
                          dt=0.5, gain=1.2, decay=0.03,
                          lr=0.005)

    print(f"Hybrid loss after one step: {loss:.4f}")

    # Demonstrate label extraction + flux scoring
    sample_text = "The quick brown-fox jumps over the lazy dog."
    label_list = ["quick", "brown fox", "lazy dog"]
    spans = label_extraction(sample_text, label_list)

    # Create a dummy node map: map character positions to node indices
    node_map = {pos: pos % dim for pos in range(len(sample_text) + 1)}

    Q = flux_matrix(W, L, pressures)
    scored_spans = score_spans_with_flux(spans, Q, node_map)

    for start, end, label, score in scored_spans:
        print(f"Span '{sample_text[start:end]}' (label='{label}') -> flux score {score:.4f}")

    # Verify that W remains symmetric and non‑negative
    assert np.allclose(W, W.T, atol=1e-9), "W lost symmetry"
    assert np.all(W >= 0), "Negative conductance detected"
    print("All assertions passed.")