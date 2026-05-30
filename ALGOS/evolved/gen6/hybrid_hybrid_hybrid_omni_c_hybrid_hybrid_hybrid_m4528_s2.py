# DARWIN HAMMER — match 4528, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m2407_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m956_s0.py (gen4)
# born: 2026-05-29T23:56:21Z

"""
Hybrid Omni-Graph State‑Space Fisher Bandit Fusion

Parents:
- hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m2407_s4.py (graph diffusion + Fisher‑regularised loss)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m956_s0.py (temperature‑dependent state transition + weak‑supervision labeling)

Mathematical bridge:
The adjacency matrix A of the graph diffusion is interpreted as a *state‑transition
operator*.  In the biological parent the transition matrix is modulated by a
temperature‑dependent scalar `developmental_rate(T)`.  We therefore scale A by
that scalar before each diffusion step, producing a temperature‑aware embedding
R^{(t+1)} = (α(T)·A) R^{(t)} where α(T)=developmental_rate(T).

The bandit component supplies a scalar propensity p∈[0,1] for each training
sample.  This propensity multiplies the Fisher information term
F(θ;μ,σ)=1/σ² (the Fisher information of a Normal(μ,σ²) w.r.t. its mean) and is
added to the standard L2 prediction loss.  The final hybrid loss is

    L = ‖Ŷ−Y‖₂² + λ·p·F(θ;μ,σ)

Thus the algorithm learns temperature‑aware graph embeddings, predicts a target,
and regularises the predictor with a confidence‑weighted Fisher term.

The module implements:
1. `graph_representation_temp` – diffusion with temperature scaling.
2. `fisher_confidence_loss` – L2 loss plus propensity‑weighted Fisher regulariser.
3. `hybrid_operation` – end‑to‑end pipeline (embedding → linear prediction →
   loss) demonstrating the fused system.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑B: Temperature‑dependent developmental rate (Schoolfield model)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (arbitrary units)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15            # K  (10 °C)
    t_high: float = 307.15           # K  (34 °C)
    delta_h_low: float = -45_000.0   # J mol⁻¹
    delta_h_high: float = 65_000.0   # J mol⁻¹
    r_cal: float = 1.987             # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Simplified Schoolfield temperature response.
    Returns a positive scalar α(T) that scales a rate.
    """
    if temp_k <= 0:
        raise ValueError("Temperature must be greater than 0 K")
    # Normalised Arrhenius term (ignoring low/high deactivation for brevity)
    k = params.rho_25 * math.exp(
        -params.delta_h_activation / (params.r_cal * (temp_k - 273.15))
    )
    # Ensure a reasonable scale (avoid overflow/underflow)
    return max(k, 1e-8)

# ----------------------------------------------------------------------
# Parent‑A: Graph diffusion embedding (row‑normalised adjacency)
# ----------------------------------------------------------------------
def _row_normalise(mat: np.ndarray) -> np.ndarray:
    """Row‑normalise a square matrix; rows that sum to zero become zero rows."""
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        normed = np.where(row_sums != 0, mat / row_sums, 0.0)
    return normed

def graph_representation_temp(
    adj: np.ndarray,
    feats: np.ndarray,
    temp_k: float,
    steps: int = 5,
) -> np.ndarray:
    """
    Temperature‑aware diffusion embedding.
    Â = α(T)·Â_row_normalised
    R^{(0)} = feats
    R^{(t+1)} = Â @ R^{(t)}
    Returns L2‑normalised node embeddings.
    """
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    if adj.shape[0] != feats.shape[0]:
        raise ValueError("Adjacency and feature row dimensions must match")
    # Row‑normalise adjacency once
    A_norm = _row_normalise(adj.astype(float))
    # Temperature scaling factor α(T)
    alpha = developmental_rate(temp_k)
    A_temp = alpha * A_norm
    R = feats.astype(float)
    for _ in range(steps):
        R = A_temp @ R
    # Unit‑length per node
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return R / norms

# ----------------------------------------------------------------------
# Fisher information for a scalar Gaussian mean (parent‑A) and bandit propensity
# ----------------------------------------------------------------------
def fisher_information(mu: float, sigma: float) -> float:
    """
    Fisher information of N(μ,σ²) w.r.t. μ is 1/σ².
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    return 1.0 / (sigma ** 2)

def bandit_propensity(scores: np.ndarray) -> np.ndarray:
    """
    Convert raw bandit scores into a confidence weight in [0,1] via softmax.
    """
    if scores.ndim != 1:
        raise ValueError("Scores must be a 1‑D array")
    max_s = np.max(scores)
    exp_shifted = np.exp(scores - max_s)
    probs = exp_shifted / exp_shifted.sum()
    return probs  # already in [0,1] and sum to 1

def fisher_confidence_loss(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    theta: np.ndarray,
    mu: float,
    sigma: float,
    propensity: np.ndarray,
    lam: float = 0.1,
) -> float:
    """
    L2 loss plus propensity‑weighted Fisher regulariser.
    theta: scalar parameters (e.g., model weights) – shape (k,).
    propensity: confidence per sample – shape (N,).
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and target shapes must match")
    if propensity.shape[0] != y_pred.shape[0]:
        raise ValueError("Propensity length must match number of samples")
    # Base L2 loss
    l2 = np.mean((y_pred - y_true) ** 2)
    # Fisher term (scalar) multiplied by mean propensity
    F = fisher_information(mu, sigma)
    weight = lam * propensity.mean() * F
    return l2 + weight

# ----------------------------------------------------------------------
# Hybrid pipeline exposing three public functions
# ----------------------------------------------------------------------
def embed_and_predict(
    adj: np.ndarray,
    feats: np.ndarray,
    temp_k: float,
    bandit_scores: np.ndarray,
    weight_vec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1. Compute temperature‑aware node embeddings.
    2. Produce a scalar prediction per node via a linear map wᵀ·h.
    3. Return predictions, propensity weights, and the embedding matrix.
    """
    embeddings = graph_representation_temp(adj, feats, temp_k)
    # Linear predictor (shared weight vector)
    preds = embeddings @ weight_vec
    prop = bandit_propensity(bandit_scores)
    return preds, prop, embeddings

def hybrid_loss(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    weight_vec: np.ndarray,
    mu: float,
    sigma: float,
    propensity: np.ndarray,
    lam: float = 0.1,
) -> float:
    """
    Wrapper that feeds the linear weight vector as the “θ” argument of the
    Fisher regulariser.
    """
    return fisher_confidence_loss(
        y_pred,
        y_true,
        theta=weight_vec,
        mu=mu,
        sigma=sigma,
        propensity=propensity,
        lam=lam,
    )

def hybrid_operation(
    adj: np.ndarray,
    feats: np.ndarray,
    temp_k: float,
    bandit_scores: np.ndarray,
    y_true: np.ndarray,
    lam: float = 0.1,
) -> float:
    """
    End‑to‑end hybrid operation:
    * random initialise linear weights,
    * embed with temperature scaling,
    * compute predictions,
    * evaluate the hybrid loss.
    Returns the scalar loss value.
    """
    d = feats.shape[1]
    # Initialise weights (seeded for reproducibility)
    rng = np.random.default_rng(42)
    weight_vec = rng.normal(scale=0.1, size=d)

    # Predict
    y_pred, propensity, _ = embed_and_predict(
        adj, feats, temp_k, bandit_scores, weight_vec
    )

    # Estimate μ and σ from predictions (simple statistics)
    mu = float(y_pred.mean())
    sigma = float(y_pred.std(ddof=1) + 1e-8)  # avoid zero

    loss = hybrid_loss(
        y_pred,
        y_true,
        weight_vec,
        mu,
        sigma,
        propensity,
        lam=lam,
    )
    return loss

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic graph (5 nodes)
    N = 5
    d = 3
    rng = np.random.default_rng(0)

    # Random adjacency (ensure connectivity)
    adj = rng.integers(0, 2, size=(N, N)).astype(float)
    np.fill_diagonal(adj, 0)  # no self‑loops
    # Random node features
    feats = rng.normal(size=(N, d))

    # Bandit scores (raw, can be negative)
    bandit_scores = rng.normal(loc=0.0, scale=1.0, size=N)

    # True targets (synthetic)
    y_true = rng.normal(loc=0.5, scale=0.2, size=N)

    # Temperature (e.g., 298 K ≈ 25 °C)
    temp_k = 298.15

    loss_val = hybrid_operation(
        adj=adj,
        feats=feats,
        temp_k=temp_k,
        bandit_scores=bandit_scores,
        y_true=y_true,
        lam=0.05,
    )
    print(f"Hybrid loss: {loss_val:.6f}")