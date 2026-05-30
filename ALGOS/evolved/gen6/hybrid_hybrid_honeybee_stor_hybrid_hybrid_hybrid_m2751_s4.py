# DARWIN HAMMER — match 2751, survivor 4
# gen: 6
# parent_a: hybrid_honeybee_store_hybrid_hybrid_hybrid_m836_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m675_s0.py (gen5)
# born: 2026-05-29T23:45:37Z

"""Hybrid Store‑SSIM & Geometric‑Product Test‑Time Training
Parents:
    - hybrid_honeybee_store_hybrid_hybrid_hybrid_m836_s2.py (store dynamics + SSIM)
    - hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m675_s0.py (geometric product + TTT + stylometry)

Mathematical Bridge
-------------------
The store delta Δs from the honeybee store model is turned into an *adaptive gain* g
by weighting it with an SSIM similarity s between the current stylometry feature vector
f and a reference vector r:

    g = s * Δs

The gain g multiplies the learning‑rate η used in the test‑time‑training (TTT) weight
matrix update

    W_{new} = W - η·g·∇_W L(W, f)

where L is the TTT loss (prediction residual) regularised by a stable hash term
h(f)∈[0,1].  Thus the two topologies are fused: the decentralized store controls the
speed of the geometric‑product‑guided weight adaptation, while the SSIM similarity
acts as a dynamic scaling factor.  The resulting system is a single feedback loop
that updates both the store state and the weight matrix in concert."""


import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants from Parent A (store + SSIM)
# ----------------------------------------------------------------------
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
K1 = 0.01
K2 = 0.03
L = 255.0

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)


# ----------------------------------------------------------------------
# Store dynamics (Parent A)
# ----------------------------------------------------------------------
def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = ALPHA,
    beta: float = BETA,
    dt: float = DT,
) -> Tuple[float, float]:
    """
    Compute the next store value and the raw delta Δstore.
    """
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta


# ----------------------------------------------------------------------
# Simplified SSIM for 1‑D vectors (Parent A)
# ----------------------------------------------------------------------
def _ssim_index(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute a simplified SSIM index between two 1‑D feature vectors.
    The formulation follows the classic SSIM components (luminance, contrast,
    structure) but is reduced to scalar operations for speed.
    """
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x = np.var(x)
    sigma_y = np.var(y)

    covariance = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * covariance + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Stylometry feature extraction (used by Parent B)
# ----------------------------------------------------------------------
def extract_features(text: str) -> np.ndarray:
    """
    Very lightweight stylometry extractor that returns a 9‑dim vector
    aligned with _FEATURE_ORDER.  Positive and negative lexical cues are
    counted and weighted accordingly.
    """
    tokens = text.lower().split()
    counts = np.zeros(len(_FEATURE_ORDER), dtype=np.int64)

    # Simple keyword maps for demonstration
    positive_map = {
        "evidence": 0, "verify": 0, "verified": 0, "confirm": 0, "confirmed": 0,
        "source": 0, "sourced": 0, "citation": 0, "receipt": 0,
        "planning": 1, "plan": 1,
        "delay": 2, "late": 2,
        "support": 3, "help": 3,
        "boundary": 4, "limit": 4,
        "outcome": 5, "result": 5,
    }
    negative_map = {
        "impulsive": 6, "rash": 6,
        "scarcity": 7, "shortage": 7,
        "risk": 8, "danger": 8,
    }

    for token in tokens:
        if token in positive_map:
            counts[positive_map[token]] += 1
        elif token in negative_map:
            counts[negative_map[token]] += 1

    # Apply the static weight vectors from Parent A
    weighted = counts * (_POSITIVE_WEIGHTS + _NEGATIVE_WEIGHTS)
    return weighted.astype(float)


# ----------------------------------------------------------------------
# Stable hashing regulariser (Parent B)
# ----------------------------------------------------------------------
def stable_hash_regulariser(vec: np.ndarray) -> float:
    """
    Produce a deterministic scalar in [0, 1] from the feature vector.
    The built‑in hash is stable within a single Python process; we normalise
    it by the maximum possible 64‑bit signed integer.
    """
    h = hash(tuple(vec.tolist()))
    # Convert signed to unsigned range
    h_unsigned = h + (1 << 63)
    return h_unsigned / float(1 << 64)


# ----------------------------------------------------------------------
# Test‑Time Training step with adaptive gain (Hybrid core)
# ----------------------------------------------------------------------
def hybrid_ttt_step(
    W: np.ndarray,
    store: float,
    inflow: List[float],
    outflow: List[float],
    x: np.ndarray,
    ref_vec: np.ndarray,
    base_lr: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """
    Perform one hybrid update:
        1. Update the store and obtain Δstore.
        2. Compute SSIM between current feature vector x and a reference vector.
        3. Form adaptive gain g = ssim * Δstore.
        4. Compute TTT loss L = ||W·x - x||² + λ·h(x) where h is the stable hash term.
        5. Gradient‑descent update W with learning‑rate η = base_lr * g.
    Returns the updated weight matrix and the new store value.
    """
    # 1. Store dynamics
    new_store, delta = update_store(store, inflow, outflow)

    # 2. SSIM similarity
    ssim = _ssim_index(x, ref_vec)

    # 3. Adaptive gain
    gain = ssim * delta

    # Guard against zero or negative gain (learning rate must be non‑negative)
    effective_lr = max(0.0, base_lr * gain)

    # 4. Loss components
    pred = W @ x
    residual = pred - x
    mse = float(residual @ residual)

    hash_reg = stable_hash_regulariser(x)
    lam = 0.1  # regularisation strength
    loss = mse + lam * hash_reg

    # 5. Gradient (same as TTT gradient)
    grad = 2.0 * np.outer(residual, x)

    # Apply adaptive learning‑rate update
    W_new = W - effective_lr * grad

    return W_new, new_store


# ----------------------------------------------------------------------
# Helper to initialise weight matrix (Parent B)
# ----------------------------------------------------------------------
def init_weight_matrix(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """
    Initialise a weight matrix with a normal distribution.
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic text to generate stylometry features
    txt = "Evidence confirms the plan but there is a risk of delay and scarcity."
    feat = extract_features(txt)

    # Reference vector (could be from a previous step; here we use a shuffled copy)
    ref = np.roll(feat, 1)

    # Initialise store
    store_val = 10.0
    inflow_vals = [random.random() for _ in range(3)]
    outflow_vals = [random.random() for _ in range(2)]

    # Initialise weight matrix
    dim = len(feat)
    W = init_weight_matrix(dim, seed=42)

    # Perform hybrid update
    W_upd, store_val = hybrid_ttt_step(
        W=W,
        store=store_val,
        inflow=inflow_vals,
        outflow=outflow_vals,
        x=feat,
        ref_vec=ref,
        base_lr=0.005,
    )

    # Simple sanity checks (no exceptions, shapes match)
    assert W_upd.shape == W.shape
    assert isinstance(store_val, float)

    print("Hybrid step completed.")
    print(f"New store value: {store_val:.4f}")
    print(f"Weight matrix norm change: {np.linalg.norm(W_upd - W):.6f}")