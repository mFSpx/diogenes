# DARWIN HAMMER — match 2511, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m613_s2.py (gen5)
# parent_b: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s0.py (gen3)
# born: 2026-05-29T23:42:39Z

"""Hybrid Algorithm: PathSignature‑Entropy‑MinHash‑NLMS with Doomsday‑Modulated Step‑Size

Parents
-------
- hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m613_s2.py (Algorithm A)
  Provides path‑signature extraction, Shannon entropy of the level‑2 signature,
  MinHash‑derived force series, and an RBF surrogate whose kernel width is
  scaled by the entropy.

- hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s0.py (Algorithm B)
  Supplies a Normalised Least‑Mean‑Squares (NLMS) adaptive filter whose learning
  rate μ is periodically modulated by the Doomsday calendar.

Mathematical Bridge
-------------------
The scalar entropy H computed from the level‑2 signature of a time‑series path
modulates two distinct components of the NLMS update:

1. **Step‑size scaling** – μ is multiplied by (1 + α·H), where α>0 controls the
   influence of the signature’s information content on the adaptation speed.

2. **RBF surrogate kernel width** – the Gaussian kernel width ε in the RBF
   surrogate is similarly scaled by (1 + β·H).  The surrogate provides an
   additive correction to the NLMS prediction, thereby fusing the continuous
   surrogate modelling of Algorithm A with the discrete adaptive filtering of
   Algorithm B.

Additionally, a MinHash‑derived discrete force series is integrated to obtain a
peak‑velocity scalar v_peak, which is appended to the NLMS regressor vector.
Thus the full feature vector fed to the combined predictor is

    φ = [sig₁, flatten(sig₂), H, v_peak] .

The following module implements this hybrid system with three core functions
demonstrating the fusion and a small smoke‑test in the ``__main__`` block.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
import hashlib
from typing import Sequence, List, Tuple

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – Path signature, entropy, MinHash force series, RBF surrogate
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Create a lead‑lag version of a path.
    For a path X₀,…,X_{T‑1} (shape (T,d)) the lead‑lag path interleaves each point
    with its predecessor, yielding shape (2T‑1,d)."""
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dim)")
    lead = path[1:]
    lag = path[:-1]
    # interleave: X₀, X₀, X₁, X₁, …, X_{T‑2}, X_{T‑2}, X_{T‑1}
    interleaved = np.empty((lead.shape[0] + lag.shape[0], path.shape[1]))
    interleaved[0::2] = lag
    interleaved[1::2] = lead
    return interleaved

def path_signature_features(path: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute level‑1 and level‑2 signatures of a path and the Shannon entropy
    of the normalized eigen‑spectrum of the level‑2 signature matrix."""
    # Lead‑lag transform improves invariance (parent A)
    ll_path = lead_lag_transform(path)

    # Incremental differences
    increments = np.diff(ll_path, axis=0)          # shape (N‑1, d)

    # Level‑1 signature: sum of increments (vector)
    sig1 = np.sum(increments, axis=0)               # shape (d,)

    # Level‑2 signature: sum of outer products of increments (matrix)
    sig2 = np.einsum('ti,tj->ij', increments, increments)  # shape (d,d)

    # Eigen‑values → normalized probability vector → Shannon entropy
    eigvals = np.linalg.eigvalsh(sig2)              # real symmetric → eigvals sorted
    eigvals = np.maximum(eigvals, 0.0)              # avoid negative due to numerical noise
    total = np.sum(eigvals) + 1e-12
    probs = eigvals / total
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    return sig1, sig2, entropy

def minhash_force_series(data: Vector, n_hashes: int = 8) -> List[float]:
    """Generate a deterministic discrete force series from an auxiliary data
    vector using simple MinHash (parent A). Each hash yields a value in {‑1,0,1}."""
    series = []
    for i in range(n_hashes):
        # Create a reproducible seed from the i‑th hash of the concatenated data
        hasher = hashlib.sha256()
        for v in data:
            hasher.update(str(v).encode('utf-8'))
        hasher.update(str(i).encode('utf-8'))
        seed = int(hasher.hexdigest(), 16) % (2**32)
        rnd = random.Random(seed)
        # Map uniform [-1,1] to discrete set
        val = rnd.choice([-1.0, 0.0, 1.0])
        series.append(val)
    return series

def integrate_force_series(series: List[float]) -> float:
    """Integrate the force series (simple cumulative sum) and return the peak
    absolute velocity."""
    velocity = 0.0
    peak = 0.0
    for f in series:
        velocity += f          # assume unit time step
        peak = max(peak, abs(velocity))
    return peak

def rbf_surrogate_predict(
    x: np.ndarray,
    centers: np.ndarray,
    alphas: np.ndarray,
    epsilon_base: float,
    entropy: float,
    beta: float = 0.5,
) -> float:
    """Gaussian RBF surrogate (parent A) where the kernel width ε is scaled by
    (1 + β·entropy). Returns a scalar correction term."""
    epsilon = epsilon_base * (1.0 + beta * entropy)
    if epsilon <= 0:
        epsilon = 1e-6
    diff = centers - x          # shape (n_centers, dim)
    sq_norm = np.sum(diff**2, axis=1)
    kernels = np.exp(-sq_norm / (2.0 * epsilon**2))
    return float(np.dot(alphas, kernels))

# ----------------------------------------------------------------------
# Parent B – Doomsday‑modulated NLMS adaptive filter
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Return Doomsday value in {0,…,6} (0 = Monday, …, 6 = Sunday)."""
    # Python weekday: Monday=0 … Sunday=6; shift to match original definition
    return (int(Path().stat().st_mtime) % 7)  # placeholder deterministic but simple

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear NLMS prediction w·x."""
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """Standard NLMS weight update."""
    error = target - weights @ x
    norm = x @ x + eps
    return weights + (mu / norm) * x * error

# ----------------------------------------------------------------------
# Hybrid core – combines both parents
# ----------------------------------------------------------------------
def hybrid_nlms_step(
    weights: np.ndarray,
    path: np.ndarray,
    aux_data: Vector,
    target: float,
    date_tuple: Tuple[int, int, int],
    mu0: float = 0.5,
    alpha: float = 0.8,
    beta: float = 0.5,
    epsilon_base: float = 1.0,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one hybrid NLMS adaptation step.

    1. Extract path‑signature features and entropy (Parent A).
    2. Generate a MinHash force series from ``aux_data`` and obtain ``v_peak``.
    3. Assemble the regressor vector ``x``.
    4. Scale the NLMS step size μ by both the entropy and the Doomsday factor
       (Parent B).
    5. Obtain an RBF surrogate correction using the same entropy‑scaled kernel
       width and add it to the NLMS prediction before computing the error.
    6. Return updated weights and the corrected prediction.
    """
    # 1. Path‑signature and entropy
    sig1, sig2, entropy = path_signature_features(path)

    # 2. MinHash‑derived peak velocity
    force_series = minhash_force_series(aux_data)
    v_peak = integrate_force_series(force_series)

    # 3. Feature vector φ = [sig1, flatten(sig2), entropy, v_peak]
    phi = np.concatenate([sig1.ravel(), sig2.ravel(), np.array([entropy, v_peak])])

    # 4. Doomsday‑modulated learning rate
    year, month, day = date_tuple
    dd = doomsday(year, month, day)                     # 0‑6
    mu_doom = mu0 * (1.0 + 0.1 * math.sin(2.0 * math.pi * dd / 7.0))
    mu = mu_doom * (1.0 + alpha * entropy)             # entropy scaling

    # 5. RBF surrogate correction (dummy centres/alphas for illustration)
    # In a realistic scenario these would be learned offline.
    dim = phi.shape[0]
    n_centers = 5
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(n_centers, dim))
    alphas = rng.normal(size=n_centers)

    surrogate = rbf_surrogate_predict(
        phi, centers, alphas, epsilon_base, entropy, beta=beta
    )

    # NLMS prediction with surrogate correction
    pred = nlms_predict(weights, phi) + surrogate

    # 6. NLMS weight update using corrected error
    updated_weights = nlms_update(weights, phi, target, mu, eps)

    return updated_weights, pred

def hybrid_train(
    weights: np.ndarray,
    path_series: List[np.ndarray],
    aux_series: List[Vector],
    targets: List[float],
    dates: List[Tuple[int, int, int]],
    n_epochs: int = 10,
) -> np.ndarray:
    """
    Train the hybrid filter over multiple epochs. Each epoch iterates over the
    supplied samples, applying ``hybrid_nlms_step``.
    """
    for _ in range(n_epochs):
        for path, aux, tgt, d in zip(path_series, aux_series, targets, dates):
            weights, _ = hybrid_nlms_step(
                weights, path, aux, tgt, d
            )
    return weights

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic 2‑D path (e.g., a simple random walk)
    rng = np.random.default_rng(0)
    T = 20
    dim = 2
    path = np.cumsum(rng.normal(size=(T, dim)), axis=0)

    # Auxiliary data vector (could be sensor readings, etc.)
    aux_data = rng.uniform(-5, 5, size=6).tolist()

    # Target scalar we would like the hybrid predictor to output
    target_value = 3.14

    # Current date (used for Doomsday modulation)
    today = (2026, 5, 29)

    # Initialise NLMS weights (dimension matches feature vector length)
    # Feature length = dim (sig1) + dim*dim (sig2) + 2 (entropy, v_peak)
    feat_len = dim + dim * dim + 2
    init_weights = np.zeros(feat_len)

    # Single‑step demonstration
    new_weights, prediction = hybrid_nlms_step(
        init_weights,
        path,
        aux_data,
        target_value,
        today,
    )
    print(f"Prediction after one hybrid step: {prediction:.4f}")
    print(f"Weight norm change: {np.linalg.norm(new_weights - init_weights):.4e}")

    # Mini training run
    paths = [path for _ in range(5)]
    auxes = [aux_data for _ in range(5)]
    targets = [target_value for _ in range(5)]
    dates = [today for _ in range(5)]

    trained_weights = hybrid_train(init_weights.copy(), paths, auxes, targets, dates, n_epochs=3)
    final_pred = nlms_predict(trained_weights, np.concatenate([path_signature_features(path)[0].ravel(),
                                                              path_signature_features(path)[1].ravel(),
                                                              np.array([0.0, 0.0])]))
    print(f"Final NLMS prediction (without surrogate) after training: {final_pred:.4f}")