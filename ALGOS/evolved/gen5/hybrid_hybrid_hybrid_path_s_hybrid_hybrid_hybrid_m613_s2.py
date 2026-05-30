# DARWIN HAMMER — match 613, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m7_s1.py (gen4)
# born: 2026-05-29T23:29:59Z

"""
Hybrid Algorithm: PathSignature‑Entropy‑MinHash‑RBF Surrogate
----------------------------------------------------------------
Parent A: hybrid_path_signature_kan_m30_s3.py – provides lead‑lag transform,
first‑ and second‑order path signatures. Entropy appears implicitly in the
pheromone decay, which we reinterpret as the Shannon entropy of the
signature’s eigen‑spectrum.

Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m7_s1.py – supplies
a MinHash‑based discrete force series, a simple drag‑limited integrator (the
Chelydrid strike), and a radial‑basis‑function (RBF) surrogate model that
maps a feature vector to a target output using a Gaussian kernel.

Mathematical Bridge
-------------------
The bridge is the *entropy* of the path signature.  We compute the normalized
eigen‑values of the level‑2 signature matrix, obtain its Shannon entropy, and
use this scalar to modulate the width (ε) of the Gaussian kernel in the RBF
surrogate.  Simultaneously, the MinHash of an auxiliary data vector is
interpreted as a discrete acceleration series; integrating it yields a peak
velocity which is appended to the signature‑based feature vector.  The final
feature vector

    Φ = [sig₁, flatten(sig₂), H(sig₂), v_peak]

is fed to the RBF surrogate, producing a unified prediction that respects
both parent algorithms’ governing equations.

The implementation below contains three core functions that demonstrate this
fusion:
    * `path_signature_features` – computes level‑1, level‑2 signatures and
      entropy.
    * `force_series_from_minhash` / `integrate_force_series` – generate a
      force series from MinHash and integrate to obtain peak velocity.
    * `rbf_surrogate_predict` – RBF prediction using entropy‑scaled kernel
      width.

All code relies only on numpy, math, random, sys, and pathlib.
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
# Parent A utilities (trimmed to essentials)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running    = path[:-1] - path[0]            # (T‑1, d)
    return running.T @ increments               # (d, d)

# ----------------------------------------------------------------------
# Entropy bridge (derived from Parent A & B)
# ----------------------------------------------------------------------
def signature_entropy(sig2: np.ndarray) -> float:
    """
    Compute Shannon entropy of the normalized eigenvalue spectrum of the
    level‑2 signature matrix.
    """
    # Ensure symmetry for eigen‑decomposition
    sym = (sig2 + sig2.T) / 2.0
    eigvals = np.linalg.eigvalsh(sym)
    # Clip negative eigenvalues caused by numerical noise
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    total = np.sum(eigvals) + 1e-12
    probs = eigvals / total
    # Shannon entropy
    return -np.sum(probs * np.log(probs + 1e-12))

def path_signature_features(path: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Return a concatenated feature vector consisting of:
        - level‑1 signature (d,)
        - flattened level‑2 signature (d*d,)
        - entropy of level‑2 signature (scalar)
    """
    lvl1 = signature_level1(path)                # (d,)
    lvl2 = signature_level2(path)                # (d, d)
    ent  = signature_entropy(lvl2)               # scalar
    features = np.concatenate([lvl1.ravel(), lvl2.ravel(), np.array([ent])])
    return features, ent

# ----------------------------------------------------------------------
# Parent B utilities (MinHash, force integration, RBF surrogate)
# ----------------------------------------------------------------------
def gaussian_kernel(r: float, epsilon: float = 1.0) -> float:
    """Gaussian (RBF) kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))

def minhash_signature(data: Vector, num_perm: int = 64) -> List[int]:
    """
    Very lightweight MinHash: for each permutation we hash the data with a
    different salt and keep the minimum 32‑bit integer.
    """
    hashes = []
    byte_data = np.asarray(data, dtype=float).tobytes()
    for i in range(num_perm):
        hasher = hashlib.sha256()
        hasher.update(i.to_bytes(4, sys.byteorder))
        hasher.update(byte_data)
        # Take first 4 bytes as unsigned int
        h = int.from_bytes(hasher.digest()[:4], sys.byteorder)
        hashes.append(h)
    return hashes

def force_series_from_minhash(minhash: List[int]) -> np.ndarray:
    """
    Map each MinHash integer to a normalized force magnitude in [0, 1).
    """
    arr = np.array(minhash, dtype=float)
    # Simple linear scaling modulo 1
    forces = (arr % 1000) / 1000.0
    return forces

def integrate_force_series(force_series: np.ndarray, dt: float = 0.01) -> float:
    """
    Treat the force series as acceleration over uniform time steps.
    Integrate twice to obtain velocity, return the peak velocity.
    """
    # Velocity = integral of acceleration
    velocity = np.cumsum(force_series) * dt
    # Peak velocity (absolute maximum)
    return float(np.max(np.abs(velocity)))

# ----------------------------------------------------------------------
# RBF surrogate model (Parent B core)
# ----------------------------------------------------------------------
def rbf_surrogate_predict(
    features: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    epsilon: float = 1.0
) -> float:
    """
    Gaussian RBF surrogate:
        y = Σ_i w_i * exp(- (ε * ||x - c_i||)^2 )
    """
    diffs = centers - features  # (n_centers, dim)
    dists = np.linalg.norm(diffs, axis=1)  # (n_centers,)
    kernels = np.exp(-((epsilon * dists) ** 2))
    return float(np.dot(weights, kernels))

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_predict(
    path: np.ndarray,
    aux_data: Vector,
    rbf_centers: np.ndarray,
    rbf_weights: np.ndarray,
    base_epsilon: float = 1.0
) -> float:
    """
    Unified prediction that fuses:
        * Path signature (levels 1 & 2) and its entropy,
        * MinHash‑derived peak velocity,
        * RBF surrogate with entropy‑scaled kernel width.
    """
    # 1. Signature + entropy
    sig_features, ent = path_signature_features(path)

    # 2. MinHash → force series → peak velocity
    mh = minhash_signature(aux_data, num_perm=32)
    forces = force_series_from_minhash(mh)
    v_peak = integrate_force_series(forces)

    # 3. Assemble final feature vector
    final_feat = np.concatenate([sig_features, np.array([v_peak])])

    # 4. Entropy modulates kernel width (larger entropy → broader kernel)
    epsilon = base_epsilon * (1.0 + ent)

    # 5. RBF surrogate prediction
    return rbf_surrogate_predict(final_feat, rbf_centers, rbf_weights, epsilon)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Random 2‑dimensional path (10 time steps)
    np.random.seed(0)
    path = np.cumsum(np.random.randn(10, 2), axis=0)

    # Auxiliary data vector (simulating external observations)
    aux_data = np.random.rand(20).tolist()

    # Build dummy RBF surrogate parameters
    # Feature dimension = level1(d) + d*d + 1 (entropy) + 1 (v_peak)
    d = path.shape[1]
    feat_dim = d + d * d + 2
    n_centers = 5
    centers = np.random.randn(n_centers, feat_dim)
    weights = np.random.randn(n_centers)

    # Run hybrid prediction
    pred = hybrid_predict(path, aux_data, centers, weights, base_epsilon=0.5)
    print(f"Hybrid prediction: {pred:.6f}")