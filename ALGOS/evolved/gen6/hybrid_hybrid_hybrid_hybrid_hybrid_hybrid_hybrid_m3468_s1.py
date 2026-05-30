# DARWIN HAMMER — match 3468, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fracti_m1474_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_hybrid_hybrid_m606_s1.py (gen5)
# born: 2026-05-29T23:50:15Z

"""
Hybrid Algorithm: Fusion of Parent A (Hyperdimensional Decision & Hoeffding) and
Parent B (Privacy‑aware Fisher‑Score JEPA).

Mathematical Bridge
-------------------
1. **Fractional causal estimate** – a scalar derived from a hyperdimensional
   vector (Parent A).  We obtain it as the absolute inner‑product between a
   random hypervector `hv` and a query hypervector `qhv`.  This scalar `c`
   quantifies the strength of a causal relationship.

2. **Hoeffding exponent** – the causal estimate `c` is used as the exponent in a
   Hoeffding‑type concentration bound  

   ``B = exp( -2 * ε² * N**c )``  

   where `N` is the number of samples and `ε` a tolerance.  The bound `B∈(0,1]`
   measures uncertainty and will modulate downstream weights.

3. **Privacy‑aware Fisher weighting** – the Fisher score matrix (Parent B) is
   multiplied by the Hoeffding bound `B` and further scaled by a reconstruction
   risk term `R` that captures privacy exposure:

   ``W = Fisher * B * (1 + R/2)``

4. **JEPA fusion** – the weighted matrix `W` is finally used in the JEPA‑style
   similarity aggregation, completing the hybrid pipeline.

The code below implements the three core functions that embody this bridge,
together with supporting utilities from both parents.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Tuple, Optional

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def random_hv(d: int = 10000, kind: str = "complex", seed: Optional[int] = None) -> np.ndarray:
    """Generate a random hyperdimensional vector."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"Unsupported kind {kind!r}")

def bind(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Bind two hypervectors using element‑wise multiplication in the Fourier domain."""
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))

def unbind(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Unbind a hypervector."""
    FY = np.fft.fft(Y)
    inv_FY = np.conj(FY) / (np.abs(FY) ** 2 + 1e-30)
    return np.fft.ifft(np.fft.fft(Z) * inv_FY)

# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Normalized reconstruction risk in [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hoeffding_bound(sample_count: int, epsilon: float, exponent: float) -> float:
    """
    Hoeffding‑type bound with a fractional exponent.
    B = exp( -2 * ε² * N**c )
    Returns a value in (0, 1].
    """
    if sample_count <= 0:
        return 1.0
    if exponent < 0:
        exponent = 0.0  # avoid exploding denominator
    bound = math.exp(-2.0 * (epsilon ** 2) * (sample_count ** exponent))
    # Clamp to avoid underflow to 0
    return max(bound, sys.float_info.min)

def compute_hybrid_fisher_matrix(
    similarity: np.ndarray,
    center: float,
    width: float,
    privacy_load: np.ndarray,
    epsilon: float = 0.1,
) -> np.ndarray:
    """
    Build a Fisher‑score matrix, modulated by the Hoeffding bound (derived from
    a fractional causal estimate) and the reconstruction risk.
    """
    # 1. Fisher scores for each similarity entry
    vec_fisher = np.vectorize(lambda th: fisher_score(th, center, width))
    fisher_vec = vec_fisher(similarity)

    # 2. Fractional causal estimate: dot product of a random hypervector with a
    #    query hypervector (here we reuse the similarity vector as a proxy).
    hv = random_hv(d=1024, kind="real")
    qhv = similarity.astype(np.complex128)  # treat as a complex probe
    causal_est = abs(np.vdot(hv, qhv))  # scalar in [0, ∞)

    # Normalise causal_est to a reasonable exponent range [0, 2]
    exponent = min(2.0, causal_est / np.linalg.norm(hv))

    # 3. Hoeffding bound
    bound = hoeffding_bound(sample_count=similarity.size, epsilon=epsilon, exponent=exponent)

    # 4. Reconstruction risk term
    risk = reconstruction_risk_score(
        unique_quasi_identifiers=int(privacy_load.sum()),
        total_records=privacy_load.size,
    )

    # 5. Combine: Fisher * bound * (1 + risk/2)
    combined = fisher_vec * bound * (1.0 + risk / 2.0)
    # Return as a diagonal matrix for later dot products
    return np.diag(combined)

def hybrid_jepa_fusion(
    similarity: np.ndarray,
    center: float,
    width: float,
    privacy_load: np.ndarray,
    epsilon: float = 0.1,
) -> np.ndarray:
    """
    JEPA‑style fusion that consumes the hybrid Fisher matrix.
    Output shape matches the input similarity vector.
    """
    W = compute_hybrid_fisher_matrix(similarity, center, width, privacy_load, epsilon)
    # JEPA aggregation: weighted sum of similarities
    return W @ similarity

def hybrid_binding_and_decision(
    input_vec: np.ndarray,
    query_vec: np.ndarray,
    center: float,
    width: float,
    privacy_load: np.ndarray,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full hybrid pipeline:
    1. Bind input and query hypervectors.
    2. Compute similarity (cosine) between bound vector and a reference HV.
    3. Apply JEPA fusion with privacy‑aware Fisher weighting.
    Returns:
        bound_vector, fused_output
    """
    # Step 1: binding
    bound_vec = bind(input_vec, query_vec)

    # Step 2: similarity to a random reference hypervector
    ref_hv = random_hv(d=bound_vec.size, kind="real")
    similarity = np.abs(np.vdot(bound_vec, ref_hv)) / (np.linalg.norm(bound_vec) * np.linalg.norm(ref_hv) + 1e-30)
    # Convert scalar similarity to a vector for the JEPA step
    sim_vec = np.full(privacy_load.shape, similarity, dtype=np.float64)

    # Step 3: JEPA fusion
    fused = hybrid_jepa_fusion(sim_vec, center, width, privacy_load, epsilon)

    return bound_vec, fused

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy data
    dim = 1024
    input_hv = random_hv(d=dim, kind="real", seed=42)
    query_hv = random_hv(d=dim, kind="real", seed=7)

    # Privacy load vector (e.g., counts of quasi‑identifiers per record)
    privacy_load = np.random.randint(0, 5, size=dim).astype(np.float64)

    # Parameters for Fisher scoring
    center = 0.5
    width = 0.2

    # Run the hybrid pipeline
    bound, output = hybrid_binding_and_decision(
        input_vec=input_hv,
        query_vec=query_hv,
        center=center,
        width=width,
        privacy_load=privacy_load,
        epsilon=0.05,
    )

    # Simple sanity checks
    print("Bound vector norm:", np.linalg.norm(bound))
    print("Fusion output shape:", output.shape)
    print("Fusion output sample values:", output[:5])