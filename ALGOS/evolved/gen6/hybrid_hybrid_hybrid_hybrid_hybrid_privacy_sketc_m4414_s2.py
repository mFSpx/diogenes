# DARWIN HAMMER — match 4414, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1223_s3.py (gen5)
# parent_b: hybrid_privacy_sketches_m15_s3.py (gen1)
# born: 2026-05-29T23:55:27Z

"""Hybrid Algorithm: Morphology‑to‑Sketch Koopman‑DP Fusion

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1223_s3.py (Morphology vector,
  Koopman operator, fractional power binding)
- hybrid_privacy_sketches_m15_s3.py (Differentially‑private Count‑Min sketch,
  reconstruction‑risk score)

Mathematical Bridge:
The morphology vector `v ∈ ℝ^{dim}` (Parent A) is injected into a Count‑Min
sketch matrix `C ∈ ℕ^{d×w}` (Parent B) by hashing each component of `v` into the
sketch.  The flattened sketch `c = vec(C)` is then transformed by a Koopman
operator matrix `K ∈ ℝ^{dim×dim}` (Parent A) yielding `c' = K·c`.  A fractional
power binding `c'' = sign(c')·|c'|^{α}` (0 < α ≤ 1) injects non‑linearity.
Laplace noise calibrated to a privacy budget `ε` is added element‑wise,
producing a differentially‑private sketch `Ĉ`.  The noisy sketch supplies a
private estimate of the number of distinct quasi‑identifiers, which is fed
into the reconstruction‑risk formula from Parent B.

The three high‑level functions below demonstrate the end‑to‑end hybrid
pipeline."""
import hashlib
import random
import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set

import numpy as np

Vector = List[float]


def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def random_vector(dim: int = 1024, seed: str | int | None = None) -> Vector:
    """Deterministic pseudo‑random vector in [0,1)."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def morphology_vector(
    length: float,
    width: float,
    height: float,
    mass: float,
    dim: int = 1024,
) -> Vector:
    """Encode a simple Morphology tuple into a high‑dimensional vector."""
    # Simple deterministic seeding from the four attributes
    seed = int(
        hashlib.sha256(f"{length:.6f}{width:.6f}{height:.6f}{mass:.6f}".encode()).hexdigest(),
        16,
    )
    return random_vector(dim, seed)


def _hash_indices(value: float, d: int, w: int, seed: int) -> Tuple[int, int]:
    """Map a float to a (row, column) pair for Count‑Min sketch."""
    # Convert float to bytes then hash with a per‑row salt
    b = f"{value:.16f}{seed}".encode()
    h = int(hashlib.sha256(b).hexdigest(), 16)
    row = h % d
    col = (h // d) % w
    return row, col


def build_count_min_sketch(
    vec: Vector, d: int = 5, w: int = 2000, seed: int = 0
) -> np.ndarray:
    """Populate a Count‑Min sketch from a dense vector."""
    C = np.zeros((d, w), dtype=np.float64)
    for i, v in enumerate(vec):
        row, col = _hash_indices(v, d, w, seed + i)
        C[row, col] += 1.0
    return C


def koopman_operator(dim: int, seed: int = 0) -> np.ndarray:
    """Generate a random orthogonal‑like Koopman matrix."""
    rng = np.random.default_rng(seed)
    # Random Gaussian matrix then QR decomposition to obtain an orthogonal matrix
    Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return Q


def apply_koopman(C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Apply Koopman operator to the flattened sketch and reshape."""
    flat = C.ravel()
    transformed = K @ flat
    return transformed.reshape(C.shape)


def fractional_power_binding(C: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Element‑wise signed fractional power: sign·|x|^{α}."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1].")
    sign = np.sign(C)
    magnitude = np.abs(C) ** alpha
    return sign * magnitude


def dp_laplace_noise_matrix(
    C: np.ndarray, epsilon: float, sensitivity: float = 1.0
) -> np.ndarray:
    """Add Laplace(0, sensitivity/ε) noise to each entry."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=C.shape)
    return C + noise


def distinct_estimate_from_sketch(C: np.ndarray) -> int:
    """Estimate number of distinct items via non‑zero columns after min‑over‑rows."""
    # min over rows for each column
    col_mins = C.min(axis=0)
    # count columns with positive estimate
    return int(np.sum(col_mins > 0))


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Deterministic risk ratio clipped to [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def hybrid_morphology_risk(
    length: float,
    width: float,
    height: float,
    mass: float,
    total_records: int,
    dim: int = 1024,
    d: int = 5,
    w: int = 2000,
    alpha: float = 0.5,
    epsilon: float = 1.0,
) -> float:
    """
    End‑to‑end hybrid pipeline:

    1. Encode morphology into a high‑dimensional vector.
    2. Insert the vector into a Count‑Min sketch.
    3. Transform the sketch with a Koopman operator.
    4. Apply fractional power binding.
    5. Add Laplace DP noise.
    6. Estimate distinct quasi‑identifiers from the noisy sketch.
    7. Compute reconstruction risk.
    """
    # Step 1
    v = morphology_vector(length, width, height, mass, dim)

    # Step 2
    C = build_count_min_sketch(v, d, w)

    # Step 3
    K = koopman_operator(d * w)
    C = apply_koopman(C, K)

    # Step 4
    C = fractional_power_binding(C, alpha)

    # Step 5
    C = dp_laplace_noise_matrix(C, epsilon)

    # Step 6
    distinct_est = distinct_estimate_from_sketch(C)

    # Step 7
    return reconstruction_risk_score(distinct_est, total_records)


def hybrid_sketch_transform(
    vec: Vector,
    d: int = 5,
    w: int = 2000,
    alpha: float = 0.7,
    epsilon: float = 0.5,
) -> np.ndarray:
    """
    Produce a differentially‑private, Koopman‑enhanced sketch from an arbitrary vector.
    Returns the noisy sketch matrix ready for downstream approximate counting.
    """
    C = build_count_min_sketch(vec, d, w)
    K = koopman_operator(d * w, seed=42)
    C = apply_koopman(C, K)
    C = fractional_power_binding(C, alpha)
    C = dp_laplace_noise_matrix(C, epsilon)
    return C


def hybrid_estimate_frequency(
    vec: Vector,
    query_value: float,
    d: int = 5,
    w: int = 2000,
    alpha: float = 0.7,
    epsilon: float = 0.5,
) -> float:
    """
    Estimate the frequency of `query_value` using the hybrid sketch.
    The estimate is the minimum count across rows for the hashed column.
    """
    C = hybrid_sketch_transform(vec, d, w, alpha, epsilon)
    row, col = _hash_indices(query_value, d, w, seed=0)
    return float(C[:, col].min())


if __name__ == "__main__":
    # Smoke test: simple morphology with modest record count
    risk = hybrid_morphology_risk(
        length=2.5,
        width=1.2,
        height=0.8,
        mass=3.4,
        total_records=10_000,
        dim=512,
        d=4,
        w=1024,
        alpha=0.6,
        epsilon=1.2,
    )
    print(f"Hybrid reconstruction risk: {risk:.4f}")

    # Demonstrate frequency estimation on a random vector
    test_vec = random_vector(256, seed=123)
    freq_est = hybrid_estimate_frequency(test_vec, query_value=0.42, d=3, w=512)
    print(f"Estimated frequency for 0.42: {freq_est:.2f}")

    # Verify that the sketch matrix is finite and DP‑noisy
    sketch = hybrid_sketch_transform(test_vec, d=3, w=512, alpha=0.8, epsilon=0.8)
    assert np.isfinite(sketch).all()
    print("Hybrid sketch generated successfully.")