# DARWIN HAMMER — match 5579, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_nlms_omni_cha_m142_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1915_s1.py (gen5)
# born: 2026-05-30T00:02:59Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_endpoi_hybrid_nlms_omni_cha_m142_s1.py
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1915_s1.py

Mathematical Bridge
-------------------
The fusion is built on three intersecting concepts:

1. **Morphology‑Driven Priority** – The sphericity and flatness indices
   (Parent A) are collapsed into a scalar priority `p ∈ (0,1]`. This priority
   rescales the NLMS learning rate `μ`, allowing the adaptive filter to
   react faster for “compact” morphologies and slower for “flat” ones.

2. **Linguistic Style Modulation** – The LSM vector (Parent B) provides a
   per‑feature weighting `w_lsm`.  The pheromone decay signal is multiplied
   by the dot‑product `w_lsm·w_pos` (positive feature weights) and reduced
   by the dot‑product with `w_neg` (negative feature weights).  This yields a
   single scalar `σ` that modulates the NLMS error term, injecting the
   linguistic context into the adaptive update.

3. **KAN‑Transformed Memory Interaction** – A simplified Kolmogorov‑Arnold
   Network (KAN) transformation is applied to an external memory matrix.
   The transformed matrix `M̂` is multiplied with the NLMS weight vector,
   feeding back a learned representation into the error calculation.

The resulting hybrid step is:


e   = d - x·w                     # prediction error
σ   = pheromone_signal(...) * (w_lsm·w_pos - w_lsm·w_neg)
μ'  = μ * p                       # morphology‑scaled step size
w←  = w + (μ' / (||x||² + ε)) * σ * e * x
w←  = M̂·w                         # KAN‑memory projection


All components respect the circuit‑breaker state: when the breaker is
open, weight updates are suppressed.  The implementation below provides
the full hybrid pipeline with three public functions demonstrating the
core operations."""

import numpy as np
import math
import random
import sys
from pathlib import Path
from datetime import datetime

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.utcnow().isoformat()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.utcnow().isoformat()

    def allow(self) -> bool:
        return not self.open


class Morphology:
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
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


def lsm_vector(text: str) -> np.ndarray:
    """Return a frequency vector aligned with _FEATURE_ORDER."""
    vocab = set(text.split())
    cnt = {w: text.count(w) for w in vocab}
    total = sum(cnt.values())
    # Simple mapping: if a feature word appears, count its frequency, else zero.
    vec = np.zeros(len(_FEATURE_ORDER), dtype=np.float64)
    for i, feat in enumerate(_FEATURE_ORDER):
        vec[i] = cnt.get(feat, 0) / total if total > 0 else 0.0
    return vec


def calculate_pheromone_signal(
    signal_value: float,
    half_life_seconds: float,
    elapsed_time: float,
) -> float:
    """Exponential‑like linear decay used in Parent B."""
    decay = max(0.0, 1.0 - (elapsed_time / half_life_seconds))
    return signal_value * decay


def kan_transformed_matrix(
    memory_matrix: np.ndarray,
    coefficients: np.ndarray,
    grids: np.ndarray,
) -> np.ndarray:
    """
    Simplified KAN transformation.
    For each column j we compute a weighted sum of cubic B‑spline basis
    functions evaluated at the grid points.  The implementation uses only
    NumPy primitives.
    """
    if memory_matrix.shape[0] != len(grids):
        raise ValueError("grid length must match memory matrix rows")
    # Cubic B‑spline basis (truncated power) φ(x) = (x - g)^3_+
    def spline_basis(x, g):
        diff = x - g
        return diff ** 3 if diff > 0 else 0.0

    transformed = np.empty_like(memory_matrix, dtype=np.float64)
    for col in range(memory_matrix.shape[1]):
        col_vec = memory_matrix[:, col]
        accum = 0.0
        for k, coeff in enumerate(coefficients):
            basis_vals = np.array([spline_basis(val, grids[k]) for val in col_vec])
            accum += coeff * basis_vals
        transformed[:, col] = accum
    return transformed


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def morphology_priority(morph: Morphology) -> float:
    """
    Combine sphericity and flatness into a priority factor p ∈ (0,1].
    Higher sphericity (more compact) yields larger p, while higher flatness
    reduces p.  The result is clipped to a minimum of 0.1 to avoid dead‑lock.
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flat = flatness_index(morph.length, morph.width, morph.height)
    p = sph / (flat + 1e-6)  # avoid division by zero
    p = max(0.1, min(p, 1.0))
    return p


def hybrid_nlms_update(
    weights: np.ndarray,
    input_vec: np.ndarray,
    desired: float,
    mu: float,
    epsilon: float,
    morph: Morphology,
    breaker: EndpointCircuitBreaker,
    lsm_vec: np.ndarray,
    pheromone_factor: float,
) -> np.ndarray:
    """
    Perform one NLMS update with morphology‑scaled step size and
    linguistic‑style modulation.
    """
    if not breaker.allow():
        breaker.record_failure()
        return weights  # no update while breaker is open

    # 1️⃣ Compute prediction and error
    pred = np.dot(input_vec, weights)
    error = desired - pred

    # 2️⃣ Morphology priority rescales μ
    p = morphology_priority(morph)
    mu_eff = mu * p

    # 3️⃣ Linguistic modulation factor σ
    sigma = pheromone_factor * (np.dot(lsm_vec, _POSITIVE_WEIGHTS) - np.dot(lsm_vec, _NEGATIVE_WEIGHTS))

    # 4️⃣ Normalized LMS step
    norm = np.dot(input_vec, input_vec) + epsilon
    step = (mu_eff / norm) * sigma * error
    new_weights = weights + step * input_vec

    # Record success (for the sake of the example we assume update succeeded)
    breaker.record_success()
    return new_weights


def hybrid_process(
    text: str,
    signal_value: float,
    half_life_seconds: float,
    elapsed_time: float,
    memory_matrix: np.ndarray,
    coefficients: np.ndarray,
    grids: np.ndarray,
    morph: Morphology,
    breaker: EndpointCircuitBreaker,
    input_vec: np.ndarray,
    desired: float,
    mu: float = 0.5,
) -> np.ndarray:
    """
    End‑to‑end hybrid pipeline:
    1. Derive LSM vector from `text`.
    2. Compute pheromone signal and apply linguistic weighting → `pheromone_factor`.
    3. Transform `memory_matrix` via KAN → `M̂`.
    4. Initialise random NLMS weights, project them through `M̂`.
    5. Perform a morphology‑aware, breaker‑protected NLMS update.
    Returns the final weight vector.
    """
    # LSM vector (Parent B)
    lsm_vec = lsm_vector(text)

    # Pheromone decay (Parent B) and linguistic modulation
    base_pheromone = calculate_pheromone_signal(signal_value, half_life_seconds, elapsed_time)
    pheromone_factor = base_pheromone  # will be scaled inside update by σ

    # KAN transformation of external memory (Parent B)
    transformed_memory = kan_transformed_matrix(memory_matrix, coefficients, grids)

    # Initialise NLMS weights and embed them via transformed memory
    init_weights = np.random.rand(input_vec.shape[0])
    # Project through memory: treat transformed_memory as a linear map
    projected_weights = transformed_memory @ init_weights[: transformed_memory.shape[1]]

    # Hybrid NLMS update (core fusion of Parent A & B)
    final_weights = hybrid_nlms_update(
        weights=projected_weights,
        input_vec=input_vec,
        desired=desired,
        mu=mu,
        epsilon=1e-6,
        morph=morph,
        breaker=breaker,
        lsm_vec=lsm_vec,
        pheromone_factor=pheromone_factor,
    )
    return final_weights


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Sample morphology
    morph = Morphology(length=2.5, width=1.8, height=0.9, mass=5.0)

    # Circuit breaker
    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # Text for LSM
    sample_text = "evidence planning evidence risk planning outcome evidence"

    # Pheromone parameters
    signal_val = 10.0
    half_life = 30.0
    elapsed = 5.0

    # Memory matrix (small for demo)
    mem_mat = np.random.rand(6, 6)
    coeffs = np.random.rand(3)  # three spline coefficients
    grid_pts = np.linspace(0, 1, 6)

    # NLMS input / desired
    x_vec = np.random.rand(6)
    d_target = 0.7
    mu_base = 0.3

    # Run the hybrid pipeline
    final_w = hybrid_process(
        text=sample_text,
        signal_value=signal_val,
        half_life_seconds=half_life,
        elapsed_time=elapsed,
        memory_matrix=mem_mat,
        coefficients=coeffs,
        grids=grid_pts,
        morph=morph,
        breaker=breaker,
        input_vec=x_vec,
        desired=d_target,
        mu=mu_base,
    )

    print("Final weight vector:", final_w)