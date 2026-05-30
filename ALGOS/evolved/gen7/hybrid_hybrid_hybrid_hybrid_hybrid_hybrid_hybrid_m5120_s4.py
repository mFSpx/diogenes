# DARWIN HAMMER — match 5120, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2623_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m1980_s2.py (gen5)
# born: 2026-05-29T23:59:57Z

"""
Hybrid Fusion of Temperature‑Dependent State Transition (Parent A) and
Ternary‑Morphology Modulation (Parent B).

Mathematical Bridge
-------------------
Parent A provides a 2 × 2 temperature‑dependent state‑transition matrix **S**
derived from the Schoolfield developmental rate.  Parent B supplies a ternary
mask **t**∈{-1,0,1}^K (K = 3) and a continuous morphology vector **m**∈ℝ^K.
The fusion interprets each pair *(t_i , m_i)* as a 2‑dimensional state vector

    **v_i** = [ t_i , m_i ]ᵀ .

For a given temperature **T**, the matrix **S(T)** acts on every **v_i**:

    **v′_i** = **S(T)** · **v_i** .

The Ollivier‑Ricci curvature of **S(T)** yields a weighting matrix **C**,
from which the diagonal entries **c_i = C_{ii}** are extracted.  These
curvature‑derived scalars weight the element‑wise product of the ternary
mask and the morphology features, producing a temperature‑aware regression
score

    ŷ = Σ_{i=1}^{K} c_i · (t_i · m_i) + b .

Thus the temperature‑dependent dynamics of Parent A modulate the sparse
selection logic of Parent B, yielding a single unified hybrid algorithm.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

__all__ = [
    "SchoolfieldParams",
    "developmental_rate",
    "temperature_dependent_state_transition",
    "ollivier_ricci_curvature",
    "generate_ternary_vector",
    "extract_morphology_features",
    "curvature_weights",
    "hybrid_predict",
    "hybrid_state_update",
    "compute_curvature_weighted_score",
]

# ----------------------------------------------------------------------
# Parent A – Temperature dependent state transition
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
    """Schoolfield equation for a temperature‑dependent developmental rate."""
    if temp_k <= 0:
        raise ValueError("Temperature must be greater than 0 K")
    rate = params.rho_25 * np.exp(
        -(params.delta_h_activation / params.r_cal) *
        (1.0 / temp_k - 1.0 / 298.15)
    )
    # Clamp to [0,1] for use as a probability‑like transition weight
    return max(0.0, min(1.0, rate))

def temperature_dependent_state_transition(
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams()
) -> np.ndarray:
    """Build the 2×2 transition matrix S(T) used by both parents."""
    rate = developmental_rate(temp_k, params)
    # Simple symmetric stochastic matrix; rows sum to 1
    S = np.array([[rate, 1.0 - rate],
                  [1.0 - rate, rate]])
    return S

def ollivier_ricci_curvature(state_transition_matrix: np.ndarray) -> np.ndarray:
    """Element‑wise Ollivier‑Ricci curvature for a stochastic matrix."""
    n = state_transition_matrix.shape[0]
    curvature = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pij = state_transition_matrix[i, j]
            pji = state_transition_matrix[j, i]
            # Avoid log(0) by adding a tiny epsilon
            eps = np.finfo(float).eps
            curvature[i, j] = pij * math.log((pij + eps) / (pji + eps))
    return curvature

# ----------------------------------------------------------------------
# Parent B – Ternary mask and morphology descriptors
# ----------------------------------------------------------------------
TERNARY_DIMS = 12          # full ternary vector length
SELECT_DIM = 3            # number of components used for masking

def _int_to_ternary(val: int, length: int) -> np.ndarray:
    """Convert a non‑negative integer to a ternary vector of given length."""
    tern = np.zeros(length, dtype=int)
    for i in range(length):
        tern[i] = (val // (3 ** i)) % 3 - 1   # map {0,1,2} → {-1,0,1}
    return tern

def generate_ternary_vector(
    raw_command: str,
    normalized_intent: str,
    context: dict,
    dims: int = TERNARY_DIMS
) -> np.ndarray:
    """
    Deterministic ternary vector derived from a SHA‑256 hash of the command
    envelope.  The hash is interpreted as a large integer which is then
    expanded in base‑3 to obtain values in {-1,0,1}.
    """
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    json_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(json_bytes).hexdigest()
    integer = int(digest, 16)
    ternary = _int_to_ternary(integer, dims)
    return ternary

def extract_morphology_features(dimensions: np.ndarray) -> np.ndarray:
    """
    Compute three simple morphology descriptors from a 3‑D size vector:
    - sphericity  (ratio of inscribed to circumscribed sphere)
    - flatness    (ratio of smallest to largest axis)
    - righting_time (mocked as inverse of volume)
    Returns a vector of length SELECT_DIM.
    """
    if dimensions.shape != (3,):
        raise ValueError("dimensions must be a length‑3 array")
    a, b, c = np.sort(dimensions)  # a ≤ b ≤ c
    volume = a * b * c + 1e-9
    sphericity = (a / c)  # simplistic proxy
    flatness = (a / c)
    righting_time = 1.0 / volume
    return np.array([sphericity, flatness, righting_time])

def curvature_weights(S: np.ndarray) -> np.ndarray:
    """
    Compute curvature matrix C for S and return its diagonal as a weight vector.
    For a 2×2 S the diagonal yields two weights; we repeat them to match SELECT_DIM.
    """
    C = ollivier_ricci_curvature(S)
    diag = np.diag(C)
    # Broadcast to SELECT_DIM (K=3) by simple repetition
    repeats = int(np.ceil(SELECT_DIM / len(diag)))
    weights = np.tile(diag, repeats)[:SELECT_DIM]
    return weights

# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def hybrid_predict(
    raw_command: str,
    normalized_intent: str,
    context: dict,
    dimensions: np.ndarray,
    temp_k: float,
    w: np.ndarray,
    b: float,
    params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    Full hybrid prediction:
    1. Build ternary vector t and take its first SELECT_DIM entries.
    2. Extract morphology vector m.
    3. Form element‑wise product x = t[:K] ⊙ m.
    4. Compute temperature‑dependent transition matrix S(T).
    5. Derive curvature‑based weights c = curvature_weights(S).
    6. Return regression score ŷ = Σ_i c_i·w_i·x_i + b.
    """
    t_full = generate_ternary_vector(raw_command, normalized_intent, context)
    t = t_full[:SELECT_DIM].astype(float)          # shape (K,)
    m = extract_morphology_features(dimensions)    # shape (K,)
    x = t * m                                      # element‑wise product
    S = temperature_dependent_state_transition(temp_k, params)
    c = curvature_weights(S)                       # shape (K,)
    if w.shape != (SELECT_DIM,):
        raise ValueError("Weight vector w must have length SELECT_DIM")
    y_hat = np.dot(c * w, x) + b
    return float(y_hat)

def hybrid_state_update(
    state: np.ndarray,
    raw_command: str,
    normalized_intent: str,
    context: dict,
    dimensions: np.ndarray,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams()
) -> np.ndarray:
    """
    Update a generic 2‑dimensional state vector using the temperature‑dependent
    matrix S(T) after it has been perturbed by the ternary‑modulated morphology.
    The perturbation vector p is constructed as the mean of the transformed
    (t_i , m_i) pairs.
    """
    if state.shape != (2,):
        raise ValueError("state must be a length‑2 vector")
    t_full = generate_ternary_vector(raw_command, normalized_intent, context)
    t = t_full[:SELECT_DIM].astype(float)
    m = extract_morphology_features(dimensions)
    # Build K transformed 2‑vectors and average them
    pairs = np.stack([t, m], axis=1)               # shape (K,2)
    S = temperature_dependent_state_transition(temp_k, params)
    transformed = pairs @ S.T                      # apply S to each pair
    p = transformed.mean(axis=0)                   # perturbation vector (2,)
    new_state = S @ state + p
    return new_state

def compute_curvature_weighted_score(
    raw_command: str,
    normalized_intent: str,
    context: dict,
    dimensions: np.ndarray,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    A utility that isolates the curvature‑weighted inner product
    Σ_i c_i·(t_i·m_i).  Useful for diagnostics or as a feature for downstream
    models.
    """
    t_full = generate_ternary_vector(raw_command, normalized_intent, context)
    t = t_full[:SELECT_DIM].astype(float)
    m = extract_morphology_features(dimensions)
    S = temperature_dependent_state_transition(temp_k, params)
    c = curvature_weights(S)
    return float(np.dot(c, t * m))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Mock inputs
    raw_cmd = "activate sensor array"
    intent = "activate"
    ctx = {"user": "operator", "timestamp": datetime.now(timezone.utc).isoformat()}
    dims = np.array([0.12, 0.08, 0.15])   # meters
    temperature = 298.15                 # K (25 °C)

    # Random regression parameters
    w_vec = np.random.randn(SELECT_DIM)
    bias = np.random.randn()

    # Run hybrid functions
    score = hybrid_predict(
        raw_cmd, intent, ctx, dims, temperature, w_vec, bias
    )
    print(f"Hybrid prediction score: {score:.4f}")

    init_state = np.array([0.5, -0.3])
    updated_state = hybrid_state_update(
        init_state, raw_cmd, intent, ctx, dims, temperature
    )
    print(f"Updated state: {updated_state}")

    curvature_score = compute_curvature_weighted_score(
        raw_cmd, intent, ctx, dims, temperature
    )
    print(f"Curvature‑weighted inner product: {curvature_score:.4f}")