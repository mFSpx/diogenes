# DARWIN HAMMER — match 2522, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_tropical_maxp_hybrid_hybrid_geomet_m927_s2.py (gen5)
# parent_b: hybrid_hybrid_krampus_brain_fractional_hdc_m240_s0.py (gen2)
# born: 2026-05-29T23:42:41Z

"""Hybrid Algorithm: Tropical‑Feature Fusion with Fractional Power Binding
=======================================================================

Parent A: *hybrid_hybrid_tropical_maxp_hybrid_hybrid_geomet_m927_s2.py* – provides
max‑plus (tropical) algebra primitives: addition (max), multiplication (sum),
matrix multiplication and tropical polynomial evaluation.

Parent B: *hybrid_hybrid_krampus_brain_fractional_hdc_m240_s0.py* – supplies a
deterministic pseudo‑random feature extractor from text and a fractional‑power
binding operation (analogous to hyper‑dimensional computing).

**Mathematical Bridge**

The feature extractor yields a dense real‑valued vector **f** ∈ ℝⁿ.  
We treat the components of **f** as tropical coefficients and evaluate a
tropical polynomial p(x) = maxᵢ (cᵢ + i·x).  
The scalar result ρ = p(λ) (with λ chosen as the mean of the feature vector)
is then used as a *fractional exponent* in a binding operation

    B(a, b; ρ) = a^{ρ} ⊙ b^{1‑ρ},

where ⊙ denotes element‑wise multiplication.  
Thus the tropical evaluation drives the analog fractional binding, creating a
single unified computation that intertwines both parent topologies.

The module below implements this hybrid pipeline together with three public
functions that showcase the combined behaviour.
"""

import numpy as np
import math
import random
import hashlib
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Parent A – Tropical algebra primitives
# ----------------------------------------------------------------------
def t_add(x, y):
    """Tropical addition (max). Broadcasts like NumPy ufunc."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication (ordinary addition). Broadcasts."""
    return np.add(x, y)


def t_matmul(A, B):
    """
    Tropical matrix multiplication.

    (A ⊗ B)[i, j] = max_k ( A[i, k] + B[k, j] )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # shape (m, p, 1) + (1, p, n) -> (m, p, n) then max over p
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


def t_polyval(coeffs, x):
    """
    Evaluate a tropical polynomial at x.

    p(x) = max_i ( coeffs[i] + i * x )
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float).reshape((-1,) + (1,) * x.ndim)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents * x
    return np.max(terms, axis=0)


# ----------------------------------------------------------------------
# Parent B – Deterministic feature extraction & fractional binding
# ----------------------------------------------------------------------
def _rng_from_text(text: str) -> random.Random:
    """Create a deterministic RNG seeded from a stable SHA‑256 hash of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Deterministic pseudo‑features for demonstration.

    Returns a dictionary with 24 keys whose values are in [0, 10).
    """
    rnd = _rng_from_text(text)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "psyche_wrath_velocity",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index",
        "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ]
    return {k: rnd.random() * 10.0 for k in keys}


def fractional_power_bind(a: np.ndarray, b: np.ndarray, rho: float) -> np.ndarray:
    """
    Fractional‑power binding of two vectors.

    B(a, b; ρ) = a^{ρ} ⊙ b^{1‑ρ}
    The operation is element‑wise; values ≤ 0 are shifted into the positive domain
    to keep the power well‑defined.
    """
    if a.shape != b.shape:
        raise ValueError("Shapes of a and b must match for binding.")
    # Shift to positive domain (add a small epsilon)
    eps = 1e-12
    a_pos = a + eps
    b_pos = b + eps
    return np.multiply(np.power(a_pos, rho), np.power(b_pos, 1.0 - rho))


# ----------------------------------------------------------------------
# Hybrid primitives – the mathematical bridge
# ----------------------------------------------------------------------
def features_to_tropical_coeffs(features: Dict[str, float],
                                degree: int) -> np.ndarray:
    """
    Convert a feature dictionary into a tropical coefficient vector of length
    ``degree + 1``.
    The largest ``degree + 1`` feature values become the coefficients; missing
    entries are filled with -inf (the tropical zero).
    """
    values = np.array(list(features.values()), dtype=float)
    # Sort descending and pick the top (degree+1) values
    top = -np.inf * np.ones(degree + 1, dtype=float)
    if len(values) > 0:
        sorted_vals = np.sort(values)[::-1]  # descending
        count = min(degree + 1, len(sorted_vals))
        top[:count] = sorted_vals[:count]
    return top


def evaluate_feature_polynomial(text: str, degree: int = 5) -> float:
    """
    Extract features from *text*, turn them into tropical coefficients, and
    evaluate the resulting tropical polynomial at the mean of the feature
    values.

    Returns the scalar tropical evaluation ρ.
    """
    feats = extract_full_features(text)
    coeffs = features_to_tropical_coeffs(feats, degree)
    mean_val = float(np.mean(list(feats.values())))
    rho = float(t_polyval(coeffs, mean_val))
    # Clamp to [0, 1] for use as a fractional exponent (optional but stable)
    rho_clamped = 1.0 / (1.0 + math.exp(-rho))  # logistic squashing
    return rho_clamped


def hybrid_process(text: str,
                   A: np.ndarray,
                   B: np.ndarray,
                   degree: int = 5) -> Tuple[np.ndarray, float]:
    """
    Full hybrid pipeline:

    1. Extract a feature vector **f** from *text*.
    2. Perform tropical matrix multiplication C = A ⊗ B.
    3. Project **f** onto C using ordinary matrix‑vector product (the result is
       a real‑valued vector ``proj``).
    4. Compute a fractional exponent ρ via tropical polynomial evaluation.
    5. Bind ``proj`` with the original feature vector using fractional‑power
       binding.

    Returns the bound vector and the exponent ρ.
    """
    # 1. Feature extraction
    feats = extract_full_features(text)
    f_vec = np.array(list(feats.values()), dtype=float)

    # 2. Tropical matrix multiplication
    C = t_matmul(A, B)                     # shape (A_rows, B_cols)

    # 3. Ordinary projection (real arithmetic) – we need compatible shapes
    #    If dimensions mismatch, truncate or pad with zeros.
    if C.shape[1] != f_vec.shape[0]:
        # Align dimensions by truncating or zero‑padding the shorter one.
        min_len = min(C.shape[1], f_vec.shape[0])
        C_adj = C[:, :min_len]
        f_adj = f_vec[:min_len]
    else:
        C_adj = C
        f_adj = f_vec
    proj = C_adj @ f_adj                    # ordinary dot product

    # 4. Tropical polynomial → fractional exponent
    rho = evaluate_feature_polynomial(text, degree)

    # 5. Fractional‑power binding between projection and original features
    #    (broadcast proj to same length as f_adj)
    if proj.shape[0] != f_adj.shape[0]:
        # Pad the shorter vector with zeros to make shapes match.
        max_len = max(proj.shape[0], f_adj.shape[0])
        proj_pad = np.zeros(max_len)
        f_pad = np.zeros(max_len)
        proj_pad[:proj.shape[0]] = proj
        f_pad[:f_adj.shape[0]] = f_adj
        proj = proj_pad
        f_adj = f_pad

    bound = fractional_power_bind(proj, f_adj, rho)
    return bound, rho


# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def tropical_feature_matrix(texts: List[str],
                            rows: int,
                            cols: int,
                            degree: int = 5) -> np.ndarray:
    """
    Build a (rows × cols) tropical matrix where each entry is the tropical
    evaluation ρ of a distinct input text. If the list is shorter than rows*cols,
    remaining entries are set to -inf (tropical zero).
    """
    size = rows * cols
    coeffs = np.full(size, -np.inf, dtype=float)
    for i, txt in enumerate(texts[:size]):
        coeffs[i] = evaluate_feature_polynomial(txt, degree)
    return coeffs.reshape((rows, cols))


def bind_vectors_from_texts(txt_a: str,
                           txt_b: str,
                           degree: int = 5) -> np.ndarray:
    """
    Extract two feature vectors, evaluate their respective tropical exponents,
    and bind the vectors using the average of the two exponents.
    """
    feats_a = np.array(list(extract_full_features(txt_a).values()), dtype=float)
    feats_b = np.array(list(extract_full_features(txt_b).values()), dtype=float)

    rho_a = evaluate_feature_polynomial(txt_a, degree)
    rho_b = evaluate_feature_polynomial(txt_b, degree)
    rho = 0.5 * (rho_a + rho_b)

    # Align lengths
    max_len = max(feats_a.shape[0], feats_b.shape[0])
    a_pad = np.zeros(max_len)
    b_pad = np.zeros(max_len)
    a_pad[:feats_a.shape[0]] = feats_a
    b_pad[:feats_b.shape[0]] = feats_b

    return fractional_power_bind(a_pad, b_pad, rho)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example tropical matrices (small sizes for quick testing)
    rng = np.random.default_rng(42)
    A = rng.integers(-5, 5, size=(3, 4)).astype(float)   # integer entries ok
    B = rng.integers(-5, 5, size=(4, 2)).astype(float)

    sample_text = "The quick brown fox jumps over the lazy dog."

    bound_vec, exponent = hybrid_process(sample_text, A, B, degree=4)
    print("Hybrid bound vector:", bound_vec)
    print("Fractional exponent (ρ):", exponent)

    # Demonstrate matrix builder
    txts = [
        "alpha", "beta", "gamma", "delta",
        "epsilon", "zeta", "eta", "theta"
    ]
    trop_mat = tropical_feature_matrix(txts, rows=2, cols=4, degree=3)
    print("\nTropical feature matrix:\n", trop_mat)

    # Demonstrate pairwise binding
    bound_pair = bind_vectors_from_texts("first text", "second text", degree=3)
    print("\nBound vector from two texts:", bound_pair)