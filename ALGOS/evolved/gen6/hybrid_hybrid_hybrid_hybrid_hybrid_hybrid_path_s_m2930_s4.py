# DARWIN HAMMER — match 2930, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_jepa_e_hybrid_hybrid_hard_t_m1309_s0.py (gen5)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s0.py (gen3)
# born: 2026-05-29T23:46:43Z

"""
Hybrid Algorithm: Text-Driven Path Signature via B‑Spline Modulation

Parents:
- hybrid_hybrid_hybrid_jepa_e_hybrid_hybrid_hard_t_m1309_s0 (Feature extraction & deterministic RNG)
- hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s0 (Lead‑lag transform, B‑spline basis, path signature)

Mathematical Bridge:
The bridge is a *feature‑conditioned B‑spline grid*. Features extracted from a
text string are deterministic (via a SHA‑256 seed) and produce a scalar scaling
vector 𝛾∈ℝⁿ.  This vector modulates the knot locations of the B‑spline basis used
to approximate the iterated‑integral (path signature).  Consequently the
signature adapts to semantic information contained in the text while retaining
the geometric encoding of the original time series via the lead‑lag transform.
"""

import sys
import math
import random
import hashlib
from datetime import date
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return a simple deterministic weekday code (0‑6)."""
    return (date(year, month, day).weekday() + 1) % 7


def _rng_from_text(text: str) -> random.Random:
    """Create a reproducible Random instance from a UTF‑8 string."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def extract_full_features(text: str) -> dict:
    """
    Deterministically generate a dictionary of 20+ pseudo‑features from `text`.
    The values are in the interval [0, 1] and are reproducible for the same
    input string.
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
    ]
    return {k: rnd.random() for k in keys}


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    Input `path` is shape (T, d). Output shape is (2*T‑1, 2*d).
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate B‑spline basis functions of order `k` at positions `x`.
    Returns a matrix B of shape (len(x), n_basis).
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Augment knot vector with clamping knots
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    N = len(x)
    B = np.zeros((N, len(t) - 1), dtype=np.float64)

    # Zeroth order (piecewise constant)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # Recursion for higher orders
    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = 0.0
            if denom_l != 0:
                term_l = ((x - t[i]) / denom_l) * B[:, i]

            term_r = 0.0
            if denom_r != 0:
                term_r = ((t[i + order] - x) / denom_r) * B[:, i + 1]

            B_new[:, i] = term_l + term_r
        B = B_new
    return B[:, :n_basis]


# ----------------------------------------------------------------------
# Hybrid Core
# ----------------------------------------------------------------------
def _grid_from_features(features: dict, base_grid: np.ndarray) -> np.ndarray:
    """
    Create a feature‑conditioned knot grid.
    The grid is scaled by a factor γ = 0.5 + 0.5 * avg(selected_features).
    This keeps the grid within a reasonable range while still reflecting the
    semantic content of the text.
    """
    # Select a subset of features that are numeric and bounded [0,1]
    selected_keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "psyche_poetic_entropy",
        "resilience_logic_crucifixion_index",
        "rainmaker_corporate_grit_tension",
    ]
    values = [features.get(k, 0.0) for k in selected_keys]
    gamma = 0.5 + 0.5 * (sum(values) / len(values))
    # Scale and shift the base grid
    scaled = base_grid * gamma
    # Ensure monotonicity
    return np.sort(scaled)


def compute_hybrid_signature(text: str, path: np.ndarray, order: int = 3) -> dict:
    """
    Compute a text‑conditioned path signature.

    Steps:
    1. Extract deterministic features from `text`.
    2. Build a feature‑modulated knot grid.
    3. Apply the lead‑lag transform to `path`.
    4. Evaluate the B‑spline basis on each dimension of the transformed path.
    5. Approximate the iterated integral by contracting the basis with the
       transformed coordinates (simple Riemann sum).

    Returns a dictionary with:
        - "features": the raw feature dict,
        - "signature": np.ndarray of shape (d, n_basis),
        - "grid": the knot grid used,
        - "gain": the scaling factor γ (for inspection).
    """
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (T, d)")

    # 1. Feature extraction
    features = extract_full_features(text)

    # 2. Feature‑conditioned knot grid
    # Base grid is uniformly spaced in [0, 1]
    base_grid = np.linspace(0.0, 1.0, num=10)  # 10 interior knots
    grid = _grid_from_features(features, base_grid)
    gamma = grid.mean() / base_grid.mean()  # for external reporting

    # 3. Lead‑lag transform
    ll_path = lead_lag_transform(path)  # shape (2T‑1, 2d)
    T_ll, d_ll = ll_path.shape

    # 4. Evaluate B‑spline basis for the temporal axis
    # Use a normalized time vector in [0,1]
    time_vec = np.linspace(0.0, 1.0, num=T_ll)
    B = bspline_basis(time_vec, grid, k=order)  # shape (T_ll, n_basis)

    # 5. Contract basis with each coordinate dimension
    n_basis = B.shape[1]
    signature = np.empty((d_ll, n_basis), dtype=np.float64)
    for dim in range(d_ll):
        # Simple inner product approximates ∫ φ_i(t) * x_dim(t) dt
        signature[dim] = B.T @ ll_path[:, dim] / T_ll

    return {
        "features": features,
        "signature": signature,
        "grid": grid,
        "gain": gamma,
    }


def hybrid_variational_free_energy(text: str, path: np.ndarray) -> float:
    """
    A lightweight surrogate for Variational Free Energy (VFE) that blends
    semantic uncertainty (entropy of extracted features) with geometric
    complexity (norm of the hybrid signature).

    VFE ≈ -H(features) + λ * ||signature||_F
    where λ is a small regularisation constant.
    """
    result = compute_hybrid_signature(text, path)
    feats = result["features"]
    # Shannon entropy of the feature distribution (treated as probabilities)
    probs = np.array(list(feats.values()))
    probs = probs / probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    sig_norm = np.linalg.norm(result["signature"], ord="fro")
    lam = 0.01
    return -entropy + lam * sig_norm


def hybrid_predictive_score(text: str, path: np.ndarray) -> float:
    """
    Produce a scalar score that could be interpreted as a prediction confidence.
    The score combines:
        * the VFE surrogate (lower is better)
        * the gain γ (higher γ → more feature influence)
    The function maps the combination into [0, 1] via a sigmoid.
    """
    vfe = hybrid_variational_free_energy(text, path)
    gain = compute_hybrid_signature(text, path)["gain"]
    raw = -vfe * gain  # negative VFE is better, multiplied by gain
    return 1.0 / (1.0 + math.exp(-raw))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic path (e.g., a 2‑D spiral)
    t = np.linspace(0, 2 * np.pi, 50)
    x = np.stack([np.cos(t), np.sin(t)], axis=1)  # shape (50, 2)

    sample_text = "The quick brown fox jumps over the lazy dog."
    hybrid_result = compute_hybrid_signature(sample_text, x)

    print("Feature sample (first 5):")
    for k, v in list(hybrid_result["features"].items())[:5]:
        print(f"  {k}: {_pct(v)}")

    print("\nSignature shape:", hybrid_result["signature"].shape)
    print("Grid (first 5 knots):", hybrid_result["grid"][:5])
    print("Gain γ:", _pct(hybrid_result["gain"]))

    vfe = hybrid_variational_free_energy(sample_text, x)
    score = hybrid_predictive_score(sample_text, x)
    print("\nSurrogate VFE:", _pct(vfe))
    print("Predictive score:", _pct(score))