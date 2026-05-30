# DARWIN HAMMER — match 25, survivor 4
# gen: 2
# parent_a: hybrid_path_signature_kan_m30_s1.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s4.py (gen1)
# born: 2026-05-29T23:25:14Z

"""Hybrid Path Signature & Feature‑KAN Fusion

Parents:
- hybrid_path_signature_kan_m30_s1.py (Path signature & KAN approximation)
- hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s4.py (Text feature extraction)

Mathematical Bridge:
The feature dictionary extracted from a piece of text is interpreted as a high‑dimensional
vector **v** ∈ ℝⁿ.  By embedding **v** into a synthetic discrete path 𝒫(t) = (t/T)·v,
t = 0,…,T‑1, we can compute the classical iterated‑integral signatures (level‑1 and level‑2)
using the lead‑lag transform from the path‑signature parent.  The resulting signature
tensors are then passed through a lightweight Kolmogorov‑Arnold Network (KAN)‑style
operator: a fixed set of univariate basis functions (identity, square, cube, sinusoid)
combined linearly with learned‑like coefficients.  This provides a differentiable,
non‑linear mapping from raw signatures to a transformed “signature‑feature” space,
thereby fusing the two algorithmic topologies into a single unified system.

The module therefore offers:
1. Extraction of a deterministic feature vector from text.
2. Construction of a synthetic path and computation of its level‑1 and level‑2 signatures.
3. A KAN‑style transformation applied to both signature components.

All operations rely only on NumPy and the Python standard library.
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent B – deterministic feature extraction (adapted)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature extraction based on the text hash."""
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric", "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index", "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax", "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density", "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio", "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline", "telemetry_manic_velocity",
    ]
    return {k: rnd.random() * 10.0 for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    """Select a subset of the full feature set that will be used as the core vector."""
    if not text.strip():
        return {}
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0),
        "directive_ratio": f.get("operator_directive_ratio", 0.0),
        "target_density": f.get("operator_target_density", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "wrath_velocity": f.get("psyche_wrath_velocity", 0.0),
        "bureaucratic_weaponization_index": f.get(
            "resilience_bureaucratic_weaponization_index", 0.0
        ),
        "resource_exhaustion_metric": f.get(
            "resilience_resource_exhaustion_metric", 0.0
        ),
        "swarm_orchestration_density": f.get(
            "resilience_swarm_orchestration_density", 0.0
        ),
        "logic_crucifixion_index": f.get(
            "resilience_logic_crucifixion_index", 0.0
        ),
        "conspiracy_grounding_ratio": f.get(
            "resilience_conspiracy_grounding_ratio", 0.0
        ),
        "chaotic_good_tax": f.get(
            "resilience_chaotic_good_tax", 0.0
        ),
        "corporate_grit_tension": f.get(
            "rainmaker_corporate_grit_tension", 0.0
        ),
        "countdown_density": f.get(
            "rainmaker_countdown_density", 0.0
        ),
        "asset_structuring_weight": f.get(
            "rainmaker_asset_structuring_weight", 0.0
        ),
        "pitch_formatting_ratio": f.get(
            "rainmaker_pitch_formatting_ratio", 0.0
        ),
        "agent_symmetry_ratio": f.get(
            "telemetry_agent_symmetry_ratio", 0.0
        ),
        "protocol_discipline": f.get(
            "telemetry_protocol_discipline", 0.0
        ),
        "manic_velocity": f.get(
            "telemetry_manic_velocity", 0.0
        ),
    }

# ----------------------------------------------------------------------
# Parent A – path‑signature utilities (adapted)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Lead‑lag transform: interleave (lead, lag) channels for causality encoding.

    Parameters
    ----------
    path : np.ndarray, shape (T, d)
        Original discrete path.

    Returns
    -------
    np.ndarray, shape (2*T‑1, 2*d)
        Lead‑lag transformed path.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """Level‑1 signature (total increment)."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path: np.ndarray) -> np.ndarray:
    """Level‑2 iterated‑integral tensor using left‑point Riemann sums."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    # Compute outer products and sum over time
    S2 = np.einsum('ti,tj->ij', running, increments)
    return S2

# ----------------------------------------------------------------------
# KAN‑style univariate basis functions
# ----------------------------------------------------------------------
def _basis_functions(x: np.ndarray) -> np.ndarray:
    """Stack of simple univariate basis functions evaluated at x.

    Returns an array of shape (B, ...) where B = number of bases.
    The bases are:
        φ₀(x) = x
        φ₁(x) = x²
        φ₂(x) = x³
        φ₃(x) = sin(x)
        φ₄(x) = cos(x)
    """
    x = np.asarray(x, dtype=float)
    return np.stack([x,
                     x ** 2,
                     x ** 3,
                     np.sin(x),
                     np.cos(x)], axis=0)


def kan_transform(tensor: np.ndarray, seed: int = 42) -> np.ndarray:
    """Apply a lightweight KAN‑style transformation to any tensor.

    The transformation is a linear combination of the basis functions with
    deterministic coefficients (drawn from a fixed random seed).  This mimics
    the learned spline coefficients of a true KAN while remaining fully
    reproducible without external ML libraries.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor (vector or matrix) to be transformed.
    seed : int
        Random seed for coefficient generation.

    Returns
    -------
    np.ndarray
        Tensor of the same shape as `tensor` after the KAN‑style mapping.
    """
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(loc=0.0, scale=0.1, size=(5,))  # one coeff per basis
    # Broadcast basis evaluation over the tensor shape
    bases = _basis_functions(tensor)                     # shape (5, *tensor.shape)
    transformed = np.tensordot(coeffs, bases, axes=1)    # shape = tensor.shape
    return transformed

# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def text_to_path(text: str) -> np.ndarray:
    """Convert extracted master features into a synthetic discrete path.

    The feature vector is ordered alphabetically, then scaled linearly over
    T = len(vector) time steps:
        P[t] = (t+1)/T * v
    Returns a path of shape (T, d) where d = number of features.
    """
    vec_dict = extract_master_vector(text)
    if not vec_dict:
        raise ValueError("Empty or whitespace‑only text provided.")
    # Deterministic ordering
    keys = sorted(vec_dict.keys())
    v = np.array([vec_dict[k] for k in keys], dtype=float)  # shape (d,)
    T = v.shape[0]
    times = np.arange(1, T + 1, dtype=float) / T
    path = times[:, None] * v[None, :]  # (T, d)
    return path


def compute_hybrid_signature(text: str) -> Dict[str, np.ndarray]:
    """Full hybrid computation: feature → path → signatures → KAN transform.

    Returns a dictionary with keys:
        'level1_raw'  – raw level‑1 signature (vector)
        'level2_raw'  – raw level‑2 signature (matrix)
        'level1_kan'  – KAN‑transformed level‑1
        'level2_kan'  – KAN‑transformed level‑2
    """
    # 1. Feature → path
    path = text_to_path(text)

    # 2. Lead‑lag transform (adds causality encoding)
    ll_path = lead_lag_transform(path)

    # 3. Compute raw signatures
    lvl1 = signature_level1(ll_path)          # shape (2*d,)
    lvl2 = signature_level2(ll_path)          # shape (2*d, 2*d)

    # 4. Apply KAN‑style non‑linear mapping
    lvl1_kan = kan_transform(lvl1, seed=123)
    lvl2_kan = kan_transform(lvl2, seed=456)

    return {
        "level1_raw": lvl1,
        "level2_raw": lvl2,
        "level1_kan": lvl1_kan,
        "level2_kan": lvl2_kan,
    }

def hybrid_signature_vector(text: str) -> np.ndarray:
    """Convenience wrapper returning a flattened hybrid signature vector."""
    hybrid = compute_hybrid_signature(text)
    # Concatenate raw and transformed components for a single descriptor
    return np.concatenate([
        hybrid["level1_raw"].ravel(),
        hybrid["level2_raw"].ravel(),
        hybrid["level1_kan"].ravel(),
        hybrid["level2_kan"].ravel(),
    ])

def similarity_between_texts(text_a: str, text_b: str) -> float:
    """Cosine similarity between the flattened hybrid signatures of two texts."""
    vec_a = hybrid_signature_vector(text_a)
    vec_b = hybrid_signature_vector(text_b)
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return dot / norm if norm != 0 else 0.0

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_a = "The quick brown fox jumps over the lazy dog."
    sample_b = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    hybrid_a = compute_hybrid_signature(sample_a)
    hybrid_b = compute_hybrid_signature(sample_b)

    print("Level‑1 raw (A):", hybrid_a["level1_raw"])
    print("Level‑2 raw (A) shape:", hybrid_a["level2_raw"].shape)
    print("Level‑1 KAN (A):", hybrid_a["level1_kan"])
    print("Similarity(A, B):", similarity_between_texts(sample_a, sample_b))
    # Ensure no exception is raised
    print("Smoke test completed successfully.")