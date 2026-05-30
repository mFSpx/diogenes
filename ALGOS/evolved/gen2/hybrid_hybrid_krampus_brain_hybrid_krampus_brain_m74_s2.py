# DARWIN HAMMER — match 74, survivor 2
# gen: 2
# parent_a: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s3.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s1.py (gen1)
# born: 2026-05-29T23:25:36Z

"""
Hybrid Krampus‑BrainMap – Ollivier‑Ricci Curvature Fusion
=======================================================

Parent A (krampus_brainmap) provides a *deterministic* feature extraction based on a
hash‑seeded ``random.Random`` instance, yielding a reproducible high‑dimensional
feature vector for a given text.  
Parent B (ollivier_ricci_curvature) supplies a *stochastic* extraction using the
global ``random.random`` generator and, conceptually, the Ollivier‑Ricci curvature
framework for analysing the geometry of a graph of such vectors.

The mathematical bridge is built by:

1. **Feature Fusion** – each raw feature appears twice (deterministic and stochastic);
   we combine them with a tunable weight ``alpha`` to obtain a single master vector.
2. **Metric Embedding** – the set of master vectors for a collection of texts is
   interpreted as points in ℝⁿ. Pairwise Euclidean distances form the metric tensor
   of the embedded “brain‑map graph”.
3. **Ollivier‑Ricci Curvature Approximation** – for two nodes *i* and *j* we define
   one‑step lazy random‑walk measures μᵢ, μⱼ (uniform over the neighbours within a
   distance cutoff). The 1‑Wasserstein distance W₁(μᵢ, μⱼ) is approximated by the
   average absolute difference of the neighbour vectors. The curvature κ(i,j) is

       κ(i,j) = 1 - W₁(μᵢ, μⱼ) / d(i,j)

   where d(i,j) is the Euclidean distance between the points. This yields a
   curvature matrix that simultaneously captures the deterministic brain‑map
   topology and the stochastic Ollivier‑Ricci perspective.

The module implements the full pipeline with three public functions and a
self‑contained smoke test.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# 1. FEATURE EXTRACTION & FUSION
# ----------------------------------------------------------------------


def _deterministic_features(text: str) -> Dict[str, float]:
    """Parent‑A style extraction – reproducible via hash‑seeded Random."""
    rnd = random.Random(hash(text))
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


def _stochastic_features(text: str) -> Dict[str, float]:
    """Parent‑B style extraction – global random state."""
    # The same key set as in parent B (a subset of A) – we reuse the full set
    # to keep the dimensionality identical.
    rnd = random.random  # shortcut
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
    return {k: rnd() for k in keys}


def extract_master_vector_hybrid(text: str, alpha: float = 0.5) -> Dict[str, float]:
    """
    Fuse deterministic and stochastic feature maps.

    Parameters
    ----------
    text : str
        Input document.
    alpha : float, optional
        Weight of deterministic component (0 ≤ α ≤ 1). The stochastic weight
        is ``1‑α``. Default is 0.5 (equal contribution).

    Returns
    -------
    dict[str, float]
        A reduced master vector containing the most salient dimensions.
    """
    if not text.strip():
        return {}

    det = _deterministic_features(text)
    sto = _stochastic_features(text)

    # Combine with linear interpolation.
    fused: Dict[str, float] = {}
    for key in det.keys():
        fused[key] = alpha * det[key] + (1.0 - alpha) * sto.get(key, 0.0)

    # Map to the concise naming used by both parents.
    return {
        "visceral_ratio": fused["operator_visceral_ratio"],
        "tech_ratio": fused["operator_tech_ratio"],
        "legal_osint_ratio": fused["operator_legal_osint_ratio"],
        "ledger_density": fused["operator_ledger_density"],
        "recursion_score": fused["operator_recursion_score"],
        "directive_ratio": fused["operator_directive_ratio"],
        "target_density": fused["operator_target_density"],
        "forensic_shield_ratio": fused["psyche_forensic_shield_ratio"],
        "poetic_entropy": fused["psyche_poetic_entropy"],
        "dissociative_index": fused["psyche_dissociative_index"],
        "wrath_velocity": fused["psyche_wrath_velocity"],
        "bureaucratic_weaponization_index": fused["resilience_bureaucratic_weaponization_index"],
        "resource_exhaustion_metric": fused["resilience_resource_exhaustion_metric"],
        "swarm_orchestration_density": fused["resilience_swarm_orchestration_density"],
        "logic_crucifixion_index": fused["resilience_logic_crucifixion_index"],
        "conspiracy_grounding_ratio": fused["resilience_conspiracy_grounding_ratio"],
        "chaotic_good_tax": fused["resilience_chaotic_good_tax"],
        "corporate_grit_tension": fused["rainmaker_corporate_grit_tension"],
        "countdown_density": fused["rainmaker_countdown_density"],
        "asset_structuring_weight": fused["rainmaker_asset_structuring_weight"],
        "pitch_formatting_ratio": fused["rainmaker_pitch_formatting_ratio"],
        "agent_symmetry_ratio": fused["telemetry_agent_symmetry_ratio"],
        "protocol_discipline": fused["telemetry_protocol_discipline"],
        "manic_velocity": fused["telemetry_manic_velocity"],
    }


def construct_feature_matrix(texts: List[str], alpha: float = 0.5) -> np.ndarray:
    """
    Build an (N, D) matrix where each row is a hybrid master vector.

    Parameters
    ----------
    texts : list[str]
        Collection of documents.
    alpha : float, optional
        Fusion weight passed to :func:`extract_master_vector_hybrid`.

    Returns
    -------
    np.ndarray
        Float matrix of shape (len(texts), D).
    """
    vectors = [extract_master_vector_hybrid(t, alpha) for t in texts]
    if not vectors:
        return np.empty((0, 0), dtype=float)

    # Preserve ordering of keys for reproducibility.
    keys = list(vectors[0].keys())
    mat = np.array([[v[k] for k in keys] for v in vectors], dtype=float)
    return mat


# ----------------------------------------------------------------------
# 2. OLLIVIER‑RICCI CURVATURE APPROXIMATION
# ----------------------------------------------------------------------


def _pairwise_euclidean(mat: np.ndarray) -> np.ndarray:
    """Efficient Euclidean distance matrix for rows of ``mat``."""
    sq = np.sum(mat ** 2, axis=1, keepdims=True)
    # Using broadcasting: ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
    dist_sq = sq + sq.T - 2.0 * mat @ mat.T
    # Numerical safety
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return np.sqrt(dist_sq)


def compute_ollivier_ricci_curvature(
    feature_mat: np.ndarray,
    radius: float = 1.0,
    lazy_prob: float = 0.2,
) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature on the point cloud defined by
    ``feature_mat``.

    The algorithm:

    1. Compute the Euclidean distance matrix *D*.
    2. For each point *i*, define a lazy random‑walk measure μᵢ:
       - With probability ``lazy_prob`` stay at *i*.
       - Otherwise distribute uniformly over all neighbours *j* with D[i,j] ≤ radius.
    3. Approximate the 1‑Wasserstein distance W₁(μᵢ, μⱼ) by the average absolute
       difference of the neighbour vectors weighted by the measures.
    4. Return κ(i,j) = 1 − W₁ / D[i,j] (with κ = 0 when D[i,j] = 0).

    Parameters
    ----------
    feature_mat : np.ndarray
        Shape (N, D) – rows are embedded points.
    radius : float, optional
        Distance threshold to consider a neighbour. Default 1.0.
    lazy_prob : float, optional
        Probability of staying at the current node in the lazy walk. Default 0.2.

    Returns
    -------
    np.ndarray
        Curvature matrix of shape (N, N).
    """
    if feature_mat.ndim != 2:
        raise ValueError("feature_mat must be a 2‑D array")

    N = feature_mat.shape[0]
    if N == 0:
        return np.empty((0, 0), dtype=float)

    D = _pairwise_euclidean(feature_mat)

    # Build neighbourhood lists respecting the radius.
    neighbourhoods: List[np.ndarray] = []
    for i in range(N):
        neigh = np.where((D[i] <= radius) & (D[i] > 0.0))[0]
        neighbourhoods.append(neigh)

    # Pre‑compute measures μᵢ as vectors of length N.
    mu = np.zeros((N, N), dtype=float)
    for i, neigh in enumerate(neighbourhoods):
        stay = lazy_prob
        move = 1.0 - stay
        if neigh.size == 0:
            # No neighbours: all probability stays.
            mu[i, i] = 1.0
            continue
        prob_per_neighbor = move / neigh.size
        mu[i, i] = stay
        mu[i, neigh] = prob_per_neighbor

    # Approximate W₁(μᵢ, μⱼ) using the linear programming relaxation:
    # W₁ ≈ Σ_k |μᵢ[k] - μⱼ[k]| * avg_norm, where avg_norm is the mean distance
    # from node k to all others (a cheap proxy).
    avg_norm = np.mean(D, axis=1)  # shape (N,)

    # Broadcast differences.
    diff = np.abs(mu[:, None, :] - mu[None, :, :])  # shape (N, N, N)
    # Weight by avg_norm (broadcast over the last axis).
    weighted = diff * avg_norm[None, None, :]

    # Sum over the measure dimension to obtain an estimate of W₁.
    W1 = np.sum(weighted, axis=2)

    # Guard against division by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = 1.0 - np.where(D > 0, W1 / D, 0.0)

    # Clamp curvature to [-1, 1] for numerical stability.
    np.clip(curvature, -1.0, 1.0, out=curvature)
    return curvature


# ----------------------------------------------------------------------
# 3. HIGH‑LEVEL HYBRID OPERATION
# ----------------------------------------------------------------------


def hybrid_brainmap_ricci(
    texts: List[str],
    alpha: float = 0.5,
    radius: float = 1.0,
    lazy_prob: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End‑to‑end pipeline:

    * Extract hybrid master vectors from ``texts``.
    * Assemble the feature matrix.
    * Compute the Ollivier‑Ricci curvature matrix.

    Returns
    -------
    (feature_matrix, curvature_matrix)
        ``feature_matrix`` – shape (N, D)
        ``curvature_matrix`` – shape (N, N)
    """
    feature_mat = construct_feature_matrix(texts, alpha)
    curvature_mat = compute_ollivier_ricci_curvature(
        feature_mat, radius=radius, lazy_prob=lazy_prob
    )
    return feature_mat, curvature_mat


# ----------------------------------------------------------------------
# 4. SMOKE TEST
# ----------------------------------------------------------------------


if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a galaxy far, far away, the stars whisper secrets.",
        "Quantum entanglement blurs the line between observer and observed.",
        "Artificial intelligence reflects humanity's deepest anxieties.",
        "",
        "   ",  # whitespace only – should produce an empty vector.
    ]

    # Run the hybrid pipeline.
    features, curvature = hybrid_brainmap_ricci(
        sample_texts, alpha=0.6, radius=2.5, lazy_prob=0.15
    )

    # Basic sanity checks – they will raise if shapes are inconsistent.
    assert features.shape[0] == len(sample_texts)
    assert curvature.shape == (len(sample_texts), len(sample_texts))

    # Print a concise summary.
    np.set_printoptions(precision=3, suppress=True)
    print("Feature matrix (first two rows):")
    print(features[:2])
    print("\nCurvature matrix:")
    print(curvature)