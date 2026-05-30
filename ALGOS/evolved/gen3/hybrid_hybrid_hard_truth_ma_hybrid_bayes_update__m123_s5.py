# DARWIN HAMMER — match 123, survivor 5
# gen: 3
# parent_a: hybrid_hard_truth_math_model_pool_m8_s2.py (gen1)
# parent_b: hybrid_bayes_update_hybrid_krampus_brain_m15_s0.py (gen2)
# born: 2026-05-29T23:26:57Z

"""Hybrid module merging hard_truth_math (Parent A) and Bayesian‑Krampus‑Ollivier‑Ricci (Parent B).

Mathematical bridge:
- Parent A yields a high‑dimensional text feature vector **v** ∈ ℝⁿ (stylometry + LSM).
- Parent B yields a low‑dimensional “master” resource vector **m** ∈ ℝᵏ (k≈9) derived from stochastic ratios.
- Compatibility is measured by a bilinear form **s = vᵀ P m**, where **P** extracts the first *k* components of **v** and projects them onto the master space.
- The scalar **s** is interpreted as Bayesian evidence and is used to update an Ollivier‑Ricci‑style curvature matrix **C** that encodes pairwise interactions among the master dimensions:
    **C′ = BayesianUpdate(C, s)**.

The three core functions below implement this pipeline:
1. `hybrid_feature_vector` – builds **v** from text.
2. `compatibility_score` – computes **s = vᵀ P m**.
3. `bayesian_curvature_update` – updates curvature **C** with evidence **s**.

The module can be used to select models under RAM/tier constraints, to analyse curvature dynamics, or as a generic hybrid representation learner.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry & LSM utilities (simplified)
# ----------------------------------------------------------------------
FUNCTION_CATS = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Return a list of lowercase alphabetic tokens."""
    return [w for w in text.lower().split() if w.isalpha()]


def stylometry_features(text: str) -> np.ndarray:
    """
    Produce a deterministic 96‑dimensional stylometry vector.
    For reproducibility we hash the text and use the hash to seed a PRNG.
    """
    h = int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\0"), "little")
    rng = random.Random(h)
    return np.array([rng.random() for _ in range(96)], dtype=float)


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a lightweight semantic‑category (LSM) vector.
    Each category receives the proportion of tokens belonging to it.
    """
    toks = words(text)
    total = len(toks) or 1
    cat_counts = {cat: 0 for cat in FUNCTION_CATS}
    for tok in toks:
        for cat, members in FUNCTION_CATS.items():
            if tok in members:
                cat_counts[cat] += 1
    return {cat: cnt / total for cat, cnt in cat_counts.items()}


def vectorize_lsm(lsm: Dict[str, float]) -> np.ndarray:
    """Order LSM entries alphabetically and return as a column vector."""
    keys = sorted(lsm.keys())
    return np.array([lsm[k] for k in keys], dtype=float)


# ----------------------------------------------------------------------
# Parent B – stochastic master feature extraction (simplified)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Generate a dictionary of random ratios; seeded by text hash for repeatability."""
    h = int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\0"), "little")
    rng = random.Random(h)
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio", "operator_legal_osint_ratio",
        "psyche_forensic_shield_ratio", "psyche_poetic_entropy", "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index", "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density", "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density", "rainmaker_asset_structuring_weight",
        "telemetry_agent_symmetry_ratio", "telemetry_protocol_discipline", "telemetry_manic_velocity",
    ]
    return {k: rng.random() for k in keys}


def extract_master_vector(text: str) -> np.ndarray:
    """
    Reduce the full feature set to a 9‑dimensional master vector.
    The selection mirrors the original parent B implementation.
    """
    f = extract_full_features(text)
    master_keys = [
        "operator_visceral_ratio", "operator_tech_ratio", "operator_legal_osint_ratio",
        "psyche_forensic_shield_ratio", "psyche_poetic_entropy", "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index", "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
    ]
    return np.array([f[k] for k in master_keys], dtype=float)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_feature_vector(text: str) -> np.ndarray:
    """
    Concatenate the stylometry vector (96), the LSM vector (|FUNCTION_CATS|),
    and the master vector (9) into a single high‑dimensional representation.
    """
    v_styl = stylometry_features(text)                     # (96,)
    v_lsm = vectorize_lsm(lsm_vector(text))               # (len(FUNCTION_CATS),)
    v_master = extract_master_vector(text)                # (9,)
    return np.concatenate([v_styl, v_lsm, v_master])


def compatibility_score(text: str, model_resource: Tuple[float, int]) -> float:
    """
    Compute the bilinear compatibility s = vᵀ P m.

    * v – hybrid feature vector.
    * m – model resource vector (RAM_scaled, tier_one_hot).
    * P – projection matrix that extracts the first *k* components of v
          where k equals the length of m (here 2) and maps them linearly.
    """
    v = hybrid_feature_vector(text)                       # shape (N,)
    # Build m: RAM scaled to [0,1] and tier encoded as one‑hot (max tier 5)
    ram_scaled, tier = model_resource
    max_tier = 5
    tier_one_hot = np.zeros(max_tier, dtype=float)
    tier_one_hot[min(tier, max_tier) - 1] = 1.0
    m = np.concatenate([np.array([ram_scaled], dtype=float), tier_one_hot])  # shape (1+max_tier,)

    # Projection matrix P extracts first len(m) entries of v and applies identity scaling.
    k = len(m)
    P = np.zeros((v.shape[0], k), dtype=float)
    P[:k, :] = np.eye(k)

    s = float(v @ P @ m)  # scalar
    # Normalise to [0,1] via sigmoid to serve as Bayesian evidence.
    return 1.0 / (1.0 + math.exp(-s))


def bayesian_curvature_update(curvature: np.ndarray, evidence: float) -> np.ndarray:
    """
    Perform a Bayesian update on a symmetric curvature matrix C.

    For each entry C_ij we treat it as a prior probability and update with
    evidence *e* (the compatibility score) using the rule:

        C'_ij = (C_ij * e) / (C_ij * e + (1 - C_ij) * (1 - e))

    The operation preserves symmetry and keeps values in (0,1).
    """
    e = np.clip(evidence, 1e-9, 1 - 1e-9)  # avoid division by zero
    numerator = curvature * e
    denominator = numerator + (1.0 - curvature) * (1.0 - e)
    updated = numerator / denominator
    # Force symmetry
    return (updated + updated.T) / 2.0


def initialise_curvature(dim: int) -> np.ndarray:
    """
    Initialise a curvature matrix with small random values in (0,1).
    The matrix is symmetric with ones on the diagonal.
    """
    rng = np.random.default_rng()
    C = rng.random((dim, dim))
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    return C


def hybrid_process(text: str, model_resource: Tuple[float, int]) -> Tuple[float, np.ndarray]:
    """
    End‑to‑end hybrid pipeline:
    1. Compute compatibility evidence from text and model resources.
    2. Initialise a curvature matrix sized to the master vector.
    3. Update the curvature with the evidence.
    Returns the evidence and the updated curvature.
    """
    evidence = compatibility_score(text, model_resource)
    master_dim = extract_master_vector(text).shape[0]
    C0 = initialise_curvature(master_dim)
    C1 = bayesian_curvature_update(C0, evidence)
    return evidence, C1


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog while analyzing quantum cryptography."
    model_res = (0.73, 3)  # 73% of RAM, tier 3
    ev, curv = hybrid_process(sample_text, model_res)

    print(f"Compatibility evidence (sigmoid‑scaled): {ev:.4f}")
    print("Updated curvature matrix (rounded):")
    print(np.round(curv, 3))
    # Simple sanity checks
    assert 0.0 < ev < 1.0, "Evidence should be a probability."
    assert curv.shape == (9, 9), "Curvature dimension must match master vector size."
    assert np.allclose(curv, curv.T), "Curvature matrix must remain symmetric."
    print("Smoke test passed.")