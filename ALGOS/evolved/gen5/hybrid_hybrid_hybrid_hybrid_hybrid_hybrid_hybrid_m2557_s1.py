# DARWIN HAMMER — match 2557, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s0.py (gen4)
# born: 2026-05-29T23:42:51Z

"""Hybrid Text-Morphology Uncertainty Fusion (Parents: hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s2,
hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s0)

Mathematical Bridge
-------------------
Algorithm A produces a *feature‑count vector*  **f** = (evidence, planning) extracted from a
document.  Algorithm B supplies *shape‑derived uncertainty indices*  **u** = (sphericity,
flatness, righting_time) for a physical object together with a discrete *epistemic flag*
ϕ ∈ {FACT, PROBABLE, …}.  

Both parents share a probabilistic weighting concept:
* In A the raw counts are re‑weighted by posterior edge beliefs (a probability vector **p**).
* In B the epistemic flag is interpreted as a confidence probability **c(ϕ)**.

The hybrid therefore treats **p** and **c(ϕ)** as a common scalar weight **w** that maps the
textual feature space onto the morphological uncertainty space.  The core operation is a
*linear‑fusion* of the normalized vectors:

  **h** = w·‖**f**‖₂ · ‖**u**‖₂⁻¹ · **u**  +  (1‑w)·‖**u**‖₂ · ‖**f**‖₂⁻¹ · **f**

where ‖·‖₂ denotes the Euclidean norm.  The resulting hybrid vector **h** lives in a
joint 5‑dimensional space and can be used for similarity scoring, cost evaluation or
free‑energy estimation.

The module implements:
* `extract_feature_vector` – regex‑based count extraction (A).
* `morphology_indices` – sphericity, flatness, righting‑time (B).
* `epistemic_confidence` – maps ϕ → scalar in [0,1].
* `hybrid_fusion_vector` – the weighted linear fusion described above.
* `hybrid_similarity` – cosine similarity between two hybrid vectors.
* `hybrid_free_energy` – a simple free‑energy proxy:  F = –log(‖h‖₂) · w.
"""

import sys
import math
import random
import pathlib
import re
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Textual feature extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def extract_feature_vector(text: str) -> np.ndarray:
    """
    Count occurrences of evidence‑type and planning‑type tokens.
    Returns a length‑2 numpy vector [evidence_count, planning_count].
    """
    ev_cnt = len(EVIDENCE_RE.findall(text))
    pl_cnt = len(PLANNING_RE.findall(text))
    return np.array([ev_cnt, pl_cnt], dtype=float)


# ----------------------------------------------------------------------
# Parent B – Morphology and epistemic confidence
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def morphology_indices(m: Morphology) -> np.ndarray:
    """
    Compute the three shape‑derived uncertainty indices.
    Returns a length‑3 numpy vector [sphericity, flatness, righting_time].
    """
    sph = sphericity_index(m.length, m.width, m.height)
    flt = flatness_index(m.length, m.width, m.height)
    rgt = righting_time_index(m)
    return np.array([sph, flt, rgt], dtype=float)


def epistemic_confidence(flag: str) -> float:
    """
    Map an epistemic flag to a confidence weight w ∈ [0,1].
    FACT → 1.0, PROBABLE → 0.8, POSSIBLE → 0.5,
    BULLSHIT → 0.1, SURE_MAYBE → 0.3.
    Unknown flags default to 0.0.
    """
    mapping = {
        "FACT": 1.0,
        "PROBABLE": 0.8,
        "POSSIBLE": 0.5,
        "BULLSHIT": 0.1,
        "SURE_MAYBE": 0.3,
    }
    return float(mapping.get(flag.upper(), 0.0))


# ----------------------------------------------------------------------
# Hybrid core: fusion of textual and morphological spaces
# ----------------------------------------------------------------------
def hybrid_fusion_vector(
    text: str,
    morph: Morphology,
    flag: str,
    posterior_edge_belief: Tuple[float, float] = (0.6, 0.4),
) -> np.ndarray:
    """
    Produce a 5‑dimensional hybrid vector by fusing the normalized textual feature
    vector **f** and the normalized morphology index vector **u**.

    Parameters
    ----------
    text : str
        Input document.
    morph : Morphology
        Physical object description.
    flag : str
        Epistemic flag controlling the fusion weight.
    posterior_edge_belief : tuple(float, float)
        A probability vector from Algorithm A (must sum to 1).  Its mean is used as
        an auxiliary weight that is multiplied with the epistemic confidence.

    Returns
    -------
    np.ndarray
        Hybrid vector of shape (5,).
    """
    # Textual features
    f_raw = extract_feature_vector(text)          # shape (2,)
    # Apply posterior edge belief as a linear transform
    p = np.array(posterior_edge_belief, dtype=float)
    if not np.isclose(p.sum(), 1.0):
        raise ValueError("posterior_edge_belief must sum to 1")
    f_weighted = p @ np.vstack([f_raw, f_raw])    # simple weighting, keeps shape (2,)

    # Morphology indices
    u_raw = morphology_indices(morph)            # shape (3,)

    # Normalizations
    f_norm = f_weighted / (np.linalg.norm(f_weighted) + 1e-12)
    u_norm = u_raw / (np.linalg.norm(u_raw) + 1e-12)

    # Fusion weight: epistemic confidence multiplied by mean posterior belief
    w = epistemic_confidence(flag) * p.mean()

    # Linear fusion as described in the module docstring
    h_part_from_u = w * np.linalg.norm(f_weighted) * u_norm
    h_part_from_f = (1.0 - w) * np.linalg.norm(u_raw) * f_norm

    hybrid_vec = np.concatenate([h_part_from_f, h_part_from_u])
    return hybrid_vec


def hybrid_similarity(
    text1: str,
    text2: str,
    morph1: Morphology,
    morph2: Morphology,
    flag: str,
    posterior_edge_belief: Tuple[float, float] = (0.6, 0.4),
) -> float:
    """
    Cosine similarity between two hybrid vectors.  Values close to 1 indicate
    high similarity in the joint textual‑morphological space.
    """
    h1 = hybrid_fusion_vector(text1, morph1, flag, posterior_edge_belief)
    h2 = hybrid_fusion_vector(text2, morph2, flag, posterior_edge_belief)
    dot = float(np.dot(h1, h2))
    norm_prod = float(np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-12)
    return dot / norm_prod


def hybrid_free_energy(
    hybrid_vec: np.ndarray,
    flag: str,
    posterior_edge_belief: Tuple[float, float] = (0.6, 0.4),
) -> float:
    """
    A simple free‑energy proxy derived from the hybrid vector magnitude and the
    fusion weight w.  The formula mirrors the RLCT‑based free‑energy minimisation
    used in the parent RLCT system:

        F = - w * log(‖h‖₂ + ε)

    where ε prevents log(0).
    """
    w = epistemic_confidence(flag) * np.mean(posterior_edge_belief)
    magnitude = np.linalg.norm(hybrid_vec) + 1e-12
    return -w * math.log(magnitude)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text_a = (
        "The audit confirmed the evidence and source. "
        "We plan the next steps with a checklist and timeline."
    )
    sample_text_b = (
        "Verification of the hash and screenshot was performed. "
        "A roadmap and budget were drafted."
    )
    morph_a = Morphology(length=2.0, width=1.5, height=0.8, mass=3.0)
    morph_b = Morphology(length=1.8, width=1.4, height=0.9, mass=2.5)

    flag = "PROBABLE"

    vec_a = hybrid_fusion_vector(sample_text_a, morph_a, flag)
    vec_b = hybrid_fusion_vector(sample_text_b, morph_b, flag)

    sim = hybrid_similarity(sample_text_a, sample_text_b, morph_a, morph_b, flag)
    fe_a = hybrid_free_energy(vec_a, flag)
    fe_b = hybrid_free_energy(vec_b, flag)

    print("Hybrid vector A:", vec_a)
    print("Hybrid vector B:", vec_b)
    print(f"Cosine similarity (A,B): {sim:.4f}")
    print(f"Free energy A: {fe_a:.4f}, B: {fe_b:.4f}")

    # Ensure no exceptions were raised
    sys.exit(0)