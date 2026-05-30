# DARWIN HAMMER — match 5017, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_semantic_neig_m1781_s0.py (gen6)
# born: 2026-05-29T23:59:15Z

"""Hybrid Morphological‑Semantic Analyzer

Parents:
- hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s1.py (Pheromone‑RLCT & Epistemic Certainty)
- hybrid_hybrid_hybrid_hybrid_hybrid_semantic_neig_m1781_s0.py (Regex Feature Extraction & Morphology)

Mathematical Bridge:
Both parents expose a *recovery priority* scalar derived from morphology
(righting‑time index → normalized ∈[0,1]).  The regex subsystem produces a
feature count vector **v**∈ℝⁿ.  By treating the recovery priority **ρ** as an
 epistemic‑certainty weight, we form the hybrid representation  

    **h** = ρ · **v**  

where ρ modulates every textual feature proportionally to the physical certainty
of the underlying object.  This unifies the information‑entropy optimisation of
the RLCT side with the semantic vector transformation of the regex side.

The module implements the combined metrics, feature extraction, and weighted
fusion, together with a simple epistemic flag mapping.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import numpy as np
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

# ----------------------------------------------------------------------
# Morphology‑based metrics (merged definitions)
# ----------------------------------------------------------------------
def sphericity_index(m: Morphology) -> float:
    """Average of the two parent definitions for robustness."""
    l, w, h = m.length, m.width, m.height
    if min(l, w, h) <= 0:
        raise ValueError("dimensions must be positive")
    # Parent A: geometric mean / length
    s1 = (l * w * h) ** (1.0 / 3.0) / l
    # Parent B: geometric mean / max dimension
    s2 = (l * w * h) ** (1.0 / 3.0) / max(l, w, h)
    return (s1 + s2) / 2.0


def flatness_index(m: Morphology) -> float:
    """Consistent flatness definition used by both parents."""
    if m.height <= 0:
        raise ValueError("height must be positive")
    return (m.length + m.width) / (2.0 * m.height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """Unified righting‑time index (identical to both parents)."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized certainty scalar ∈[0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Regex‑based semantic feature extraction (Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def extract_regex_features(text: str) -> Dict[str, int]:
    """Count occurrences of each regex category in *text*."""
    return {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
    }


# ----------------------------------------------------------------------
# Epistemic flag mapping (Parent A)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "BULLSHIT",
    "SURE_MAYBE",
)


def epistemic_flag(score: float) -> str:
    """Map a normalized score ∈[0,1] to a qualitative epistemic flag."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be within [0, 1]")
    # Partition the interval into equal segments
    idx = int(score * len(EPISTEMIC_FLAGS))
    idx = min(idx, len(EPISTEMIC_FLAGS) - 1)
    return EPISTEMIC_FLAGS[idx]


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def weighted_feature_vector(text: str, morph: Morphology) -> np.ndarray:
    """
    Produce the hybrid representation **h** = ρ · **v**.

    *v*  – regex count vector (evidence, planning) as float32.
    ρ    – recovery priority derived from morphology.
    """
    counts = extract_regex_features(text)
    v = np.array([counts["evidence"], counts["planning"]], dtype=np.float32)
    rho = recovery_priority(morph)
    return rho * v


def hybrid_score(text: str, morph: Morphology) -> Dict[str, float]:
    """
    Compute a dictionary containing:
      - raw counts,
      - weighted counts,
      - recovery priority,
      - epistemic flag.
    """
    raw = extract_regex_features(text)
    weighted = weighted_feature_vector(text, morph)
    rho = recovery_priority(morph)
    flag = epistemic_flag(rho)
    return {
        "evidence_raw": raw["evidence"],
        "planning_raw": raw["planning"],
        "evidence_weighted": float(weighted[0]),
        "planning_weighted": float(weighted[1]),
        "recovery_priority": rho,
        "epistemic_flag": flag,
    }


def simulate_pheromone_update(
    current: np.ndarray, morph: Morphology, decay: float = 0.1, boost: float = 0.5
) -> np.ndarray:
    """
    Mimic a simple pheromone‑like reinforcement:
      - decay all entries,
      - add a boost proportional to recovery priority.
    This provides a stochastic bridge to the RLCT‑grokking spirit.
    """
    if not 0.0 <= decay <= 1.0 or not 0.0 <= boost <= 1.0:
        raise ValueError("decay and boost must be within [0,1]")
    rho = recovery_priority(morph)
    decayed = current * (1.0 - decay)
    boost_vec = np.full_like(current, rho * boost)
    return decayed + boost_vec + np.random.rand(*current.shape) * 0.01  # tiny noise


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The plan includes a checklist and timeline. "
        "Evidence was verified by the audit log and a SHA256 hash."
    )
    sample_morph = Morphology(length=2.0, width=1.5, height=0.5, mass=3.0)

    print("Morphology metrics:")
    print(f"  sphericity_index   = {sphericity_index(sample_morph):.4f}")
    print(f"  flatness_index     = {flatness_index(sample_morph):.4f}")
    print(f"  righting_time_index= {righting_time_index(sample_morph):.4f}")
    print(f"  recovery_priority  = {recovery_priority(sample_morph):.4f}")

    print("\nRegex raw features:")
    print(extract_regex_features(sample_text))

    print("\nHybrid score:")
    for k, v in hybrid_score(sample_text, sample_morph).items():
        print(f"  {k}: {v}")

    # Demonstrate pheromone update
    init_vec = weighted_feature_vector(sample_text, sample_morph)
    updated_vec = simulate_pheromone_update(init_vec, sample_morph)
    print("\nPheromone‑style update:")
    print(f"  before = {init_vec}")
    print(f"  after  = {updated_vec}")

    sys.exit(0)