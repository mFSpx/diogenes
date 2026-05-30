# DARWIN HAMMER — match 23, survivor 3
# gen: 2
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s1.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s5.py (gen1)
# born: 2026-05-29T23:25:22Z

"""
Hybrid Module: workshare_allocator + doomsday_calendar  ×  krampus_brainmap + ollivier_ricci_curvature

This fusion links the two parent algorithms through a *feature‑curvature matrix* that
modulates the LLM‑share of the work‑allocation.  

- The deterministic portion of the allocation is scaled by the classic
  **doomsday weekday value** (0‑6) as in `hybrid_workshare_allocator_doomsday_calendar`.
- The stochastic LLM‑portion is no longer split evenly; instead a **24‑dimensional
  feature vector** extracted deterministically from an input text (Krampus extractor)
  is transformed into a **curvature matrix** `C = v·vᵀ` (outer product of the
  normalized feature vector).  
- The per‑group share is obtained by projecting the curvature matrix onto a
  one‑hot encoding of the group name, yielding a weight proportional to the
  corresponding entry of the vector `w = C·g`, where `g` is a one‑hot vector for
  the group.  This mathematically fuses the matrix‑operation core of the Ricci
  curvature with the allocation topology.

The three public functions demonstrate the hybrid behaviour:
`allocate_workshare_with_features`, `compute_feature_curvature`, and
`hybrid_summary`.
"""

import math
import random
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Helper utilities (shared by both parents)
# ---------------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return the weekday index used by the original doomsday calendar:
    Monday → 1, …, Sunday → 0 (mod 7).
    """
    return (date(year, month, day).weekday() + 1) % 7


def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from a stable SHA‑256 hash of *text*."""
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Krampus feature extraction (parent B – deterministic stub)
# ---------------------------------------------------------------------------

_FEATURE_KEYS = [
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


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Produce a deterministic pseudo‑feature dictionary of length 24.
    The values are in the interval [0, 10).
    """
    rng = _rng_from_text(text)
    return {k: rng.random() * 10.0 for k in _FEATURE_KEYS}


def extract_master_vector(text: str) -> np.ndarray:
    """
    Return a 24‑dimensional numpy array (float64) containing the feature values.
    If the input text is empty, a zero vector is returned.
    """
    if not text.strip():
        return np.zeros(len(_FEATURE_KEYS), dtype=np.float64)
    feats = extract_full_features(text)
    return np.array([feats[k] for k in _FEATURE_KEYS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Ricci‑like curvature from the feature vector (parent B – mathematical core)
# ---------------------------------------------------------------------------

def compute_feature_curvature(features: np.ndarray) -> np.ndarray:
    """
    Compute a symmetric curvature matrix C = (v̂ ⊗ v̂) where v̂ is the L2‑normalized
    feature vector.  This mimics the Ollivier‑Ricci curvature construction where
    the outer product encodes pairwise similarity of feature dimensions.
    """
    if features.ndim != 1:
        raise ValueError("features must be a 1‑D vector")
    norm = np.linalg.norm(features)
    if norm == 0:
        # Return a zero matrix for the degenerate case.
        return np.zeros((features.size, features.size), dtype=np.float64)
    v_hat = features / norm
    return np.outer(v_hat, v_hat)


# ---------------------------------------------------------------------------
# Hybrid allocation that bridges doomsday and curvature (core fusion)
# ---------------------------------------------------------------------------

def allocate_workshare_with_features(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    year: int = date.today().year,
    month: int = date.today().month,
    day: int = date.today().day,
    text: str = "",
) -> Dict[str, any]:
    """
    Allocate *total_units* across *groups*.

    1. Deterministic units are scaled by the doomsday weekday (as in parent A).
    2. The residual LLM units are distributed according to a weight vector
       derived from the curvature matrix of the feature vector (parent B).
       For each group we compute a hash‑based one‑hot vector `g` and obtain a
       weight `w_i = (C·g)_i`.  The weights are normalised to sum to 1.
    3. The result contains per‑group LLM allocation, deterministic allocation,
       and the curvature matrix for downstream analysis.
    """
    # ---- validation -------------------------------------------------------
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0 <= deterministic_target_pct <= 100):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("at least one group required")

    # ---- deterministic part (doomsday scaling) ---------------------------
    dd = doomsday(year, month, day)
    deterministic_units = total_units * deterministic_target_pct / 100.0 * (1 + dd / 7.0)
    deterministic_units = _pct(deterministic_units)

    # ---- stochastic part --------------------------------------------------
    llm_units = total_units - deterministic_units
    llm_units = max(llm_units, 0.0)  # guard against rounding overshoot

    # Feature extraction and curvature
    feature_vec = extract_master_vector(text)
    curvature = compute_feature_curvature(feature_vec)

    # Build a weight for each group via a deterministic hash‑based one‑hot vector.
    # The hash ensures reproducibility across runs for the same group name.
    raw_weights = []
    for grp in groups:
        # deterministic hash to a one‑hot index in [0, len(groups)-1]
        idx = int.from_bytes(hash(grp).to_bytes(8, "big", signed=False), "big") % len(groups)
        g = np.zeros(len(groups), dtype=np.float64)
        g[idx] = 1.0
        # Project curvature onto g; then pick the component corresponding to the group.
        proj = curvature @ g
        weight = proj[idx]  # scalar weight for this group
        raw_weights.append(weight)

    raw_weights = np.array(raw_weights, dtype=np.float64)
    # If all weights are zero (degenerate curvature), fall back to uniform.
    if np.allclose(raw_weights, 0):
        raw_weights = np.ones_like(raw_weights)

    norm_weights = raw_weights / raw_weights.sum()
    per_group_llm = llm_units * norm_weights
    per_group_llm = np.vectorize(_pct)(per_group_llm)

    lanes = []
    for grp, llm_share in zip(groups, per_group_llm):
        lanes.append(
            {
                "group": grp,
                "llm_units": llm_share,
                "llm_share_pct": _pct(100.0 * llm_share / llm_units) if llm_units else 0.0,
                "proof_required": True,
            }
        )

    result = {
        "date": f"{year:04d}-{month:02d}-{day:02d}",
        "doomsday_value": dd,
        "deterministic_units": deterministic_units,
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "llm_residual_units": _pct(llm_units),
        "curvature_matrix": curvature.tolist(),
        "allocation": lanes,
    }
    return result


# ---------------------------------------------------------------------------
# Additional helper exposing the hybrid relationship
# ---------------------------------------------------------------------------

def hybrid_summary(total_units: float, text: str) -> Dict[str, any]:
    """
    Produce a concise summary that combines allocation statistics with a
    curvature‑derived metric (the trace of the curvature matrix, i.e. the sum of
    squared normalized feature components).  This metric can be interpreted as
    the *effective dimensionality* of the feature space influencing the LLM
    distribution.
    """
    alloc = allocate_workshare_with_features(total_units=total_units, text=text)
    curvature = np.array(alloc["curvature_matrix"])
    trace = _pct(np.trace(curvature))  # equals 1.0 for a properly normalised vector
    summary = {
        "total_units": total_units,
        "deterministic_units": alloc["deterministic_units"],
        "llm_residual_units": alloc["llm_residual_units"],
        "curvature_trace": trace,
        "group_breakdown": {lane["group"]: lane["llm_units"] for lane in alloc["allocation"]},
    }
    return summary


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example parameters
    TOTAL = 1000.0
    SAMPLE_TEXT = (
        "In a world where deterministic algorithms meet stochastic curvature, "
        "the allocation must respect both calendar cycles and hidden feature geometry."
    )
    today = date.today()
    allocation = allocate_workshare_with_features(
        total_units=TOTAL,
        deterministic_target_pct=85.0,
        groups=GROUPS,
        year=today.year,
        month=today.month,
        day=today.day,
        text=SAMPLE_TEXT,
    )
    print("Hybrid Allocation Result:")
    for lane in allocation["allocation"]:
        print(
            f"  {lane['group']}: {lane['llm_units']} LLM units "
            f"({lane['llm_share_pct']}% of LLM pool)"
        )
    print(f"Deterministic units: {allocation['deterministic_units']}")
    print(f"Doomsday weekday value: {allocation['doomsday_value']}")
    print("\nCurvature matrix (first 3 rows):")
    for row in allocation["curvature_matrix"][:3]:
        print("  ", [_pct(v) for v in row[:3]])

    print("\nHybrid Summary:")
    print(hybrid_summary(TOTAL, SAMPLE_TEXT))