# DARWIN HAMMER — match 23, survivor 4
# gen: 2
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s1.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s5.py (gen1)
# born: 2026-05-29T23:25:22Z

"""
Hybrid Workshare‑Feature Allocator
=================================

This module fuses the *workshare_allocator* (Parent A) and the
*krampus + ollivier‑ricci* feature extraction pipeline (Parent B).

Mathematical bridge
-------------------
1. **Deterministic core** – The deterministic portion of the workshare
   (`deterministic_units`) is computed exactly as in *workshare_allocator*,
   i.e. it is scaled by the Doomsday weekday factor `1 + d/7`.

2. **Feature‑driven residual** – The remaining units (`llm_units`) are
   distributed among the model groups using a weight vector derived from
   the 24‑dimensional feature vector produced by the Krampus/Ollivier‑Ricci
   extractor.  The selected subset of features is normalised to sum to 1,
   providing a probability distribution that drives the allocation.

3. **Curvature feedback** – An optional curvature metric is obtained from
   the outer‑product matrix of the full feature vector; its trace is used
   as a scalar “curvature adjustment” that can further modulate the
   deterministic target (demonstrated in `allocate_workshare_hybrid`).

Thus the hybrid system treats the deterministic allocation as a
baseline and refines the stochastic, LLM‑driven share with a mathematically
grounded feature weighting, achieving a single unified allocation routine.
"""

import math
import random
import sys
import hashlib
from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants & helpers (shared)
# ---------------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return the Doomsday weekday index for a given Gregorian date.
    Monday → 0, …, Sunday → 6 (the original code used (weekday+1)%7).
    """
    return (date(year, month, day).weekday() + 1) % 7


# ---------------------------------------------------------------------------
# Parent B – deterministic feature extraction
# ---------------------------------------------------------------------------


def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from a stable SHA‑256 hash of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Produce a deterministic 24‑dimensional pseudo‑feature vector.
    Each entry is a float in [0, 10) generated from a text‑derived RNG.
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


def extract_master_vector(text: str) -> Dict[str, float]:
    """
    Human‑readable 24‑dimensional vector derived from the full feature set.
    The keys are shortened for ergonomics.
    """
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
        "chaotic_good_tax": f.get("resilience_chaotic_good_tax", 0.0),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
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
        "manic_velocity": f.get("telemetry_manic_velocity", 0.0),
    }


def compute_curvature_metric(features: Dict[str, float]) -> float:
    """
    Simple curvature proxy: trace of the outer‑product matrix of the feature vector.
    This yields Σ_i v_i², a scalar that grows with feature magnitude.
    """
    vec = np.array(list(features.values()), dtype=float)
    if vec.size == 0:
        return 0.0
    outer = np.outer(vec, vec)
    return float(np.trace(outer))


# ---------------------------------------------------------------------------
# Hybrid allocation logic (core of the fusion)
# ---------------------------------------------------------------------------


def _feature_weights_for_groups(
    master_vector: Dict[str, float], groups: Tuple[str, ...] = GROUPS
) -> Dict[str, float]:
    """
    Map a subset of the master vector onto the supplied *groups*.
    The first ``len(groups)`` entries (ordered as in the dictionary) are used.
    The raw values are normalised to sum to 1.0; if the sum is zero,
    an equal distribution is returned.
    """
    if not master_vector:
        return {g: 1.0 / len(groups) for g in groups}

    # Preserve deterministic ordering of dict (Python 3.7+)
    raw_vals = list(master_vector.values())[: len(groups)]
    raw_sum = sum(raw_vals)
    if raw_sum == 0.0:
        return {g: 1.0 / len(groups) for g in groups}
    return {g: v / raw_sum for g, v in zip(groups, raw_vals)}


def allocate_workshare_hybrid(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    text: str,
    year: int = date.today().year,
    month: int = date.today().month,
    day: int = date.today().day,
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, any]:
    """
    Compute a unified allocation:

    * ``deterministic_units`` – baseline share scaled by Doomsday.
    * ``llm_units`` – residual share split among *groups* according to
      feature‑derived weights.
    * ``curvature_adj`` – a scalar derived from the feature curvature that
      optionally nudges the deterministic target (demonstrated here by a
      simple linear blend).

    The function returns a dictionary mirroring the structure of the
    original allocator but enriched with feature metadata.
    """
    # ---- input validation -------------------------------------------------
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0 <= deterministic_target_pct <= 100):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("at least one group required")

    # ---- deterministic core ------------------------------------------------
    d_value = doomsday(year, month, day)
    # curvature metric provides a small adjustment (max ±5 % of target)
    master_vec = extract_master_vector(text)
    curvature = compute_curvature_metric(master_vec)
    curvature_adj = (curvature % 10) / 200.0  # yields [0, 0.05)
    adj_target = deterministic_target_pct * (1 + curvature_adj)

    deterministic_units = total_units * adj_target / 100.0 * (1 + d_value / 7.0)
    deterministic_units = min(deterministic_units, total_units)  # safety clamp
    llm_units = total_units - deterministic_units

    # ---- feature‑driven residual -------------------------------------------
    weights = _feature_weights_for_groups(master_vec, groups)
    lanes = []
    for grp in groups:
        grp_llm = llm_units * weights[grp]
        lanes.append(
            {
                "group": grp,
                "llm_units": _pct(grp_llm),
                "llm_share_pct": _pct(100.0 * weights[grp]),
                "proof_required": True,
                "feature_weight": _pct(weights[grp]),
            }
        )

    # ---- assemble final payload --------------------------------------------
    allocation = {
        "date": {"year": year, "month": month, "day": day, "doomsday": d_value},
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "curvature_adjustment_pct": _pct(curvature_adj * 100),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
        "metadata": {
            "curvature_metric": _pct(curvature),
            "feature_vector": {k: _pct(v) for k, v in master_vec.items()},
        },
    }
    return allocation


def summarize_allocation(allocation: Dict[str, any]) -> str:
    """Pretty‑print the most relevant fields of a hybrid allocation."""
    lines = [
        f"Date (Y-M-D): {allocation['date']['year']}-{allocation['date']['month']:02d}-{allocation['date']['day']:02d}",
        f"Doomsday index: {allocation['date']['doomsday']}",
        f"Total units: {allocation['total_units']}",
        f"Deterministic target (%): {allocation['deterministic_target_pct']}",
        f"Curvature adjustment (%): {allocation['curvature_adjustment_pct']}",
        f"Deterministic units: {allocation['deterministic_units']}",
        f"LLM residual units: {allocation['llm_units']}",
        "Group allocations:",
    ]
    for lane in allocation["lanes"]:
        lines.append(
            f"  - {lane['group']}: {lane['llm_units']} units "
            f"({lane['llm_share_pct']}% of LLM share, weight={lane['feature_weight']})"
        )
    lines.append(f"Curvature metric (trace): {allocation['metadata']['curvature_metric']}")
    return "\n".join(lines)


def allocate_and_report(
    total_units: float,
    deterministic_target_pct: float,
    text: str,
    year: int = date.today().year,
    month: int = date.today().month,
    day: int = date.today().day,
) -> None:
    """
    Convenience wrapper that runs the hybrid allocator and prints the summary.
    """
    alloc = allocate_workshare_hybrid(
        total_units=total_units,
        deterministic_target_pct=deterministic_target_pct,
        text=text,
        year=year,
        month=month,
        day=day,
    )
    print(summarize_allocation(alloc))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_TEXT = (
        "In the twilight of algorithmic governance, the "
        "interplay of deterministic scheduling and stochastic feature "
        "extraction defines the future of resource allocation."
    )
    allocate_and_report(
        total_units=1_000_000,
        deterministic_target_pct=85.0,
        text=SAMPLE_TEXT,
        year=2026,
        month=5,
        day=29,
    )