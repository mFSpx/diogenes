# DARWIN HAMMER — match 23, survivor 5
# gen: 2
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s1.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s5.py (gen1)
# born: 2026-05-29T23:25:22Z

import math
import random
import sys
import hashlib
from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7

def _rng_from_text(text: str) -> random.Random:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)

def extract_full_features(text: str) -> Dict[str, float]:
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
    vec = np.array(list(features.values()), dtype=float)
    if vec.size == 0:
        return 0.0
    outer = np.outer(vec, vec)
    return float(np.trace(outer))

def _feature_weights_for_groups(
    master_vector: Dict[str, float], groups: Tuple[str, ...] = GROUPS
) -> Dict[str, float]:
    if not master_vector:
        return {g: 1.0 / len(groups) for g in groups}

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
) -> Dict[str, float]:
    doomsday_factor = 1 + doomsday(year, month, day) / 7
    deterministic_units = (deterministic_target_pct / 100) * total_units * doomsday_factor
    llm_units = total_units - deterministic_units

    master_vector = extract_master_vector(text)
    feature_weights = _feature_weights_for_groups(master_vector, groups)

    curvature_metric = compute_curvature_metric(master_vector)
    adjusted_deterministic_units = deterministic_units * (1 + curvature_metric / (1 + curvature_metric))

    allocation = {g: (feature_weights[g] * llm_units) for g in groups}
    allocation["deterministic"] = adjusted_deterministic_units

    # Normalize to ensure total units are allocated
    total_allocated = sum(allocation.values())
    if abs(total_allocated - total_units) > 1e-6:
        scaling_factor = total_units / total_allocated
        allocation = {k: v * scaling_factor for k, v in allocation.items()}

    return allocation

def main():
    total_units = 100.0
    text = "example text"
    allocation = allocate_workshare_hybrid(total_units=total_units, text=text)
    print(allocation)

if __name__ == "__main__":
    main()