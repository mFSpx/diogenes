#!/usr/bin/env python3
"""Krampus brain-map projection algorithms. No runtime orchestration here."""
from __future__ import annotations

from ALGOS.krampus_stickers import (
    dbstream_features,
    extract_operator_telemetry,
    extract_operator_vibes,
    extract_psyche_vibes,
    extract_rainmaker_vibes,
    extract_resilience_vibes,
)


def extract_full_features(text: str) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(extract_operator_vibes(text))
    features.update(extract_psyche_vibes(text))
    features.update(extract_resilience_vibes(text))
    features.update(extract_rainmaker_vibes(text))
    features.update(extract_operator_telemetry(text))
    return features


def extract_master_vector(text: str) -> dict[str, float]:
    """Human-readable 20+ dimension vector for exports/maps."""
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
        "bureaucratic_weaponization_index": f.get("resilience_bureaucratic_weaponization_index", 0.0),
        "resource_exhaustion_metric": f.get("resilience_resource_exhaustion_metric", 0.0),
        "swarm_orchestration_density": f.get("resilience_swarm_orchestration_density", 0.0),
        "logic_crucifixion_index": f.get("resilience_logic_crucifixion_index", 0.0),
        "conspiracy_grounding_ratio": f.get("resilience_conspiracy_grounding_ratio", 0.0),
        "chaotic_good_tax": f.get("resilience_chaotic_good_tax", 0.0),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": f.get("rainmaker_asset_structuring_weight", 0.0),
        "pitch_formatting_ratio": f.get("rainmaker_pitch_formatting_ratio", 0.0),
        "agent_symmetry_ratio": f.get("telemetry_agent_symmetry_ratio", 0.0),
        "protocol_discipline": f.get("telemetry_protocol_discipline", 0.0),
        "manic_velocity": f.get("telemetry_manic_velocity", 0.0),
    }


def brain_xyz(master: dict[str, float]) -> dict[str, float]:
    """Tiny deterministic 3-axis projection for initial plotting."""
    x_architect_operator = (
        master.get("visceral_ratio", 0.0) * 8
        + master.get("ledger_density", 0.0) * 6
        + min(master.get("directive_ratio", 0.0), 8.0) / 8
        + master.get("recursion_score", 0.0) * 4
    )
    y_psyche_resilience = (
        master.get("forensic_shield_ratio", 0.0) * 6
        + master.get("poetic_entropy", 0.0) * 4
        + min(master.get("dissociative_index", 0.0), 8.0) / 8
        + master.get("resource_exhaustion_metric", 0.0) * 6
        + master.get("bureaucratic_weaponization_index", 0.0) * 4
    )
    z_rainmaker_sprint = (
        master.get("corporate_grit_tension", 0.0) * 6
        + master.get("countdown_density", 0.0) * 6
        + master.get("asset_structuring_weight", 0.0) * 4
        + master.get("swarm_orchestration_density", 0.0) * 4
        + master.get("chaotic_good_tax", 0.0) * 4
        + master.get("agent_symmetry_ratio", 0.0) * 0.5
        + master.get("protocol_discipline", 0.0) * 0.2
        + master.get("manic_velocity", 0.0) * 0.4
    )
    return {"x": x_architect_operator, "y": y_psyche_resilience, "z": z_rainmaker_sprint}


__all__ = ["dbstream_features", "extract_full_features", "extract_master_vector", "brain_xyz"]
