# DARWIN HAMMER — match 17, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s4.py (gen3)
# parent_b: hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s2.py (gen2)
# born: 2026-05-29T23:26:28Z

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

import numpy as np
import re

# ----------------------------------------------------------------------
# Parent A – regexes and weighted cue extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|delay|postpone|defer)\b", re.I
)

W_POS = np.array([1.2, 0.8, 0.5])   
W_NEG = np.array([0.3, 0.2, 1.0])   

def _count_cues(text: str) -> np.ndarray:
    return np.array([
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
    ], dtype=float)

def compute_load_privacy(text: str) -> Tuple[float, float]:
    c = _count_cues(text)
    load = float(c @ (W_POS - W_NEG))
    privacy = float(c[2] * 0.7)  
    return load, privacy

# ----------------------------------------------------------------------
# Parent B – feature extraction (deterministic pseudo‑random)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
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
    return {k: rnd.random() * 10 for k in keys}

def extract_master_vector(text: str) -> np.ndarray:
    full = extract_full_features(text)
    master_keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
    ]
    return np.array([full[k] for k in master_keys], dtype=float)

# ----------------------------------------------------------------------
# Spatial utilities (used by Parent A for load computation)
# ----------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0  
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def spatial_load(lat: float, lon: float, ref: Tuple[float, float]) -> float:
    return haversine_distance(lat, lon, *ref)

# ----------------------------------------------------------------------
# Path‑signature utilities (Parent B mathematical core)
# ----------------------------------------------------------------------
def discrete_path_signature(path: np.ndarray, level: int = 2) -> np.ndarray:
    if path.shape[0] < 2:
        raise ValueError("Path must contain at least two points.")
    increments = np.diff(path, axis=0)  
    sig1 = increments.sum(axis=0)  

    if level == 1:
        return sig1

    d = increments.shape[1]
    sig2 = np.zeros((d, d), dtype=float)
    m = increments.shape[0]
    for i in range(m):
        for j in range(i, m):
            sig2 += np.outer(increments[i], increments[j])
    idx = np.triu_indices(d)
    sig2_flat = sig2[idx]
    return np.concatenate([sig1, sig2_flat])

# ----------------------------------------------------------------------
# Greedy budget selector (Parent A)
# ----------------------------------------------------------------------
def select_under_budget(
    resources: np.ndarray,
    spatial_budget: float,
    privacy_budget: float,
) -> List[int]:
    if resources.shape[1] != 2:
        raise ValueError("Resources must have exactly two columns (load, privacy).")
    n = resources.shape[0]
    order = np.argsort(resources.sum(axis=1))
    chosen = []
    load_used = 0.0
    privacy_used = 0.0
    for idx in order:
        load_i, priv_i = resources[idx]
        if load_used + load_i <= spatial_budget and privacy_used + priv_i <= privacy_budget:
            chosen.append(idx)
            load_used += load_i
            privacy_used += priv_i
    return chosen

def hybrid_text_spatial_path_signature_fusion(
    texts: List[str],
    spatial_ref: Tuple[float, float],
    spatial_budget: float,
    privacy_budget: float,
) -> Tuple[List[int], np.ndarray]:
    resources = np.array([compute_load_privacy(text) for text in texts])
    master_vectors = np.array([extract_master_vector(text) for text in texts])
    augmented_points = np.concatenate((resources, master_vectors), axis=1)
    path_signature = discrete_path_signature(augmented_points)
    chosen_indices = select_under_budget(resources, spatial_budget, privacy_budget)
    return chosen_indices, path_signature

def main():
    texts = ["example text 1", "example text 2", "example text 3"]
    spatial_ref = (40.7128, -74.0060)  
    spatial_budget = 100.0
    privacy_budget = 100.0
    chosen_indices, path_signature = hybrid_text_spatial_path_signature_fusion(texts, spatial_ref, spatial_budget, privacy_budget)
    print("Chosen indices:", chosen_indices)
    print("Path signature:", path_signature)

if __name__ == "__main__":
    main()