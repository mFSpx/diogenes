# DARWIN HAMMER — match 17, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s4.py (gen3)
# parent_b: hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s2.py (gen2)
# born: 2026-05-29T23:26:28Z

"""hybrid_text_spatial_path_signature_fusion.py
Hybrid algorithm merging:

* **Parent A** – regex‑based textual cue extraction that yields a 2‑dimensional
  resource vector **R_i = [L_i, P_i]** (load, privacy) and a greedy budget
  selector.
* **Parent B** – high‑dimensional feature extraction whose output is treated as a
  discrete path; a path‑signature (iterated‑integral) is computed to capture
  structural interactions.

**Mathematical bridge**

Each datum (a textual snippet possibly linked to a spatial entity) is first
converted into the 2‑D resource vector of Parent A.  The same datum is also
mapped to a high‑dimensional master‑feature vector **v_i** (from Parent B).
By concatenating **[L_i, P_i]** with **v_i** we obtain a point **p_i ∈
ℝ^{2+d}**.  The ordered sequence **{p_i}** defines a discrete path in this
space.  Applying the (truncated) path signature to the sequence yields a
compact descriptor **S** that aggregates all pairwise interactions of the
augmented resources.  The selector of Parent A then operates on the original
resource matrix while the signature can be used for downstream ranking or
as an additional budget‑aware constraint.

The module implements the full pipeline:
1. textual cue extraction → load / privacy,
2. master‑feature extraction,
3. construction of the augmented path,
4. discrete path‑signature computation,
5. greedy budget selection.
"""

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

# Example positive / negative weight vectors (length = 3 for the three cue types)
W_POS = np.array([1.2, 0.8, 0.5])   # evidence, planning, delay
W_NEG = np.array([0.3, 0.2, 1.0])   # same order, penalising delay more

def _count_cues(text: str) -> np.ndarray:
    """Return raw cue counts for evidence, planning, delay."""
    return np.array([
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
    ], dtype=float)

def compute_load_privacy(text: str) -> Tuple[float, float]:
    """
    Convert textual cues into the 2‑dimensional resource vector.
    Load  =  c·W_POS  –  c·W_NEG
    Privacy = sum of risk‑related cues (here we treat delay as privacy penalty)
    """
    c = _count_cues(text)
    load = float(c @ (W_POS - W_NEG))
    privacy = float(c[2] * 0.7)  # delay cues weighted as privacy penalty
    return load, privacy

# ----------------------------------------------------------------------
# Parent B – feature extraction (deterministic pseudo‑random)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature generation based on the text hash."""
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
    """
    Produce a fixed‑size master vector (dim = 5) from the full feature dict.
    The selection mirrors the original Parent B reduction.
    """
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
    """Return distance in kilometres between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def spatial_load(lat: float, lon: float, ref: Tuple[float, float]) -> float:
    """Load contribution derived from spatial distance."""
    return haversine_distance(lat, lon, *ref)

# ----------------------------------------------------------------------
# Path‑signature utilities (Parent B mathematical core)
# ----------------------------------------------------------------------
def discrete_path_signature(path: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Compute a truncated discrete signature of a path.
    *level = 1* returns the total increment (∑Δp).
    *level = 2* also returns the second‑order iterated integral approximated by
    Σ_{i≤j} Δp_i ⊗ Δp_j.
    The result is flattened to a 1‑D vector.
    """
    if path.shape[0] < 2:
        raise ValueError("Path must contain at least two points.")
    increments = np.diff(path, axis=0)  # shape (m, d)
    # Level‑1 term
    sig1 = increments.sum(axis=0)  # shape (d,)

    if level == 1:
        return sig1

    # Level‑2 term (symmetric tensor)
    d = increments.shape[1]
    sig2 = np.zeros((d, d), dtype=float)
    m = increments.shape[0]
    for i in range(m):
        for j in range(i, m):
            sig2 += np.outer(increments[i], increments[j])
    # Flatten upper triangle (including diagonal) for compactness
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
    """
    Greedy selector that returns indices of chosen items.
    Resources shape: (n, 2) where column 0 = load, column 1 = privacy.
    Items are ordered by ascending load+privacy (a simple heuristic).
    """
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
            chosen.append(int(idx))
            load_used += load_i
            privacy_used += priv_i
    return chosen

# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def build_augmented_path(
    texts: List[str],
    spatial_coords: List[Tuple[float, float]],
    spatial_ref: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each datum produce:
      * a 2‑D resource vector (load, privacy) – Parent A,
      * a master feature vector (dim = 5) – Parent B.
    Concatenate them into points of dimension 7, forming a discrete path.
    Returns:
        path   – array of shape (n, 7),
        R2d    – the original 2‑D resource matrix (n, 2) for budget selection.
    """
    if len(texts) != len(spatial_coords):
        raise ValueError("texts and spatial_coords must have the same length.")
    points = []
    R2d = []
    for txt, (lat, lon) in zip(texts, spatial_coords):
        # Parent A contribution
        load_txt, priv_txt = compute_load_privacy(txt)
        load_spatial = spatial_load(lat, lon, spatial_ref)
        load = load_txt + load_spatial
        privacy = priv_txt  # spatial privacy is not added in this example
        R2d.append([load, privacy])

        # Parent B contribution
        master_vec = extract_master_vector(txt)  # shape (5,)

        # Concatenate
        points.append(np.concatenate([[load, privacy], master_vec]))
    return np.array(points, dtype=float), np.array(R2d, dtype=float)

def hybrid_fusion(
    texts: List[str],
    spatial_coords: List[Tuple[float, float]],
    spatial_ref: Tuple[float, float],
    spatial_budget: float,
    privacy_budget: float,
    signature_level: int = 2,
) -> Dict[str, object]:
    """
    Execute the full hybrid algorithm:
      1. Build augmented path and 2‑D resource matrix.
      2. Compute discrete path signature.
      3. Perform greedy budget selection on the 2‑D resources.
    Returns a dictionary with keys:
        'signature' – np.ndarray,
        'selected_indices' – List[int],
        'selected_resources' – np.ndarray (rows of resources matrix).
    """
    path, R2d = build_augmented_path(texts, spatial_coords, spatial_ref)
    signature = discrete_path_signature(path, level=signature_level)
    selected_idx = select_under_budget(R2d, spatial_budget, privacy_budget)
    selected_resources = R2d[selected_idx] if selected_idx else np.empty((0, 2))
    return {
        "signature": signature,
        "selected_indices": selected_idx,
        "selected_resources": selected_resources,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example inputs
    sample_texts = [
        "The evidence confirms the source and includes a screenshot.",
        "We need to plan the next steps and schedule the test.",
        "Please wait until tomorrow, we will defer the audit.",
        "Audit completed, all records verified and logged.",
    ]
    # Random lat/lon pairs around a reference point (e.g., London)
    reference_point = (51.5074, -0.1278)  # London
    random.seed(0)
    sample_coords = [
        (51.5 + random.uniform(-0.1, 0.1), -0.12 + random.uniform(-0.1, 0.1))
        for _ in sample_texts
    ]

    # Budgets
    spatial_budget = 500.0   # km‑equivalent load budget
    privacy_budget = 2.0     # privacy penalty budget

    result = hybrid_fusion(
        texts=sample_texts,
        spatial_coords=sample_coords,
        spatial_ref=reference_point,
        spatial_budget=spatial_budget,
        privacy_budget=privacy_budget,
        signature_level=2,
    )

    print("Signature vector (len={}):".format(len(result["signature"])))
    print(result["signature"])
    print("\nSelected indices:", result["selected_indices"])
    print("Selected resources (load, privacy):")
    print(result["selected_resources"])
    sys.exit(0)