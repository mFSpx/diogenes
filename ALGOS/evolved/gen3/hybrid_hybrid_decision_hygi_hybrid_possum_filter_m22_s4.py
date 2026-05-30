# DARWIN HAMMER — match 22, survivor 4
# gen: 3
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s5.py (gen1)
# parent_b: hybrid_possum_filter_hybrid_privacy_model_m53_s2.py (gen2)
# born: 2026-05-29T23:25:23Z

"""hybrid_text_spatial_privacy_fusion.py
Hybrid algorithm merging:

* **Parent A** – regex‑based textual cue extraction with positive/negative
  weight vectors (decision hygiene & Shannon entropy).
* **Parent B** – spatial‑signature resource vectors (distance, privacy load) and a
  linear‑budget selection model.

**Mathematical bridge**

Both parents can be expressed as *resource vectors* that are summed into a
single matrix **R** ∈ ℝⁿˣ²:


R_i = [ L_i , P_i ]


* **L_i** – “load” dimension.
    – For textual inputs (Parent A) we collapse the weighted cue vector **c**
      (length = 9) into a scalar load using the dot‑product with the positive
      weight vector **w⁺** minus the negative weight vector **w⁻**.
    – For spatial entities (Parent B) the load is the haversine distance
      *d* from a reference point.

* **P_i** – “privacy” dimension.
    – For textual inputs we treat risk‑related cues (impulsive, scarcity,
      risk) as a privacy penalty, summed with the same weight scheme.
    – For spatial entities the privacy term is β·σ where σ∈{0,1} flags a
      signature collision.

The combined matrix **R** is then fed to a greedy linear‑budget selector that
chooses a subset **x** (binary vector) satisfying


Rᵀ·x ≤ [ spatial_budget , privacy_budget ] .


The three core functions below realise this fusion:
`extract_text_features`, `entity_resource_vector`, and `select_under_budget`. """

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Set

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
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
)

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 2000, 2500, 3000], dtype=np.int64)

_REGEX_MAP = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
    "risk": RISK_RE,
}


def _count_matches(pattern: re.Pattern, text: str) -> int:
    """Return non‑overlapping match count for *pattern* in *text*."""
    return len(pattern.findall(text))


def extract_text_features(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyse *text* with the Parent A regexes.

    Returns
    -------
    cue_counts : np.ndarray, shape (9,)
        Raw occurrence counts ordered according to ``_FEATURE_ORDER``.
    weighted_score : (load, privacy) tuple
        *load*  = Σ (positive_weight – negative_weight) * count  
        *privacy* = Σ (negative_weight) * count for the risk‑related cues
        (impulsive, scarcity, risk).
    """
    counts = np.array([_count_matches(_REGEX_MAP[name], text) for name in _FEATURE_ORDER], dtype=np.int64)

    # Load contribution: positive minus negative (negative only penalises load for risk cues)
    load = np.dot(_POSITIVE_WEIGHTS, counts) - np.dot(_NEGATIVE_WEIGHTS, counts)

    # Privacy contribution: only the risk‑related negative weights
    privacy = np.dot(_NEGATIVE_WEIGHTS, counts)

    return counts, (float(load), float(privacy))


# ----------------------------------------------------------------------
# Parent B – spatial / signature utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """Simple geographic entity used by Parent B."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000 * math.sqrt(h)


def entity_resource_vector(
    entity: Entity,
    reference: Tuple[float, float],
    colliding_signatures: Set[str],
    beta: float = 1.0,
) -> Tuple[float, float]:
    """
    Build the 2‑dimensional resource vector for *entity*.

    Parameters
    ----------
    entity : Entity
    reference : (lat, lon) tuple used as the spatial origin.
    colliding_signatures : set of signatures that appear more than once.
    beta : scaling factor for the privacy term.

    Returns
    -------
    (load, privacy) tuple where
        load = haversine distance from *reference*,
        privacy = beta if the entity's signature collides, else 0.
    """
    load = haversine_m((entity.lat, entity.lon), reference)
    privacy = beta if entity.address_signature in colliding_signatures else 0.0
    return load, privacy


# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------
def build_resource_matrix(
    texts: List[str],
    entities: List[Entity],
    reference_point: Tuple[float, float],
    beta: float = 1.0,
) -> np.ndarray:
    """
    Construct the combined resource matrix **R** ∈ ℝⁿˣ².

    Rows 0 … len(texts)‑1 correspond to textual inputs (Parent A),
    rows after that correspond to entities (Parent B).

    Returns
    -------
    np.ndarray of shape (n_rows, 2) with columns [load, privacy].
    """
    rows: List[Tuple[float, float]] = []

    # ---- Textual rows -------------------------------------------------
    for txt in texts:
        _, (load, privacy) = extract_text_features(txt)
        rows.append((load, privacy))

    # ---- Entity rows --------------------------------------------------
    # Determine colliding signatures once for the whole batch.
    signature_counts: Dict[str, int] = {}
    for e in entities:
        signature_counts[e.address_signature] = signature_counts.get(e.address_signature, 0) + 1
    colliding = {sig for sig, cnt in signature_counts.items() if cnt > 1}

    for ent in entities:
        rows.append(entity_resource_vector(ent, reference_point, colliding, beta=beta))

    return np.asarray(rows, dtype=np.float64)


def greedy_select_under_budget(
    resource_matrix: np.ndarray,
    spatial_budget: float,
    privacy_budget: float,
) -> List[int]:
    """
    Greedy selection of rows that respect the two‑dimensional budgets.

    The heuristic picks the row with the smallest *load*+*privacy* ratio
    among those that still fit inside the remaining budget.

    Parameters
    ----------
    resource_matrix : np.ndarray, shape (n,2)
        Combined [load, privacy] matrix.
    spatial_budget : float
        Upper bound on the sum of loads.
    privacy_budget : float
        Upper bound on the sum of privacy terms.

    Returns
    -------
    List[int]
        Indices of the selected rows (in the original order).
    """
    n = resource_matrix.shape[0]
    selected: List[int] = []
    remaining_load = spatial_budget
    remaining_priv = privacy_budget
    available = set(range(n))

    while available:
        # Compute feasibility mask
        feasible = [
            i
            for i in available
            if resource_matrix[i, 0] <= remaining_load and resource_matrix[i, 1] <= remaining_priv
        ]
        if not feasible:
            break

        # Choose the feasible row with minimal (load+privacy)/1 (simple ratio)
        best = min(feasible, key=lambda i: resource_matrix[i, 0] + resource_matrix[i, 1])
        selected.append(best)

        # Update budgets
        remaining_load -= resource_matrix[best, 0]
        remaining_priv -= resource_matrix[best, 1]

        # Remove from pool
        available.remove(best)

    return selected


def summarize_selection(
    selected_indices: List[int],
    resource_matrix: np.ndarray,
) -> Dict[str, float]:
    """
    Compute aggregate statistics for a selection.

    Returns a dictionary with total load, total privacy and count.
    """
    total_load = float(resource_matrix[selected_indices, 0].sum())
    total_priv = float(resource_matrix[selected_indices, 1].sum())
    return {
        "selected_count": len(selected_indices),
        "total_load": total_load,
        "total_privacy": total_priv,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample textual inputs
    sample_texts = [
        "I have evidence that the plan is solid, but I'm scared and might act impulsively.",
        "We need to check the source, verify the data, and then ship the product.",
        "I'm broke, can't afford rent, and I'm feeling hopeless.",
    ]

    # Sample entities
    sample_entities = [
        Entity(id="e1", lat=37.7749, lon=-122.4194, category="store", address_signature="sigA"),
        Entity(id="e2", lat=34.0522, lon=-118.2437, category="cafe", address_signature="sigB"),
        Entity(id="e3", lat=40.7128, lon=-74.0060, category="office", address_signature="sigA"),  # collides with e1
    ]

    # Reference point (e.g., user location)
    ref_point = (36.0, -120.0)

    # Build matrix
    R = build_resource_matrix(sample_texts, sample_entities, ref_point, beta=5000.0)

    # Define generous budgets for demonstration
    spatial_budget = 2.0e7  # ~20 000 km total distance allowance
    privacy_budget = 1.0e4  # arbitrary privacy units

    chosen = greedy_select_under_budget(R, spatial_budget, privacy_budget)
    stats = summarize_selection(chosen, R)

    print("Resource matrix (load, privacy):")
    print(R)
    print("\nChosen row indices:", chosen)
    print("Selection summary:", stats)