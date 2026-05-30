# DARWIN HAMMER — match 22, survivor 5
# gen: 3
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s5.py (gen1)
# parent_b: hybrid_possum_filter_hybrid_privacy_model_m53_s2.py (gen2)
# born: 2026-05-29T23:25:23Z

"""Hybrid Decision‑Hygiene & Spatial‑Privacy Model

Parents
-------
* **Parent A** – `hybrid_decision_hygiene_shannon_entropy_m12_s5.py`  
  Provides regex‑based extraction of psychological cues from free‑form text and
  maps them to a weighted feature vector (positive and negative contributions).

* **Parent B** – `hybrid_possum_filter_hybrid_privacy_model_m53_s2.py`  
  Defines entities with a 2‑dimensional resource vector `[spatial_load,
  privacy_load]` and selects a feasible subset under linear budgets.

Mathematical Bridge
-------------------
The bridge is the *privacy_load* dimension.  
Parent A yields a scalar “cognitive‑risk” score `c_i` for each textual record
by a linear combination of feature counts with the weight vectors
`w⁺` (positive) and `w⁻` (negative):


c_i = w⁺·f_i⁺ + w⁻·f_i⁻


We reinterpret `c_i` as the privacy‑load `p_i` of an entity, i.e. the
privacy‑impact of exposing that textual evidence.  
Parent B already supplies a spatial load `d_i` (haversine distance).  
Thus each entity `e_i` obtains the unified resource vector


r_i = [ d_i ,  p_i ] .


All vectors are stacked into a matrix **A** and a binary indicator **x**
must satisfy the linear budget constraints


Aᵀ·x ≤ [ spatial_budget , privacy_budget ] .


The implementation below builds this matrix, computes the cognitive‑risk
score from Parent A, and runs a greedy selector that respects both budgets,
 thereby fusing the two topologies into a single decision system.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import re

# ----------------------------------------------------------------------
# Parent A – regexes and weighted feature extraction
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

# Positive contributions (desired cues)
_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)

# Negative contributions (undesired cues) – chosen to penalise risky language
_NEGATIVE_WEIGHTS = np.array([-800, -600, -700, -500, -600, -400, -200, -300, -500], dtype=np.int64)

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


def extract_feature_counts(text: str) -> Dict[str, int]:
    """
    Scan *text* with the nine regexes and return a dict mapping each feature
    name to the number of matches.
    """
    counts: Dict[str, int] = {}
    for name, regex in _REGEX_MAP.items():
        matches = regex.findall(text)
        counts[name] = len(matches)
    return counts


def compute_cognitive_risk(counts: Dict[str, int]) -> int:
    """
    Convert raw counts into a single scalar risk score using the linear
    combination of positive and negative weight vectors.
    """
    # Build ordered count vectors for positive and negative terms
    pos_counts = np.array([counts[name] for name in _FEATURE_ORDER], dtype=np.int64)
    neg_counts = pos_counts.copy()  # same counts used for both weight sets
    risk = int(_POSITIVE_WEIGHTS @ pos_counts + _NEGATIVE_WEIGHTS @ neg_counts)
    return risk


# ----------------------------------------------------------------------
# Parent B – spatial‑signature utilities and budgeted selection
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """
    Represents a candidate item that carries both a geographic location and a
    textual description whose cognitive‑risk score will be used as the
    privacy‑load component.
    """
    id: str
    lat: float
    lon: float
    description: str
    category: str = ""
    score: float = 0.0  # optional auxiliary score (e.g., relevance)


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Great‑circle distance in metres between two (lat,lon) points.
    """
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000 * math.sqrt(h)


def build_resource_matrix(
    entities: List[Entity],
    reference_point: Tuple[float, float],
) -> np.ndarray:
    """
    For each *entity* compute the 2‑dimensional resource vector
    `[spatial_load, privacy_load]`:

    * spatial_load = haversine distance from *reference_point*.
    * privacy_load = cognitive‑risk score derived from the entity's description.

    Returns a matrix ``A`` of shape (len(entities), 2).
    """
    rows = []
    for ent in entities:
        spatial = haversine_m((ent.lat, ent.lon), reference_point)
        counts = extract_feature_counts(ent.description)
        privacy = compute_cognitive_risk(counts)
        rows.append([spatial, privacy])
    return np.array(rows, dtype=np.float64)


def greedy_budget_selector(
    resource_matrix: np.ndarray,
    spatial_budget: float,
    privacy_budget: float,
) -> List[int]:
    """
    Greedy algorithm that selects a subset of rows (entities) such that the
    cumulative spatial and privacy loads stay within the supplied budgets.
    The heuristic sorts candidates by the ratio of (spatial+privacy) to a
    random tie‑breaker to favour low‑cost items.

    Returns the list of selected row indices.
    """
    n = resource_matrix.shape[0]
    indices = list(range(n))

    # Compute a simple cost metric; lower is better.
    costs = resource_matrix.sum(axis=1) + np.random.rand(n) * 1e-6
    indices.sort(key=lambda i: costs[i])

    selected: List[int] = []
    cum_spatial = 0.0
    cum_privacy = 0.0

    for i in indices:
        d, p = resource_matrix[i]
        if cum_spatial + d <= spatial_budget and cum_privacy + p <= privacy_budget:
            selected.append(i)
            cum_spatial += d
            cum_privacy += p
    return selected


# ----------------------------------------------------------------------
# Hybrid API – three demonstrative functions
# ----------------------------------------------------------------------
def evaluate_entities(
    entities: List[Entity],
    reference_point: Tuple[float, float],
    spatial_budget: float,
    privacy_budget: float,
) -> List[Entity]:
    """
    End‑to‑end pipeline:
    1. Build the unified resource matrix.
    2. Run the greedy selector.
    3. Return the concrete ``Entity`` objects that survive the budgets.
    """
    A = build_resource_matrix(entities, reference_point)
    chosen_idx = greedy_budget_selector(A, spatial_budget, privacy_budget)
    return [entities[i] for i in chosen_idx]


def summarize_selection(selected: List[Entity], reference_point: Tuple[float, float]) -> Dict[str, Any]:
    """
    Produce a short report for a list of selected entities:
    total distance, total cognitive risk, and a breakdown by category.
    """
    total_distance = sum(haversine_m((e.lat, e.lon), reference_point) for e in selected)
    total_privacy = sum(compute_cognitive_risk(extract_feature_counts(e.description)) for e in selected)
    by_category: Dict[str, int] = {}
    for e in selected:
        by_category[e.category] = by_category.get(e.category, 0) + 1
    return {
        "count": len(selected),
        "total_distance_m": total_distance,
        "total_privacy_score": total_privacy,
        "category_breakdown": by_category,
    }


def random_entities(num: int, center: Tuple[float, float], radius_km: float) -> List[Entity]:
    """
    Helper that creates *num* synthetic ``Entity`` objects around *center*
    within *radius_km*. Descriptions are randomly populated with a few cue words.
    """
    cue_pool = list(_REGEX_MAP.keys())
    entities: List[Entity] = []
    for i in range(num):
        # Random polar offset
        angle = random.random() * 2 * math.pi
        distance = random.random() * radius_km * 1000  # metres
        dlat = (distance / 6_371_000) * math.cos(angle)
        dlon = (distance / 6_371_000) * math.sin(angle) / math.cos(math.radians(center[0]))
        lat = center[0] + math.degrees(dlat)
        lon = center[1] + math.degrees(dlon)

        # Random description with 0‑3 cue words
        words = random.sample(cue_pool, k=random.randint(0, 3))
        description = " ".join(words) + " normal text."

        entities.append(
            Entity(
                id=f"ent_{i}",
                lat=lat,
                lon=lon,
                description=description,
                category=random.choice(["A", "B", "C"]),
            )
        )
    return entities


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reference location – arbitrary (e.g., city centre)
    REF_POINT = (37.7749, -122.4194)  # San Francisco approx.

    # Generate synthetic data
    synthetic = random_entities(num=30, center=REF_POINT, radius_km=20)

    # Define generous budgets to ensure at least a few selections
    SPATIAL_BUDGET = 150_000.0   # 150 km in metres
    PRIVACY_BUDGET = 20_000      # arbitrary risk units

    selected = evaluate_entities(
        entities=synthetic,
        reference_point=REF_POINT,
        spatial_budget=SPATIAL_BUDGET,
        privacy_budget=PRIVACY_BUDGET,
    )

    report = summarize_selection(selected, REF_POINT)

    print("Selected entities:", [e.id for e in selected])
    print("Report:", report)
    sys.exit(0)