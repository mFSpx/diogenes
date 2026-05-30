# DARWIN HAMMER — match 996, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s2.py (gen3)
# parent_b: hybrid_hdc_hybrid_hybrid_decisi_m131_s1.py (gen3)
# born: 2026-05-29T23:32:12Z

"""Hybrid algorithm merging:

- Parent A: decision‑hygiene feature scoring, spatial‑signature filtering, privacy‑aware linear budgeting.
- Parent B: Hyperdimensional Computing (HDC) bipolar vectors combined with decision‑hygiene regex features.

Mathematical bridge:
Both parents rely on the same set of nine decision‑hygiene regex features.
We first extract raw counts for these features (Parent A & B common step).  
Each feature i is assigned a deterministic random bipolar hypervector h_i∈{‑1,+1}^D (HDC from Parent B).  
A weighted aggregation produces a single high‑dimensional representation:

    v = sign( Σ_i w_i·c_i·h_i )    (1)

where c_i is the count of feature i, w_i = +P_i − N_i combines the positive (P) and negative (N) weights
defined in Parent B, and sign maps positive entries to +1 and non‑positive to ‑1.
The resulting vector v is used as the “entity score” in Parent A’s spatial‑signature filter.
Selection of a subset of entities then obeys linear privacy‑ and resource‑budget constraints,
exactly as in Parent A.

The code below implements this fused pipeline."""
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants (shared with Parent B)
# ----------------------------------------------------------------------
DIM = 10000  # HDC dimensionality

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
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

# ----------------------------------------------------------------------
# Regex patterns (identical to those in both parents)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|"
    r"criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|"
    r"first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|"
    r"support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|"
    r"safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|"
    r"destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|"
    r"poverty|need money)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:risk|danger|dangerous|threat|unsafe|hazard|expose|vulnerable|peril)\b",
    re.I,
)

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

# ----------------------------------------------------------------------
# Helper: deterministic bipolar hypervector per feature
# ----------------------------------------------------------------------
_hv_cache: Dict[str, np.ndarray] = {}


def _bipolar_hv(name: str) -> np.ndarray:
    """Return a deterministic bipolar (+1 / -1) hypervector for *name*."""
    if name not in _hv_cache:
        # Use a stable seed derived from the feature name
        seed = abs(hash(name)) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.integers(0, 2, size=DIM, dtype=np.int8) * 2 - 1  # {‑1, +1}
        _hv_cache[name] = vec.astype(np.int8)
    return _hv_cache[name]


# ----------------------------------------------------------------------
# Core Functions (Hybrid Operations)
# ----------------------------------------------------------------------


def extract_feature_counts(text: str) -> Dict[str, int]:
    """
    Count occurrences of each decision‑hygiene regex in *text*.
    Mirrors the feature extraction steps of both parents.
    """
    counts: Dict[str, int] = {}
    for feat, pattern in _REGEX_MAP.items():
        matches = pattern.findall(text)
        counts[feat] = len(matches)
    return counts


def hyperdimensional_representation(counts: Dict[str, int]) -> np.ndarray:
    """
    Build the high‑dimensional vector v (Eq. 1) from feature counts.
    Positive and negative weights from Parent B are combined as
        w_i = +P_i – N_i.
    The aggregation is a weighted sum of deterministic bipolar hypervectors,
    followed by a sign‑binarization to obtain a bipolar result.
    """
    # Prepare weight vector w_i
    pos = np.array([_POSITIVE_WEIGHTS[i] for i, f in enumerate(_FEATURE_ORDER)], dtype=np.int64)
    neg = np.array([_NEGATIVE_WEIGHTS[i] for i, f in enumerate(_FEATURE_ORDER)], dtype=np.int64)
    w = pos - neg  # shape (9,)

    # Accumulate
    agg = np.zeros(DIM, dtype=np.int64)
    for i, feat in enumerate(_FEATURE_ORDER):
        c = counts.get(feat, 0)
        if c == 0:
            continue
        hv = _bipolar_hv(feat).astype(np.int64)  # cast for multiplication
        agg += w[i] * c * hv

    # Sign‑binarize to bipolar (+1 / -1)
    v = np.where(agg >= 0, 1, -1).astype(np.int8)
    return v


@dataclass
class Entity:
    """
    Represents an entity in the spatial‑signature space.
    *vector* is the hyperdimensional representation derived from its textual content.
    *resource* and *privacy* are scalar costs used in the linear budget constraints
    (mirroring Parent A's privacy‑aware model‑resource formulation).
    """
    name: str
    vector: np.ndarray
    resource: float
    privacy: float


def build_entity(name: str, text: str, resource: float, privacy: float) -> Entity:
    """
    Construct an Entity from raw *text* and associated costs.
    The text is transformed into a hyperdimensional vector using the shared
    decision‑hygiene features.
    """
    counts = extract_feature_counts(text)
    vec = hyperdimensional_representation(counts)
    return Entity(name=name, vector=vec, resource=resource, privacy=privacy)


def select_entities(
    entities: List[Entity],
    query_vec: np.ndarray,
    max_privacy: float,
    max_resource: float,
) -> List[Tuple[Entity, float]]:
    """
    Spatial‑signature filtering with linear budget constraints.

    1. Compute cosine‑like similarity s_i = (q·v_i) / D.
    2. Sort entities by descending similarity.
    3. Greedily pick entities while respecting total privacy ≤ max_privacy
       and total resource ≤ max_resource.

    Returns a list of (Entity, similarity) tuples for the selected subset.
    """
    # Similarity (dot product normalized by dimension)
    sims = [(e, float(np.dot(query_vec, e.vector)) / DIM) for e in entities]
    sims.sort(key=lambda pair: pair[1], reverse=True)

    selected: List[Tuple[Entity, float]] = []
    used_privacy = 0.0
    used_resource = 0.0

    for ent, sim in sims:
        if used_privacy + ent.privacy > max_privacy:
            continue
        if used_resource + ent.resource > max_resource:
            continue
        selected.append((ent, sim))
        used_privacy += ent.privacy
        used_resource += ent.resource

    return selected


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example texts (short snippets)
    txt_a = "The evidence was verified and the plan was approved. No delay expected."
    txt_b = "I feel impulsive, need support now, cannot afford the risk."
    txt_c = "Boundary was set, outcome successful, privacy respected."
    txt_query = "We need evidence, a solid plan and no delay."

    # Build entities with arbitrary resource/privacy costs
    entities = [
        build_entity("Alpha", txt_a, resource=3.2, privacy=1.0),
        build_entity("Beta", txt_b, resource=5.5, privacy=2.5),
        build_entity("Gamma", txt_c, resource=2.1, privacy=0.8),
    ]

    # Build query vector
    query_counts = extract_feature_counts(txt_query)
    query_vec = hyperdimensional_representation(query_counts)

    # Apply hybrid selection
    chosen = select_entities(
        entities,
        query_vec,
        max_privacy=3.0,
        max_resource=8.0,
    )

    # Output results
    print("Selected entities (name, similarity):")
    for ent, sim in chosen:
        print(f"- {ent.name}: {sim:.4f} (resource={ent.resource}, privacy={ent.privacy})")