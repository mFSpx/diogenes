# DARWIN HAMMER — match 22, survivor 6
# gen: 3
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s5.py (gen1)
# parent_b: hybrid_possum_filter_hybrid_privacy_model_m53_s2.py (gen2)
# born: 2026-05-29T23:25:23Z

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

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


def _count_matches(pattern, text):
    return len(pattern.findall(text))


def extract_text_features(text: str) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.array([_count_matches(_REGEX_MAP[name], text) for name in _FEATURE_ORDER], dtype=np.int64)

    load = np.dot(_POSITIVE_WEIGHTS, counts) - np.dot(_NEGATIVE_WEIGHTS, counts)
    privacy = np.dot(_NEGATIVE_WEIGHTS[[6, 7, 8]], counts)

    return counts, np.array([load, privacy])


# ----------------------------------------------------------------------
# Parent B – spatial / signature utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
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
) -> np.ndarray:
    load = haversine_m((entity.lat, entity.lon), reference)
    privacy = beta if entity.address_signature in colliding_signatures else 0.0
    return np.array([load, privacy])


# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------
def build_resource_matrix(
    texts: List[str],
    entities: List[Entity],
    reference_point: Tuple[float, float],
    beta: float = 1.0,
) -> np.ndarray:
    rows = []

    for txt in texts:
        _, features = extract_text_features(txt)
        rows.append(features)

    signature_counts = {}
    for e in entities:
        signature_counts[e.address_signature] = signature_counts.get(e.address_signature, 0) + 1
    colliding = {sig for sig, cnt in signature_counts.items() if cnt > 1}

    for ent in entities:
        rows.append(entity_resource_vector(ent, reference_point, colliding, beta))

    return np.array(rows)


def select_under_budget(R: np.ndarray, spatial_budget: float, privacy_budget: float) -> np.ndarray:
    num_resources = R.shape[0]
    x = np.zeros(num_resources, dtype=np.int64)

    remaining_spatial_budget = spatial_budget
    remaining_privacy_budget = privacy_budget

    for i in np.argsort(np.linalg.norm(R, axis=1))[::-1]:
        load, privacy = R[i]
        if load <= remaining_spatial_budget and privacy <= remaining_privacy_budget:
            x[i] = 1
            remaining_spatial_budget -= load
            remaining_privacy_budget -= privacy

    return x


import re

def main():
    texts = ["This is a test text with evidence and planning."]
    entities = [Entity("1", 37.7749, -122.4194, "test", 0.0, "sig1")]
    reference_point = (37.7749, -122.4194)

    R = build_resource_matrix(texts, entities, reference_point)
    x = select_under_budget(R, 1000000.0, 1000.0)

    print(R)
    print(x)


if __name__ == "__main__":
    main()