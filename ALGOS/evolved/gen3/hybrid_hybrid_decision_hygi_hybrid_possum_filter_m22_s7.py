# DARWIN HAMMER — match 22, survivor 7
# gen: 3
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s5.py (gen1)
# parent_b: hybrid_possum_filter_hybrid_privacy_model_m53_s2.py (gen2)
# born: 2026-05-29T23:25:23Z

import math
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

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
    return len(pattern.findall(text))

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

def extract_text_features(text: str) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.array([_count_matches(_REGEX_MAP[name], text) for name in _FEATURE_ORDER], dtype=np.int64)
    load = np.dot(_POSITIVE_WEIGHTS, counts) - np.dot(_NEGATIVE_WEIGHTS, counts)
    privacy = np.dot(_NEGATIVE_WEIGHTS, counts)
    return counts, (float(load), float(privacy))

def entity_resource_vector(
    entity: Entity,
    reference: Tuple[float, float],
    colliding_signatures: Set[str],
    beta: float = 1.0,
) -> Tuple[float, float]:
    load = haversine_m((entity.lat, entity.lon), reference)
    privacy = beta if entity.address_signature in colliding_signatures else 0.0
    return load, privacy

def build_resource_matrix(
    texts: List[str],
    entities: List[Entity],
    reference_point: Tuple[float, float],
    beta: float = 1.0,
) -> np.ndarray:
    rows: List[Tuple[float, float]] = []
    for txt in texts:
        _, (load, privacy) = extract_text_features(txt)
        rows.append((load, privacy))

    signature_counts: Dict[str, int] = {}
    for e in entities:
        signature_counts[e.address_signature] = signature_counts.get(e.address_signature, 0) + 1
    colliding = {sig for sig, cnt in signature_counts.items() if cnt > 1}

    for ent in entities:
        load, privacy = entity_resource_vector(ent, reference_point, colliding, beta)
        rows.append((load, privacy))

    return np.array(rows)

def select_under_budget(resource_matrix: np.ndarray, spatial_budget: float, privacy_budget: float) -> np.ndarray:
    num_rows = resource_matrix.shape[0]
    best_selection = np.zeros(num_rows)
    best_value = 0.0

    for i in range(1 << num_rows):
        selection = np.array([bool(int(x)) for x in bin(i)[2:].zfill(num_rows)])
        total_load = np.sum(resource_matrix[:, 0] * selection)
        total_privacy = np.sum(resource_matrix[:, 1] * selection)

        if total_load <= spatial_budget and total_privacy <= privacy_budget:
            value = np.sum(selection)
            if value > best_value:
                best_value = value
                best_selection = selection

    return best_selection

def improved_hybrid_algorithm(
    texts: List[str],
    entities: List[Entity],
    reference_point: Tuple[float, float],
    beta: float = 1.0,
    spatial_budget: float = 10000.0,
    privacy_budget: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    resource_matrix = build_resource_matrix(texts, entities, reference_point, beta)
    selection = select_under_budget(resource_matrix, spatial_budget, privacy_budget)
    return resource_matrix, selection

texts = ["example text 1", "example text 2"]
entities = [Entity("id1", 37.7749, -122.4194, "category1"), Entity("id2", 34.0522, -118.2437, "category2")]
reference_point = (37.7749, -122.4194)
beta = 1.0
spatial_budget = 10000.0
privacy_budget = 10.0

resource_matrix, selection = improved_hybrid_algorithm(texts, entities, reference_point, beta, spatial_budget, privacy_budget)
print(resource_matrix)
print(selection)