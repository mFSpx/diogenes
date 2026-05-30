# DARWIN HAMMER — match 1482, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_possum_filter_m323_s2.py (gen4)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s1.py (gen2)
# born: 2026-05-29T23:36:52Z

import math
import numpy as np
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Iterable

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    lat: float = 0.0
    lon: float = 0.0

@dataclass(frozen=True)
class Entity:
    identifier: str
    score: float
    lat: float
    lon: float

class PheromoneEntry:
    __slots__ = ("uuid", "key", "kind", "value", "half_life_seconds", "created_at", "last_decay")

    def __init__(self, key: str, kind: str, value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.key = key
        self.kind = kind
        self.value = value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        age = self.age_seconds()
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (age / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.value *= factor
        self.last_decay = datetime.now(timezone.utc)

class PheromoneStore:
    def __init__(self):
        self.entries: Dict[str, PheromoneEntry] = {}

    def add_or_update(self, key: str, kind: str, value: float, half_life_seconds: int = 3600) -> None:
        if key in self.entries:
            entry = self.entries[key]
            entry.apply_decay()
            entry.value += value
            entry.last_decay = datetime.now(timezone.utc)
        else:
            self.entries[key] = PheromoneEntry(key, kind, value, half_life_seconds)

    def decay_all(self) -> None:
        for entry in self.entries.values():
            entry.apply_decay()

    def total_entropy(self) -> float:
        values = np.array([e.value for e in self.entries.values() if e.value > 0])
        if values.size == 0:
            return 0.0
        probs = values / values.sum()
        return -np.sum(probs * np.log(probs + 1e-12))

EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)

def hygiene_feature_vector(text: str) -> np.ndarray:
    counts = np.array([
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
        len(SUPPORT_RE.findall(text)),
        len(BOUNDARY_RE.findall(text)),
        len(OUTCOME_RE.findall(text)),
        len(IMPULSIVE_RE.findall(text)),
    ], dtype=float)
    return counts

def normalized_shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    H = -np.sum(probs * np.log(probs + 1e-12))
    k = np.count_nonzero(counts)
    H_max = math.log(k) if k > 1 else 0.0
    return H / H_max if H_max > 0 else 0.0

def ternary_lens_factor(audit_report: Dict[str, Any]) -> float:
    mapping = {"low": 0.9, "medium": 1.0, "high": 1.2}
    risk = audit_report.get("risk", "medium").lower()
    return mapping.get(risk, 1.0)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def joint_weight(span: Span, entity: Entity, text: str, audit_report: Dict[str, Any], λ: float, α: float) -> float:
    d = haversine_distance(span.lat, span.lon, entity.lat, entity.lon)
    counts = hygiene_feature_vector(text)
    Ĥ = normalized_shannon_entropy(counts)
    return (span.score * entity.score) * math.exp(-d/λ) * (1 + α*Ĥ) * ternary_lens_factor(audit_report)

def update_pheromones(pheromone_store: PheromoneStore, key: str, kind: str, value: float, half_life_seconds: int = 3600) -> None:
    pheromone_store.add_or_update(key, kind, value, half_life_seconds)

def hybrid_document_score(document: str, spans: List[Span], entities: List[Entity], audit_reports: List[Dict[str, Any]], λ: float, α: float) -> float:
    score = 0.0
    for span, entity, audit_report in zip(spans, entities, audit_reports):
        weight = joint_weight(span, entity, document, audit_report, λ, α)
        score += weight
    return score