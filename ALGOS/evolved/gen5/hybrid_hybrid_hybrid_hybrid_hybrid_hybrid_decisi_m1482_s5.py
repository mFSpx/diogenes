# DARWIN HAMMER — match 1482, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_possum_filter_m323_s2.py (gen4)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s1.py (gen2)
# born: 2026-05-29T23:36:52Z

"""Hybrid Algorithm Fusion of:
- Parent A: deterministic Span matcher with pheromone‑based information gain.
- Parent B: regex‑driven decision hygiene, Shannon entropy, and ternary lens audit.

Mathematical Bridge
-------------------
Both parents manipulate scalar fields over discrete objects:

* Parent A assigns each *Span–Entity* pair a pheromone signal `s` whose
  decay follows a half‑life law and whose information weight is the entropy
  `IG = - Σ p_i log p_i` of the pheromone distribution.

* Parent B extracts a feature‑count vector `c` from textual evidence using
  regexes, builds a probability distribution `p_i = c_i / Σ c_i`,
  and computes a normalized Shannon entropy `Ĥ = H / H_max`.

The hybrid treats every Span–Entity pair as a joint random variable.
Its joint weight `w` is


w = (span.score * entity.score) * exp(-d/λ) * (1 + α·Ĥ)


where `d` is the haversine distance between the span’s anchor position and the
entity’s geolocation, `λ` is a spatial scale, and `α` scales the entropy
contribution derived from the hygiene regexes.  The weight `w` becomes the
new pheromone signal value, which is inserted into the pheromone store and
subject to exponential decay according to its half‑life.

The three core functions below implement:
1. Joint weight computation (`joint_weight`).
2. Pheromone decay and insertion (`update_pheromones`).
3. End‑to‑end hybrid scoring for a document (`hybrid_document_score`).
"""

import math
import random
import sys
import pathlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np
import re

# ----------------------------------------------------------------------
# Data structures (derived from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """A textual span with a confidence score."""
    start: int
    end: int
    text: str
    label: str
    score: float
    # optional geographic anchor (latitude, longitude) for distance calculations
    lat: float = 0.0
    lon: float = 0.0


@dataclass(frozen=True)
class Entity:
    """An external entity that can be linked to a Span."""
    identifier: str
    score: float
    lat: float
    lon: float


class PheromoneEntry:
    """Stores a pheromone signal with half‑life decay."""
    __slots__ = ("uuid", "key", "kind", "value", "half_life_seconds",
                 "created_at", "last_decay")

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
        """Multiplicative decay factor based on half‑life."""
        age = self.age_seconds()
        if self.half_life_seconds <= 0:
            return 0.0
        # classic exponential decay: factor = 0.5 ** (age / half_life)
        return 0.5 ** (age / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Container for multiple PheromoneEntry objects."""
    def __init__(self):
        self.entries: Dict[str, PheromoneEntry] = {}

    def add_or_update(self, key: str, kind: str, value: float,
                      half_life_seconds: int = 3600) -> None:
        if key in self.entries:
            entry = self.entries[key]
            entry.apply_decay()
            entry.value += value
            entry.last_decay = datetime.now(timezone.utc)
        else:
            self.entries[key] = PheromoneEntry(key, kind, value,
                                               half_life_seconds)

    def decay_all(self) -> None:
        for entry in self.entries.values():
            entry.apply_decay()

    def total_entropy(self) -> float:
        """Shannon entropy of the normalized pheromone value distribution."""
        values = np.array([e.value for e in self.entries.values() if e.value > 0])
        if values.size == 0:
            return 0.0
        probs = values / values.sum()
        return -np.sum(probs * np.log(probs + 1e-12))


# ----------------------------------------------------------------------
# Regex‑based hygiene extraction (Parent B)
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


def hygiene_feature_vector(text: str) -> np.ndarray:
    """Count matches for each hygiene regex and return a vector."""
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
    """Return H / H_max where H_max = log(k) with k = number of non‑zero bins."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    H = -np.sum(probs * np.log(probs + 1e-12))
    k = np.count_nonzero(counts)
    H_max = math.log(k) if k > 1 else 0.0
    return H / H_max if H_max > 0 else 0.0


# ----------------------------------------------------------------------
# Ternary Lens Audit (simplified stub)
# ----------------------------------------------------------------------
def ternary_lens_factor(audit_report: Dict[str, Any]) -> float:
    """
    Convert a ternary audit report into a multiplicative factor.
    The report is expected to contain a `'risk'` field with values
    `'low'`, `'medium'`, `'high'`.  Mapping:
        low    → 0.9
        medium → 1.0
        high   → 1.2
    Missing or unknown values default to 1.0.
    """
    mapping = {"low": 0.9, "medium": 1.0, "high": 1.2}
    risk = audit_report.get("risk", "medium").lower()
    return mapping.get(risk, 1.0)


# ----------------------------------------------------------------------
# Core hybrid mathematics
# ----------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Return distance in kilometres between two (lat,lon) points."""
    R = 6371.0  # Earth radius in km
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)

    a = math.sin(Δφ / 2) ** 2 + \
        math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def joint_weight(span: Span,
                 entity: Entity,
                 text: str,
                 λ: float = 50.0,
                 α: float = 0.5,
                 audit_report: Dict[str, Any] = None) -> float:
    """
    Compute the hybrid joint weight for a Span–Entity pair.

    Parameters
    ----------
    span : Span
        The textual span.
    entity : Entity
        The linked entity.
    text : str
        Full document text (used for hygiene entropy).
    λ : float
        Spatial attenuation length scale (km). Default 50 km.
    α : float
        Entropy scaling coefficient. Default 0.5.
    audit_report : dict | None
        Optional ternary lens audit report.

    Returns
    -------
    float
        Joint weight ready to be stored as a pheromone signal.
    """
    # 1. Base product of confidence scores
    base = span.score * entity.score

    # 2. Spatial attenuation
    d = haversine_distance(span.lat, span.lon, entity.lat, entity.lon)
    spatial_factor = math.exp(-d / λ)

    # 3. Hygiene entropy contribution
    counts = hygiene_feature_vector(text)
    Ĥ = normalized_shannon_entropy(counts)

    # 4. Ternary lens factor (defaults to 1.0)
    lens_factor = ternary_lens_factor(audit_report or {})

    # Combined weight
    weight = base * spatial_factor * (1.0 + α * Ĥ) * lens_factor
    return weight


def update_pheromones(store: PheromoneStore,
                     span: Span,
                     entity: Entity,
                     text: str,
                     half_life_seconds: int = 3600,
                     **joint_kwargs) -> None:
    """
    Compute the joint weight and insert/update it in the pheromone store.
    The store key is a deterministic concatenation of span and entity IDs.
    """
    weight = joint_weight(span, entity, text, **joint_kwargs)
    key = f"{span.start}:{span.end}|{entity.identifier}"
    store.add_or_update(key, kind="joint_signal",
                        value=weight,
                        half_life_seconds=half_life_seconds)


def hybrid_document_score(spans: List[Span],
                          entities: List[Entity],
                          text: str,
                          store: PheromoneStore,
                          **joint_kwargs) -> float:
    """
    Process all Span–Entity combinations, update the pheromone store,
    and return the total entropy of the store after decay.
    """
    # Decay existing pheromones first
    store.decay_all()

    # Update with new joint weights
    for span in spans:
        for entity in entities:
            update_pheromones(store, span, entity, text, **joint_kwargs)

    # Return the current entropy as a global quality metric
    return store.total_entropy()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample document
    doc_text = """
    The evidence was verified and the plan was approved.
    However, we need to wait until tomorrow before proceeding.
    Contact the support team if any issues arise.
    """

    # Create a few spans with geographic anchors
    spans = [
        Span(start=0, end=20, text="The evidence was verified", label="VERIF",
             score=0.92, lat=37.7749, lon=-122.4194),
        Span(start=50, end=70, text="wait until tomorrow", label="DELAY",
             score=0.78, lat=34.0522, lon=-118.2437),
    ]

    # Create entities
    entities = [
        Entity(identifier="ENT-001", score=0.88, lat=37.7749, lon=-122.4194),
        Entity(identifier="ENT-002", score=0.65, lat=40.7128, lon=-74.0060),
    ]

    # Simple audit report
    audit = {"risk": "medium"}

    # Initialise pheromone store
    store = PheromoneStore()

    # Run hybrid scoring
    entropy = hybrid_document_score(
        spans, entities, doc_text, store,
        λ=100.0, α=0.6, audit_report=audit, half_life_seconds=7200
    )

    print(f"Post‑update pheromone entropy: {entropy:.4f}")
    # Show a few stored entries
    for k, e in list(store.entries.items())[:5]:
        print(f"Key={k} | Value={e.value:.4f} | Age={e.age_seconds():.1f}s")