# DARWIN HAMMER — match 3635, survivor 1
# gen: 4
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s2.py (gen1)
# parent_b: hybrid_hybrid_gliner_zero_s_hybrid_krampus_brain_m30_s2.py (gen3)
# born: 2026-05-29T23:50:58Z

"""Hybrid Decision Hygiene & Pheromone Entropy Module.

Parents:
- hybrid_decision_hygiene_shannon_entropy_m12_s2 (regex‑based feature counts + Shannon entropy)
- hybrid_hybrid_gliner_zero_s_hybrid_krampus_brain_m30_s2 (span matcher + pheromone decay)

Mathematical Bridge:
Both parents expose a *probability‑like* vector:
* Decision Hygiene provides a count vector **c** over semantic categories; after normalisation **p = c / Σc** it is a discrete probability distribution used in Shannon entropy **H(p)**.
* The Pheromone system stores a signal vector **φ** over surface keys; each entry decays exponentially and can be interpreted as a weighted probability mass.

The hybrid algorithm maps the hygiene probability **p** onto the pheromone surface keys (using spans as keys) and lets the pheromone signal be modulated by the entropy factor **γ = H(p)/H_max**.  The resulting signal drives a combined decision score:

S = (w·c) * γ * Σ_i φ_i

where **w·c** is the original hygiene dot‑product score and **Σ_i φ_i** is the total pheromone mass after decay.  This fuses the two topologies into a single unified system.
"""

from __future__ import annotations

import math
import random
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regexes and raw count extraction
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
    r"\b(?:rage|impulsive|panic|pan|hasty|rash|reckless|spontaneou?s?)\b",
    re.I,
)

REGEXES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
}

WEIGHTS = np.array([1.2, 1.0, 0.8, 0.9, 1.1, 1.3, 0.7])  # arbitrary but fixed


def extract_feature_vector(text: str) -> Dict[str, int]:
    """Return raw counts for each semantic regex category."""
    counts = {}
    for name, pattern in REGEXES.items():
        counts[name] = len(pattern.findall(text))
    return counts


def hygiene_dot_score(counts: Dict[str, int]) -> float:
    """Weighted dot‑product of counts with predefined WEIGHTS."""
    vec = np.array([counts[k] for k in REGEXES.keys()], dtype=float)
    return float(np.dot(WEIGHTS, vec))


def shannon_entropy(counts: Dict[str, int]) -> Tuple[float, float]:
    """
    Compute Shannon entropy H and its normalised form H/H_max.
    Returns (H, H_norm).
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    probs = np.array([c / total for c in counts.values()], dtype=float)
    # avoid log2(0) by masking zeros
    mask = probs > 0
    H = -np.sum(probs[mask] * np.log2(probs[mask]))
    H_max = math.log2(len(probs))
    return float(H), float(H / H_max)


# ----------------------------------------------------------------------
# Parent B – span matcher (very lightweight) and pheromone store
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """Simple span representation used as a bridge to pheromones."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    """A single pheromone signal that decays exponentially."""
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory store for pheromone entries keyed by surface_key."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PheromoneStore, cls).__new__(cls)
            cls._instance._entries: Dict[str, PheromoneEntry] = {}
        return cls._instance

    def get(self, key: str) -> PheromoneEntry | None:
        return self._entries.get(key)

    def upsert(self, key: str, kind: str, delta: float, half_life: int = 300) -> None:
        entry = self._entries.get(key)
        if entry is None:
            entry = PheromoneEntry(surface_key=key, signal_kind=kind, signal_value=delta, half_life_seconds=half_life)
            self._entries[key] = entry
        else:
            entry.apply_decay()
            entry.signal_value += delta
            entry.last_decay = datetime.now(timezone.utc)

    def decay_all(self) -> None:
        for entry in list(self._entries.values()):
            entry.apply_decay()
            if entry.signal_value < 1e-6:  # prune near‑zero signals
                del self._entries[entry.surface_key]

    def total_signal(self) -> float:
        return sum(e.signal_value for e in self._entries.values())


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def generate_spans(text: str) -> List[Span]:
    """
    Very lightweight matcher that turns each regex match into a Span.
    The label is the regex category name; the score is the length of the match.
    """
    spans: List[Span] = []
    for label, pattern in REGEXES.items():
        for m in pattern.finditer(text):
            span = Span(
                start=m.start(),
                end=m.end(),
                text=m.group(),
                label=label,
                score=float(len(m.group())),
            )
            spans.append(span)
    # deterministic ordering for reproducibility
    spans.sort(key=lambda s: (s.start, s.end))
    return spans


def update_pheromones_from_text(text: str, store: PheromoneStore) -> None:
    """
    Extract hygiene features, compute entropy factor γ,
    generate spans, and feed each span into the pheromone store.
    The delta signal for a span is:
        Δ = hygiene_score * γ * span.score
    """
    counts = extract_feature_vector(text)
    hygiene_score = hygiene_dot_score(counts)
    _, gamma = shannon_entropy(counts)  # γ ∈ [0,1]

    if hygiene_score == 0:
        return  # nothing to propagate

    spans = generate_spans(text)
    for span in spans:
        delta = hygiene_score * gamma * span.score
        # Use the span text as the surface key (could be any deterministic identifier)
        key = f"{span.label}:{span.text.lower()}"
        store.upsert(key=key, kind="hygiene", delta=delta, half_life=600)


def hybrid_decision_score(text: str, store: PheromoneStore) -> float:
    """
    Compute the final hybrid decision score:
        S = hygiene_score * γ * total_pheromone_mass
    """
    counts = extract_feature_vector(text)
    hygiene_score = hygiene_dot_score(counts)
    _, gamma = shannon_entropy(counts)
    total_pheromone = store.total_signal()
    return hygiene_score * gamma * total_pheromone


def decay_pheromones(store: PheromoneStore) -> None:
    """Apply exponential decay to all stored pheromones."""
    store.decay_all()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We need to verify the source, plan the roadmap, and schedule a test. "
        "If there is a delay, we will wait tomorrow. "
        "Support from the team is essential, and we must keep the boundary safe."
    )
    store = PheromoneStore()
    update_pheromones_from_text(sample_text, store)
    print("Total pheromone after first update:", store.total_signal())
    score1 = hybrid_decision_score(sample_text, store)
    print("Hybrid decision score (iteration 1):", score1)

    # Simulate passage of time and another update
    decay_pheromones(store)
    print("Total pheromone after decay:", store.total_signal())

    # Second round with slightly altered text
    second_text = "Confirm the evidence, then finalize the outcome. No further delay."
    update_pheromones_from_text(second_text, store)
    print("Total pheromone after second update:", store.total_signal())
    score2 = hybrid_decision_score(second_text, store)
    print("Hybrid decision score (iteration 2):", score2)