# DARWIN HAMMER — match 5484, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1482_s6.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m1839_s0.py (gen6)
# born: 2026-05-30T00:02:21Z

"""Hybrid Fusion of Pheromone Entropy and SSIM-Weighted Feature Scoring

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1482_s6.py (Pheromone store, entropy)
- hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m1839_s0.py (Regex feature extraction, SSIM weighting, Fisher‑information‑like score)

Mathematical Bridge:
Both parents operate on *vectors* derived from textual input.
Parent A provides a decay‑driven scalar field (pheromone values) whose
Shannon entropy  H  = –∑p·log p  quantifies distributional uncertainty.
Parent B produces a feature count vector **f** and a similarity measure
SSIM(**f**, **g**) ∈ [0, 1] between two such vectors.

The fusion treats the SSIM value as a *mixing coefficient* λ that blends
the entropy‑derived uncertainty with a Fisher‑information‑like quadratic
form Q(**f**) = **f**ᵀ W **f** (W diagonal, width‑controlled Gaussian beam).
The hybrid score is

    S = λ·Q(**f**) + (1–λ)·H

where H is the current pheromone entropy.  This unifies the stochastic
decay dynamics of Parent A with the deterministic similarity‑driven
scoring of Parent B in a single mathematically coherent system.
"""

import math
import random
import sys
import re
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Pheromone infrastructure (trimmed & completed)
# ----------------------------------------------------------------------
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
    """Collects pheromone entries and provides a Shannon‑entropy metric."""
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
        """Shannon entropy of positive pheromone values."""
        values = np.array([e.value for e in self.entries.values() if e.value > 0])
        if values.size == 0:
            return 0.0
        probs = values / values.sum()
        return -np.sum(probs * np.log(probs + 1e-12))

# ----------------------------------------------------------------------
# Parent B – Regex feature extraction & SSIM‑weighted scoring
# ----------------------------------------------------------------------
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
IMPULSIVE_RE = re.compile(r"\b(?:impulse|rash|hasty|spontan(?:eous|ous))\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:scarcity|limited|rare|few|shortage)\b", re.I)
RISK_RE = re.compile(r"\b(?:risk|danger|hazard|threat|exposure)\b", re.I)

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

def extract_feature_vector(text: str) -> np.ndarray:
    """Count occurrences of each feature regex in the supplied text."""
    counts = [len(_REGEX_MAP[name].findall(text)) for name in _FEATURE_ORDER]
    return np.array(counts, dtype=float)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """A simple Gaussian beam used as a weighting kernel (Fisher‑like)."""
    if width <= 0:
        return 0.0
    return math.exp(-0.5 * ((theta - center) / width) ** 2)

def fisher_like_score(vec: np.ndarray, center: float = 0.0, width: float = 1.0) -> float:
    """Quadratic form where each component is weighted by a Gaussian beam."""
    weights = np.array([gaussian_beam(v, center, width) for v in vec])
    return float(vec @ np.diag(weights) @ vec)

def ssim_vector(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """
    Simplified SSIM for 1‑D vectors.
    Returns a value in [0, 1] where 1 means identical.
    """
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / (denominator + 1e-12))

# ----------------------------------------------------------------------
# Hybrid Core – blending pheromone entropy with SSIM‑weighted Fisher score
# ----------------------------------------------------------------------
class HybridRouter:
    """Encapsulates the fused algorithm."""
    def __init__(self):
        self.pheromones = PheromoneStore()
        # Reference vector can be static or updated; we start with zeros.
        self.reference_vec = np.zeros(len(_FEATURE_ORDER), dtype=float)

    def update_reference(self, text: str) -> None:
        """Replace the internal reference vector with the feature vector of *text*."""
        self.reference_vec = extract_feature_vector(text)

    def ingest_span(self, span: Span) -> None:
        """
        Use span metadata to update pheromone store.
        The span label becomes the key; its score is added as pheromone value.
        """
        self.pheromones.add_or_update(key=span.label, kind="span", value=span.score)

    def hybrid_score(self, text: str) -> float:
        """
        Compute the hybrid score for *text*:
        λ = SSIM(feature_vec, reference_vec)
        Q = fisher_like_score(feature_vec)
        H = current pheromone entropy
        S = λ·Q + (1‑λ)·H
        """
        vec = extract_feature_vector(text)
        # If reference vector is all zeros, fall back to λ = 0.5
        if np.all(self.reference_vec == 0):
            lam = 0.5
        else:
            lam = ssim_vector(vec, self.reference_vec)
        Q = fisher_like_score(vec, center=0.0, width=2.0)
        H = self.pheromones.total_entropy()
        return lam * Q + (1.0 - lam) * H

    def decay_cycle(self) -> None:
        """Apply decay to all pheromones – mimics the temporal forgetting."""
        self.pheromones.decay_all()

# ----------------------------------------------------------------------
# Demonstration functions (fulfil requirement of ≥3 functions)
# ----------------------------------------------------------------------
def demo_feature_extraction():
    txt = "The audit confirmed the evidence and the plan was scheduled after a short delay."
    vec = extract_feature_vector(txt)
    print("Feature vector:", dict(zip(_FEATURE_ORDER, vec.astype(int))))

def demo_ssim_and_fisher():
    txt1 = "Evidence was verified, and the outcome was successful."
    txt2 = "Evidence was verified, and the outcome was successful."
    v1 = extract_feature_vector(txt1)
    v2 = extract_feature_vector(txt2)
    print("SSIM:", ssim_vector(v1, v2))
    print("Fisher‑like score:", fisher_like_score(v1))

def demo_hybrid_router():
    router = HybridRouter()
    router.update_reference("Initial reference mentioning evidence and planning.")
    span = Span(start=0, end=10, text="test", label="evidence", score=1.2)
    router.ingest_span(span)
    score = router.hybrid_score("The evidence was verified after a short delay.")
    print("Hybrid score:", score)
    router.decay_cycle()
    print("Entropy after decay:", router.pheromones.total_entropy())

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Demo: Feature Extraction ---")
    demo_feature_extraction()
    print("\n--- Demo: SSIM & Fisher ---")
    demo_ssim_and_fisher()
    print("\n--- Demo: Hybrid Router ---")
    demo_hybrid_router()