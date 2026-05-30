# DARWIN HAMMER — match 432, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_gliner_zero_s_hybrid_krampus_brain_m30_s2.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_ternary_lens__m71_s1.py (gen3)
# born: 2026-05-29T23:28:57Z

"""Hybrid Algorithm combining:
- Parent A: hybrid_gliner_zero_shot_ext_minimum_cost_tree (deterministic span matcher) &
  hybrid_krampus_brainmap_hybrid_pheromone_inf (entropy/pheromone decay).
- Parent B: hybrid_fisher_localization (Gaussian beam, Fisher information) &
  hybrid_ternary_lens_router (regex‑driven ternary dimension & SSIM).

Mathematical Bridge:
The bridge is the notion of *information gain*.  Parent A models it via
entropy‑based pheromone values; Parent B quantifies it analytically with
Fisher information of a Gaussian beam.  This module maps each deterministic
Span (Parent A) to a Gaussian beam whose intensity is the Span score.
The Fisher information of that beam becomes the raw pheromone signal.
A ternary dimension vector derived from regex matches on the Span text
modulates the pheromone half‑life, while the SSIM between vectorised
Span texts provides a similarity measure that can be used to reinforce
or attenuate pheromone updates.  Thus the core topologies of both parents
are fused into a single unified system."""

import math
import random
import sys
import pathlib
import uuid
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks (adapted)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """Deterministic span produced by the label matcher."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
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
        """Multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory store for pheromone entries."""
    def __init__(self):
        self._entries: Dict[str, PheromoneEntry] = {}

    def add(self, entry: PheromoneEntry) -> None:
        self._entries[entry.uuid] = entry

    def all(self) -> List[PheromoneEntry]:
        return list(self._entries.values())

    def decay_all(self) -> None:
        for entry in self._entries.values():
            entry.apply_decay()


# ----------------------------------------------------------------------
# Parent B building blocks (adapted)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float,
                 eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """One‑dimensional Structural Similarity Index."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    cov = np.cov(x, y, ddof=0)[0, 1]
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Ternary dimension catalogue (Parent B – part B)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12

_REGEX_CATALOG: List[re.Pattern] = [
    re.compile(r"\b(evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I),  # 0
    re.compile(r"\b(plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I),  # 1
    re.compile(r"\b(pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|first|after|review)\b", re.I),  # 2
    re.compile(r"\b(ask|call|text|friend|colleague|peer|expert|consult|reach out)\b", re.I),  # 3
    re.compile(r"\b(error|fail|exception|bug|crash|stacktrace|traceback|panic)\b", re.I),  # 4
    re.compile(r"\b(success|ok|done|completed|finished|passed|green)\b", re.I),  # 5
    re.compile(r"\b(critical|high|urgent|blocker|severity\s*1)\b", re.I),  # 6
    re.compile(r"\b(low|minor|info|severity\s*3|notice)\b", re.I),  # 7
    re.compile(r"\b(user|client|customer|end[- ]?user)\b", re.I),  # 8
    re.compile(r"\b(server|backend|database|api|service)\b", re.I),  # 9
    re.compile(r"\b(timeout|latency|performance|slow|lag)\b", re.I),  #10
    re.compile(r"\b(security|auth|encryption|access|privilege|token)\b", re.I)   #11
]


def ternary_vector_from_text(text: str) -> np.ndarray:
    """
    Produce a ternary (0/1) vector of length TERNARY_DIMS where each dimension
    is 1 if the corresponding regex matches the text, else 0.
    """
    vec = np.zeros(TERNARY_DIMS, dtype=np.int8)
    for idx, pattern in enumerate(_REGEX_CATALOG):
        if pattern.search(text):
            vec[idx] = 1
    return vec


# ----------------------------------------------------------------------
# Hybrid core functions (mathematical fusion)
# ----------------------------------------------------------------------
def fisher_from_span(span: Span,
                     center: float = 0.5,
                     width: float = 0.1) -> float:
    """
    Map a Span's score to a Gaussian beam and return its Fisher information.
    The score is interpreted as the beam angle (theta) in [0,1].
    """
    theta = max(0.0, min(1.0, span.score))  # clamp to [0,1]
    return fisher_score(theta, center, width)


def vectorise_span_text(span: Span, length: int = 64) -> np.ndarray:
    """
    Simple deterministic vectorisation: ASCII codes of characters,
    padded/truncated to ``length`` and normalised to [0,255].
    """
    codes = [ord(c) for c in span.text[:length]]
    if len(codes) < length:
        codes.extend([0] * (length - len(codes)))
    arr = np.array(codes, dtype=np.float32)
    return arr


def update_pheromone_store_from_spans(spans: List[Span],
                                      store: PheromoneStore,
                                      base_half_life: int = 300) -> None:
    """
    For each Span:
      1. Compute Fisher information → raw pheromone value.
      2. Derive a ternary dimension vector from the Span text.
      3. Modulate half‑life: base_half_life * (1 + sum(ternary_vector)).
      4. Create a PheromoneEntry keyed by a deterministic surface_key
         (label + start + end) and add it to the store.
    """
    for span in spans:
        raw_value = fisher_from_span(span)
        tern_vec = ternary_vector_from_text(span.text)
        half_life = base_half_life * (1 + int(tern_vec.sum()))
        surface_key = f"{span.label}:{span.start}-{span.end}"
        entry = PheromoneEntry(
            surface_key=surface_key,
            signal_kind="fisher_pheromone",
            signal_value=raw_value,
            half_life_seconds=half_life
        )
        store.add(entry)


def span_similarity(span_a: Span, span_b: Span) -> float:
    """
    Compute similarity between two spans using SSIM on their vectorised texts.
    The SSIM value (0‑1) can be used to weight pheromone reinforcement.
    """
    vec_a = vectorise_span_text(span_a)
    vec_b = vectorise_span_text(span_b)
    # Normalise to typical dynamic range for SSIM
    return ssim(vec_a, vec_b, dynamic_range=255.0)


def reinforce_pheromones(store: PheromoneStore,
                        span_pairs: List[Tuple[Span, Span]],
                        reinforcement_factor: float = 0.2) -> None:
    """
    For each (span_a, span_b) pair, compute similarity.
    If similarity exceeds 0.7, increase the signal_value of matching
    pheromone entries by ``reinforcement_factor * similarity``.
    Matching is performed by surface_key equality.
    """
    for span_a, span_b in span_pairs:
        sim = span_similarity(span_a, span_b)
        if sim < 0.7:
            continue
        key_a = f"{span_a.label}:{span_a.start}-{span_a.end}"
        key_b = f"{span_b.label}:{span_b.start}-{span_b.end}"
        for entry in store.all():
            if entry.surface_key in (key_a, key_b):
                entry.signal_value *= (1.0 + reinforcement_factor * sim)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a few deterministic spans
    spans = [
        Span(start=0, end=5, text="The evidence was verified.", label="FACT", score=0.85),
        Span(start=10, end=20, text="Please plan the next steps.", label="ACTION", score=0.40),
        Span(start=30, end=45, text="Server timeout occurred during query.", label="ERROR", score=0.65),
    ]

    # Initialise pheromone store
    store = PheromoneStore()

    # Populate store using the hybrid update
    update_pheromone_store_from_spans(spans, store)

    # Print initial pheromone values
    print("Initial pheromone entries:")
    for e in store.all():
        print(f"{e.surface_key} | value={e.signal_value:.6f} | half_life={e.half_life_seconds}s")

    # Simulate a similarity reinforcement step
    reinforce_pheromones(store, [(spans[0], spans[1]), (spans[0], spans[2])])

    # Decay all entries once (e.g., after some time)
    store.decay_all()

    # Final state
    print("\nAfter reinforcement and decay:")
    for e in store.all():
        print(f"{e.surface_key} | value={e.signal_value:.6f} | half_life={e.half_life_seconds}s")