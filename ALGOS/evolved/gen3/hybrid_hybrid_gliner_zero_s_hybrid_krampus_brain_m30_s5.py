# DARWIN HAMMER — match 30, survivor 5
# gen: 3
# parent_a: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s4.py (gen1)
# parent_b: hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s1.py (gen2)
# born: 2026-05-29T23:25:25Z

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities (derived from Parent A)
# ----------------------------------------------------------------------
DEFAULT_LABELS = [
    "Operator", "Rainmaker", "Paladin / God-Mode", "Psyche / State-Collapse",
    "Forensic Shield", "Infinite Sink", "Anchor Weight", "Server Wipe",
    "API Rate Limiting", "Environment Migration", "Cruelty Protocols",
    "Master’s Eye", "Chrono-Ledger", "KRAMPUSCHEWING", "KORPUS",
    "DIOGENES", "FairyFuse", "Job Fair Allocator", "Darwinian Surfaces",
    "Command Envelope Protocol",
]


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


def now_iso() -> str:
    """Current UTC timestamp in ISO‑8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    """SHA‑256 hash of the supplied Unicode text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def parse_labels(raw: str | None) -> List[str]:
    """Parse a JSON file, comma‑separated string or fallback to defaults."""
    if not raw:
        return list(DEFAULT_LABELS)
    p = Path(raw)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            labels = data.get("required_exact_labels") or data.get("labels") or []
        else:
            labels = data
        return [str(x).strip() for x in labels if str(x).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def literal_fallback(text: str, labels: List[str], *, case_sensitive: bool = False) -> List[Span]:
    """Pure‑Python label matcher that returns deterministic spans."""
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: List[Span] = []
    seen: set[Tuple[int, int, str]] = set()
    for label in labels:
        # simple normalisation: replace slashes and hyphens with spaces
        pattern = re.escape(label).replace(r"\ ", r"\s+").replace(r"\-", r"\s+")
        for m in re.finditer(pattern, text, flags):
            start, end = m.span()
            key = (start, end, label)
            if key in seen:
                continue
            seen.add(key)
            span = Span(start=start, end=end, text=m.group(), label=label, score=1.0)
            spans.append(span)
    return spans


# ----------------------------------------------------------------------
# Pheromone infrastructure (derived from Parent B)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: int,
    ) -> None:
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
        """Multiplicative decay since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory singleton store for pheromone entries."""

    _entries: dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def all_entries(cls) -> List[PheromoneEntry]:
        return list(cls._entries.values())

    @classmethod
    def decay_all(cls) -> None:
        for e in cls._entries.values():
            e.apply_decay()


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def extract_spans(text: str, label_source: str | None = None) -> List[Span]:
    """
    Wrapper around the literal fallback matcher.
    `label_source` can be a path to a JSON file, a comma‑separated string,
    or ``None`` to use the default label set.
    """
    labels = parse_labels(label_source)
    return literal_fallback(text, labels)


def vector_from_spans(spans: List[Span], label_order: List[str] | None = None) -> np.ndarray:
    """
    Build a count vector **v** ∈ ℝⁿ where n = len(label_order).
    Each component v[i] is the number of spans whose label equals label_order[i].
    If ``label_order`` is ``None`` the default label list is used.
    """
    if label_order is None:
        label_order = DEFAULT_LABELS
    vec = np.zeros(len(label_order), dtype=float)
    label_to_idx = {lbl: i for i, lbl in enumerate(label_order)}
    for span in spans:
        idx = label_to_idx.get(span.label)
        if idx is not None:
            vec[idx] += 1.0
    return vec


def entropy(prob: np.ndarray) -> float:
    """Shannon entropy H(p) for a probability vector (base e)."""
    # Guard against zero probabilities
    mask = prob > 0
    return -float(np.sum(prob[mask] * np.log(prob[mask])))


def normalise(vector: np.ndarray) -> np.ndarray:
    """L1 normalise a vector."""
    return vector / np.sum(np.abs(vector))


def pheromone_update_and_decide(
    vector: np.ndarray,
    label_order: List[str],
    half_life_seconds: int = 300,
    signal_kind: str = "span_count",
) -> Tuple[str, float]:
    """
    1. Decay all existing pheromones.
    2. For each dimension i, inject ``vector[i]`` as a new pheromone entry
       (or augment an existing one with the same surface key).
    3. Compute the probability distribution over surfaces and its entropy.
    4. For each surface compute the *expected* entropy after a hypothetical
       injection of the current vector component and pick the surface that
       yields the maximal information gain.
    """
    PheromoneStore.decay_all()

    # Create or update pheromone entries
    for i, label in enumerate(label_order):
        existing_entries = PheromoneStore.get_by_surface(label)
        if existing_entries:
            # Update existing entry
            entry = existing_entries[0]
            entry.signal_value += vector[i]
        else:
            # Create new entry
            entry = PheromoneEntry(label, signal_kind, vector[i], half_life_seconds)
            PheromoneStore.add(entry)

    # Compute current probability distribution
    pheromone_values = np.array([e.signal_value for e in PheromoneStore.all_entries()])
    prob = normalise(pheromone_values)

    # Compute current entropy
    current_entropy = entropy(prob)

    # Compute expected entropy after injection
    expected_entropies = []
    for i, label in enumerate(label_order):
        # Hypothetical injection of vector component
        injected_prob = prob.copy()
        injected_prob[i] += vector[i]
        injected_prob = normalise(injected_prob)
        expected_entropy = entropy(injected_prob)
        expected_entropies.append(expected_entropy)

    # Select label with maximal information gain
    info_gains = current_entropy - np.array(expected_entropies)
    best_label_idx = np.argmax(info_gains)
    best_label = label_order[best_label_idx]

    return best_label, info_gains[best_label_idx]


if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog."
    spans = extract_spans(text)
    vector = vector_from_spans(spans)
    best_label, info_gain = pheromone_update_and_decide(vector, DEFAULT_LABELS)
    print(f"Best label: {best_label}, Info gain: {info_gain}")