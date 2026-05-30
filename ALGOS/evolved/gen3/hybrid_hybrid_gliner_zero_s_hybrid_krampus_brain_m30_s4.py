# DARWIN HAMMER — match 30, survivor 4
# gen: 3
# parent_a: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s4.py (gen1)
# parent_b: hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s1.py (gen2)
# born: 2026-05-29T23:25:25Z

"""Hybrid algorithm combining literal label span extraction (Parent A) with pheromone‑based infotaxis (Parent B).

Mathematical bridge:
- Parent A yields a high‑dimensional sparse vector **v** ∈ ℝⁿ where each component counts occurrences of a label in the input text.
- Parent B maintains a pheromone store **s** = (s₁,…,sₙ) with exponential decay. Normalising **s** gives a probability distribution **p** over labels.
- Entropy **H(p) = – Σ pᵢ log pᵢ** quantifies uncertainty. Updating the pheromone store with **v** changes **s**, thus **p**, and the resulting entropy reduction ΔH = H_before – H_after is the *information gain*.
- The hybrid algorithm uses ΔH as a decision signal: the label (surface) with maximal expected gain is selected for the next action, and the pheromone values are updated proportionally to the extracted span counts.

The implementation below provides:
1. `extract_spans` – deterministic literal label matcher (Parent A core).
2. `vector_from_spans` – builds the label count vector **v**.
3. `pheromone_update_and_decide` – decays existing pheromones, injects **v**, computes entropy, and returns the label with maximal information gain.

Running the module as a script executes a smoke test on a sample sentence."""

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


def pheromone_update_and_decide(
    vector: np.ndarray,
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
       yields the maximal information gain ΔH.
    Returns a tuple ``(selected_surface, information_gain)``.
    """
    # ------------------------------------------------------------------
    # Step 1 – decay
    # ------------------------------------------------------------------
    PheromoneStore.decay_all()

    # ------------------------------------------------------------------
    # Step 2 – inject current vector as pheromone entries
    # ------------------------------------------------------------------
    label_order = DEFAULT_LABELS
    for idx, count in enumerate(vector):
        if count == 0:
            continue
        surface = label_order[idx]
        # Look for an existing entry for this surface
        existing = PheromoneStore.get_by_surface(surface)
        if existing:
            # Simple aggregation: add to the first matching entry
            entry = existing[0]
            entry.signal_value += count
            entry.last_decay = datetime.now(timezone.utc)
        else:
            entry = PheromoneEntry(
                surface_key=surface,
                signal_kind=signal_kind,
                signal_value=count,
                half_life_seconds=half_life_seconds,
            )
            PheromoneStore.add(entry)

    # ------------------------------------------------------------------
    # Step 3 – compute current distribution and entropy
    # ------------------------------------------------------------------
    entries = PheromoneStore.all_entries()
    total_signal = sum(e.signal_value for e in entries) + 1e-12  # avoid div‑zero
    probs = np.array([e.signal_value / total_signal for e in entries])
    current_entropy = entropy(probs)

    # ------------------------------------------------------------------
    # Step 4 – evaluate information gain for each surface
    # ------------------------------------------------------------------
    best_gain = -math.inf
    best_surface = ""
    for entry in entries:
        # Simulate adding one more unit of signal for this surface
        simulated_signal = entry.signal_value + 1.0
        simulated_total = total_signal + 1.0
        simulated_probs = np.array(
            [
                (e.signal_value + (1.0 if e is entry else 0.0)) / simulated_total
                for e in entries
            ]
        )
        simulated_entropy = entropy(simulated_probs)
        gain = current_entropy - simulated_entropy
        if gain > best_gain:
            best_gain = gain
            best_surface = entry.surface_key

    return best_surface, best_gain


def hybrid_process(text: str, label_source: str | None = None) -> Tuple[List[Span], np.ndarray, str, float]:
    """
    End‑to‑end demonstration:
    1. Extract spans from ``text``.
    2. Convert spans to a label count vector.
    3. Update pheromones and obtain the surface with maximal information gain.
    Returns ``(spans, vector, chosen_surface, gain)``.
    """
    spans = extract_spans(text, label_source)
    vec = vector_from_spans(spans)
    surface, gain = pheromone_update_and_decide(vec)
    return spans, vec, surface, gain


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The Operator initiated an API Rate Limiting sequence while the "
        "Server Wipe protocol was engaged. Meanwhile, the KRAMPUSCHEWING "
        "module reported an infinite sink."
    )
    spans, vec, surface, gain = hybrid_process(sample_text)
    print("Extracted spans:")
    for s in spans:
        print(f"  [{s.start}:{s.end}] {s.label!r} → {s.text!r}")
    print("\nLabel count vector:", vec)
    print(f"\nChosen surface (max info gain): {surface!r} with ΔH = {gain:.4f}")
    # Verify that the store contains entries
    print("\nPheromone store snapshot:")
    for e in PheromoneStore.all_entries():
        print(
            f"  {e.surface_key}: value={e.signal_value:.3f}, "
            f"age={e.age_seconds():.1f}s"
        )