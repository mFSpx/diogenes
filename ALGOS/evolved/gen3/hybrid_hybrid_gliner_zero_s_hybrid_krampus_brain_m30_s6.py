# DARWIN HAMMER — match 30, survivor 6
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
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def parse_labels(raw: str | None) -> List[str]:
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
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: List[Span] = []
    seen: set[Tuple[int, int, str]] = set()
    for label in labels:
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
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)

class PheromoneStore:
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

def extract_spans(text: str, label_source: str | None = None) -> List[Span]:
    labels = parse_labels(label_source)
    return literal_fallback(text, labels)

def vector_from_spans(spans: List[Span], label_order: List[str] | None = None) -> np.ndarray:
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
    mask = prob > 0
    return -float(np.sum(prob[mask] * np.log(prob[mask])))

def pheromone_update_and_decide(
    vector: np.ndarray,
    half_life_seconds: int = 300,
    signal_kind: str = "span_count",
) -> Tuple[str, float]:
    PheromoneStore.decay_all()
    for i, value in enumerate(vector):
        surface_key = DEFAULT_LABELS[i]
        existing_entries = PheromoneStore.get_by_surface(surface_key)
        if existing_entries:
            entry = existing_entries[0]
            entry.signal_value += value
        else:
            entry = PheromoneEntry(
                surface_key=surface_key,
                signal_kind=signal_kind,
                signal_value=value,
                half_life_seconds=half_life_seconds,
            )
            PheromoneStore.add(entry)
    pheromone_values = np.array([entry.signal_value for entry in PheromoneStore.all_entries()])
    prob = pheromone_values / pheromone_values.sum()
    entropy_before = entropy(prob)
    max_info_gain = 0
    best_label = None
    for i, label in enumerate(DEFAULT_LABELS):
        new_prob = prob.copy()
        new_prob[i] += 1
        new_prob /= new_prob.sum()
        entropy_after = entropy(new_prob)
        info_gain = entropy_before - entropy_after
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_label = label
    return best_label, max_info_gain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--labels", type=str, default=None)
    args = parser.parse_args()
    spans = extract_spans(args.text, args.labels)
    vector = vector_from_spans(spans)
    label, info_gain = pheromone_update_and_decide(vector)
    print(f"Best label: {label}, Info gain: {info_gain}")

if __name__ == "__main__":
    main()