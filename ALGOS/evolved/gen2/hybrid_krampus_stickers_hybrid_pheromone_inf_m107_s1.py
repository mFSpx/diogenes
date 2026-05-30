# DARWIN HAMMER — match 107, survivor 1
# gen: 2
# parent_a: krampus_stickers.py (gen0)
# parent_b: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# born: 2026-05-29T23:25:41Z

"""
This module fuses the krampus_stickers.py and hybrid_pheromone_infotaxis_m3_s4.py algorithms.
The mathematical bridge between these two algorithms lies in the concept of information entropy and pheromone decay.
The krampus_stickers.py algorithm calculates the entropy of a given text, while the hybrid_pheromone_infotaxis_m3_s4.py algorithm simulates pheromone decay over time.
The fusion of these two algorithms creates a hybrid system that associates pheromone signals with the entropy of text data, allowing for the simulation of information diffusion and decay.
"""

import numpy as np
import math
import random
import sys
import pathlib
import re
from datetime import datetime, timezone, timedelta
import uuid

MAX_COMPONENT_TOKENS = 500

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
    def get_by_surface(cls, surface_key: str) -> list[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> list[dict]:
        rows = []
        for entry in cls.get_by_surface(surface_key):
            before = entry.signal_value
            entry.apply_decay()
            rows.append({
                "pheromone_uuid": entry.uuid,
                "surface_key": entry.surface_key,
                "signal_kind": entry.signal_kind,
                "signal_value_before": before,
                "signal_value_after": entry.signal_value
            })
        return rows


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def entropy_for_text(text: str) -> float:
    return float(shannon_entropy(list((text or "")[:10000]))) if text else 0.0


def shannon_entropy(probability_vector: list) -> float:
    probabilities = [p / sum(probability_vector) for p in probability_vector]
    return -sum([p * math.log(p, 2) for p in probabilities if p != 0])


def links_from_text(text: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for m in re.finditer(r"\[([^\]]{0,240})\]\(([^)\s]{1,1000})\)", text or ""):
        target = m.group(2).strip()
        kind = "url" if target.lower().startswith(("http://", "https://")) else "markdown"
        key = (kind, target, m.group(1))
        if key not in seen:
            seen.add(key)
            links.append({"link_kind": kind, "raw_target": target, "anchor_text": m.group(1), "source": "markdown_link"})
    for m in re.finditer(r"\[\[([^\]|#]{1,500})(?:#[^\]|]+)?(?:\|([^\]]{1,240}))?\]\]", text or ""):
        target = m.group(1).strip()
        anchor = (m.group(2) or target).strip()
        key = ("wikilink", target, anchor)
        if key not in seen:
            seen.add(key)
            links.append({"link_kind": "wikilink", "raw_target": target, "anchor_text": anchor, "source": "wikilink"})
    for m in re.finditer(r'\bhttps?://[^\s<>\')]+', text or "", re.I):
        target = m.group(0).rstrip(".,;")
        key = ("url", target, target)
        if key not in seen:
            seen.add(key)
            links.append({"link_kind": "url", "raw_target": target, "anchor_text": target[:240], "source": "bare_url"})
    return links


def associate_pheromone_with_text(phero_signal: float, text: str) -> PheromoneEntry:
    entropy = entropy_for_text(text)
    half_life_seconds = int(entropy * 10)
    return PheromoneEntry(surface_key=text, signal_kind="text_entropy", signal_value=phero_signal, half_life_seconds=half_life_seconds)


def simulate_pheromone_diffusion(phero_signals: list[float], texts: list[str]) -> list[PheromoneEntry]:
    pheromones: list[PheromoneEntry] = []
    for phero, text in zip(phero_signals, texts):
        phero_entry = associate_pheromone_with_text(phero, text)
        PheromoneStore.add(phero_entry)
        pheromones.append(phero_entry)
    return pheromones


def apply_pheromone_decay(pheromones: list[PheromoneEntry]) -> list[dict]:
    decay_results: list[dict] = []
    for phero in pheromones:
        PheromoneStore.decay_surface(phero.surface_key)
        decay_results.append({
            "pheromone_uuid": phero.uuid,
            "surface_key": phero.surface_key,
            "signal_kind": phero.signal_kind,
            "signal_value": phero.signal_value
        })
    return decay_results


if __name__ == "__main__":
    texts = ["This is a test text.", "Another test text.", "Text with high entropy: abcdefghijklmnopqrstuvwxyz"]
    phero_signals = [1.0, 0.5, 0.8]
    pheromones = simulate_pheromone_diffusion(phero_signals, texts)
    decay_results = apply_pheromone_decay(pheromones)
    print(decay_results)