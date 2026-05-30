# DARWIN HAMMER — match 107, survivor 0
# gen: 2
# parent_a: krampus_stickers.py (gen0)
# parent_b: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# born: 2026-05-29T23:25:41Z

"""
This module represents a novel hybrid algorithm that fuses the core topologies of 
'krampus_stickers.py' and 'hybrid_pheromone_infotaxis_m3_s4.py'. The mathematical bridge 
between the two structures lies in the application of information theory and pheromone 
dynamics to text analysis. We integrate the Shannon entropy calculation from 'krampus_stickers.py' 
with the pheromone decay mechanism from 'hybrid_pheromone_infotaxis_m3_s4.py' to create a 
hybrid system that analyzes text data while considering the temporal dynamics of information.
"""

import numpy as np
import math
import random
import sys
import pathlib
import re
from typing import Any, List, Dict

MAX_COMPONENT_TOKENS = 500

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(random.getrandbits(128))
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        self.created_at = None
        self.last_decay = None

    def age_seconds(self) -> float:
        if self.last_decay is None:
            return 0.0
        return (self.last_decay - self.created_at).total_seconds()

    def decay_factor(self) -> float:
        """Return the multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = self.created_at

class PheromoneStore:
    """Singleton-like in-memory store for demo purposes."""
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))

def entropy_for_text(text: str) -> float:
    return float(shannon_entropy(list((text or "")[:10000]))) if text else 0.0

def shannon_entropy(data: List[str]) -> float:
    if not data:
        return 0.0
    entropy = 0.0
    for x in set(data):
        p_x = data.count(x) / len(data)
        entropy += - p_x * math.log(p_x, 2)
    return entropy

def links_from_text(text: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
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
    for m in re.finditer(r'''\bhttps?://[^\s<>'")]+''', text or "", re.I):
        target = m.group(0).rstrip(".,;")
        key = ("url", target, target)
        if key not in seen:
            seen.add(key)
            links.append({"link_kind": "url", "raw_target": target, "anchor_text": target[:240], "source": "bare_url"})
    return links

def calculate_pheromone(text: str, half_life_seconds: int) -> PheromoneEntry:
    entropy = entropy_for_text(text)
    entry = PheromoneEntry("text", "entropy", entropy, half_life_seconds)
    entry.created_at = sys.float_info.max
    entry.last_decay = sys.float_info.max
    return entry

def apply_pheromone_decay(entry: PheromoneEntry) -> None:
    entry.apply_decay()

def analyze_text_with_pheromone(text: str, half_life_seconds: int) -> float:
    entry = calculate_pheromone(text, half_life_seconds)
    apply_pheromone_decay(entry)
    return entry.signal_value

if __name__ == "__main__":
    text = "This is a sample text with some entropy and links."
    half_life_seconds = 10
    result = analyze_text_with_pheromone(text, half_life_seconds)
    print(f"Result: {result}")