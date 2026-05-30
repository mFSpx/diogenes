# DARWIN HAMMER — match 107, survivor 2
# gen: 2
# parent_a: krampus_stickers.py (gen0)
# parent_b: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# born: 2026-05-29T23:25:41Z

"""Hybrid module combining Krampus sticker text analytics (Parent A) with
Pheromone infotaxis dynamics (Parent B).

Mathematical bridge:
- Parent A extracts a feature vector **f(text)** = (tokens, entropy,
  link_counts, …).
- Parent B treats each scalar feature as a pheromone signal **s** with
  exponential decay:  s(t) = s₀·½^{Δt/τ}.
- The hybrid maps **f(text)** → a set of **PheromoneEntry** objects where
  the initial signal value is the normalized feature magnitude and the
  half‑life τ is a monotonic function of the text entropy (high entropy →
  slower decay).  Decay and aggregation are then performed on the combined
  signal space, providing a time‑aware document metric.

The code below implements this fusion with three public functions:
`extract_features`, `inject_pheromones`, and `decay_and_aggregate`.
"""

import re
import math
import random
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – text feature extraction (simplified)
# ----------------------------------------------------------------------


def normalize_ws(text: str) -> str:
    """Collapse whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def token_count(text: str) -> int:
    """Count whitespace‑separated tokens."""
    return len(re.findall(r"\S+", text or ""))


def shannon_entropy(symbols: List[str]) -> float:
    """Classic Shannon entropy H = -Σ p·log₂(p) for a list of symbols."""
    if not symbols:
        return 0.0
    total = len(symbols)
    freq = Counter(symbols)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def entropy_for_text(text: str, max_len: int = 10_000) -> float:
    """Entropy of the first `max_len` characters of `text`."""
    if not text:
        return 0.0
    snippet = list(text[:max_len])
    return shannon_entropy(snippet)


def links_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract markdown links, wikilinks and bare URLs."""
    links: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    # Markdown links [anchor](url)
    for m in re.finditer(r"\[([^\]]{0,240})\]\(([^)\s]{1,1000})\)", text or ""):
        target = m.group(2).strip()
        kind = "url" if target.lower().startswith(("http://", "https://")) else "markdown"
        key = (kind, target, m.group(1))
        if key not in seen:
            seen.add(key)
            links.append(
                {
                    "link_kind": kind,
                    "raw_target": target,
                    "anchor_text": m.group(1),
                    "source": "markdown_link",
                }
            )

    # Wikilinks [[Page|Alias]]
    for m in re.finditer(r"\[\[([^\]|#]{1,500})(?:#[^\]|]+)?(?:\|([^\]]{1,240}))?\]\]", text or ""):
        target = m.group(1).strip()
        anchor = (m.group(2) or target).strip()
        key = ("wikilink", target, anchor)
        if key not in seen:
            seen.add(key)
            links.append(
                {
                    "link_kind": "wikilink",
                    "raw_target": target,
                    "anchor_text": anchor,
                    "source": "wikilink",
                }
            )

    # Bare URLs
    for m in re.finditer(r"""\bhttps?://[^\s<>'")]+""", text or "", re.I):
        target = m.group(0).rstrip(".,;")
        key = ("url", target, target)
        if key not in seen:
            seen.add(key)
            links.append(
                {
                    "link_kind": "url",
                    "raw_target": target,
                    "anchor_text": target[:240],
                    "source": "bare_url",
                }
            )
    return links


def extract_features(text: str) -> Dict[str, float]:
    """Compute a lightweight numeric feature vector from `text`."""
    norm = normalize_ws(text)
    tokens = token_count(norm)
    entropy = entropy_for_text(norm)
    link_objs = links_from_text(norm)
    link_counts = Counter(l["link_kind"] for l in link_objs)

    # Normalizations (avoid division by zero)
    token_norm = tokens / 1000.0  # assume 1000 tokens ≈ scale 1
    entropy_norm = entropy / 8.0  # max entropy for ASCII ≈ 8 bits
    url_norm = link_counts.get("url", 0) / 10.0
    wikilink_norm = link_counts.get("wikilink", 0) / 10.0
    markdown_norm = link_counts.get("markdown", 0) / 10.0

    return {
        "tokens": token_norm,
        "entropy": entropy_norm,
        "url_links": url_norm,
        "wikilink_links": wikilink_norm,
        "markdown_links": markdown_norm,
    }


# ----------------------------------------------------------------------
# Parent B – pheromone store with exponential decay
# ----------------------------------------------------------------------


class PheromoneEntry:
    """A single pheromone signal with exponential decay."""

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
        self.half_life_seconds = max(1, half_life_seconds)  # enforce >0
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative factor since last_decay based on half‑life."""
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory singleton store for pheromone entries."""

    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> List[Dict[str, Any]]:
        rows = []
        for entry in cls.get_by_surface(surface_key):
            before = entry.signal_value
            entry.apply_decay()
            rows.append(
                {
                    "pheromone_uuid": entry.uuid,
                    "surface_key": entry.surface_key,
                    "signal_kind": entry.signal_kind,
                    "signal_value_before": before,
                    "signal_value_after": entry.signal_value,
                }
            )
        return rows

    @classmethod
    def aggregate_surface(cls, surface_key: str) -> Dict[str, float]:
        """Sum remaining signal values per kind after decay."""
        agg: Dict[str, float] = {}
        for entry in cls.get_by_surface(surface_key):
            entry.apply_decay()
            agg[entry.signal_kind] = agg.get(entry.signal_kind, 0.0) + entry.signal_value
        return agg


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def inject_pheromones(surface_key: str, text: str) -> List[PheromoneEntry]:
    """
    Convert extracted text features into pheromone entries.

    Half‑life τ is proportional to (1 + entropy) to give high‑entropy
    documents longer‑lived signals.
    """
    feats = extract_features(text)
    entries: List[PheromoneEntry] = []
    base_half_life = 3600  # 1 hour baseline

    for kind, value in feats.items():
        # Scale half‑life: more entropy → slower decay
        half_life = int(base_half_life * (1.0 + feats["entropy"]))
        entry = PheromoneEntry(
            surface_key=surface_key,
            signal_kind=kind,
            signal_value=value,
            half_life_seconds=half_life,
        )
        PheromoneStore.add(entry)
        entries.append(entry)
    return entries


def decay_and_aggregate(surface_key: str) -> Dict[str, float]:
    """
    Apply decay to all pheromones on `surface_key` and return an aggregated
    feature vector.
    """
    # Decay is already performed inside `aggregate_surface`
    aggregated = PheromoneStore.aggregate_surface(surface_key)
    return aggregated


def surface_score(surface_key: str, weights: Dict[str, float] | None = None) -> float:
    """
    Compute a scalar score for a surface by weighting the aggregated
    pheromone signals.  If `weights` is omitted, equal weighting is used.
    """
    agg = decay_and_aggregate(surface_key)
    if not agg:
        return 0.0
    if weights is None:
        # Equal weight for each present signal kind
        weights = {k: 1.0 for k in agg}
    # Ensure every key has a weight (default 0)
    total = 0.0
    for kind, val in agg.items():
        w = weights.get(kind, 0.0)
        total += w * val
    return total


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    sample_text = """
    # Example Document

    This is a test document containing several links:
    - Markdown link: [OpenAI](https://openai.com)
    - Wikilink: [[HomePage|Start]]
    - Bare URL: https://example.org/resource

    The quick brown fox jumps over the lazy dog. Repeated words words words.
    """
    surface = "doc_001"

    print("Extracted features:")
    print(extract_features(sample_text))

    print("\nInjecting pheromones...")
    injected = inject_pheromones(surface, sample_text)
    for e in injected:
        print(f"  {e.signal_kind}: value={e.signal_value:.4f}, half_life={e.half_life_seconds}s")

    print("\nAggregated after immediate decay (should be near initial):")
    agg_now = decay_and_aggregate(surface)
    print(agg_now)

    # Simulate passage of time by manually adjusting timestamps
    for entry in PheromoneStore.get_by_surface(surface):
        entry.last_decay -= timedelta(seconds=7200)  # 2 hours ago

    print("\nAggregated after 2‑hour simulated decay:")
    agg_later = decay_and_aggregate(surface)
    print(agg_later)

    print("\nSurface score (equal weights):")
    print(surface_score(surface))