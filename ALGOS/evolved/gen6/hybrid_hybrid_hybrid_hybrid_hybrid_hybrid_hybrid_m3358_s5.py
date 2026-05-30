# DARWIN HAMMER — match 3358, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m530_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1482_s5.py (gen5)
# born: 2026-05-29T23:49:29Z

"""Hybrid Fusion of:
- Parent A: hash‑based signature generation, similarity, and ternary vector algebra.
- Parent B: span‑entity joint weighting with spatial decay, regex‑derived entropy, and pheromone dynamics.

Mathematical Bridge
-------------------
Both parents manipulate scalar fields over discrete objects.  
Parent A provides a deterministic similarity `S ∈ [0,1]` between two
hash‑signatures.  This similarity can be interpreted as a probability
estimate for a joint span–entity event.  By converting `S` into a binary
distribution `{S, 1‑S}` we obtain an entropy `H(S) = -S·log S -(1‑S)·log(1‑S)`.
Parent B already incorporates an entropy term `Ĥ` (derived from regex
counts) into its joint weight:

    w = (span.score * entity.score) * exp(-d/λ) * (1 + α·Ĥ)

The hybrid replaces `Ĥ` with the normalized entropy `Ĥ_S = H(S)/H_max`,
thereby fusing the deterministic signature similarity of A with the
probabilistic weighting of B.  The resulting weight becomes the new
pheromone signal, which is stored and decays exponentially according to
a configurable half‑life.

The module implements the combined pipeline:
1. Generate signatures for token sets (A).
2. Compute similarity and derived entropy (A → B).
3. Evaluate joint weight with spatial decay (B).
4. Insert/update pheromones with exponential decay (B).

All operations rely only on NumPy and the Python standard library.
"""

import math
import random
import sys
import pathlib
import time
import hashlib
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Min‑hash style signature of length *k*."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

# ----------------------------------------------------------------------
# Data structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """A textual span with optional geographic anchor."""
    start: int
    end: int
    text: str
    label: str
    score: float
    lat: float = 0.0
    lon: float = 0.0

@dataclass(frozen=True)
class Entity:
    """External entity that can be linked to a Span."""
    identifier: str
    score: float
    lat: float = 0.0
    lon: float = 0.0

# ----------------------------------------------------------------------
# Hybrid mathematical components
# ----------------------------------------------------------------------
def entropy_from_similarity(sim: float) -> float:
    """
    Convert a similarity value `sim ∈ [0,1]` into a normalized entropy term.
    The binary distribution {sim, 1‑sim} yields entropy H = -∑p·log p.
    Normalization divides by the maximal binary entropy `log 2`.
    """
    if not 0.0 <= sim <= 1.0:
        raise ValueError("similarity must be in [0,1]")
    # Guard against log(0)
    eps = 1e-12
    p1 = max(sim, eps)
    p2 = max(1.0 - sim, eps)
    h = -(p1 * math.log(p1) + p2 * math.log(p2))
    h_max = math.log(2.0)
    return h / h_max  # ∈ [0,1]

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great‑circle distance in kilometers."""
    R = 6371.0  # Earth radius
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def joint_weight(
    span: Span,
    entity: Entity,
    entropy_term: float,
    spatial_scale: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """
    Hybrid joint weight:
        w = (span.score * entity.score) *
            exp(-d / λ) *
            (1 + α·Ĥ_S)

    where d is haversine distance, λ is spatial_scale,
    and Ĥ_S is the normalized entropy derived from signature similarity.
    """
    d = haversine_distance(span.lat, span.lon, entity.lat, entity.lon)
    decay = math.exp(-d / max(spatial_scale, 1e-6))
    base = span.score * entity.score
    return base * decay * (1.0 + alpha * entropy_term)

# ----------------------------------------------------------------------
# Pheromone store handling (Parent B style)
# ----------------------------------------------------------------------
@dataclass
class PheromoneEntry:
    value: float
    timestamp: float  # seconds since epoch

class PheromoneStore:
    """Simple in‑memory pheromone repository with exponential decay."""
    def __init__(self, half_life_seconds: float = 3600.0):
        self.half_life = max(half_life_seconds, 1e-6)
        self.store: Dict[Tuple[int, int], PheromoneEntry] = {}

    def _decay_factor(self, elapsed: float) -> float:
        """Exponential decay factor based on half‑life."""
        return 0.5 ** (elapsed / self.half_life)

    def insert_or_update(self, key: Tuple[int, int], weight: float) -> None:
        now = time.time()
        entry = self.store.get(key)
        if entry is None:
            self.store[key] = PheromoneEntry(value=weight, timestamp=now)
        else:
            elapsed = now - entry.timestamp
            decayed = entry.value * self._decay_factor(elapsed)
            self.store[key] = PheromoneEntry(value=decayed + weight, timestamp=now)

    def get(self, key: Tuple[int, int]) -> float:
        entry = self.store.get(key)
        if entry is None:
            return 0.0
        elapsed = time.time() - entry.timestamp
        return entry.value * self._decay_factor(elapsed)

    def prune(self, threshold: float = 1e-6) -> None:
        """Remove entries whose decayed value falls below *threshold*."""
        to_delete = [k for k, v in self.store.items() if self.get(k) < threshold]
        for k in to_delete:
            del self.store[k]

# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_document_score(
    spans: List[Span],
    entities: List[Entity],
    reference_tokens: Iterable[str],
    k: int = 64,
    spatial_scale: float = 5.0,
    alpha: float = 0.7,
    pheromone_half_life: float = 1800.0,
) -> Tuple[float, PheromoneStore]:
    """
    Compute an aggregate score for a document by:
    1. Building a reference signature from *reference_tokens* (Parent A).
    2. For each (span, entity) pair:
       a. Build a combined token set = span.text tokens ∪ entity.identifier tokens.
       b. Compute signature similarity with the reference.
       c. Derive normalized entropy from that similarity.
       d. Evaluate the hybrid joint weight.
       e. Insert the weight into a pheromone store.
    3. Return the sum of decayed pheromone values as the document score.
    """
    ref_sig = signature(reference_tokens, k=k)
    pher_store = PheromoneStore(half_life_seconds=pheromone_half_life)
    total_score = 0.0

    for span in spans:
        span_tokens = span.text.lower().split()
        for entity in entities:
            entity_tokens = entity.identifier.lower().split("_")
            joint_tokens = set(span_tokens) | set(entity_tokens)
            joint_sig = signature(joint_tokens, k=k)
            sim = similarity(ref_sig, joint_sig)
            ent = entropy_from_similarity(sim)
            w = joint_weight(
                span,
                entity,
                entropy_term=ent,
                spatial_scale=spatial_scale,
                alpha=alpha,
            )
            key = (hash(span), hash(entity))
            pher_store.insert_or_update(key, w)
            total_score += w

    # Apply decay to all stored pheromones before returning
    pher_store.prune()
    return total_score, pher_store

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic data
    spans = [
        Span(start=0, end=5, text="Paris", label="LOC", score=0.9, lat=48.8566, lon=2.3522),
        Span(start=10, end=15, text="Eiffel Tower", label="LOC", score=0.85, lat=48.8584, lon=2.2945),
    ]

    entities = [
        Entity(identifier="city_paris", score=0.95, lat=48.8566, lon=2.3522),
        Entity(identifier="monument_eiffel", score=0.92, lat=48.8584, lon=2.2945),
    ]

    reference = ["travel", "tourism", "landmark", "city", "france"]

    doc_score, store = hybrid_document_score(
        spans,
        entities,
        reference_tokens=reference,
        k=32,
        spatial_scale=10.0,
        alpha=0.6,
        pheromone_half_life=3600.0,
    )

    print(f"Document hybrid score: {doc_score:.4f}")
    print(f"Pheromone entries: {len(store.store)}")
    for key, entry in store.store.items():
        decayed = store.get(key)
        print(f"  Key {key}: raw={entry.value:.4f}, decayed={decayed:.4f}")