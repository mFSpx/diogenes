# DARWIN HAMMER — match 323, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_gliner_zero_s_hybrid_krampus_brain_m30_s2.py (gen3)
# parent_b: hybrid_possum_filter_hybrid_privacy_model_m53_s1.py (gen2)
# born: 2026-05-29T23:28:17Z

"""Hybrid algorithm merging:
- Parent A: hybrid_gliner_zero_shot_ext_minimum_cost_tree (deterministic Span matcher + pheromone information gain)
- Parent B: hybrid_possum_filter_hybrid_privacy_model (spatial diversity filter using haversine distance)

Mathematical bridge:
Both parents manipulate weighted objects (spans, entities) and apply a scalar field:
* A uses entropy‑based information gain `IG = - Σ p_i log p_i` on pheromone signal values.
* B uses a Euclidean‑like metric `d = haversine` to enforce diversity.

The hybrid treats each Span–Entity pair as a joint random variable.
Its joint “information weight” is the product of the Span score and the Entity score,
scaled by a distance‑based attenuation factor `α = exp(-d/λ)`.
The resulting weight feeds the pheromone store, where decay follows the half‑life law of A.
Thus the entropy of the pheromone distribution is continuously reshaped by the spatial
diversity constraints of B."""

import math
import random
import sys
import pathlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
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
        """Multiplicative decay since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Very small in‑memory singleton‑like store."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PheromoneStore, cls).__new__(cls)
            cls._instance._entries: Dict[str, PheromoneEntry] = {}
        return cls._instance

    def add(self, entry: PheromoneEntry) -> None:
        self._entries[entry.uuid] = entry

    def all(self) -> List[PheromoneEntry]:
        return list(self._entries.values())

    def decay_all(self) -> None:
        for e in self._entries.values():
            e.apply_decay()


# ----------------------------------------------------------------------
# Core data structures (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()


def keep_candidate(candidate: Entity, selected: List[Entity],
                  delta_m: float) -> bool:
    """Return True if `candidate` is sufficiently distant from all `selected`."""
    for existing in selected:
        same_kind = (signature(candidate) == signature(existing) or
                     candidate.category.strip().lower() ==
                     existing.category.strip().lower())
        if same_kind and haversine_m((candidate.lat, candidate.lon),
                                    (existing.lat, existing.lon)) <= delta_m:
            return False
    return True


def filter_entities(entities: Iterable[Entity],
                    delta_m: float = 75.0,
                    sort_by_score: bool = True) -> List[Entity]:
    """Diversity filter à la Possum."""
    if delta_m < 0:
        raise ValueError("delta_m must be non‑negative")
    ordered = list(entities)
    if sort_by_score:
        ordered.sort(key=lambda e: (-e.score, e.id))
    selected: List[Entity] = []
    for entity in ordered:
        if keep_candidate(entity, selected, delta_m):
            selected.append(entity)
    return selected


# ----------------------------------------------------------------------
# Hybrid operations (new)
# ----------------------------------------------------------------------
def compute_information_gain(values: List[float]) -> float:
    """Entropy‑based information gain of a list of non‑negative scores."""
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    probs = np.array(values) / total
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log2(probs))
    # Information gain is reduction from maximum entropy (log2 N)
    max_entropy = math.log2(len(values))
    return max_entropy - entropy


def select_hybrid_candidates(spans: List[Span],
                             entities: List[Entity],
                             delta_m: float = 75.0,
                             lambda_dist: float = 200.0) -> List[Tuple[Span, Entity, float]]:
    """
    Pair each Span with the nearest Entity that satisfies the Possum diversity
    constraint.  The returned weight w is

        w = (span.score * entity.score) * exp(-d / λ)

    where d is the haversine distance between the Span's pseudo‑location
    (derived from its start index) and the Entity's (lat, lon).  The exponential
    term implements the mathematical bridge between spatial diversity and
    information gain.
    """
    # First, enforce diversity on entities.
    diverse_entities = filter_entities(entities, delta_m=delta_m, sort_by_score=True)

    # Map Span start index to a pseudo‑coordinate on Earth for demo purposes.
    # We place them on a latitude line at 0° latitude, longitude = start * 0.001°
    def span_coord(span: Span) -> Tuple[float, float]:
        return (0.0, span.start * 0.001)

    matches: List[Tuple[Span, Entity, float]] = []
    for span in spans:
        s_coord = span_coord(span)
        best_entity = None
        best_weight = -1.0
        for ent in diverse_entities:
            d = haversine_m(s_coord, (ent.lat, ent.lon))
            attenuation = math.exp(-d / lambda_dist)
            weight = span.score * ent.score * attenuation
            if weight > best_weight:
                best_weight = weight
                best_entity = ent
        if best_entity is not None:
            matches.append((span, best_entity, best_weight))
    return matches


def apply_hybrid_pheromones(matches: List[Tuple[Span, Entity, float]],
                           half_life_seconds: int = 3600) -> None:
    """
    For every (Span, Entity, weight) triple, create a pheromone entry whose
    signal value equals the weight multiplied by the joint information gain
    of the two scores.  The entry is stored in the global PheromoneStore.
    """
    store = PheromoneStore()
    # Compute a global information gain factor from all weights.
    weights = [w for _, _, w in matches]
    ig_factor = compute_information_gain(weights) + 1e-6  # avoid zero
    for span, ent, weight in matches:
        signal_value = weight * ig_factor
        surface_key = f"{span.start}-{span.end}:{ent.id}"
        entry = PheromoneEntry(surface_key=surface_key,
                               signal_kind="hybrid",
                               signal_value=signal_value,
                               half_life_seconds=half_life_seconds)
        store.add(entry)


def decay_and_report() -> None:
    """Decay all pheromones and print a short report."""
    store = PheromoneStore()
    before = [e.signal_value for e in store.all()]
    store.decay_all()
    after = [e.signal_value for e in store.all()]
    print(f"Pheromone count: {len(before)}")
    for i, (b, a) in enumerate(zip(before, after), 1):
        print(f"  Entry {i}: {b:.6f} → {a:.6f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy spans
    dummy_spans = [
        Span(start=10, end=20, text="foo", label="A", score=0.8),
        Span(start=30, end=45, text="bar", label="B", score=0.6),
        Span(start=70, end=85, text="baz", label="C", score=0.9),
    ]

    # Create dummy entities spread over a small geographic area
    dummy_entities = [
        Entity(id="e1", lat=0.0, lon=0.010, category="type1", score=0.7),
        Entity(id="e2", lat=0.0, lon=0.030, category="type1", score=0.5),
        Entity(id="e3", lat=0.0, lon=0.060, category="type2", score=0.9),
        Entity(id="e4", lat=0.0, lon=0.100, category="type2", score=0.4),
    ]

    # Hybrid selection
    pairs = select_hybrid_candidates(dummy_spans, dummy_entities,
                                     delta_m=75.0, lambda_dist=200.0)
    print("Selected Span‑Entity pairs with weights:")
    for span, ent, w in pairs:
        print(f"  Span[{span.start}:{span.end}] ↔ Entity[{ent.id}] weight={w:.6f}")

    # Apply pheromones
    apply_hybrid_pheromones(pairs, half_life_seconds=1800)

    # Decay and report
    decay_and_report()