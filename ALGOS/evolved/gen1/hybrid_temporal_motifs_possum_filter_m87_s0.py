# DARWIN HAMMER — match 87, survivor 0
# gen: 1
# parent_a: temporal_motifs.py (gen0)
# parent_b: possum_filter.py (gen0)
# born: 2026-05-29T23:25:37Z

import math
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


@dataclass(frozen=True)
class BurstSignal:
    key: str
    count: int
    z_score: float


@dataclass(frozen=True)
class TemporalMotif:
    pattern: tuple[str, ...]
    support: int


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()


def keep_candidate(candidate: Entity, selected: list[Entity], delta_m: float) -> bool:
    for existing in selected:
        same_kind = signature(candidate) == signature(existing) or candidate.category.strip().lower() == existing.category.strip().lower()
        if same_kind and haversine_m((candidate.lat, candidate.lon), (existing.lat, existing.lon)) <= delta_m:
            return False
    return True


class HybridTemporalMotif:
    def __init__(self, entities: Iterable[Entity], delta_m: float, gap_seconds: float = 1800.0):
        self.entities = entities
        self.delta_m = delta_m
        self.gap_seconds = gap_seconds

    def sessionize_events(self, events: list[dict]) -> list[list[dict]]:
        return sessionize_events(events, self.gap_seconds)

    def detect_bursts(self, events: list[dict], key: str = 'category') -> list[BurstSignal]:
        return detect_bursts(events, key)

    def mine_temporal_motifs(self, sessions: list[list[dict]], key: str = 'category', min_support: int = 2) -> list[TemporalMotif]:
        return mine_temporal_motifs(sessions, key, min_support)

    def filter_entities(self) -> Iterable[Entity]:
        ordered = self.entities
        if isinstance(ordered, list):
            ordered.sort(key=lambda e: (-e.score, e.id))
        selected: list[Entity] = []
        for entity in ordered:
            if keep_candidate(entity, selected, self.delta_m):
                selected.append(entity)
        return selected

    def hybrid(self) -> Iterable[TemporalMotif]:
        sessions = self.sessionize_events([e._asdict() for e in self.entities])
        bursts = self.detect_bursts([e._asdict() for e in self.entities if keep_candidate(e, [e], self.delta_m)])
        motifs = self.mine_temporal_motifs(sessions)
        filtered_entities = self.filter_entities()
        return [TemporalMotif(p, v if v <= len(filtered_entities) else sum(1 for e, x in zip(filtered_entities, bursts) if e.category in p)) for p, v in [(tuple(str(e.get('category','')) for e in s), len(s)) for s in sessions]]


def sessionize_events(events: list[dict], gap_seconds: float = 1800.0) -> list[list[dict]]:
    sessions = []
    cur = []
    last = None
    for e in sorted(events, key=lambda x: x.get('t', 0)):
        t = float(e.get('t', 0))
        if cur and last is not None and t - last > gap_seconds:
            sessions.append(cur)
            cur = []
        cur.append(e)
        last = t
    if cur:
        sessions.append(cur)
    return sessions


def detect_bursts(events: list[dict], key: str = 'type') -> list[BurstSignal]:
    c = Counter(str(e.get(key, '')) for e in events)
    if not c:
        return []
    mean = sum(c.values()) / len(c)
    sd = math.sqrt(sum((v - mean) ** 2 for v in c.values()) / len(c)) or 1.0
    return [BurstSignal(k, v, (v - mean) / sd) for k, v in c.items() if v >= mean]


def mine_temporal_motifs(sessions: list[list[dict]], key: str = 'type', min_support: int = 2) -> list[TemporalMotif]:
    c = Counter(tuple(str(e.get(key, '')) for e in s) for s in sessions)
    return [TemporalMotif(p, v) for p, v in c.items() if v >= min_support]


if __name__ == "__main__":
    import random

    entities = [
        Entity(f'{i}', random.uniform(37.7749, 37.7859), random.uniform(-122.4194, -122.4294), f'E{i}', 0.1, f'{i}')
        for i in range(15)
    ]

    hybrid_model = HybridTemporalMotif(entities, delta_m=75.0)

    sessions = hybrid_model.sessionize_events([e._asdict() for e in entities])
    for s in sessions:
        print(s)

    bursts = hybrid_model.detect_bursts([e._asdict() for e in entities])
    for b in bursts:
        print(b)

    motifs = hybrid_model.mine_temporal_motifs(sessions)
    for m in motifs:
        print(m)

    filtered_entities = hybrid_model.filter_entities()
    for e in filtered_entities:
        print(e)

    final_motifs = hybrid_model.hybrid()
    for m in final_motifs:
        print(m)