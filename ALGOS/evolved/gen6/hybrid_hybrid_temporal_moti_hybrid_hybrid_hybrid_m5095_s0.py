# DARWIN HAMMER — match 5095, survivor 0
# gen: 6
# parent_a: hybrid_temporal_motifs_hybrid_hybrid_hybrid_m867_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_tempor_hybrid_hybrid_doomsd_m676_s3.py (gen3)
# born: 2026-05-29T23:59:41Z

"""
This module fuses the core topologies of hybrid_temporal_motifs_hybrid_hybrid_hybrid_m867_s2.py and 
hybrid_hybrid_hybrid_tempor_hybrid_hybrid_doomsd_m676_s3.py. The mathematical bridge between the two 
structures lies in the use of numpy arrays to represent vectors and matrices. Specifically, the 
temporal motif detection in the first parent can be combined with the spatial-temporal utilities and 
hyperdimensional primitives in the second parent by representing temporal motifs as vectors and 
applying spatial-temporal operations to these vectors. The resulting hybrid algorithm enables the 
detection of spatial-temporal patterns in data.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class BurstSignal: 
    key: str; 
    count: int; 
    z_score: float

@dataclass(frozen=True)
class TemporalMotif: 
    pattern: tuple[str,...]; 
    support: int

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

def haversine_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    c = 2 * math.atan2(math.sqrt(math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2), math.sqrt(1-math.sin(dlat/2)**2 - math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2))
    return 6371 * c  # radius of the Earth in kilometers

def doomsday_numpy(date: str) -> int:
    year, month, day = map(int, date.split('-'))
    t = [year // 100, year % 100]
    return (t[0] // 4 - 2 * t[0] + t[1] // 4 + 26 * (t[1] % 28 + 26 + 28 * ((t[1] + 27) % 28)) // 28) % 7

def gini_coefficient(values: list[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    n = len(xs)
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * xs)) / (n * np.sum(xs)))

def random_vector(dim: int = 10000, seed: str | int | None = None) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() * 2 - 1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)

def bind(a: list[float], b: list[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: list[list[float]]) -> list[float]:
    vecs = list(vectors)
    if not vecs:
        return []
    result = vecs[0]
    for vec in vecs[1:]:
        result = [x + y for x, y in zip(result, vec)]
    return result

def detect_bursts(events: list[dict], key: str='type') -> list[BurstSignal]:
    c = {str(e.get(key,'')): 0 for e in events}
    for e in events:
        c[str(e.get(key,''))] += 1
    if not c: 
        return []
    mean = sum(c.values()) / len(c)
    sd = math.sqrt(sum((v - mean) ** 2 for v in c.values()) / len(c)) or 1.0
    return [BurstSignal(k, v, (v - mean) / sd) for k, v in c.items() if v >= mean]

def mine_temporal_motifs(sessions: list[list[dict]], min_support: int = 2) -> list[TemporalMotif]:
    motifs = []
    for session in sessions:
        for i in range(len(session)):
            for j in range(i + 1, len(session)):
                pattern = tuple(e.get('type') for e in session[i:j + 1])
                support = sum(1 for sess in sessions if any(tuple(e.get('type') for e in sess[k:k + len(pattern)]) == pattern for k in range(len(sess) - len(pattern) + 1)))
                if support >= min_support:
                    motifs.append(TemporalMotif(pattern, support))
    return motifs

def spatial_temporal_analysis(entities: list[Entity], events: list[dict]) -> list[TemporalMotif]:
    sessions = []
    for entity in entities:
        entity_events = [e for e in events if e.get('entity_id') == entity.id]
        entity_sessions = []
        cur_session = []
        last_time = None
        for event in sorted(entity_events, key=lambda x: x.get('t', 0)):
            time = float(event.get('t', 0))
            if cur_session and last_time is not None and time - last_time > 1800:
                entity_sessions.append(cur_session)
                cur_session = []
            cur_session.append(event)
            last_time = time
        if cur_session:
            entity_sessions.append(cur_session)
        sessions.extend(entity_sessions)
    return mine_temporal_motifs(sessions)

if __name__ == "__main__":
    entities = [Entity('1', 37.7749, -122.4194, 'category1'), Entity('2', 34.0522, -118.2437, 'category2')]
    events = [{'t': 1643723400, 'entity_id': '1', 'type': 'event1'}, {'t': 1643723405, 'entity_id': '1', 'type': 'event2'}, 
              {'t': 1643723410, 'entity_id': '2', 'type': 'event1'}, {'t': 1643723415, 'entity_id': '2', 'type': 'event3'}]
    motifs = spatial_temporal_analysis(entities, events)
    print(motifs)