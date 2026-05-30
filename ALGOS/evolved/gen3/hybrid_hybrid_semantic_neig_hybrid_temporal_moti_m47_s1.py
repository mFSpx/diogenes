# DARWIN HAMMER — match 47, survivor 1
# gen: 3
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s1.py (gen2)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s1.py (gen1)
# born: 2026-05-29T23:26:30Z

"""
Hybrid algorithm merging the semantic neighbor search and morphology-based recovery priority (hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s1.py)
with temporal motif mining and possum-style spatial diversity filtering (hybrid_temporal_motifs_possum_filter_m87_s1.py).

Mathematical bridge:
- Parent A defines a recovery priority based on morphology and a semantic similarity metric.
- Parent B generates temporal motifs and applies a possum filter for spatial diversity.
- The fusion builds a score S(m) = r(m) * z_s, where r(m) is the recovery priority and z_s is the z-score of the motif support distribution.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from random import random
from sys import exit
from pathlib import Path
from typing import Any, Dict, List, Tuple

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def semantic_neighbors(doc_id: str, vector: list[float], k: int=5) -> list[tuple[str,float]]:
    den= np.sqrt(sum(x*x for x in vector))*np.sqrt(sum(y*y for y in vector)); 
    return sorted(((d,_cos(vector,w)) for d,w in [(doc_id,vector)]+[("doc"+str(i),np.random.rand(5)) for i in range(1,k+1)] if d!=doc_id), key=lambda x:(-x[1],x[0]))[:k]

def _cos(a,b):
    den= np.sqrt(sum(x*x for x in a))*np.sqrt(sum(y*y for y in b)); 
    return 0.0 if den==0 else sum(x*y for x,y in zip(a,b))/den

@dataclass(frozen=True)
class BurstSignal:
    key: str
    count: int
    z_score: float

@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int

@dataclass(frozen=True)
class HybridMotif:
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    score: float  

def sessionize_events(events: List[dict], gap_seconds: float = 1800.0) -> List[List[dict]]:
    sessions: List[List[dict]] = []
    cur: List[dict] = []
    last: float | None = None
    for e in sorted(events, key=lambda x: float(x.get('t', 0))):
        t = float(e.get('t', 0))
        if cur and last is not None and t - last > gap_seconds:
            sessions.append(cur)
            cur = []
        cur.append(e)
        last = t
    if cur:
        sessions.append(cur)
    return sessions

def calculate_hybrid_score(morphology: Morphology, motif: TemporalMotif, max_recovery_priority: float = 10.0) -> float:
    recovery_priority_score = recovery_priority(morphology, max_recovery_priority)
    z_score = (motif.support - np.mean([m.support for m in [TemporalMotif(("a",), 10), TemporalMotif(("b",), 20)]])) / np.std([m.support for m in [TemporalMotif(("a",), 10), TemporalMotif(("b",), 20)]])
    return recovery_priority_score * z_score

def get_hybrid_motif(morphology: Morphology, motif: TemporalMotif, centroid_lat: float, centroid_lon: float) -> HybridMotif:
    score = calculate_hybrid_score(morphology, motif)
    return HybridMotif(motif.pattern, motif.support, centroid_lat, centroid_lon, score)

def filter_motifs(motifs: List[HybridMotif], threshold: float) -> List[HybridMotif]:
    return [m for m in motifs if m.score > threshold]

if __name__ == "__main__":
    morphology = Morphology(10.0, 5.0, 3.0, 100.0)
    motif = TemporalMotif(("a",), 10)
    hybrid_motif = get_hybrid_motif(morphology, motif, 37.7749, -122.4194)
    print(hybrid_motif)
    filtered_motifs = filter_motifs([hybrid_motif], 0.5)
    print(filtered_motifs)