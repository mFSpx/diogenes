# DARWIN HAMMER — match 47, survivor 2
# gen: 3
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s1.py (gen2)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s1.py (gen1)
# born: 2026-05-29T23:26:30Z

"""
Hybrid algorithm merging DARWIN HAMMER's hybrid semantic neighbors with hybrid endpoint circuit 
(parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s1.py) and 
DARWIN HAMMER's hybrid temporal motifs with possum filter (parent_b: hybrid_temporal_motifs_possum_filter_m87_s1.py).

The mathematical bridge:
- Parent A produces a recovery priority score based on morphology and 
  semantic neighbors.
- Parent B defines a joint score S(p) = s(p) · (1 + z_s) where z_s is the 
  z-score of the support distribution across patterns.
- The fusion builds a unified score that combines the recovery priority 
  with the joint score, enabling a spatio-temporal analysis of 
  morphologies and their semantic neighbors.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, sqrt
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
    den=sqrt(sum(x*x for x in vector))*sqrt(sum(y*y for y in vector)); 
    return sorted(((d,_cos(vector,w)) for d,w in [(doc_id,vector)]+[("doc"+str(i),np.random.rand(5)) for i in range(1,k+1)] if d!=doc_id), key=lambda x:(-x[1],x[0]))[:k]

def _cos(a,b):
    den=sqrt(sum(x*x for x in a))*sqrt(sum(y*y for y in b)); 
    return 0.0 if den==0 else sum(x*y for x,y in zip(a,b))/den

class HybridEndpointCircuit():
    def __init__(self, failure_threshold: int = 3, morphology: Morphology = None, alpha: float = 0.5):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""
        self.morphology = morphology
        self.alpha = alpha

    def calculate_recovery_priority(self) -> float:
        if self.morphology:
            return recovery_priority(self.morphology)

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

def calculate_joint_score(support: int, z_score: float) -> float:
    return support * (1 + z_score)

def hybrid_fusion(morphology: Morphology, support: int, z_score: float) -> float:
    recovery_priority_score = recovery_priority(morphology)
    joint_score = calculate_joint_score(support, z_score)
    return recovery_priority_score * joint_score

def main():
    morphology = Morphology(10.0, 5.0, 2.0, 100.0)
    circuit = HybridEndpointCircuit(morphology=morphology)
    recovery_priority_score = circuit.calculate_recovery_priority()
    print("Recovery Priority Score:", recovery_priority_score)

    events = [{'t': 1643723400, 'x': 10.0, 'y': 20.0}, 
              {'t': 1643723405, 'x': 15.0, 'y': 25.0}, 
              {'t': 1643723410, 'x': 20.0, 'y': 30.0}]
    sessions = sessionize_events(events)
    print("Sessions:", sessions)

    support = 10
    z_score = 2.0
    joint_score = calculate_joint_score(support, z_score)
    print("Joint Score:", joint_score)

    hybrid_score = hybrid_fusion(morphology, support, z_score)
    print("Hybrid Fusion Score:", hybrid_score)

if __name__ == "__main__":
    main()