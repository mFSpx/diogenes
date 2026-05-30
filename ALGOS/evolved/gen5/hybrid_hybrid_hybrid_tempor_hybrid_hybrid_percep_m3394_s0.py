# DARWIN HAMMER — match 3394, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_temporal_moti_hybrid_doomsday_cale_m218_s1.py (gen2)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_ternar_m1907_s5.py (gen4)
# born: 2026-05-29T23:49:44Z

"""
Hybrid Temporal Motif & Perceptual-RBF Voronoi-Ternary Router Fusion

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – hybrid_hybrid_temporal_moti_hybrid_doomsday_cale_m218_s1.py 
  (Hybrid Temporal Motif & Weekday Gini Fusion)
* **Parent B** – hybrid_hybrid_perceptual_de_hybrid_hybrid_ternar_m1907_s5.py 
  (Hybrid Perceptual-RBF Voronoi-Ternary Router)

The mathematical bridge is built by injecting the temporal motif's 
weekday distribution into the RBF surrogate of the perceptual hashing 
component. The motif's support and Gini coefficient are used to 
modulate the failure probability estimate of the RBF surrogate.

The fusion thus creates a unified system that simultaneously respects 
spatial proximity, temporal ordering, weekday inequality, and 
perceptual hashing.

The module implements three public functions that showcase this hybrid 
behaviour: `sessionize_events`, `hybrid_motif_gini_score`, and 
`hybrid_route`.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – spatial‑temporal utilities
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
    lat1, lon1 = a
    lat2, lon2 = b
    earth_radius = 6371  # kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius * c

def doomsday_numpy(date: dt.datetime) -> int:
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    year = date.year
    month = date.month
    day = date.day
    if month < 3:
        year -= 1
        month += 10
    return (year + int(year/4) - int(year/100) + int(year/400) + t[month-1] + day) % 7

def gini_coefficient_numpy(vector: np.ndarray) -> float:
    vector = np.sort(vector)
    index = np.arange(1, vector.size+1)
    n = vector.size
    return ((np.sum((2 * index - n  - 1) * vector)) / (n * np.sum(vector)))

# ----------------------------------------------------------------------
# Parent B – perceptual hashing utilities
# ----------------------------------------------------------------------

def compute_dhash(values: List[float]) -> int:
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    avg = sum(values) / len(values)
    bits = 0
    for v in values:
        bits = (bits << 1) | int(v >= avg)
    return bits

def rbf_surrogate(x: np.ndarray, center: np.ndarray, sigma: float) -> float:
    return math.exp(-((x - center) ** 2).sum() / (2 * sigma ** 2))

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------

def sessionize_events(entities: Iterable[Entity]) -> Dict[str, List[Entity]]:
    sessions = {}
    for entity in entities:
        if entity.id not in sessions:
            sessions[entity.id] = []
        sessions[entity.id].append(entity)
    return sessions

def hybrid_motif_gini_score(sessions: Dict[str, List[Entity]]) -> float:
    weekday_counts = np.zeros(7)
    for session in sessions.values():
        for entity in session:
            weekday_counts[doomsday_numpy(entity.timestamp)] += 1
    gini = gini_coefficient_numpy(weekday_counts)
    support = len(sessions)
    return support * (1 - gini)

def hybrid_route(points: List[Tuple[float, float]], 
                   hash_values: List[List[float]], 
                   lambda_: float, 
                   mu: float) -> List[Tuple[float, float]]:
    seeds = []
    for hash_value in hash_values:
        seed = np.mean([point for point, hv in zip(points, hash_values) if compute_dhash(hv) == compute_dhash(hash_value)], axis=0)
        seeds.append(seed)
    costs = np.zeros((len(points), len(seeds)))
    for i, point in enumerate(points):
        for j, seed in enumerate(seeds):
            costs[i, j] = lambda_ * haversine_m(point, seed) + mu * (1 - rbf_surrogate(np.array(point), np.array(seed), 1.0))
    route = []
    current_point = points[0]
    for _ in range(len(points)):
        min_cost = float('inf')
        next_seed = None
        for i, seed in enumerate(seeds):
            cost = costs[points.index(current_point), i]
            if cost < min_cost:
                min_cost = cost
                next_seed = seed
        route.append(next_seed)
        current_point = next_seed
    return route

if __name__ == "__main__":
    entities = [Entity("1", 37.7749, -122.4194, "A"), 
                Entity("1", 37.7859, -122.4364, "B"), 
                Entity("2", 37.7963, -122.4574, "A")]
    sessions = sessionize_events(entities)
    score = hybrid_motif_gini_score(sessions)
    print(score)
    points = [(37.7749, -122.4194), (37.7859, -122.4364), (37.7963, -122.4574)]
    hash_values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    route = hybrid_route(points, hash_values, 1.0, 1.0)
    print(route)