# DARWIN HAMMER — match 4372, survivor 1
# gen: 2
# parent_a: chelydrid_ambush.py (gen0)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s0.py (gen1)
# born: 2026-05-29T23:55:21Z

"""Hybrid of chelydrid_ambush.py and hybrid_temporal_motifs_possum_filter_m87_s0.py.

This module fuses the ambush-strike kinematics of chelydrid_ambush.py with the 
temporal motif mining of hybrid_temporal_motifs_possum_filter_m87_s0.py. The 
mathematical bridge between the two parents lies in the use of optimization 
techniques to maximize the ambush-strike score (from chelydrid_ambush.py) 
subject to the constraints imposed by the temporal motifs (from 
hybrid_temporal_motifs_possum_filter_m87_s0.py).

The ambush-strike score is calculated based on the velocity, distance, and 
peak velocity of the strike, which are influenced by the force series, drag, 
and other physical parameters. The temporal motifs, on the other hand, 
represent patterns of behavior that occur over time.

By combining these two concepts, we can create a hybrid system that 
optimizes the ambush-strike score with respect to the temporal motifs. This 
is achieved by using the haversine distance (from hybrid_temporal_motifs_possum_filter_m87_s0.py) 
to calculate the distance between entities, and then using this distance to 
inform the ambush-strike score calculation.

The resulting hybrid system provides a more comprehensive understanding of 
the complex interactions between entities and their behavior over time.
"""

import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

@dataclass(frozen=True)
class StrikeState:
    velocity: float
    distance: float
    peak_velocity: float

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

def integrate_strike(force_series: list[float], dt: float, m_head: float, drag_cd: float = 0.3, fluid_density: float = 1000.0, area: float = 0.001, v0: float = 0.0) -> StrikeState:
    if dt <= 0 or m_head <= 0 or drag_cd < 0 or fluid_density < 0 or area < 0:
        raise ValueError("invalid strike parameters")
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        x += v * dt
        peak = max(peak, v)
    return StrikeState(v, x, peak)

def pulse_force(peak_force: float, steps: int) -> list[float]:
    if peak_force < 0 or steps <= 0:
        raise ValueError("peak_force non-negative and steps positive required")
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)]

def optimize_strike(entities: list[Entity], peak_force: float, steps: int, dt: float, m_head: float) -> StrikeState:
    max_score = 0.0
    best_state = StrikeState(0.0, 0.0, 0.0)
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            distance = haversine_m((entities[i].lat, entities[i].lon), (entities[j].lat, entities[j].lon))
            force_series = pulse_force(peak_force, steps)
            state = integrate_strike(force_series, dt, m_head)
            score = state.distance / distance
            if score > max_score:
                max_score = score
                best_state = state
    return best_state

def detect_bursts(events: list[dict], key: str = 'category') -> list[BurstSignal]:
    counts = Counter(event[key] for event in events)
    mean = np.mean(list(counts.values()))
    std = np.std(list(counts.values()))
    bursts = []
    for key, count in counts.items():
        z_score = (count - mean) / std
        bursts.append(BurstSignal(key, count, z_score))
    return bursts

def mine_temporal_motifs(sessions: list[list[dict]], key: str = 'category', min_support: int = 2) -> list[TemporalMotif]:
    motifs = defaultdict(int)
    for session in sessions:
        for i in range(len(session)):
            for j in range(i + 1, len(session)):
                motif = tuple(event[key] for event in session[i:j+1])
                motifs[motif] += 1
    return [TemporalMotif(motif, support) for motif, support in motifs.items() if support >= min_support]

def hybrid_ambush(entities: list[Entity], events: list[dict], peak_force: float, steps: int, dt: float, m_head: float) -> StrikeState:
    bursts = detect_bursts(events)
    sessions = [[event for event in events if event['category'] == burst.key] for burst in bursts]
    motifs = mine_temporal_motifs(sessions)
    state = optimize_strike(entities, peak_force, steps, dt, m_head)
    return state

if __name__ == "__main__":
    entities = [Entity("1", 37.7749, -122.4194, "A"), Entity("2", 37.7859, -122.4364, "B")]
    events = [{"category": "A"}, {"category": "B"}, {"category": "A"}, {"category": "B"}]
    peak_force = 10.0
    steps = 10
    dt = 0.01
    m_head = 1.0
    state = hybrid_ambush(entities, events, peak_force, steps, dt, m_head)
    print(state)