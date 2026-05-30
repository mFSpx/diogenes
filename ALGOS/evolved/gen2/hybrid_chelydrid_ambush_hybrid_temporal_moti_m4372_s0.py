# DARWIN HAMMER — match 4372, survivor 0
# gen: 2
# parent_a: chelydrid_ambush.py (gen0)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s0.py (gen1)
# born: 2026-05-29T23:55:21Z

import math
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

class HybridKinematics:
    def __init__(self, m_head: float, drag_cd: float, fluid_density: float, area: float, dt: float, max_steps: int):
        self.m_head = m_head
        self.drag_cd = drag_cd
        self.fluid_density = fluid_density
        self.area = area
        self.dt = dt
        self.max_steps = max_steps

    def haversine_m(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, a)
        lat2, lon2 = map(math.radians, b)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

    def integrate_strike(self, force_series: Iterable[float]) -> tuple[float, float, float]:
        v = 0.0
        x = 0.0
        peak = 0.0
        for force in force_series:
            drag = self.drag_cd * self.fluid_density * self.area * v * abs(v) / (2.0 * self.m_head)
            acc = force / self.m_head - drag
            v = max(0.0, v + acc * self.dt)
            x += v * self.dt
            peak = max(peak, v)
        return v, x, peak

    def keep_candidate(self, candidate: Entity, selected: list[Entity]) -> bool:
        for existing in selected:
            same_kind = (candidate.address_signature or candidate.category).strip().lower() == (existing.address_signature or existing.category).strip().lower()
            if same_kind and self.haversine_m((candidate.lat, candidate.lon), (existing.lat, existing.lon)) <= self.delta_m:
                return False
        return True

    def hybrid_burst_admission_score(self, entities: Iterable[Entity], work_value: float, cost_drag: float, urgency_force: float, key: str = 'category') -> float:
        delta_m = max(self.haversine_m((entity.lat, entity.lon), (entities[0].lat, entities[0].lon)) for entity in entities)
        selected = []
        for entity in entities:
            if self.keep_candidate(entity, selected):
                selected.append(entity)
        burst_signals = self.detect_bursts(selected, key)
        distance = 0.0
        for signal in burst_signals:
            force_series = self.pulse_force(max(0.0, urgency_force), 12)
            v, x, peak = self.integrate_strike(force_series)
            distance += x
        return work_value * distance

    def pulse_force(self, peak_force: float, steps: int) -> list[float]:
        mid = (steps - 1) / 2.0
        denom = max(1.0, mid)
        return [peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)]

    def detect_bursts(self, entities: list[Entity], key: str = 'category') -> list[BurstSignal]:
        bursts = Counter()
        for entity in entities:
            bursts[entity.category] += 1
        return [BurstSignal(key=category, count=count, z_score=count / len(entities)) for category, count in bursts.items()]

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

if __name__ == "__main__":
    np.random.seed(0)
    entities = [Entity(str(i), np.random.uniform(0, 180), np.random.uniform(-180, 180), f"category_{i}") for i in range(100)]
    work_value = 10.0
    cost_drag = 0.3
    urgency_force = 1.0
    delta_m = 0.1
    dt = 0.01
    max_steps = 100
    hybrid = HybridKinematics(1.0, 0.3, 1000.0, 0.001, dt, max_steps)
    print(hybrid.hybrid_burst_admission_score(entities, work_value, cost_drag, urgency_force))