# DARWIN HAMMER — match 1243, survivor 6
# gen: 5
# parent_a: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s1.py (gen4)
# born: 2026-05-29T23:34:51Z

import sys
import math
import random
import pathlib
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import numpy as np

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
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> List[Dict]:
        rows = []
        for entry in cls.get_by_surface(surface_key):
            before = entry.signal_value
            entry.apply_decay()
            rows.append({
                "pheromone_uuid": entry.uuid,
                "surface_key": entry.surface_key,
                "signal_kind": entry.signal_kind,
                "signal_value_before": before,
                "signal_value_after": entry.signal_value,
                "age_seconds": entry.age_seconds(),
            })
        return rows


class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc)

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc)

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc)

    def allow(self) -> bool:
        return not self.open


class Morphology:
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

    def priority(self) -> float:
        volume = self.length * self.width * self.height
        return math.tanh(volume / (self.mass + 1e-6))


_NLMS_W = None
_NLMS_BIAS = 0.0
_MU = 0.5
_EPS = 1e-6


def decay_surface_pheromones(surface_key: str) -> np.ndarray:
    rows = PheromoneStore.decay_surface(surface_key)
    rows.sort(key=lambda r: r["pheromone_uuid"])
    values = np.array([r["signal_value_after"] for r in rows], dtype=float)
    return values


def nlms_weight_update(x: np.ndarray, I: np.ndarray, s: float,
                      desired: float) -> Tuple[np.ndarray, float]:
    global _NLMS_W, _NLMS_BIAS

    z = np.concatenate([x, I, np.array([s], dtype=float)])

    if _NLMS_W is None:
        _NLMS_W = np.zeros_like(z)

    y = np.dot(_NLMS_W, z) + _NLMS_BIAS
    e = desired - y

    norm_sq = np.dot(z, z)
    step = _MU / (norm_sq + _EPS)
    _NLMS_W += step * e * z
    _NLMS_BIAS += step * e  

    return _NLMS_W.copy(), e


def hybrid_step(surface_key: str,
                x: np.ndarray,
                desired_output: float,
                morphology: Morphology,
                breaker: EndpointCircuitBreaker,
                diffusion_schedule: List[float],
                t: int) -> Dict:
    I_raw = decay_surface_pheromones(surface_key)

    bar_alpha = diffusion_schedule[min(t, len(diffusion_schedule) - 1)]
    I = np.sqrt(bar_alpha) * I_raw + np.sqrt(1 - bar_alpha) * np.random.normal(size=I_raw.shape)

    s = morphology.priority()
    if breaker.allow():
        w, e = nlms_weight_update(x, I, s, desired_output)
        if abs(e) > 1.0:
            breaker.record_failure()
        else:
            breaker.record_success()
    else:
        w = _NLMS_W.copy() if _NLMS_W is not None else None
        e = np.nan

    return {
        "surface_key": surface_key,
        "weight_vector": w,
        "error": e,
        "circuit_breaker_open": breaker.open,
    }


def main():
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    breaker = EndpointCircuitBreaker()
    diffusion_schedule = [0.9 ** i for i in range(100)]

    surface_key = "test_surface"
    PheromoneStore.add(PheromoneEntry(surface_key, "test", 1.0, 10))

    for t in range(100):
        x = np.random.normal(size=5)
        desired_output = np.sin(t)
        output = hybrid_step(surface_key, x, desired_output, morphology, breaker, diffusion_schedule, t)
        print(output)


if __name__ == "__main__":
    main()