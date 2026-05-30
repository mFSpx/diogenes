# DARWIN HAMMER — match 1243, survivor 7
# gen: 5
# parent_a: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s1.py (gen4)
# born: 2026-05-29T23:34:51Z

import sys
import math
import random
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

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


class NLMSLearner:
    def __init__(self, mu: float = 0.5, eps: float = 1e-6):
        self.mu = mu
        self.eps = eps
        self.w = None
        self.bias = 0.0

    def update(self, x: np.ndarray, I: np.ndarray, s: float, desired: float) -> Tuple[np.ndarray, float]:
        z = np.concatenate([x, I, np.array([s], dtype=float)])
        if self.w is None:
            self.w = np.zeros_like(z)
        y = np.dot(self.w, z) + self.bias
        e = desired - y
        norm_sq = np.dot(z, z)
        step = self.mu / (norm_sq + self.eps)
        self.w += step * e * z
        self.bias += step * e
        return self.w.copy(), e


def decay_surface_pheromones(surface_key: str) -> np.ndarray:
    rows = PheromoneStore.decay_surface(surface_key)
    rows.sort(key=lambda r: r["pheromone_uuid"])
    values = np.array([r["signal_value_after"] for r in rows], dtype=float)
    return values


def hybrid_step(surface_key: str,
                x: np.ndarray,
                desired_output: float,
                morphology: Morphology,
                breaker: EndpointCircuitBreaker,
                diffusion_schedule: List[float],
                nlms_learner: NLMSLearner) -> Dict:
    I_raw = decay_surface_pheromones(surface_key)
    alpha = diffusion_schedule[0] if diffusion_schedule else 0.0
    I = np.sqrt(alpha) * I_raw + np.sqrt(1 - alpha) * np.random.normal(size=I_raw.shape)
    s = morphology.priority()
    if breaker.allow():
        w, e = nlms_learner.update(x, I, s, desired_output)
        breaker.record_success() if abs(e) < 1e-3 else breaker.record_failure()
    else:
        w = nlms_learner.w
        e = np.nan
    return {"w": w, "e": e, "breaker_open": breaker.open}


def main():
    surface_key = "surface1"
    x = np.array([1.0, 2.0, 3.0])
    desired_output = 10.0
    morphology = Morphology(1.0, 2.0, 3.0, 10.0)
    breaker = EndpointCircuitBreaker()
    diffusion_schedule = [0.5]
    nlms_learner = NLMSLearner()

    pheromone = PheromoneEntry(surface_key, "kind1", 10.0, 10)
    PheromoneStore.add(pheromone)

    result = hybrid_step(surface_key, x, desired_output, morphology, breaker, diffusion_schedule, nlms_learner)
    print(result)


if __name__ == "__main__":
    main()