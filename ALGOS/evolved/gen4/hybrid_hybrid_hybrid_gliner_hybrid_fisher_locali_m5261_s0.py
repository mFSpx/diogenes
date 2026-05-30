# DARWIN HAMMER — match 5261, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_gliner_zero_s_hybrid_krampus_brain_m30_s3.py (gen3)
# parent_b: hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s2.py (gen3)
# born: 2026-05-30T00:00:50Z

import numpy as np
import math
import random
import sys
import pathlib
from datetime import datetime, timezone

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
        self.uuid = str(random.getrandbits(128))
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
        """Return the multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Singleton-like in-memory store for demo purposes."""
    _entries: dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list[PheromoneEntry]:
        return list(cls._entries.values())


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def hybrid_entropy(angle: float, center: float, width: float, pheromone_entry: PheromoneEntry) -> float:
    pheromone_weight = pheromone_entry.signal_value
    fisher_feature = fisher_score(angle, center, width)
    return -pheromone_weight * fisher_feature * math.log(fisher_feature, 2)

def hybrid_fusion(candidates: list[float], center: float, width: float, pheromone_entry: PheromoneEntry) -> np.ndarray:
    angles = np.array(candidates)
    entropies = np.array([hybrid_entropy(angle, center, width, pheromone_entry) for angle in angles])
    lead_lag_path = np.column_stack((angles, entropies))
    return lead_lag_transform(lead_lag_path)

def best_angle(candidates: list[float], center: float, width: float, pheromone_entry: PheromoneEntry) -> float:
    if not candidates:
        raise ValueError('candidates required')
    return max(candidates, key=lambda t: (hybrid_entropy(t, center, width, pheromone_entry), -abs(t-center)))

if __name__ == "__main__":
    # Smoke test
    pheromone_entry = PheromoneEntry("test_surface_key", "test_signal_kind", 1.0, 10)
    candidates = [0.0, 1.0, 2.0]
    center = 1.0
    width = 1.0
    result = hybrid_fusion(candidates, center, width, pheromone_entry)
    print(result)