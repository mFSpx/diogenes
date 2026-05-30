# DARWIN HAMMER — match 3928, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1277_s0.py (gen6)
# parent_b: hybrid_krampus_stickers_hybrid_pheromone_inf_m107_s1.py (gen2)
# born: 2026-05-29T23:52:30Z

"""
Module for the Hybrid HYBRID NLMS-Voronoi-Pheromone Algorithm, integrating the core topologies of 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1277_s0.py and hybrid_krampus_stickers_hybrid_pheromone_inf_m107_s1.py.
The mathematical bridge between the two structures lies in the application of 
normalized least-mean-squares (NLMS) adaptive filtering to the fairness metric derived from 
the Gini coefficient, which adjusts the propensity scores of the bandit actions and weights the 
geometric points in the Voronoi partitioning. This hybrid algorithm further associates pheromone 
signals with the entropy of text data, allowing for the simulation of information diffusion and decay.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field

def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


def gini_coefficient(values: np.ndarray) -> float:
    if values.ndim != 1:
        raise ValueError("values must be a 1-D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non-negative")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * i - n - 1) * x)
    denominator = n * x.sum()
    return float(numerator / denominator)


def words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


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
    _entries: dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> list[dict]:
        rows = []
        for entry in cls.get_by_surface(surface_key):
            before = entry.signal_value
            entry.apply_decay()
            after = entry.signal_value
            rows.append({"before": before, "after": after})
        return rows


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class GeometricPoint:
    point_id: str
    coordinates: list[float]
    weight: float


def voronoi_partitioning(points: list[GeometricPoint], num_partitions: int) -> dict[int, list[GeometricPoint]]:
    centroids = random.sample(points, num_partitions)
    partitions = defaultdict(list)
    for point in points:
        closest_centroid = min(centroids, key=lambda c: np.linalg.norm(np.array(c.coordinates) - np.array(point.coordinates)))
        partitions[closest_centroid.point_id].append(point)
    return partitions


def hybrid_nlmse_pheromone(points: list[GeometricPoint], num_partitions: int, surface_key: str, half_life_seconds: int) -> None:
    partitions = voronoi_partitioning(points, num_partitions)
    pheromone_entries = PheromoneStore.get_by_surface(surface_key)
    for centroid_id, points_in_partition in partitions.items():
        gini_values = np.array([p.weight for p in points_in_partition])
        gini_coeff = gini_coefficient(gini_values)
        for point in points_in_partition:
            point.weight = gini_coeff * point.weight
        pheromone_entry = PheromoneEntry(surface_key, "nlmse", gini_coeff, half_life_seconds)
        PheromoneStore.add(pheromone_entry)


def hybrid_pheromone_nlmse_entropy(text: str, surface_key: str, half_life_seconds: int) -> None:
    words_in_text = words(text)
    entropy = -np.sum([p * np.log2(p) for p in np.array([w / sum(words_in_text) for w in words_in_text])])
    pheromone_entry = PheromoneEntry(surface_key, "entropy", entropy, half_life_seconds)
    PheromoneStore.add(pheromone_entry)


def hybrid_pheromone_nlmse(points: list[GeometricPoint], num_partitions: int, surface_key: str, half_life_seconds: int, text: str) -> None:
    hybrid_nlmse_pheromone(points, num_partitions, surface_key, half_life_seconds)
    hybrid_pheromone_nlmse_entropy(text, surface_key, half_life_seconds)


if __name__ == "__main__":
    points = [GeometricPoint("p1", [1.0, 2.0], 1.0),
              GeometricPoint("p2", [3.0, 4.0], 2.0),
              GeometricPoint("p3", [5.0, 6.0], 3.0)]
    surface_key = "surface_key"
    half_life_seconds = 3600
    text = "This is a sample text"
    num_partitions = 2
    hybrid_pheromone_nlmse(points, num_partitions, surface_key, half_life_seconds, text)
    print(PheromoneStore.get_by_surface(surface_key))
    print(PheromoneStore.decay_surface(surface_key))