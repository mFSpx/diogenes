# DARWIN HAMMER — match 3620, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_fisher_m1097_s0.py (gen4)
# parent_b: hybrid_voronoi_partition_poikilotherm_schoolf_m49_s1.py (gen1)
# born: 2026-05-29T23:51:01Z

import numpy as np
import math
import random
import sys
import pathlib

class HybridPheromoneEntry:
    """
    Represents a pheromone entry with additional properties to integrate with Voronoi partition.

    :param surface_key: Key for the surface where the pheromone is deposited.
    :param signal_kind: Kind of signal associated with the pheromone.
    :param signal_value: Value of the pheromone signal.
    :param half_life_seconds: Time it takes for the pheromone signal to decay by half.
    :param created_at: Timestamp when the pheromone was created.
    :param last_decay: Timestamp of the last decay.
    :param point: Point where the pheromone entry is located.
    """
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay", "point")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int,
                 point: Tuple[float, float]):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now
        self.point = point

    def age_seconds(self) -> float:
        """
        Calculate the age of the pheromone in seconds.

        :return: Age of the pheromone in seconds.
        """
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """
        Calculate the multiplicative decay factor since last_decay.

        :return: Decay factor.
        """
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        """
        Apply the decay factor to the pheromone signal.
        """
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)

class HybridPheromoneStore:
    """
    Singleton-like in-memory store for demo purposes, integrating with Voronoi partition.

    :param points: List of points to assign to regions.
    :param seeds: List of seeds to use for Voronoi partition.
    """
    _entries: dict[str, HybridPheromoneEntry]
    _points: List[Tuple[float, float]]
    _seeds: List[Tuple[float, float]]

    def __init__(self, points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]):
        self._entries = {}
        self._points = points
        self._seeds = seeds

    def assign_pheromone(self, surface_key: str, signal_kind: str,
                         signal_value: float, half_life_seconds: int,
                         point: Tuple[float, float]) -> HybridPheromoneEntry:
        """
        Assign a pheromone entry to the store.

        :param surface_key: Key for the surface where the pheromone is deposited.
        :param signal_kind: Kind of signal associated with the pheromone.
        :param signal_value: Value of the pheromone signal.
        :param half_life_seconds: Time it takes for the pheromone signal to decay by half.
        :param point: Point where the pheromone entry is located.
        :return: Assigned pheromone entry.
        """
        entry = HybridPheromoneEntry(surface_key, signal_kind, signal_value,
                                     half_life_seconds, point)
        self._entries[entry.uuid] = entry
        return entry

    def assign_voronoi(self, point: Tuple[float, float]) -> int:
        """
        Assign a point to a Voronoi region.

        :param point: Point to assign.
        :return: ID of the Voronoi region.
        """
        return nearest(point, self._seeds)

    def assign_hybrid(self, point: Tuple[float, float]) -> Dict[int, Dict[int, List[Tuple[float, float]]]]:
        """
        Assign a point to a hybrid region, integrating both pheromone and Voronoi partitions.

        :param point: Point to assign.
        :return: Assigned regions.
        """
        region_id = self.assign_voronoi(point)
        pheromone_entry = self.assign_pheromone("surface_key", "signal_kind", 1.0, 3600, point)
        assigned_region = assign(self._points, self._seeds)
        thermal_rates = [normalized_activity(c_to_k(temp_c), low_c=5.0, high_c=40.0, samples=141)
                         for temp_c in [10.0, 20.0, 30.0]]
        hybrid_region = {region_id: {i: [p for p in assigned_region[region_id] if thermal_rates[i] > 0.5]
                                    for i in range(len(thermal_rates))}}
        return hybrid_region

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    """
    Find the nearest seed to a given point.

    :param point: Point to find the nearest seed for.
    :param seeds: List of seeds to choose from.
    :return: ID of the nearest seed.
    """
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    :param p1: First point.
    :param p2: Second point.
    :return: Distance between the points.
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """
    Assign points to Voronoi regions.

    :param points: Points to assign.
    :param seeds: Seeds to use for Voronoi partition.
    :return: Assigned regions.
    """
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def c_to_k(celsius: float) -> float:
    """
    Convert Celsius to Kelvin.

    :param celsius: Temperature in Celsius.
    :return: Temperature in Kelvin.
    """
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Calculate the developmental rate of a system based on temperature.

    :param temp_k: Temperature in Kelvin.
    :param params: Parameters for the developmental rate calculation.
    :return: Developmental rate.
    """
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * np.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = np.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = np.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def normalized_activity(temp_c: float, low_c: float = 5.0, high_c: float = 40.0, samples: int = 141) -> float:
    """
    Normalize the activity of a system based on temperature.

    :param temp_c: Temperature in Celsius.
    :param low_c: Lower bound of the temperature range.
    :param high_c: Upper bound of the temperature range.
    :param samples: Number of samples to use for the calculation.
    :return: Normalized activity.
    """
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    max_rate = max(developmental_rate(c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)), params) for i in range(samples))
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))

def assign_hybrid_region(points: List[Tuple[float, float]], seeds1: List[Tuple[float, float]], seeds2: List[Tuple[float, float]]) -> Dict[int, Dict[int, List[Tuple[float, float]]]]:
    """
    Assign points to a hybrid region, integrating both pheromone and Voronoi partitions.

    :param points: Points to assign.
    :param seeds1: Seeds for the first Voronoi partition.
    :param seeds2: Seeds for the second Voronoi partition.
    :return: Assigned regions.
    """
    assigned_region = assign(points, seeds1)
    thermal_rates = [normalized_activity(c_to_k(temp_c), low_c=5.0, high_c=40.0, samples=141)
                     for temp_c in [10.0, 20.0, 30.0]]
    hybrid_region = {region_id: {i: [p for p in assigned_region[region_id] if thermal_rates[i] > 0.5]
                                for i in range(len(thermal_rates))}}
    return hybrid_region

if __name__ == "__main__":
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    seeds1 = [(0.0, 0.5), (1.0, 0.5), (0.5, 1.0)]
    seeds2 = [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)]
    store = HybridPheromoneStore(points, seeds1)
    region = store.assign_hybrid((0.5, 0.5))
    print(region)