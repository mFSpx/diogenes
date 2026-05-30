# DARWIN HAMMER — match 4565, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_semant_m431_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2010_s3.py (gen4)
# born: 2026-05-29T23:56:32Z

"""
This module combines the hybrid geometric temporal motif fusion from 
hybrid_hybrid_geometric_pro_hybrid_hybrid_semant_m431_s0 and the 
bandit update mechanism with epistemic certainty flags from 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2010_s3. The 
mathematical bridge is formed by incorporating the epistemic certainty 
flags into the geometric distance calculation in the Voronoi 
partitioning, and using the TemporalMotif support as a weight for the 
bandit update mechanism. The Morphology is used to define the shape and 
size of the Voronoi regions, and the bandit update is used to adapt the 
propensity scores based on both physical distances and epistemic certainty.
"""

import math
import random
import sys
import pathlib
from typing import Tuple, List, Dict
import numpy as np
from dataclasses import dataclass, field

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate pair
                del lst[j:j + 2]
                n -= 2
                sign *= 1  # e_i*e_i = 1 contributes +1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int


@dataclass(frozen=True)
class HybridMotif:
    """Entity representing a spatio-temporal motif with morphology."""
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    morphology: Morphology
    vector: Tuple[float, ...]          # semantic feature vector
    score: float        
    certainty: float


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""

    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Encapsulates the honeybee‑style store and its derived control signal."""

    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """
        Apply the store equation and recompute the dance duration.

        Returns
        -------
        new_level, delta
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ (computed lazily)."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float], certainty: float) -> float:
    """Calculate the Euclidean distance between two points, modified by the epistemic certainty."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) * certainty


def update_bandit(action: BanditAction, update: BanditUpdate, motif: HybridMotif) -> BanditAction:
    """Update the bandit action based on the TemporalMotif support and epistemic certainty."""
    new_propensity = action.propensity + update.reward * motif.support * motif.certainty
    return BanditAction(action.action_id, new_propensity, action.expected_reward, action.confidence_bound, action.algorithm)


def calculate_voronoi(motifs: List[HybridMotif]) -> Dict[Tuple[float, float], List[HybridMotif]]:
    """Calculate the Voronoi partitioning based on the HybridMotif morphologies and epistemic certainty."""
    voronoi = {}
    for motif in motifs:
        closest_points = []
        for other_motif in motifs:
            if motif != other_motif:
                distance = calculate_distance((motif.centroid_lat, motif.centroid_lon), (other_motif.centroid_lat, other_motif.centroid_lon), motif.certainty)
                closest_points.append((other_motif, distance))
        closest_points.sort(key=lambda x: x[1])
        voronoi[(motif.centroid_lat, motif.centroid_lon)] = [x[0] for x in closest_points]
    return voronoi


if __name__ == "__main__":
    motif1 = HybridMotif(("A", "B"), 10, 0.0, 0.0, Morphology(1.0, 1.0, 1.0, 1.0), (1.0, 2.0), 0.5, 0.8)
    motif2 = HybridMotif(("C", "D"), 20, 1.0, 1.0, Morphology(2.0, 2.0, 2.0, 2.0), (3.0, 4.0), 0.7, 0.9)
    voronoi = calculate_voronoi([motif1, motif2])
    print(voronoi)

    action = BanditAction("action1", 0.5, 10.0, 5.0, "algorithm1")
    update = BanditUpdate("context1", "action1", 5.0, 0.5)
    new_action = update_bandit(action, update, motif1)
    print(new_action)

    store = StoreState()
    inflow = [1.0, 2.0]
    outflow = [0.5, 1.0]
    new_level, delta = store.update(inflow, outflow)
    print(new_level, delta)