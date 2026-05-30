# DARWIN HAMMER — match 3472, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_pheromone_inf_m1240_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1528_s0.py (gen6)
# born: 2026-05-29T23:50:25Z

"""Hybrid Fusion Module

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_pheromone_inf_m1240_s2.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1528_s0.py (Algorithm B)

Mathematical Bridge:
Algorithm A generates a sinusoidal weight vector `w_i = 1 + A·sin(θ_i+φ)` that is row‑stochastic.
Algorithm B partitions a 2‑D plane into Voronoi cells and extracts a normalized stylometry probability vector `p_j`.

The fusion treats each Voronoi seed as a “pheromone site”. The sinusoidal weight vector supplies
site‑specific scaling factors, while the stylometry probabilities modulate the pheromone signal
strengths. The combined signal for site *i* is

    s_i = w_i · p_i · v_i

where `v_i` is a raw pheromone value (e.g. count of points in the cell). This creates a unified
matrix‑like operation: element‑wise multiplication of three stochastic vectors, after which
entropy of the resulting distribution is evaluated.

The module provides three core hybrid functions:
1. `sinusoidal_weights` – builds the sinusoidal weight vector.
2. `voronoi_partition` – assigns points to seeds (Voronoi cells).
3. `update_pheromones_hybrid` – integrates weights, stylometry, and decay, then returns entropy.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Helper geometry (from Parent B)
# ----------------------------------------------------------------------
Vector = List[float]

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """Assign each point to the index of its nearest seed."""
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Stylometry extraction (from Parent B)
# ----------------------------------------------------------------------
def stylometry_features(text: str) -> np.ndarray:
    """Return a normalized frequency vector over predefined function word categories."""
    FUNCTION_CATS: dict[str, set[str]] = {
        "pronoun": set(
            "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
        ),
        "article": set("a an the".split()),
        "preposition": set(
            "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
        ),
        "auxiliary": set(
            "am are be been being can could did do does had has have is may might must shall should was were will would".split()
        ),
        "conjunction": set(
            "and but or nor so yet because although while if when where whereas unless until".split()
        ),
        "negation": set("no not never none neither can't won't".split()),
    }

    tokens = [t.strip(".,!?;:()[]\"'").lower() for t in text.split()]
    cat_counts = np.zeros(len(FUNCTION_CATS), dtype=float)

    for i, cat in enumerate(FUNCTION_CATS.values()):
        cat_counts[i] = sum(1 for t in tokens if t in cat)

    total = cat_counts.sum()
    if total == 0:
        return np.full_like(cat_counts, 1.0 / len(cat_counts))
    return cat_counts / total

# ----------------------------------------------------------------------
# Sinusoidal weight generation (from Parent A)
# ----------------------------------------------------------------------
def sinusoidal_weights(num: int, amplitude: float = 0.3, phase: float = 0.0) -> np.ndarray:
    """
    Generate a row‑stochastic weight vector of length `num` using a sinusoidal rotation.

    w_i = 1 + amplitude * sin(2π i / num + phase)
    The vector is shifted to be positive and finally normalized to sum to 1.
    """
    base_angles = np.linspace(0, 2 * np.pi, num, endpoint=False)
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    # Ensure positivity (sin can be -1)
    raw = raw - raw.min() + 1e-9
    return raw / raw.sum()

# ----------------------------------------------------------------------
# Hybrid pheromone system (extends Parent A)
# ----------------------------------------------------------------------
class HybridPheromoneSystem:
    """
    Stores pheromone records per Voronoi seed. Each record contains:
        - signal_kind: identifier string
        - signal_value: float (raw signal before weighting)
        - half_life_seconds: decay constant
        - created_time: timestamp of last update
    """

    def __init__(self):
        self.pheromones: Dict[int, dict] = {}

    def _decay(self, previous_value: float, half_life: float, elapsed: float) -> float:
        """Exponential half‑life decay."""
        return previous_value * math.pow(0.5, elapsed / half_life)

    def update_signal(self,
                      site_idx: int,
                      signal_kind: str,
                      raw_value: float,
                      half_life_seconds: float,
                      weight: float) -> float:
        """
        Apply decay to any existing signal, then blend the new weighted signal.
        Returns the stored (post‑decay) signal value.
        """
        now = datetime.now(timezone.utc)
        if site_idx in self.pheromones:
            rec = self.pheromones[site_idx]
            elapsed = (now - rec['created_time']).total_seconds()
            decayed = self._decay(rec['signal_value'], rec['half_life_seconds'], elapsed)
        else:
            decayed = 0.0

        # Weighted addition
        new_signal = decayed + weight * raw_value

        self.pheromones[site_idx] = {
            'signal_kind': signal_kind,
            'signal_value': new_signal,
            'half_life_seconds': half_life_seconds,
            'created_time': now
        }
        return new_signal

    def get_signal_vector(self) -> np.ndarray:
        """Return an array of current signal values ordered by site index."""
        if not self.pheromones:
            return np.array([])
        max_idx = max(self.pheromones.keys())
        vec = np.zeros(max_idx + 1, dtype=float)
        for idx, rec in self.pheromones.items():
            vec[idx] = rec['signal_value']
        return vec

    @staticmethod
    def calculate_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
        """Shannon entropy of a probability distribution."""
        probs = probabilities.astype(float)
        total = probs.sum()
        if total <= eps:
            return 0.0
        probs = probs / total
        probs = np.clip(probs, eps, 1.0)
        return -np.sum(probs * np.log2(probs))

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def voronoi_partition(points: List[Tuple[float, float]],
                      seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """Wraps `assign` for clearer API."""
    return assign(points, seeds)


def update_pheromones_hybrid(text: str,
                             points: List[Tuple[float, float]],
                             seeds: List[Tuple[float, float]],
                             half_life_seconds: float = 60.0,
                             amplitude: float = 0.4,
                             phase: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Full hybrid pipeline:
        1. Extract stylometry probabilities from `text`.
        2. Build sinusoidal weights for each seed.
        3. Partition `points` into Voronoi cells.
        4. For each cell i:
               raw_value = number of points in cell i
               weighted_signal = w_i * p_i * raw_value
               update pheromone system.
        5. Return the final signal vector and its entropy.
    """
    # 1. Stylometry
    stylometry_vec = stylometry_features(text)
    if len(stylometry_vec) != len(seeds):
        # If dimensions differ, repeat or truncate to match seed count
        repeats = math.ceil(len(seeds) / len(stylometry_vec))
        stylometry_vec = np.tile(stylometry_vec, repeats)[:len(seeds)]

    # 2. Sinusoidal weights
    weight_vec = sinusoidal_weights(len(seeds), amplitude=amplitude, phase=phase)

    # 3. Voronoi assignment
    regions = voronoi_partition(points, seeds)

    # 4. Pheromone updates
    system = HybridPheromoneSystem()
    for idx, cell_points in regions.items():
        raw_val = float(len(cell_points))
        combined_weight = weight_vec[idx] * stylometry_vec[idx]
        system.update_signal(site_idx=idx,
                             signal_kind='hybrid',
                             raw_value=raw_val,
                             half_life_seconds=half_life_seconds,
                             weight=combined_weight)

    signal_vec = system.get_signal_vector()
    entropy = system.calculate_entropy(signal_vec)
    return signal_vec, entropy


def generate_random_points(num_points: int,
                           x_range: Tuple[float, float] = (0.0, 100.0),
                           y_range: Tuple[float, float] = (0.0, 100.0)) -> List[Tuple[float, float]]:
    """Utility to generate a list of random 2‑D points."""
    return [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(num_points)]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy text for stylometry
    sample_text = ("I think that we should not forget the importance of "
                   "the article that explains how we can use the function "
                   "of conjunctions and prepositions in a sentence.")

    # Generate seeds and points
    seed_coords = generate_random_points(6, (0, 50), (0, 50))
    point_coords = generate_random_points(200, (0, 50), (0, 50))

    # Run hybrid pipeline
    signals, ent = update_pheromones_hybrid(sample_text,
                                            point_coords,
                                            seed_coords,
                                            half_life_seconds=120.0,
                                            amplitude=0.5,
                                            phase=0.3)

    print("Final pheromone signals per site:", signals)
    print("Entropy of signal distribution:", ent)