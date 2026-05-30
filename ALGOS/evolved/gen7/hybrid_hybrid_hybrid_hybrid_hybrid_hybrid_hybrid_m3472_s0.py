# DARWIN HAMMER — match 3472, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_pheromone_inf_m1240_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1528_s0.py (gen6)
# born: 2026-05-29T23:50:25Z

"""
Module hybrid_fusion: A hybrid algorithm combining the Stylometry features from 
hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py and the Pheromone 
System from hybrid_pheromone_infotaxis_m3_s3.py. The mathematical bridge between 
the two structures lies in the use of Voronoi cell centers as input to the 
pheromone signal calculations, enabling the integration of geometric and 
probabilistic reasoning.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

class HybridPheromoneSystem:
    def __init__(self):
        self.pheromones = {}
        self.voronoi_regions = {}

    def calculate_pheromone_signal(self, surface_key, signal_kind, signal_value, half_life_seconds, voronoi_center):
        current_time = datetime.now()
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time, 'voronoi_center': voronoi_center}
        else:
            previous_signal_value = self.pheromones[surface_key]['signal_value']
            previous_half_life_seconds = self.pheromones[surface_key]['half_life_seconds']
            previous_created_time = self.pheromones[surface_key]['created_time']
            previous_voronoi_center = self.pheromones[surface_key]['voronoi_center']
            elapsed_time = (current_time - previous_created_time).total_seconds()
            decayed_signal_value = previous_signal_value * math.pow(0.5, elapsed_time / previous_half_life_seconds)
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time, 'voronoi_center': voronoi_center}
        return signal_value

    def calculate_entropy(self, probabilities, eps=1e-12):
        total = sum(probabilities)
        if total <= 0:
            return np.inf
        return -np.sum([p / total * np.log(p / total) for p in probabilities])

    def stylometry_features(self, text: str) -> np.ndarray:
        """
        Extract a normalized frequency vector over FUNCTION_CATS.
        Returns a (NUM_CATS,) float array that sums to 1 (or zeros if no matches).
        """
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
            "negation": set("no not never none neither can't cant couldn't".split())
        }
        words = text.split()
        features = {}
        for word in words:
            for category, cat_words in FUNCTION_CATS.items():
                if word in cat_words:
                    features[category] = features.get(category, 0) + 1
        total = sum(features.values())
        if total > 0:
            features = {k: v / total for k, v in features.items()}
        return np.array(list(features.values()))

def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def calculate_voronoi_center(regions: Dict[int, List[Tuple[float, float]]]) -> Tuple[float, float]:
    x_coords = []
    y_coords = []
    for points in regions.values():
        for point in points:
            x_coords.append(point[0])
            y_coords.append(point[1])
    return np.mean(x_coords), np.mean(y_coords)

def hybrid_function(surface_key, signal_kind, signal_value, half_life_seconds, points, seeds):
    voronoi_regions = assign(points, seeds)
    voronoi_center = calculate_voronoi_center(voronoi_regions)
    pheromone_signal = HybridPheromoneSystem().calculate_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds, voronoi_center)
    stylometry_features_vector = HybridPheromoneSystem().stylometry_features("This is a test sentence.")
    return pheromone_signal, stylometry_features_vector

if __name__ == "__main__":
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    seeds = [(0.0, 0.0), (10.0, 10.0)]
    surface_key = "test_key"
    signal_kind = "test_kind"
    signal_value = 1.0
    half_life_seconds = 3600
    pheromone_signal, stylometry_features_vector = hybrid_function(surface_key, signal_kind, signal_value, half_life_seconds, points, seeds)
    print("Pheromone Signal:", pheromone_signal)
    print("Stylometry Features Vector:", stylometry_features_vector)