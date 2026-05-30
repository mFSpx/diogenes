# DARWIN HAMMER — match 3472, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_pheromone_inf_m1240_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1528_s0.py (gen6)
# born: 2026-05-29T23:50:25Z

"""
Module hybrid_fusion: A hybrid algorithm combining the Pheromone System from 
hybrid_hybrid_hybrid_hybrid_hybrid_pheromone_inf_m1240_s2.py and the Stylometry 
features and Voronoi partition from hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1528_s0.py.
The mathematical bridge between the two structures lies in the use of Pheromone 
signals as weights for the Stylometry features, enabling the integration of 
probabilistic and geometric reasoning.

The Pheromone System calculates pheromone signals and entropy, while the 
Stylometry features extract a normalized frequency vector over FUNCTION_CATS. 
The hybrid algorithm combines these two concepts by using the Pheromone signals 
as weights for the Stylometry features and calculating the entropy of the 
resulting features.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Dict

Vector = List[float]

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def stylometry_features(text: str) -> np.ndarray:
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
        "negation": set("no not never none neither cannot".split())
    }
    text = text.lower()
    freqs = np.zeros(len(FUNCTION_CATS))
    for cat, words in FUNCTION_CATS.items():
        count = sum(1 for word in text.split() if word in words)
        freqs[list(FUNCTION_CATS.keys()).index(cat)] = count
    total = sum(freqs)
    if total > 0:
        freqs = freqs / total
    return freqs

class HybridPheromoneSystem:
    def __init__(self):
        self.pheromones = {}

    def calculate_pheromone_signal(self, surface_key, signal_kind, signal_value, half_life_seconds):
        current_time = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
        else:
            previous_signal_value = self.pheromones[surface_key]['signal_value']
            previous_half_life_seconds = self.pheromones[surface_key]['half_life_seconds']
            previous_created_time = self.pheromones[surface_key]['created_time']
            elapsed_time = (current_time - previous_created_time).total_seconds()
            decayed_signal_value = previous_signal_value * math.pow(0.5, elapsed_time / previous_half_life_seconds)
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
        return signal_value

    def calculate_entropy(self, probabilities, eps=1e-12):
        total = sum(probabilities)
        if total <= 0:
            return 0
        return -sum([p * math.log(p + eps) for p in probabilities])

def hybrid_stylometry(pheromone_system, text, surface_key, signal_kind, signal_value, half_life_seconds):
    pheromone_signal = pheromone_system.calculate_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds)
    stylometry_vec = stylometry_features(text)
    weighted_stylometry_vec = stylometry_vec * pheromone_signal
    entropy = pheromone_system.calculate_entropy(weighted_stylometry_vec)
    return weighted_stylometry_vec, entropy

def generate_points(num_points, num_seeds):
    points = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(num_points)]
    seeds = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(num_seeds)]
    return points, seeds

def hybrid_voronoi(pheromone_system, points, seeds, surface_key, signal_kind, signal_value, half_life_seconds):
    pheromone_signal = pheromone_system.calculate_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds)
    regions = assign(points, seeds)
    weighted_regions = {i: len(region) * pheromone_signal for i, region in regions.items()}
    entropy = pheromone_system.calculate_entropy(list(weighted_regions.values()))
    return weighted_regions, entropy

from datetime import datetime, timezone

if __name__ == "__main__":
    pheromone_system = HybridPheromoneSystem()
    text = "This is a test sentence with multiple words."
    surface_key = "test_surface"
    signal_kind = "test_signal"
    signal_value = 1.0
    half_life_seconds = 3600
    weighted_stylometry_vec, entropy = hybrid_stylometry(pheromone_system, text, surface_key, signal_kind, signal_value, half_life_seconds)
    print("Weighted Stylometry Vector:", weighted_stylometry_vec)
    print("Entropy:", entropy)

    num_points = 100
    num_seeds = 5
    points, seeds = generate_points(num_points, num_seeds)
    weighted_regions, entropy = hybrid_voronoi(pheromone_system, points, seeds, surface_key, signal_kind, signal_value, half_life_seconds)
    print("Weighted Regions:", weighted_regions)
    print("Entropy:", entropy)