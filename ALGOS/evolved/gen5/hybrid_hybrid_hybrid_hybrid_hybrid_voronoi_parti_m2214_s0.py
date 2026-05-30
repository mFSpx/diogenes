# DARWIN HAMMER — match 2214, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_ollivier_ricci_curva_m393_s1.py (gen4)
# parent_b: hybrid_voronoi_partition_poikilotherm_schoolf_m49_s1.py (gen1)
# born: 2026-05-29T23:41:16Z

"""
This module implements a novel hybrid algorithm, fusing the stylometry analysis and Voronoi partition from 
'hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s0.py' and 'poikilotherm_schoolfield.py'. The 
mathematical bridge between these two structures lies in the representation of text data as a weighted 
graph, where the stylometry features are used as edge weights and a Voronoi partition is applied to 
analyze the local connectivity of the graph. The Voronoi partition is used to cluster the graph into 
regions, and the stylometry analysis is used to calculate the edge weights between nodes in each region.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from collections import Counter, OrderedDict
from dataclasses import dataclass

# Define stylometry categories and punctuation
FUNCTION_CATS: Dict[str, set[str]] = {
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
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def words(text: str) -> List[str]:
    """Return a list of lowercase words (ASCII letters + optional apostrophe)."""
    return text.lower().replace('-', ' ').replace('/', ' ').replace('\\', ' ').split()

def stylometry_features(text: str) -> Dict[str, int]:
    """Return a dictionary of stylometry features for the given text."""
    words_list = words(text)
    features = Counter()
    for word in words_list:
        for cat in FUNCTION_CATS:
            if word in FUNCTION_CATS[cat]:
                features[cat] += 1
    return dict(features)

def voronoi_partition(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * np.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = np.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = np.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def normalized_activity(temp_c: float, low_c: float = 5.0, high_c: float = 40.0, samples: int = 141) -> float:
    params = SchoolfieldParams(t_low=math.pow(10, c_to_k(low_c)), t_high=math.pow(10, c_to_k(high_c)))
    rate = developmental_rate(math.pow(10, c_to_k(temp_c)), params)
    max_rate = max(developmental_rate(math.pow(10, c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1))), params) for i in range(samples))
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

def assign_hybrid_region(points: List[Tuple[float, float]], seeds1: List[Tuple[float, float]], seeds2: List[Tuple[float, float]]) -> Dict[int, Dict[int, List[Tuple[float, float]]]]:
    assigned_region = voronoi_partition(points, seeds1)
    thermal_rates = [normalized_activity(temp_c) for temp_c in [c_to_k(temp) for temp in [point[0] for point in points]]]
    for i, point in enumerate(points):
        assigned_region[nearest(point, seeds2)][i].append((point, thermal_rates[i]))
    return assigned_region

def smoke_test():
    points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    seeds1 = [(0.5, 0.5), (2.5, 2.5)]
    seeds2 = [(1.0, 1.0), (3.0, 3.0)]
    assigned_region = assign_hybrid_region(points, seeds1, seeds2)
    print(assigned_region)

if __name__ == "__main__":
    smoke_test()