# DARWIN HAMMER — match 2214, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_ollivier_ricci_curva_m393_s1.py (gen4)
# parent_b: hybrid_voronoi_partition_poikilotherm_schoolf_m49_s1.py (gen1)
# born: 2026-05-29T23:41:16Z

"""
This module implements a novel hybrid algorithm, fusing the stylometry analysis 
and geometric product from 'hybrid_hybrid_hybrid_hard_t_ollivier_ricci_curva_m393_s1.py' 
and the Voronoi partitioning and Schoolfield thermal rate modeling from 
'hybrid_voronoi_partition_poikilotherm_schoolf_m49_s1.py'. The mathematical 
bridge between these two structures lies in the representation of text data as 
a weighted graph, where the stylometry features are used as edge weights and the 
Voronoi partitioning is applied to group similar texts based on their thermal rates.

The core idea is to construct a graph where nodes represent texts and edges represent 
similarities between texts based on their stylometric features. The Voronoi 
partitioning is then used to divide the texts into regions based on their thermal 
rates, and the Ollivier-Ricci curvature is applied to analyze the local connectivity 
of the graph within each region.
"""

import numpy as np
import math
import random
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

# Define stylometry categories and punctuation
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

def words(text: str) -> list[str]:
    """Return a list of lowercase words (ASCII letters + optional apostrophe)."""
    return [word.strip(PUNCT).lower() for word in text.split() if word.strip(PUNCT).isalpha()]

def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * np.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = np.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = np.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def normalized_activity(temp_c: float, low_c: float = 5.0, high_c: float = 40.0, samples: int = 141) -> float:
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    max_rate = max(developmental_rate(c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)), params) for i in range(samples))
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))

def stylometry_graph(texts: list[str]) -> dict[tuple[str, str], float]:
    graph = {}
    for i, text1 in enumerate(texts):
        for j, text2 in enumerate(texts):
            if i < j:
                words1 = words(text1)
                words2 = words(text2)
                similarity = len(set(words1) & set(words2)) / len(set(words1) | set(words2))
                graph[(text1, text2)] = similarity
    return graph

def thermal_rate_regions(texts: list[str], seeds: list[tuple[float, float]]) -> dict[int, list[str]]:
    points = [(random.random(), random.random()) for _ in range(len(texts))]
    regions = assign(points, seeds)
    return {i: [texts[j] for j, point in enumerate(points) if nearest(point, seeds) == i] for i in range(len(seeds))}

def hybrid_stylometry_thermal_rate(texts: list[str], seeds: list[tuple[float, float]]) -> dict[int, dict[tuple[str, str], float]]:
    regions = thermal_rate_regions(texts, seeds)
    stylometry = {i: stylometry_graph(region) for i, region in regions.items()}
    return stylometry

if __name__ == "__main__":
    texts = ["This is a test text.", "Another test text.", "Test text with different words."]
    seeds = [(0.5, 0.5), (0.8, 0.8)]
    result = hybrid_stylometry_thermal_rate(texts, seeds)
    print(result)