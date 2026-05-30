# DARWIN HAMMER — match 1528, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m892_s0.py (gen5)
# born: 2026-05-29T23:37:06Z

"""
Module hybrid_fusion: A hybrid algorithm combining the Voronoi partition 
from hybrid_hybrid_voronoi_parti_hybrid_rbf_surrogate_m85_s0.py and the 
stylometry analysis from hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py.
The mathematical bridge between the two structures lies in the use of Voronoi 
cell centers as input to the stylometry analysis, enabling the integration of 
geometric and linguistic reasoning.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
import re
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

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
    "negation": set("no not never none neither cannot can't won't don't".split()),
}

CATEGORY_ORDER = list(FUNCTION_CATS.keys())
NUM_CATS = len(CATEGORY_ORDER)

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

def _tokenize(text: str) -> List[str]:
    """Very simple word tokenizer."""
    return re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())

def stylometry_features(text: str) -> np.ndarray:
    """
    Extract a normalized frequency vector over FUNCTION_CATS.
    Returns a (NUM_CATS,) float array that sums to 1 (or zeros if no matches).
    """
    tokens = _tokenize(text)
    counts = Counter(tokens)
    vec = np.zeros(NUM_CATS, dtype=float)
    for idx, cat in enumerate(CATEGORY_ORDER):
        cat_words = FUNCTION_CATS[cat]
        cat_count = sum(counts[w] for w in cat_words if w in counts)
        vec[idx] = cat_count
    total = vec.sum()
    if total > 0.0:
        vec /= total
    return vec

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

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * np.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = np.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = np.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def voronoi_stylometry(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], text: str) -> Dict[int, np.ndarray]:
    regions = assign(points, seeds)
    result = {}
    for i, region in regions.items():
        centroid = np.mean([p[0] for p in region]), np.mean([p[1] for p in region])
        result[i] = stylometry_features(text)
    return result

def stylometry_voronoi_integration(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], text: str) -> np.ndarray:
    regions = assign(points, seeds)
    result = np.zeros(NUM_CATS)
    for region in regions.values():
        centroid_text = ",".join(f"{p[0]} {p[1]}" for p in region)
        result += stylometry_features(centroid_text)
    return result / len(regions)

def hybrid_fusion(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], text: str, temp_celsius: float) -> Tuple[np.ndarray, float]:
    voronoi_result = stylometry_voronoi_integration(points, seeds, text)
    temp_k = c_to_k(temp_celsius)
    developmental_rate_result = developmental_rate(temp_k)
    return voronoi_result, developmental_rate_result

if __name__ == "__main__":
    points = [(0, 0), (1, 1), (2, 2), (3, 3)]
    seeds = [(0.5, 0.5), (2.5, 2.5)]
    text = "This is a test text."
    temp_celsius = 25.0
    voronoi_result, developmental_rate_result = hybrid_fusion(points, seeds, text, temp_celsius)
    print(voronoi_result)
    print(developmental_rate_result)