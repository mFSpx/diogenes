# DARWIN HAMMER — match 1528, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m892_s0.py (gen5)
# born: 2026-05-29T23:37:06Z

"""
Module hybrid_fusion: A hybrid algorithm combining the stylometry features 
from hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py and the 
Voronoi partition and developmental rate calculations from 
hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m892_s0.py.
The mathematical bridge between the two structures lies in the use of 
stylometry features as weights for the Voronoi cell assignments, 
enabling the integration of textual and geometric reasoning.

The core idea is to use the stylometry features to influence the 
assignment of points to Voronoi cells, and then use the developmental 
rate calculation to evaluate the cells.
"""

import math
import numpy as np
import random
import sys
from dataclasses import dataclass
from typing import Tuple, List, Dict
from collections import Counter
import re
import pathlib

# Stylometry – function word categories
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

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]], weights: List[float]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]) * weights[i], i))

def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], weights: List[float]) -> Dict[int, List[Tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds, weights)].append(p)
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

def hybrid_operation(text: str, points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    features = stylometry_features(text)
    weights = features / features.sum()
    return assign(points, seeds, weights)

def evaluate_cells(regions: Dict[int, List[Tuple[float, float]]], temp_k: float) -> Dict[int, float]:
    result = {}
    for i, region in regions.items():
        # Simple evaluation: average x-coordinate of points in region
        avg_x = sum(p[0] for p in region) / len(region)
        result[i] = developmental_rate(temp_k) * avg_x
    return result

if __name__ == "__main__":
    text = "This is a test sentence with multiple words and some pronouns."
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    seeds = [(0.0, 0.0), (10.0, 10.0)]
    regions = hybrid_operation(text, points, seeds)
    result = evaluate_cells(regions, 298.15)
    print(result)