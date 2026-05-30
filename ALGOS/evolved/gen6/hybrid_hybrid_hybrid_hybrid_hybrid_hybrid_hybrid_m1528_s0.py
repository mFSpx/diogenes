# DARWIN HAMMER — match 1528, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m892_s0.py (gen5)
# born: 2026-05-29T23:37:06Z

"""
Module hybrid_fusion: A hybrid algorithm combining the Stylometry features from 
hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py and the Voronoi 
partition from hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m892_s0.py.
The mathematical bridge between the two structures lies in the use of Voronoi 
cell centers as input to the sphericity index calculations, enabling the 
integration of geometric and probabilistic reasoning.
"""

import math
import numpy as np
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
        "negation": set("no not never none neither cannot can't won't don't".split()),
    }
    CATEGORY_ORDER = list(FUNCTION_CATS.keys())
    NUM_CATS = len(CATEGORY_ORDER)
    
    tokens = text.lower().split()
    counts = {}
    for word in tokens:
        if word in FUNCTION_CATS['pronoun']:
            counts['pronoun'] = counts.get('pronoun', 0) + 1
        elif word in FUNCTION_CATS['article']:
            counts['article'] = counts.get('article', 0) + 1
        elif word in FUNCTION_CATS['preposition']:
            counts['preposition'] = counts.get('preposition', 0) + 1
        elif word in FUNCTION_CATS['auxiliary']:
            counts['auxiliary'] = counts.get('auxiliary', 0) + 1
        elif word in FUNCTION_CATS['conjunction']:
            counts['conjunction'] = counts.get('conjunction', 0) + 1
        elif word in FUNCTION_CATS['negation']:
            counts['negation'] = counts.get('negation', 0) + 1
        else:
            continue
    vec = np.zeros(NUM_CATS, dtype=float)
    for idx, cat in enumerate(CATEGORY_ORDER):
        if cat in counts:
            vec[idx] = counts[cat]
    total = vec.sum()
    if total > 0.0:
        vec /= total
    return vec

def sphericity_index(length: float, width: float, height: float) -> float:
    return (length + width + height) / (3 * max(length, width, height))

def hybrid_operation(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], text: str) -> np.ndarray:
    regions = assign(points, seeds)
    features = {}
    for i, region in regions.items():
        length = max(np.ptp(region, axis=0))
        width = max(np.ptp(region, axis=1))
        height = length
        features[i] = sphericity_index(length, width, height)
    text_features = stylometry_features(text)
    return np.concatenate((text_features, np.array(list(features.values()))))

def developmental_rate(temp_k: float, params: dict = field(default_factory=lambda: {'rho_25': 1.0, 'delta_h_activation': 12000.0, 't_low': 283.15, 't_high': 307.15, 'delta_h_low': -45000.0, 'delta_h_high': 65000.0, 'r_cal': 1.987})) -> float:
    numerator = params['rho_25'] * (temp_k / 298.15) * np.exp((params['delta_h_activation'] / params['r_cal']) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = np.exp((params['delta_h_low'] / params['r_cal']) * ((1.0 / params['t_low']) - (1.0 / temp_k)))
    high = np.exp((params['delta_h_high'] / params['r_cal']) * ((1.0 / params['t_high']) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

if __name__ == "__main__":
    points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    seeds = [(0.5, 0.5), (2.5, 2.5)]
    text = "This is a sample text."
    result = hybrid_operation(points, seeds, text)
    print(result)
    print(devolutional_rate(300.0))