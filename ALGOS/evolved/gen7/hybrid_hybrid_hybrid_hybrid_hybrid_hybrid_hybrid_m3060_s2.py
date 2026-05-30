# DARWIN HAMMER — match 3060, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2707_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1268_s4.py (gen5)
# born: 2026-05-29T23:47:29Z

"""
This module integrates the stylometry features from the DARWIN HAMMER 
algorithm (hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s2.py) 
with the tropical linear and fractional decay operations from the 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1268_s4.py algorithm.

The mathematical bridge between these two structures is formed by using the 
tropical linear operation to compute the maximum weighted sum of the 
stylometry features, and then applying the fractional decay operation to 
model the temporal evolution of these features.

The governing equations of the stylometry features are used to compute the 
feature vectors, which are then used as inputs to the tropical linear and 
fractional decay operations.
"""

import math
import numpy as np
import random
import sys
import pathlib
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import Counter
import re

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't".split()),
}

CATEGORY_ORDER = list(FUNCTION_CATS.keys())
NUM_CATS = len(CATEGORY_ORDER)

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())

def stylometry_features(text: str) -> np.ndarray:
    tokens = _tokenize(text)
    counts = Counter(tokens)
    vec = np.zeros(NUM_CATS, dtype=float)
    for cat in FUNCTION_CATS:
        for token in FUNCTION_CATS[cat]:
            if token in counts:
                vec[CATEGORY_ORDER.index(cat)] += counts[token]
    return vec

def tropical_linear(weights: np.ndarray, features: np.ndarray) -> float:
    if weights.shape != features.shape:
        raise ValueError("weights and features must have the same shape")
    return float(np.max(weights * features))

def fractional_decay(alpha: float, t: float) -> float:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if t < 0:
        raise ValueError("time t must be non-negative")
    return (t ** (alpha - 1)) / math.gamma(alpha)

def hybrid_stylometry_tropical(weights: np.ndarray, text: str, alpha: float, t: float) -> float:
    features = stylometry_features(text)
    return tropical_linear(weights, features) * fractional_decay(alpha, t)

def compute_distance(weights: np.ndarray, text1: str, text2: str) -> float:
    features1 = stylometry_features(text1)
    features2 = stylometry_features(text2)
    return np.linalg.norm(features1 - features2) * tropical_linear(weights, np.abs(features1 - features2))

def get_closest_text(weights: np.ndarray, text: str, texts: List[str]) -> str:
    distances = [compute_distance(weights, text, t) for t in texts]
    return texts[np.argmin(distances)]

if __name__ == "__main__":
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "The dog runs quickly, but the fox is lazy."
    weights = np.random.rand(NUM_CATS)
    alpha = 2.0
    t = 1.0
    result = hybrid_stylometry_tropical(weights, text1, alpha, t)
    print(result)
    closest_text = get_closest_text(weights, text1, [text1, text2])
    print(closest_text)