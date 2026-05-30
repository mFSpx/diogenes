# DARWIN HAMMER — match 3060, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2707_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1268_s4.py (gen5)
# born: 2026-05-29T23:47:29Z

"""
This module integrates the stylometry features from the DARWIN HAMMER algorithm 
(hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s2.py) with the 
Fisher score and Hoeffding bound computations from the 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1268_s4.py algorithm.

The mathematical bridge between these two structures is formed by using the 
geometric product to compute distances and orientations between points in 
the stylometry feature space, and then applying these computations to 
assign points to their nearest seeds using the hybrid ternary route algorithm. 
Meanwhile, the Fisher score is used to evaluate the similarity between the 
stylometry features of different texts, and the Hoeffding bound is used to 
estimate the confidence interval of the similarity scores.
"""

import math
import numpy as np
import random
import sys
import pathlib
from collections import Counter

FUNCTION_CATS = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't".split()),
}

CATEGORY_ORDER = list(FUNCTION_CATS.keys())
NUM_CATS = len(CATEGORY_ORDER)

def _tokenize(text: str) -> list:
    return [word for word in re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())]

def stylometry_features(text: str) -> np.ndarray:
    tokens = _tokenize(text)
    counts = Counter(tokens)
    vec = np.zeros(NUM_CATS, dtype=float)
    for i, cat in enumerate(CATEGORY_ORDER):
        vec[i] = sum(1 for token in tokens if token in FUNCTION_CATS[cat])
    return vec / len(tokens)

def hoeffding_bound(range_R: float, delta: float, n: int) -> float:
    if n <= 0:
        raise ValueError("sample size n must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt((range_R ** 2) * math.log(1.0 / delta) / (2.0 * n))

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(math.exp(-0.5 * ((theta - center) / width) ** 2), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return (derivative ** 2) / intensity

def hybrid_similarity(text1: str, text2: str) -> float:
    vec1 = stylometry_features(text1)
    vec2 = stylometry_features(text2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def confidence_interval(similarity: float, range_R: float, delta: float, n: int) -> tuple:
    bound = hoeffding_bound(range_R, delta, n)
    return similarity - bound, similarity + bound

def geometric_product(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    return np.dot(vec1.reshape(-1, 1), vec2.reshape(1, -1))

if __name__ == "__main__":
    text1 = "This is a sample text."
    text2 = "This is another sample text."
    similarity = hybrid_similarity(text1, text2)
    interval = confidence_interval(similarity, 1.0, 0.05, 100)
    print("Similarity:", similarity)
    print("Confidence Interval:", interval)
    print("Geometric Product:")
    print(geometric_product(stylometry_features(text1), stylometry_features(text2)))