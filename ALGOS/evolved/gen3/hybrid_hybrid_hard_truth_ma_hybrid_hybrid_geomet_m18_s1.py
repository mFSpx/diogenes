# DARWIN HAMMER — match 18, survivor 1
# gen: 3
# parent_a: hybrid_hard_truth_math_model_pool_m8_s5.py (gen1)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s4.py (gen2)
# born: 2026-05-29T23:25:07Z

"""
This module defines a hybrid algorithm, named hybrid_darwin_hammer, 
which combines the stylometry analysis from 'hybrid_hard_truth_math_model_pool_m8_s5.py' 
and the geometric product and Voronoi partitioning from 'hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s4.py'. 
The mathematical bridge between the two parent algorithms is the use of vectors and points, 
which are utilized in both stylometry analysis and geometric product calculations. 
The hybrid algorithm integrates these concepts to create a novel system for analyzing and clustering textual data.

Parent A: hybrid_hard_truth_math_model_pool_m8_s5.py
Parent B: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s4.py
"""

import datetime as dt
import hashlib
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import math
import itertools
import random

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
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Dict[str, float]:
    """Return a normalized frequency vector for each FUNCTION_CATS."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """Produce a 96‑dimensional stylometric fingerprint."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    features = np.zeros(dim)
    for i, (cat, vocab) in enumerate(FUNCTION_CATS.items()):
        features[i] = sum(cnt[w] for w in vocab) / total
    return features


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Standard Euclidean distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest_point(point: List[float], seeds: List[List[float]]) -> int:
    """Return the index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean_distance(point, seeds[i]), i))


def voronoi_partition(seeds: List[List[float]], points: List[List[float]]) -> Dict[int, List[List[float]]]:
    """Assign each point to the Voronoi cell of the nearest seed."""
    regions: Dict[int, List[List[float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest_point(p, seeds)].append(p)
    return regions


def text_to_vector(text: str) -> List[float]:
    """Convert text to a vector using stylometry features."""
    return stylometry_features(text).tolist()


def hybrid_darwin_hammer(texts: List[str], num_seeds: int) -> Dict[int, List[str]]:
    """Hybrid algorithm combining stylometry analysis and Voronoi partitioning."""
    vectors = [text_to_vector(text) for text in texts]
    seeds = random.sample(vectors, num_seeds)
    partition = voronoi_partition(seeds, vectors)
    result = {i: [] for i in range(num_seeds)}
    for i, vector in enumerate(vectors):
        result[nearest_point(vector, seeds)].append(texts[i])
    return result


def test_hybrid_darwin_hammer():
    texts = ["This is a test sentence.", "Another test sentence.", "A different sentence."]
    num_seeds = 2
    result = hybrid_darwin_hammer(texts, num_seeds)
    print(result)


if __name__ == "__main__":
    test_hybrid_darwin_hammer()