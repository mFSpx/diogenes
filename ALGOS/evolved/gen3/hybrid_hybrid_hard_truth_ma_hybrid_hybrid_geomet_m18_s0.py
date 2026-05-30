# DARWIN HAMMER — match 18, survivor 0
# gen: 3
# parent_a: hybrid_hard_truth_math_model_pool_m8_s5.py (gen1)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s4.py (gen2)
# born: 2026-05-29T23:25:07Z

"""
This module implements a novel hybrid algorithm, fusing the stylometry analysis from 
'hybrid_hard_truth_math_model_pool_m8_s5.py' with the geometric product and Voronoi 
partitioning from 'hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s4.py'. 
The mathematical bridge between these two structures lies in the representation of 
text data as geometric points, where the stylometry features are used as coordinates 
in a high-dimensional space. The Voronoi partitioning is then applied to cluster 
similar texts based on their stylometric features.
"""

import datetime as dt
import hashlib
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import math
import itertools
import random

# ----------------------------------------------------------------------
# Parent A – stylometry / LSM utilities
# ----------------------------------------------------------------------
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


def stable_hash(text: str) -> int:
    """Deterministic hash used for trigram indexing."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """Produce a 96‑dimensional stylometric fingerprint."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    features = {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }
    # Pad with zeros to reach the desired dimension
    padded_features = list(features.values())
    padded_features += [0.0] * (dim - len(padded_features))
    return np.array(padded_features)


# ----------------------------------------------------------------------
# Parent B – geometric product and Voronoi partitioning
# ----------------------------------------------------------------------
Point = Tuple[float, ...]  # n‑dimensional point


def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest_point(point: Point, seeds: List[Point]) -> int:
    """Return the index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean_distance(point, seeds[i]), i))


def voronoi_partition(seeds: List[Point], points: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the Voronoi cell of the nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest_point(p, seeds)].append(p)
    return regions


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def text_to_point(text: str) -> Point:
    """Convert a text to a geometric point using stylometry features."""
    features = stylometry_features(text)
    return tuple(features)


def cluster_texts(texts: List[str], num_seeds: int = 5) -> Dict[int, List[str]]:
    """Cluster texts based on their stylometry features using Voronoi partitioning."""
    points = [text_to_point(text) for text in texts]
    seeds = random.sample(points, num_seeds)
    partition = voronoi_partition(seeds, points)
    clusters = {i: [texts[j] for j, p in enumerate(points) if nearest_point(p, seeds) == i] for i in range(num_seeds)}
    return clusters


def stylometry_distance(text1: str, text2: str) -> float:
    """Compute the Euclidean distance between two texts based on their stylometry features."""
    point1 = text_to_point(text1)
    point2 = text_to_point(text2)
    return euclidean_distance(point1, point2)


if __name__ == "__main__":
    texts = ["This is a sample text.", "Another text for clustering.", "Texts can be clustered based on their stylometry features."]
    clusters = cluster_texts(texts)
    for i, cluster in clusters.items():
        print(f"Cluster {i}: {cluster}")
    print(f"Distance between first two texts: {stylometry_distance(texts[0], texts[1])}")