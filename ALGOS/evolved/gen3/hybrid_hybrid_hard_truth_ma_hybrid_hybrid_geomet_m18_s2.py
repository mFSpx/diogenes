# DARWIN HAMMER — match 18, survivor 2
# gen: 3
# parent_a: hybrid_hard_truth_math_model_pool_m8_s5.py (gen1)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s4.py (gen2)
# born: 2026-05-29T23:25:07Z

"""
Hybrid Algorithm: 
This module represents a novel fusion of two mathematical algorithms: 
- hybrid_hard_truth_math_model_pool_m8_s5.py (Parent A), a stylometry and LSM utility
- hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s4.py (Parent B), a geometric product and Voronoi partition algorithm

The mathematical bridge between these two structures is the application of Voronoi partitions to stylometric fingerprints, 
enabling the clustering of texts based on their stylistic features. 
This fusion integrates the governing equations of both parents, creating a unified system for text analysis and geometric modeling.
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
    vector = np.array([
        sum(cnt[w] for w in vocab) / total
        for vocab in FUNCTION_CATS.values()
    ])
    # Pad the vector with zeros to match the desired dimension
    padded_vector = np.pad(vector, (0, dim - len(vector)))
    return padded_vector


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest_point(point: np.ndarray, seeds: List[np.ndarray]) -> int:
    """Return the index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean_distance(point, seeds[i]), i))


def voronoi_partition(seeds: List[np.ndarray], points: List[np.ndarray]) -> Dict[int, List[np.ndarray]]:
    """Assign each point to the Voronoi cell of the nearest seed."""
    regions: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest_point(p, seeds)].append(p)
    return regions


def cluster_texts(texts: List[str], num_seeds: int) -> Dict[int, List[str]]:
    """Cluster texts based on their stylometric features."""
    features = [stylometry_features(text) for text in texts]
    seeds = random.sample(features, num_seeds)
    regions = voronoi_partition(seeds, features)
    clustered_texts = {}
    for i, region in regions.items():
        clustered_texts[i] = [texts[j] for j, feature in enumerate(features) if feature in region]
    return clustered_texts


def analyze_text_clusters(texts: List[str], num_seeds: int) -> Dict[int, Dict[str, float]]:
    """Analyze the stylometric features of each text cluster."""
    clustered_texts = cluster_texts(texts, num_seeds)
    analysis = {}
    for i, cluster in clustered_texts.items():
        features = np.mean([stylometry_features(text) for text in cluster], axis=0)
        analysis[i] = {f"feature_{j}": feature for j, feature in enumerate(features)}
    return analysis


def visualize_text_clusters(texts: List[str], num_seeds: int) -> None:
    """Visualize the text clusters using their stylometric features."""
    clustered_texts = cluster_texts(texts, num_seeds)
    for i, cluster in clustered_texts.items():
        print(f"Cluster {i}: {', '.join(cluster)}")


if __name__ == "__main__":
    texts = ["This is a sample text.", "Another sample text.", "A text with different style."]
    num_seeds = 2
    clustered_texts = cluster_texts(texts, num_seeds)
    analysis = analyze_text_clusters(texts, num_seeds)
    visualize_text_clusters(texts, num_seeds)
    print("Analysis:")
    for i, cluster in analysis.items():
        print(f"Cluster {i}: {cluster}")