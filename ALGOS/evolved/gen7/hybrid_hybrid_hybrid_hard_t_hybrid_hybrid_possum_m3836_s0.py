# DARWIN HAMMER — match 3836, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_hybrid_m1853_s2.py (gen6)
# parent_b: hybrid_hybrid_possum_filter_hybrid_hybrid_hybrid_m1779_s1.py (gen5)
# born: 2026-05-29T23:51:49Z

"""
This module represents a novel hybrid algorithm that fuses the core topologies of two parent algorithms: 
hybrid_hybrid_hard_truth_ma_hybrid_hybrid_hybrid_m1853_s2 and hybrid_hybrid_possum_filter_hybrid_hybrid_hybrid_m1779_s1.
The mathematical bridge between the two structures is the integration of the feature matrix built from the graph topology 
with the joint feature vector consisting of normalized spatial coordinates, textual signature, and recovery priority.
The hybrid algorithm utilizes the word categorization from the first parent, the graph topology and Schoolfield rate from the second parent,
and combines them with the spatial diversity, textual similarity, and morphological robustness from the second parent.
"""

import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

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
        "no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

class Entity:
    """Represents a spatial, textual and morphological object."""
    def __init__(self, id: str, lat: float, lon: float, category: str, text: str, score: float = 0.0, length: float = 1.0, width: float = 1.0, height: float = 1.0, mass: float = 1.0):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.category = category
        self.text = text
        self.score = score
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {cat: sum(cnt[w] for w in FUNCTION_CATS[cat]) / total for cat in FUNCTION_CATS}

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0  # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = φ2 - φ1
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def joint_feature_vector(entity: Entity) -> np.ndarray:
    lat, lon = math.radians(entity.lat), math.radians(entity.lon)
    x = math.cos(lat) * math.cos(lon)
    y = math.cos(lat) * math.sin(lon)
    z = math.sin(lat)
    lsm = lsm_vector(entity.text)
    return np.array([x, y, z, *lsm.values()])

def composite_distance(entity1: Entity, entity2: Entity) -> float:
    vec1 = joint_feature_vector(entity1)
    vec2 = joint_feature_vector(entity2)
    haversine_dist = haversine_distance(entity1.lat, entity1.lon, entity2.lat, entity2.lon)
    lsm_dist = np.linalg.norm(vec1[3:] - vec2[3:])
    spatial_dist = np.linalg.norm(vec1[:3] - vec2[:3])
    return 0.4 * haversine_dist + 0.3 * lsm_dist + 0.3 * spatial_dist

if __name__ == "__main__":
    entity1 = Entity("id1", 37.7749, -122.4194, "category1", "text1")
    entity2 = Entity("id2", 38.8977, -77.0365, "category2", "text2")
    print(composite_distance(entity1, entity2))