# DARWIN HAMMER — match 2474, survivor 2
# gen: 4
# parent_a: hybrid_hard_truth_math_model_pool_m8_s3.py (gen1)
# parent_b: hybrid_infotaxis_hybrid_semantic_neig_m739_s4.py (gen3)
# born: 2026-05-29T23:42:29Z

"""
Hybrid Stylometry-Infotaxis System

Parents:
- hybrid_hard_truth_math_model_pool_m8_s3.py (stylometry / LSM utilities)
- hybrid_infotaxis_hybrid_semantic_neig_m739_s4.py (morphology-driven hybrid affinity)

Mathematical Bridge:
The stylometry features derived from a document's text are used to inform the morphology 
of the recovery priority, while the morphology-driven hybrid affinity is used to scale 
the cosine similarity of the stylometry features. This creates a feedback loop where the 
text analysis drives the morphology, and the morphology drives the text analysis.
"""
from __future__ import annotations

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re
import hashlib
import numpy as np

FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))
    vals: List[float] = [
        total_words / 500.0,
        sum(len(w) for w in ws) / total_words / 12.0,
        (text.count("\n") + 1) / 200.0,
        sum(text.count(p) for p in "!?") / total_chars,
        sum(text.count(p) for p in ";:") / total_chars,
    ]
    return np.array(vals)

def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def words(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)

def text_morphology(text: str) -> Morphology:
    features = stylometry_features(text)
    length = features[0] * 10
    width = features[1] * 10
    height = features[2] * 10
    mass = features[3] * 10
    return Morphology(length, width, height, mass)

def hybrid_affinity(text1: str, text2: str) -> float:
    m1 = text_morphology(text1)
    m2 = text_morphology(text2)
    p1 = recovery_priority(m1)
    p2 = recovery_priority(m2)
    features1 = lsm_vector(text1)
    features2 = lsm_vector(text2)
    dot_product = sum(features1[cat] * features2[cat] for cat in features1)
    magnitude1 = math.sqrt(sum(x ** 2 for x in features1.values()))
    magnitude2 = math.sqrt(sum(x ** 2 for x in features2.values()))
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity * p2

def infotaxis_entropy(action: str, texts: List[str]) -> float:
    affinities = [hybrid_affinity(action, text) for text in texts]
    affinities = np.array(affinities)
    probabilities = affinities / affinities.sum()
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return entropy

if __name__ == "__main__":
    text1 = "This is a test text."
    text2 = "This is another test text."
    print(hybrid_affinity(text1, text2))
    print(infotaxis_entropy(text1, [text2]))