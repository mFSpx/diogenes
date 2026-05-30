# DARWIN HAMMER — match 2474, survivor 4
# gen: 4
# parent_a: hybrid_hard_truth_math_model_pool_m8_s3.py (gen1)
# parent_b: hybrid_infotaxis_hybrid_semantic_neig_m739_s4.py (gen3)
# born: 2026-05-29T23:42:29Z

"""
Hybrid Algorithm: Fusing Stylometry (hard_truth_math_model_pool_m8_s3.py) 
and Hybrid Infotaxis–Semantic Morphology (hybrid_infotaxis_hybrid_semantic_neig_m739_s4.py)

The mathematical bridge between the two parents lies in the use of information-theoretic 
measures to analyze text and morphology. We fuse the stylometry features with the 
morphology-modulated semantic topology to derive a hybrid affinity.

The stylometry features (e.g., LSM vector) are used to compute a probability-like mass, 
which is then fed into the infotaxis entropy machinery. The morphology of a document 
is used to modulate the semantic similarity, yielding a hybrid affinity.

This hybrid algorithm integrates the governing equations of both parents, enabling 
a unified analysis of text and morphology.
"""

import numpy as np
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import re
import hashlib

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

def words(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)

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
    return np.array(vals[:dim])

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
    """Maps righting time index to a normalized priority in [0,1]."""
    rti = righting_time_index(m)
    return max(0.0, min(1.0, rti / max_index))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)

def hybrid_affinity(stylometry_features: np.ndarray, morphology: Morphology) -> float:
    recovery_p = recovery_priority(morphology)
    similarity = cosine_similarity(stylometry_features, np.array([1.0]*len(stylometry_features)))
    return similarity * recovery_p

def expected_entropy(action_affinities: List[float]) -> float:
    affinities = np.array(action_affinities) / sum(action_affinities)
    return -np.sum(affinities * np.log2(affinities))

if __name__ == "__main__":
    text = "This is a sample text for stylometry analysis."
    morphology = Morphology(10.0, 5.0, 2.0, 1.0)
    stylometry_feat = stylometry_features(text)
    hybrid_aff = hybrid_affinity(stylometry_feat, morphology)
    print("Hybrid Affinity:", hybrid_aff)
    action_affinities = [hybrid_affinity(stylometry_feat, morphology) for _ in range(10)]
    print("Expected Entropy:", expected_entropy(action_affinities))