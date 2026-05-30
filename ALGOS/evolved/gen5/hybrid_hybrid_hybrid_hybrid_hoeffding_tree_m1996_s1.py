# DARWIN HAMMER — match 1996, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s3.py (gen4)
# parent_b: hoeffding_tree.py (gen0)
# born: 2026-05-29T23:40:18Z

"""
This module fuses the structures of 'hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s3.py' and 'hoeffding_tree.py' 
by integrating their mathematical operations. The core idea is to apply the Hoeffding bound 
from the 'hoeffding_tree.py' to the stylometry features and LSM vector calculations in 'hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s3.py'. 
The mathematical bridge between the two structures lies in the application of statistical bounds to the 
feature extraction process, allowing for a more informed decision-making process in the hybrid algorithm.
"""

import numpy as np
import random
import math
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass

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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def _deterministic_hash(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)

def words(text: str) -> list[str]:
    return [w for w in text.lower().split() if w.isalpha()]

def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    seed = _deterministic_hash(text)
    rng = np.random.default_rng(seed)
    return rng.random(dim)

def lsm_vector(text: str) -> np.ndarray:
    word_list = words(text)
    lsm = np.zeros(len(FUNCTION_CATS))
    for i, (cat, words) in enumerate(FUNCTION_CATS.items()):
        lsm[i] = sum(1 for w in word_list if w in words) / len(word_list) if word_list else 0
    return lsm

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

def hybrid_feature_extraction(text: str, dim: int = 96) -> tuple[np.ndarray, np.ndarray]:
    stylometry_feat = stylometry_features(text, dim)
    lsm_feat = lsm_vector(text)
    return stylometry_feat, lsm_feat

def hybrid_hoeffding_bound(stylometry_feat: np.ndarray, lsm_feat: np.ndarray, delta: float, n: int) -> float:
    r = np.mean(stylometry_feat)
    return hoeffding_bound(r, delta, n)

def hybrid_split_decision(stylometry_feat: np.ndarray, lsm_feat: np.ndarray, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    best_gain = np.mean(stylometry_feat)
    second_best_gain = np.mean(lsm_feat)
    eps = hybrid_hoeffding_bound(stylometry_feat, lsm_feat, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

if __name__ == "__main__":
    text = "This is a sample text for testing the hybrid algorithm."
    stylometry_feat, lsm_feat = hybrid_feature_extraction(text)
    delta = 0.1
    n = 100
    split_decision = hybrid_split_decision(stylometry_feat, lsm_feat, delta, n)
    print(split_decision)