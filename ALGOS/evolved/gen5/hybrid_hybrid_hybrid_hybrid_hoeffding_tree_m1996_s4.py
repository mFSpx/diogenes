# DARWIN HAMMER — match 1996, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s3.py (gen4)
# parent_b: hoeffding_tree.py (gen0)
# born: 2026-05-29T23:40:18Z

"""
This module represents a novel hybrid algorithm that fuses the core topologies of two parent algorithms: 
'hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s3.py' and 'hoeffding_tree.py'. 
The mathematical bridge between these two structures lies in the application of Hoeffding bounds to the 
stylometry features and LSM vectors extracted from text data. This fusion enables the dynamic adaptation of 
the feature extraction process based on the Hoeffding bound calculations.

The hybrid algorithm integrates the governing equations of both parents by using the Hoeffding bound 
calculations to determine the optimal number of features to extract from text data. The 'should_split' 
decision from the Hoeffding tree algorithm is used to decide whether to split the feature space or not, 
based on the calculated epsilon value. This adaptive feature extraction process is then used to extract 
stylometry features and LSM vectors from text data.
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


def extract_full_features(text: str, num_features: int = 15) -> dict[str, float]:
    seed = _deterministic_hash(text) % (2**32)
    rnd = random.Random(seed)
    keys = [f"feature_{i}" for i in range(num_features)]
    return {k: rnd.random() for k in keys}


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


def hybrid_feature_extraction(text: str, num_features: int = 15, r: float = 0.1, delta: float = 0.05, n: int = 1000) -> dict[str, float]:
    """
    This function extracts features from text data using a hybrid approach. It uses the Hoeffding bound 
    calculation to determine the optimal number of features to extract based on the provided parameters.
    """
    seed = _deterministic_hash(text)
    rnd = random.Random(seed)
    best_gain = rnd.random()
    second_best_gain = rnd.random()
    split_decision = should_split(best_gain, second_best_gain, r, delta, n)
    if split_decision.should_split:
        num_features = min(num_features, int(split_decision.epsilon * num_features))
    return extract_full_features(text, num_features)


def adaptive_lsm(text: str, r: float = 0.1, delta: float = 0.05, n: int = 1000) -> np.ndarray:
    """
    This function calculates the LSM vector from text data using an adaptive approach based on the Hoeffding 
    bound calculation.
    """
    seed = _deterministic_hash(text)
    rnd = random.Random(seed)
    best_gain = rnd.random()
    second_best_gain = rnd.random()
    split_decision = should_split(best_gain, second_best_gain, r, delta, n)
    if split_decision.should_split:
        word_list = words(text)
        lsm = np.zeros(len(FUNCTION_CATS))
        for i, (cat, words) in enumerate(FUNCTION_CATS.items()):
            lsm[i] = sum(1 for w in word_list if w in words) / len(word_list) if word_list else 0
        return lsm
    else:
        return np.zeros(len(FUNCTION_CATS))


def hybrid_stylometry(text: str, r: float = 0.1, delta: float = 0.05, n: int = 1000) -> np.ndarray:
    """
    This function calculates the stylometry features from text data using a hybrid approach based on the 
    Hoeffding bound calculation.
    """
    seed = _deterministic_hash(text)
    rng = np.random.default_rng(seed)
    best_gain = rng.random()
    second_best_gain = rng.random()
    split_decision = should_split(best_gain, second_best_gain, r, delta, n)
    if split_decision.should_split:
        return stylometry_features(text)
    else:
        return np.zeros(96)


if __name__ == "__main__":
    text = "This is a test text."
    print(hybrid_feature_extraction(text))
    print(adaptive_lsm(text))
    print(hybrid_stylometry(text))