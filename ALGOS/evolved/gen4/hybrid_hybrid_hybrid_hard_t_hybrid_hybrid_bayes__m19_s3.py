# DARWIN HAMMER — match 19, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py (gen3)
# parent_b: hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s4.py (gen3)
# born: 2026-05-29T23:26:23Z

import numpy as np
import random
import math
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

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
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _deterministic_hash(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def words(text: str) -> List[str]:
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


def extract_full_features(text: str, num_features: int = 15) -> Dict[str, float]:
    seed = _deterministic_hash(text) % (2**32)
    rnd = random.Random(seed)
    keys = [f"feature_{i}" for i in range(num_features)]
    return {k: rnd.random() for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    f = extract_full_features(text)
    return {k: v for k, v in f.items()}


def hybrid_features(text: str, stylometry_dim: int = 96, num_full_features: int = 15) -> Tuple[np.ndarray, Dict[str, float]]:
    stylometry = stylometry_features(text, stylometry_dim)
    full_features = extract_full_features(text, num_full_features)
    return stylometry, full_features


def hybrid_master_vector(text: str, stylometry_dim: int = 96, num_master_features: int = 8) -> Tuple[np.ndarray, Dict[str, float]]:
    stylometry = stylometry_features(text, stylometry_dim)
    master_vector = {f"feature_{i}": v for i, v in enumerate(extract_full_features(text, num_master_features).values())}
    return stylometry, master_vector


if __name__ == "__main__":
    text = "This is a sample text for demonstration purposes."
    stylometry = stylometry_features(text)
    lsm = lsm_vector(text)
    full_features = extract_full_features(text)
    master_vector = extract_master_vector(text)
    hybrid = hybrid_features(text)
    hybrid_master = hybrid_master_vector(text)
    print("Stylometry Features:", stylometry)
    print("LSM Vector:", lsm)
    print("Full Features:", full_features)
    print("Master Vector:", master_vector)
    print("Hybrid Features:", hybrid)
    print("Hybrid Master Vector:", hybrid_master)