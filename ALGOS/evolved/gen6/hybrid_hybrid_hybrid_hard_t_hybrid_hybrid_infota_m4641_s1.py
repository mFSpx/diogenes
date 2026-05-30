# DARWIN HAMMER — match 4641, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hard_truth_ma_kan_m27_s1.py (gen2)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m1066_s0.py (gen5)
# born: 2026-05-29T23:57:09Z

"""
This module implements a hybrid algorithm that combines the stylometry and LSM utilities 
from the 'hybrid_hybrid_hard_truth_ma_kan_m27_s1' algorithm with the entropic MinHash 
topology of the 'hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m1066_s0' algorithm.

The mathematical bridge between the two algorithms is found in the representation of 
the stylometry features as a probability distribution, which can be estimated using 
the entropic MinHash method. Specifically, the stylometry features are used to compute 
the entropic MinHash signature, and the similarity between the signatures is used to 
inform the stylometry analysis.

This hybrid algorithm allows for the representation of stylometry features in a more 
flexible and powerful way, using the entropic MinHash method to estimate the similarity 
between the features.
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Parent A – stylometry / LSM utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
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


def words(text: str) -> List[str]:
    return [word for word in (text or "").lower().split() if word.isalpha()]


def lsm_vector(text: str) -> dict[str, float]:
    word_list = words(text)
    word_count = len(word_list)
    lsm_vector = {}
    for func_cat, func_words in FUNCTION_CATS.items():
        func_word_count = sum(1 for word in word_list if word in func_words)
        lsm_vector[func_cat] = func_word_count / word_count if word_count > 0 else 0
    return lsm_vector


# ----------------------------------------------------------------------
# Parent B – entropic MinHash
# ----------------------------------------------------------------------
def entropy(probabilities: list[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)


def signature(tokens: list[str], k: int = 128) -> list[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')


def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def entropic_minhash(probabilities: list[float], k: int = 128) -> list[int]:
    tokens = [str(p) for p in probabilities]
    return signature(tokens, k)


def stylometry_analysis(text: str) -> dict[str, float]:
    lsm_vector_result = lsm_vector(text)
    probabilities = list(lsm_vector_result.values())
    entropic_minhash_result = entropic_minhash(probabilities)
    return {
        'lsm_vector': lsm_vector_result,
        'entropic_minhash': entropic_minhash_result,
    }


def compare_stylometry(text_a: str, text_b: str) -> float:
    stylometry_analysis_a = stylometry_analysis(text_a)
    stylometry_analysis_b = stylometry_analysis(text_b)
    similarity_result = similarity(stylometry_analysis_a['entropic_minhash'], stylometry_analysis_b['entropic_minhash'])
    return similarity_result


def main():
    text_a = "This is a sample text."
    text_b = "This is another sample text."
    stylometry_analysis_result = stylometry_analysis(text_a)
    compare_stylometry_result = compare_stylometry(text_a, text_b)
    print(stylometry_analysis_result)
    print(compare_stylometry_result)


if __name__ == "__main__":
    main()