# DARWIN HAMMER — match 3193, survivor 0
# gen: 3
# parent_a: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py (gen2)
# parent_b: hybrid_shannon_entropy_rsa_cipher_m51_s1.py (gen1)
# born: 2026-05-29T23:48:19Z

"""
This module combines the Least Squares Metric (LSM) calculation from hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py 
with the Shannon entropy and RSA cipher from hybrid_shannon_entropy_rsa_cipher_m51_s1.py.
The mathematical bridge between the two is the use of entropy as a measure of the uncertainty of the text data, 
which is then used to adjust the LSM calculation.

The LSM calculation provides a vector representation of text data based on part-of-speech (POS) categories. 
The Shannon entropy calculation provides a measure of the uncertainty or randomness of the text data. 
The RSA cipher provides a secure way to encrypt the text data.

In this hybrid algorithm, we use the Shannon entropy to analyze the randomness of the text data, 
and then adjust the LSM calculation based on the entropy value. 
We also use the RSA cipher to encrypt the text data and then calculate the LSM and entropy of the encrypted text.

The interface between the two algorithms is through the use of the entropy value, 
which is used to adjust the LSM calculation.
"""

import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
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

def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    detail: Dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
    overall = sum(detail.values()) / len(FUNCTION_CATS)
    return overall, detail

def shannon_entropy(observations: List[Any], is_distribution: bool = False) -> float:
    xs = list(observations)
    if not xs: return 0.0
    if is_distribution:
        probs=[float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs)-1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        c=Counter(xs); total=sum(c.values()); probs=[v/total for v in c.values()]
    return -sum(p*math.log2(p) for p in probs if p > 0)

def rsa_encrypt(message: int, e: int, n: int) -> int:
    if not 0 <= message < n: raise ValueError("message must be in [0, n)")
    return pow(message, e, n)

def hybrid_lsm_entropy(text: str, e: int, n: int) -> Tuple[Dict[str, float], float]:
    vector = lsm_vector(text)
    encrypted_text = rsa_encrypt(int.from_bytes(text.encode(), 'big'), e, n)
    encrypted_text_str = bin(encrypted_text)[2:]
    observations = [int(x) for x in encrypted_text_str]
    entropy = shannon_entropy(observations)
    return vector, entropy

def adjust_lsm_by_entropy(vector: Dict[str, float], entropy: float) -> Dict[str, float]:
    adjusted_vector = {}
    for cat, value in vector.items():
        adjusted_value = value * (1 - entropy / 8)
        adjusted_vector[cat] = max(0.0, min(1.0, adjusted_value))
    return adjusted_vector

def hybrid_operation(text: str, e: int, n: int) -> Tuple[Dict[str, float], Dict[str, float], float]:
    vector, entropy = hybrid_lsm_entropy(text, e, n)
    adjusted_vector = adjust_lsm_by_entropy(vector, entropy)
    score, detail = lsm_score(vector, adjusted_vector)
    return vector, detail, score

if __name__ == "__main__":
    text = "This is a test sentence."
    e = 17
    n = 257
    vector, detail, score = hybrid_operation(text, e, n)
    print("LSM Vector:", vector)
    print("Adjusted LSM Detail:", detail)
    print("LSM Score:", score)