# DARWIN HAMMER — match 3193, survivor 1
# gen: 3
# parent_a: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py (gen2)
# parent_b: hybrid_shannon_entropy_rsa_cipher_m51_s1.py (gen1)
# born: 2026-05-29T23:48:20Z

"""
This module combines the LSM vector calculation from hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py with the RSA cipher from hybrid_shannon_entropy_rsa_cipher_m51_s1.py.
The mathematical bridge between the two is the use of entropy as a measure of the uncertainty of the encrypted message.
In this hybrid algorithm, we use the LSM vectors to analyze the randomness of the encrypted message and adjust the RSA cipher parameters accordingly.
"""

from __future__ import annotations
import math
from collections import Counter
from collections.abc import Hashable, Iterable
import numpy as np
import random
import sys
import pathlib

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

Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    # This function is used from parent A, but we don't need it in the hybrid algorithm.
    # We can remove it or modify it to fit the hybrid algorithm.
    pass

def calculate_entropy(message: int, n: int) -> float:
    binary_message = np.array([int(x) for x in bin(message)[2:].zfill(n.bit_length())])
    observations = binary_message.tolist()
    return shannon_entropy(observations, is_distribution=False)

def shannon_entropy(observations: Iterable[Hashable | float], is_distribution: bool = False) -> float:
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

def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    if not 0 <= ciphertext < n: raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)

def hybrid_entropy_lsm(text: str, n: int) -> Tuple[float, Dict[str, float]]:
    lsm = lsm_vector(text)
    entropy = calculate_entropy(random.randint(0, 2**n), n)
    return entropy, lsm

def hybrid_rsa_lsm(message: int, e: int, n: int, text: str) -> Tuple[int, float]:
    lsm = lsm_vector(text)
    encrypted_message = rsa_encrypt(message, e, n)
    return encrypted_message, lsm

def hybrid_optimal_key(n: int, text: str) -> int:
    max_entropy = 0
    optimal_key = 0
    for e in range(2, n):
        if math.gcd(e, (n-1)) == 1:
            message = random.randint(0, n-1)
            _, lsm = hybrid_entropy_lsm(text, n)
            encrypted_message, _ = hybrid_rsa_lsm(message, e, n, text)
            entropy = calculate_entropy(encrypted_message, n)
            if entropy > max_entropy:
                max_entropy = entropy
                optimal_key = e
    return optimal_key

if __name__ == "__main__":
    n = 257
    e = 17
    d = 77
    message = 123
    text = "This is a test message."
    _, lsm = hybrid_entropy_lsm(text, n)
    print(lsm)
    encrypted_message, _ = hybrid_rsa_lsm(message, e, n, text)
    print(encrypted_message)
    optimal_key = hybrid_optimal_key(n, text)
    print(optimal_key)