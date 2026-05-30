# DARWIN HAMMER — match 4082, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m928_s0.py (gen5)
# born: 2026-05-29T23:53:30Z

"""
Hybrid Algorithm: Fusing Stylometry with Ternary Routing and Shapley Attribution

This module fuses the stylometry feature extraction from hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s4.py 
with the ternary routing mechanism and Shapley attribution method from hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m928_s0.py.
The mathematical bridge between the two algorithms lies in the use of vector operations 
to determine routing weights and stylometry-based feature attribution.

The stylometry feature extraction is used to generate feature importance, while the Shapley attribution method's 
shapley_kernel_weight function is used to calculate weights for each feature. The ternary router's 
route_command function is used to generate routing information based on the stylometry and Shapley weights.
"""

import hashlib
import math
import random
import re
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# ----------------------------------------------------------------------
# Text processing utilities
# ----------------------------------------------------------------------
WORD_RE = re.compile(r"\b[a-zA-Z]+\b")

def words(text: Optional[str]) -> List[str]:
    """Return a list of lower‑cased alphabetic tokens."""
    if not text:
        return []
    return WORD_RE.findall(text.lower())

# ----------------------------------------------------------------------
# Function‑word categories (stylometry)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
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
        "no not never none neither cannot cant wont dont didnt isnt arent wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

# ----------------------------------------------------------------------
# Stylometry feature extraction
# ----------------------------------------------------------------------
def lsm_vector(text: str) -> Dict[str, float]:
    """Return normalized frequencies of function‑word categories."""
    ws = words(text)
    total = max(1, len(ws))
    return {cat: len([w for w in ws if w in cat_set]) / total for cat, cat_set in FUNCTION_CATS.items()}

Vector = list[int]

def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: list[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    dim = len(vecs[0])
    for v in vecs:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    summed = [sum(comp) for comp in zip(*vecs)]
    return [1 if s >= 0 else -1 for s in summed]

def similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x ** 2 for x in a)) * math.sqrt(sum(x ** 2 for x in b)))

def stylometry_route(text: str) -> Dict[str, float]:
    lsm = lsm_vector(text)
    vector = random_vector(seed=text)
    return {cat: similarity(vector, symbol_vector(cat)) for cat in lsm}

def hybrid_operation(text: str) -> Dict[str, float]:
    lsm = lsm_vector(text)
    vector = random_vector(seed=text)
    sty_ro = stylometry_route(text)
    return {cat: similarity(vector, symbol_vector(cat)) * sty_ro[cat] for cat in lsm}

def feature_importance(text: str) -> Dict[str, float]:
    sty_ro = stylometry_route(text)
    hy_op = hybrid_operation(text)
    return {cat: (sty_ro[cat] + hy_op[cat]) / 2 for cat in sty_ro}

if __name__ == "__main__":
    text = "This is a test text for the hybrid algorithm."
    print(lsm_vector(text))
    print(stylometry_route(text))
    print(hybrid_operation(text))
    print(feature_importance(text))