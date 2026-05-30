# DARWIN HAMMER — match 2889, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_infotaxis_hyb_m2474_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_sparse_wta_hy_m1093_s2.py (gen3)
# born: 2026-05-29T23:46:25Z

"""
This module implements a novel hybrid algorithm, fusing the core topologies of 
'hybrid_hybrid_hard_truth_ma_hybrid_infotaxis_hyb_m2474_s1' and 
'hybrid_hybrid_hybrid_hard_t_hybrid_sparse_wta_hy_m1093_s2'. 
The mathematical bridge between the two parents is established by integrating 
the B-spline basis from 'hybrid_hybrid_hybrid_hard_t_hybrid_sparse_wta_hy_m1093_s2' 
with the stylometry features and LSM utilities from 'hybrid_hybrid_hard_truth_ma_hybrid_infotaxis_hyb_m2474_s1'. 
The B-spline basis is used to scale the stylometry features, 
yielding a hybrid representation that drives the information-theoretic action ranking.
"""

from __future__ import annotations
import datetime as dt
import hashlib
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Stylometry / LSM Utilities
# ----------------------------------------------------------------------
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

def words(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {w: c / total for w, c in cnt.items()}

def stylometry_features(text: str) -> np.ndarray:
    """Return a fixed‑size numeric vector describing ``text``."""
    tokens = text.lower().split()
    n_words = len(tokens)
    n_chars = len(text)
    avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0.0

    # Category frequencies
    cat_counts = {cat: 0 for cat in FUNCTION_CATS}
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                cat_counts[cat] += 1

    cat_freqs = [cat_counts[cat] / n_words if n_words else 0.0 for cat in sorted(FUNCTION_CATS)]

    return np.array([n_words, n_chars, avg_word_len, *cat_freqs], dtype=float)

# ----------------------------------------------------------------------
# Parent B – B‑spline basis (Cox‑de Boor recursion)
# ----------------------------------------------------------------------
def _cox_de_boor(x: float, k: int, i: int, knots: List[float]) -> float:
    """Recursive definition of B‑spline basis N_{i,k}(x)."""
    if k == 0:
        return 1.0 if knots[i] <= x < knots[i + 1] else 0.0
    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]
    term1 = 0.0
    term2 = 0.0
    if denom1 > 0:
        term1 = (x - knots[i]) / denom1 * _cox_de_boor(x, k - 1, i, knots)
    if denom2 > 0:
        term2 = (knots[i + k + 1] - x) / denom2 * _cox_de_boor(x, k - 1, i + 1, knots)
    return term1 + term2

def bspline_basis(x: float, knots: List[float], degree: int) -> np.ndarray:
    """Evaluate B-spline basis at ``x``."""
    n_basis = len(knots) - degree - 1
    basis = np.zeros(n_basis)
    for i in range(n_basis):
        basis[i] = _cox_de_boor(x, degree, i, knots)
    return basis

# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------
def hybrid_representation(text: str, knots: List[float], degree: int) -> np.ndarray:
    """Return a hybrid representation of ``text``."""
    stylometry = stylometry_features(text)
    basis = bspline_basis(0.5, knots, degree)  # Evaluate basis at a fixed point
    return stylometry * basis

def hybrid_action_ranking(texts: List[str], knots: List[float], degree: int) -> List[int]:
    """Return a ranking of ``texts`` based on their hybrid representations."""
    representations = [hybrid_representation(text, knots, degree) for text in texts]
    return np.argsort(-np.array(representations).sum(axis=1))

def smoke_test():
    texts = ["This is a test.", "Another test.", "Test again."]
    knots = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    degree = 2
    ranking = hybrid_action_ranking(texts, knots, degree)
    print(ranking)

if __name__ == "__main__":
    smoke_test()