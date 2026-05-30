# DARWIN HAMMER — match 27, survivor 2
# gen: 2
# parent_a: hybrid_cockpit_metrics_rectified_flow_m10_s2.py (gen1)
# parent_b: hard_truth_math.py (gen0)
# born: 2026-05-29T23:26:23Z

"""hybrid_truth_cockpit.py
Hybrid module unifying the hard-truth math (Parent B) with the cockpit honesty/evidence metrics (Parent A).

Mathematical bridge
-------------------
The core of the hard-truth math is the linguistic style matching (LSM) vector

    lsm_vector(text) = {cat: sum(cnt[w] for w in vocab) / total}     (1)

which characterizes the linguistic style of a given text.
The cockpit metrics provide a scalar *trust* value in the interval [0,1]
(e.g. ``cockpit_honesty`` or ``anti_slop_ratio``).  By treating this scalar as a
weighting factor on the LSM vector, we obtain a *trust-weighted* LSM vector

    lsm_hybrid(text; h) = h · lsm_vector(text)                      (2)

where h ∈ [0,1] is any cockpit metric.  Equation (2) fuses the two topologies:
the linguistic style matching of hard-truth math is modulated by the evidence‑coverage
quality of the cockpit.

The LSM score between two vectors can then be computed as

    lsm_score_hybrid(a, b; h) = lsm_score(h · a, b)                  (3)

or

    lsm_score_hybrid(a, b; h) = lsm_score(a, h · b)                  (4)

depending on the application.

All functions remain pure NumPy and respect the import restrictions.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – cockpit metrics (re‑implemented for internal use)
# ---------------------------------------------------------------------------

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0,
                claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))


def audit_debt(exports_missing_audit_step: int, total_exports: int) -> float:
    """Fraction of exports missing an audit step, clamped to [0, 1]."""
    return 1.0 if total_exports <= 0 else max(0.0, min(1.0,
                exports_missing_audit_step / total_exports))


# ---------------------------------------------------------------------------
# Parent B – hard-truth math (re‑implemented for internal use)
# ---------------------------------------------------------------------------

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> list[str]:
    return [w.lower() for w in text.split() if w.isalpha()]


def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = {w: ws.count(w) for w in set(ws)}
    return {cat: sum(cnt[w] for w in vocab) / total for cat, vocab in FUNCTION_CATS.items()}


def lsm_score(a: dict[str, float], b: dict[str, float]) -> float:
    score = 0.0
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score += 1.0 - (abs(av - bv) / (av + bv + 1e-6))
    return score / len(FUNCTION_CATS)


# ---------------------------------------------------------------------------
# Hybrid functions
# ---------------------------------------------------------------------------

def lsm_hybrid(text: str, trust: float) -> dict[str, float]:
    """Trust-weighted LSM vector."""
    return {cat: trust * val for cat, val in lsm_vector(text).items()}


def lsm_score_hybrid(a: dict[str, float], b: dict[str, float], trust: float) -> float:
    """Trust-weighted LSM score."""
    return lsm_score({cat: trust * val for cat, val in a.items()}, b)


def hybrid_analysis(text_a: str, text_b: str, trust: float) -> Tuple[float, dict[str, float]]:
    """Hybrid analysis of two texts with trust weighting."""
    vec_a = lsm_hybrid(text_a, trust)
    vec_b = lsm_vector(text_b)
    score = lsm_score(vec_a, vec_b)
    return score, vec_a


if __name__ == "__main__":
    text_a = "This is a test sentence."
    text_b = "This sentence is another test."
    trust = 0.8
    score, vec_a = hybrid_analysis(text_a, text_b, trust)
    print(f"LSM score: {score:.4f}")
    print(f"Trust-weighted LSM vector: {vec_a}")