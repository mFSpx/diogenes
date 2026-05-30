# DARWIN HAMMER — match 27, survivor 3
# gen: 2
# parent_a: hybrid_cockpit_metrics_rectified_flow_m10_s2.py (gen1)
# parent_b: hard_truth_math.py (gen0)
# born: 2026-05-29T23:26:23Z

"""hybrid_truth_cockpit.py
Hybrid module unifying the hard-truth math (Parent B) with the cockpit honesty/evidence metrics (Parent A).

Mathematical bridge
-------------------
The core of the hard-truth math is the linguistic similarity measure (LSM) vector

    lsm_vector(text) = {cat: sum(cnt[w] for w in vocab) / total}    (1)

and the LSM score

    lsm_score(a, b) = {cat: 1.0 - (abs(av - bv) / (av + bv + 1e-6))}  (2)

The cockpit metrics provide a scalar *trust* value in the interval [0,1]
(e.g. ``cockpit_honesty`` or ``anti_slop_ratio``).  By treating this scalar as a
weight on the LSM score, we obtain a *trust-weighted* LSM score

    trust_weighted_lsm_score(a, b; h) = h * lsm_score(a, b)          (3)

where h ∈ [0,1] is any cockpit metric.  Equation (3) fuses the two topologies:
the linguistic similarity of hard-truth math is modulated by the evidence‑coverage
quality of the cockpit.

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
PERSONA_PATTERNS: dict[str, re.Pattern[str]] = {
    "ponyboy": re.compile(r"\bponyboy\b", re.I),
    "northern_strike": re.compile(r"\bnorthern\.?strike\b", re.I),
    "lucidota": re.compile(r"\blucidota\b|\bluci\b", re.I),
    "indy_reads": re.compile(r"\bindy[_ -]?reads\b|\bindy\b", re.I),
    "zachary": re.compile(r"\bzachary\b|\bzach\b", re.I),
    "operator": re.compile(r"\boperator\b", re.I),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {cat: sum(cnt[w] for w in vocab) / total for cat, vocab in FUNCTION_CATS.items()}


def lsm_score(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    detail: dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
    return detail


# ---------------------------------------------------------------------------
# Hybrid functions
# ---------------------------------------------------------------------------

def trust_weighted_lsm_score(a: dict[str, float], b: dict[str, float], h: float) -> dict[str, float]:
    """Trust-weighted LSM score."""
    lsm = lsm_score(a, b)
    return {cat: h * score for cat, score in lsm.items()}


def hybrid_lsm_vector(text: str, h: float) -> dict[str, float]:
    """Trust-weighted LSM vector."""
    lsm = lsm_vector(text)
    return {cat: h * value for cat, value in lsm.items()}


def hybrid_audit_debt(text1: str, text2: str, h: float) -> float:
    """Trust-weighted audit debt."""
    lsm = lsm_score(lsm_vector(text1), lsm_vector(text2))
    return 1.0 - sum(lsm.values()) / len(lsm)


if __name__ == "__main__":
    text1 = "This is a test sentence."
    text2 = "This is another test sentence."
    h = cockpit_honesty(10, 2)
    print(trust_weighted_lsm_score(lsm_vector(text1), lsm_vector(text2), h))
    print(hybrid_lsm_vector(text1, h))
    print(hybrid_audit_debt(text1, text2, h))