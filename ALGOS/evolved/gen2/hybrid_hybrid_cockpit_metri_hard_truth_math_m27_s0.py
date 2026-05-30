# DARWIN HAMMER — match 27, survivor 0
# gen: 2
# parent_a: hybrid_cockpit_metrics_rectified_flow_m10_s2.py (gen1)
# parent_b: hard_truth_math.py (gen0)
# born: 2026-05-29T23:26:23Z

"""
Hybrid module unifying the cockpit honesty/evidence metrics from 'hybrid_cockpit_metrics_rectified_flow_m10_s2.py'
with the hard-truth telemetry algorithms for LUCIDOTA from 'hard_truth_math.py'.

The mathematical bridge between the two structures is found in the modulation of the ideal velocity
from the rectified flow transport framework by the evidence-coverage quality of the cockpit metrics.
This modulation is achieved by treating the scalar trust value from the cockpit metrics as a multiplicative
factor on the ideal velocity, resulting in a trust-weighted velocity field.

The hard-truth telemetry algorithms are used to analyze the text data and generate a stylometry feature vector,
which is then used to compute the trust value. This trust value is used to modulate the ideal velocity,
resulting in a hybrid velocity field that takes into account both the rectified flow and the stylometry features.
"""

import math
import random
import sys
from pathlib import Path
from typing import Callable, Tuple, Dict
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


# ---------------------------------------------------------------------------
# Parent B – hard-truth telemetry algorithms
# ---------------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {cat: sum(cnt[w] for w in vocab) / total for cat, vocab in FUNCTION_CATS.items()}

def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    detail: Dict[str, float] = {}
    vals: list[float] = []
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
        vals.append(score)
    return np.mean(vals), detail

# ---------------------------------------------------------------------------
# Hybrid functions
# ---------------------------------------------------------------------------
def hybrid_flow_target(text: str, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Metric-scaled target velocity."""
    lsm_vec = lsm_vector(text)
    trust = np.mean(list(lsm_vec.values()))
    return trust * (x1 - x0)

def hybrid_flow_loss(text: str, x0: np.ndarray, x1: np.ndarray, prediction: np.ndarray) -> float:
    """MSE between a model prediction and the scaled target."""
    target = hybrid_flow_target(text, x0, x1)
    return np.mean((target - prediction) ** 2)

def hybrid_euler_solve(text: str, x0: np.ndarray, x1: np.ndarray, step_size: float) -> np.ndarray:
    """Euler integration that adapts the step size using the audit-debt count as a regulariser."""
    lsm_vec = lsm_vector(text)
    trust = np.mean(list(lsm_vec.values()))
    step_size *= trust
    return x0 + step_size * (x1 - x0)

if __name__ == "__main__":
    import re
    from collections import Counter
    text = "This is a test sentence."
    x0 = np.array([1.0, 2.0])
    x1 = np.array([3.0, 4.0])
    prediction = np.array([2.5, 3.5])
    print(hybrid_flow_target(text, x0, x1))
    print(hybrid_flow_loss(text, x0, x1, prediction))
    print(hybrid_euler_solve(text, x0, x1, 0.1))