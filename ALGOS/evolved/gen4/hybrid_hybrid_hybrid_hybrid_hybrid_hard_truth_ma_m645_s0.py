# DARWIN HAMMER — match 645, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s2.py (gen3)
# parent_b: hybrid_hard_truth_math_model_pool_m8_s3.py (gen1)
# born: 2026-05-29T23:30:08Z

"""
Hybrid Bandit-Capybara-Stylometry Algorithm.

This module fuses the contextual multi-armed bandit router from 
hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s2.py with the 
stylometry utilities from hybrid_hard_truth_math_model_pool_m8_s3.py.
The mathematical bridge is the **matrix representation** of the stylometry features,
which is used to rescale the reward function of the bandit algorithm.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

# ----------------------------------------------------------------------
# Bandit core
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}          # virtual VRAM store per key

# ----------------------------------------------------------------------
# Stylometry utilities
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
    return [word for word in re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower()) if word not in PUNCT]

def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = {}
    for w in ws:
        if w not in cnt:
            cnt[w] = 0
        cnt[w] += 1
    return {
        cat: sum(cnt.get(w, 0) for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))
    vals: List[float] = [
        total_words / 500.0,
        sum(len(w) for w in ws) / total_words / 12.0,
        (text.count("\n") + 1) / 200.0,
        sum(text.count(p) for p in "!?") / total_chars,
        sum(text.count(p) for p in ";:") / total_chars,
    ]
    return np.array(vals + [0.0] * (dim - len(vals)))

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_update(context_id: str, action_id: str, reward: float, propensity: float, text: str) -> None:
    """
    Update the bandit policy using the stylometry features of the given text.
    """
    features = stylometry_features(text)
    reward *= np.linalg.norm(features)
    _POLICY[action_id] = [_POLICY.get(action_id, [0.0])[0] + reward, _POLICY.get(action_id, [0.0])[0] + propensity]

def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """
    Exponential decay schedule for evasion magnitude.
    """
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)

def clamp(x: List[float], lower: float, upper: float) -> List[float]:
    """
    Clamp each component of a vector to [lower, upper].
    """
    return [min(upper, max(lower, xi)) for xi in x]

import re

if __name__ == "__main__":
    text = "This is a test text."
    action_id = "test_action"
    _POLICY[action_id] = [0.0, 0.0]
    hybrid_update("test_context", action_id, 1.0, 0.5, text)
    print(_POLICY[action_id])
    print(evasion_delta(10, 100))
    print(clamp([0.2, 0.5, 0.8], 0.0, 1.0))