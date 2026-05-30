# DARWIN HAMMER — match 2474, survivor 0
# gen: 4
# parent_a: hybrid_hard_truth_math_model_pool_m8_s3.py (gen1)
# parent_b: hybrid_infotaxis_hybrid_semantic_neig_m739_s4.py (gen3)
# born: 2026-05-29T23:42:29Z

"""
Hybrid Algorithm: Hard Truth Math Model + Infotaxis-Semantic Morphology
Parents:
- hard_truth_math_model_pool_m8_s3.py (stylometry / LSM utilities)
- hybrid_infotaxis_hybrid_semantic_neig_m739_s4.py (Hybrid Infotaxis–Semantic Morphology System)

Mathematical Bridge:
The recovery priority p∈[0,1] derived from a document’s morphology is used to scale the LSM vectors, yielding a hybrid affinity for each candidate neighbor. This affinity is then treated as a probability-like mass and fed into the infotaxis entropy machinery.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


# ----------------------------------------------------------------------
# Parent A – stylometry / LSM utilities
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
    morphology = Morphology(
        length=len(text) / 100.0,
        width=len(text) / 100.0,
        height=len(text) / 100.0,
        mass=len(text),
    )
    recovery_priority_m = recovery_priority(morphology)
    return {
        cat: sum(cnt[w] for w in vocab) / total * recovery_priority_m
        for cat, vocab in FUNCTION_CATS.items()
    }


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


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
    return np.array(vals)


# ----------------------------------------------------------------------
# Parent B – Morphology & Recovery Priority
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting time index to a normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m, b=1.0/3.0, k=0.35, neck_lever=1.0) / max_index))


def hybrid_affinity(text: str) -> float:
    """Hybrid affinity for a given text"""
    morphology = Morphology(
        length=len(text) / 100.0,
        width=len(text) / 100.0,
        height=len(text) / 100.0,
        mass=len(text),
    )
    recovery_priority_m = recovery_priority(morphology)
    lsm_vector_m = lsm_vector(text)
    return recovery_priority_m * np.mean(list(lsm_vector_m.values()))


def test_hybrid_affinity():
    text = "This is a test text"
    print(hybrid_affinity(text))

if __name__ == "__main__":
    test_hybrid_affinity()