# DARWIN HAMMER — match 2979, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_semantic_neig_hybrid_sketches_hybr_m57_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hoeffding_tre_m1469_s0.py (gen6)
# born: 2026-05-29T23:47:05Z

import numpy as np
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass

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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class Document:
    id: str
    vector: list[float]
    text: str

@dataclass(frozen=True)
class BanditAction:
    action_id: str; 
    propensity: float; 
    expected_reward: float; 
    confidence_bound: float; 
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str; 
    action_id: str; 
    reward: float; 
    propensity: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = sphericity_index(m.length, m.width, m.height)
    return (m.mass ** (1.0 / 3.0)) / (b * (fi ** k) * neck_lever)

def calculate_semantic_recovery_priority(document: Document) -> float:
    ws = words(document.text)
    lsm = lsm_vector(document.text)
    semantic_neighbors = np.array([lsm.get(w, 0) for w in ws])
    return np.mean(semantic_neighbors)

def hoeffding_bound(confidence: float, n: int, delta: float) -> float:
    return math.sqrt((math.log(2 / confidence)) / (2 * n * delta))

def calculate_hoeffding_bound(bandit_action: BanditAction, semantic_recovery_priority: float) -> float:
    confidence = 0.95  
    delta = 0.01  
    n = max(1, int(bandit_action.propensity * 100))  
    bound = hoeffding_bound(confidence, n, delta) * (1 + semantic_recovery_priority)
    return bound

def hybrid_operation(document: Document, bandit_action: BanditAction) -> BanditAction:
    semantic_recovery_priority = calculate_semantic_recovery_priority(document)
    confidence_bound = calculate_hoeffding_bound(bandit_action, semantic_recovery_priority)
    return BanditAction(
        action_id=bandit_action.action_id,
        propensity=bandit_action.propensity,
        expected_reward=bandit_action.expected_reward,
        confidence_bound=confidence_bound,
        algorithm=bandit_action.algorithm
    )

def words(text: str) -> list[str]:
    return [word for word in (text or "").lower().split() if word.isalpha()]

def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    return {w: count / total for w, count in Counter(ws).items()}

if __name__ == "__main__":
    document = Document(id="test_doc", vector=[0.5, 0.3, 0.2], text="This is a test document.")
    bandit_action = BanditAction(
        action_id="test_action",
        propensity=0.5,
        expected_reward=10.0,
        confidence_bound=0.1,
        algorithm="test_algorithm"
    )
    hybrid_action = hybrid_operation(document, bandit_action)
    print(hybrid_action)