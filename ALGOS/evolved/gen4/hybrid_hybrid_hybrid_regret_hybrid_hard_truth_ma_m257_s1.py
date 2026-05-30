# DARWIN HAMMER — match 257, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s1.py (gen3)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s5.py (gen2)
# born: 2026-05-29T23:27:56Z

import numpy as np
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class MathAction:
    id: str; expected_value: float; cost: float=0.0; risk: float=0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str; outcome_value: float; probability: float=1.0

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

def words(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = {}
    for w in ws:
        cnt[w] = cnt.get(w, 0) + 1
    return {
        cat: sum(cnt.get(w, 0) for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def lsm_score(a: dict[str, float], b: dict[str, float]) -> tuple[float, dict[str, float]]:
    detail: dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
    overall = sum(detail.values()) / len(FUNCTION_CATS)
    return overall, detail

def hybrid_rw_tl_lsm(action: MathAction, lsm_vec: dict[str, float], 
                      reference_actions: list[MathAction], 
                      reference_lsm_vecs: list[dict[str, float]]) -> float:
    similarities = []
    for ref_action, ref_lsm_vec in zip(reference_actions, reference_lsm_vecs):
        similarity = int(action.id == ref_action.id)
        lsm_similarity = lsm_score(lsm_vec, ref_lsm_vec)[0]
        similarities.append(similarity * lsm_similarity)
    return np.mean(similarities)

def regret_weighted_strategy(actions: list[MathAction], 
                              lsm_vecs: list[dict[str, float]], 
                              reference_actions: list[MathAction], 
                              reference_lsm_vecs: list[dict[str, float]]) -> MathAction:
    best_action = actions[0]
    best_score = -np.inf
    for action, lsm_vec in zip(actions, lsm_vecs):
        score = hybrid_rw_tl_lsm(action, lsm_vec, reference_actions, reference_lsm_vecs)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action

def improved_regret_weighted_strategy(actions: list[MathAction], 
                                      lsm_vecs: list[dict[str, float]], 
                                      reference_actions: list[MathAction], 
                                      reference_lsm_vecs: list[dict[str, float]]) -> MathAction:
    weights = np.array([action.expected_value for action in actions])
    scores = np.array([hybrid_rw_tl_lsm(action, lsm_vec, reference_actions, reference_lsm_vecs) for action, lsm_vec in zip(actions, lsm_vecs)])
    weighted_scores = weights * scores
    best_action_index = np.argmax(weighted_scores)
    return actions[best_action_index]

if __name__ == "__main__":
    actions = [MathAction("action1", 0.5), MathAction("action2", 0.7), MathAction("action3", 0.3)]
    lsm_vecs = [lsm_vector("This is a test sentence"), 
                lsm_vector("This sentence is another test"), 
                lsm_vector("Test sentence with different words")]
    reference_actions = [MathAction("ref_action1", 0.2), MathAction("ref_action2", 0.6)]
    reference_lsm_vecs = [lsm_vector("Reference sentence 1"), lsm_vector("Reference sentence 2")]
    best_action = improved_regret_weighted_strategy(actions, lsm_vecs, reference_actions, reference_lsm_vecs)
    print(best_action.id)