# DARWIN HAMMER — match 551, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s2.py (gen3)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py (gen2)
# born: 2026-05-29T23:29:46Z

"""Hybrid LSM‑Signature Store‑Bandit Fusion

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s2.py (Bandit‑Router + Path‑Signature‑KAN)
- hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py (Function‑word LSM similarity)

Mathematical Bridge
-------------------
The LSM similarity between two texts yields a *category‑wise* score vector
`detail[cat] ∈ [0,1]`.  We interpret each category score as a *flow*:
high similarity → inflow, low similarity → outflow.  The inflow/outflow
vectors are fed into the store dynamics of Parent A:

    Δ   = α·Σ(inflow) – β·Σ(outflow)
    lvl = max(0, lvl + Δ·dt)
    dance = tanh(gain·Δ)

The scalar `dance` then rescales the raw bandit propensities, closing the
loop.  Thus textual similarity modulates a stochastic decision policy
through a shared store, mathematically fusing the two topologies into a
single system.

The module provides three core hybrid functions:
1. `compute_lsm_flow` – turn two texts into inflow/outflow vectors.
2. `store_update_from_flow` – update the store and emit the dance signal.
3. `adjust_bandit_propensities` – rescale bandit propensities with dance.

A tiny smoke test at the bottom runs a full hybrid cycle."""
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float = 0.0
    confidence_bound: float = 0.0
    algorithm: str = "hybrid_lsm_store_bandit"


@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee‑style store and derived control signal."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0  # used as gain for the dance non‑linearity


# ----------------------------------------------------------------------
# Parent‑B lexical similarity utilities
# ----------------------------------------------------------------------
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
        "no not never none neither cannot can't won't don't didn't isn't aren't wasnt weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def words(text: str) -> List[str]:
    """Tokenise a string into lower‑case words."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a low‑dimensional lexical functional word (LSM) vector.
    Each entry is the relative frequency of a functional‑word category.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Return overall similarity and per‑category detail.
    Similarity per category is 1 - normalized absolute difference.
    """
    detail: Dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
    overall = sum(detail.values()) / len(FUNCTION_CATS)
    return overall, detail


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_lsm_flow(text_a: str, text_b: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert two texts into inflow and outflow vectors for the store.

    For each functional category we treat the LSM similarity `s ∈ [0,1]` as
    an inflow magnitude.  The complementary outflow is `1‑s`.  The resulting
    vectors have shape (C,) where C = number of categories.
    """
    vec_a = lsm_vector(text_a)
    vec_b = lsm_vector(text_b)
    _, detail = lsm_score(vec_a, vec_b)
    inflow = np.array([detail[cat] for cat in sorted(FUNCTION_CATS.keys())], dtype=float)
    outflow = 1.0 - inflow
    return inflow, outflow


def store_update_from_flow(store: StoreState, inflow: np.ndarray, outflow: np.ndarray) -> float:
    """
    Update the store level using the hybrid equation from Parent A and
    return the generated `dance` signal.

    Δ = α·Σ(inflow) – β·Σ(outflow)
    level_{t+1} = max(0, level_t + Δ·dt)
    dance = tanh(gain·Δ)   with gain = store.base
    """
    delta = store.alpha * inflow.sum() - store.beta * outflow.sum()
    store.level = max(0.0, store.level + delta * store.dt)
    dance = math.tanh(store.base * delta)
    return dance


def adjust_bandit_propensities(actions: List[BanditAction], dance: float) -> List[BanditAction]:
    """
    Rescale each action's raw propensity by a linear factor derived from
    the store's `dance` signal.

    new_propensity = propensity * (1 + λ·dance)
    where λ is a modest scaling constant (here 0.5) that keeps propensities
    positive and bounded.
    """
    lam = 0.5
    adjusted: List[BanditAction] = []
    for act in actions:
        factor = 1.0 + lam * dance
        new_prop = max(0.0, act.propensity * factor)
        adjusted.append(
            BanditAction(
                action_id=act.action_id,
                propensity=new_prop,
                expected_reward=act.expected_reward,
                confidence_bound=act.confidence_bound,
                algorithm=act.algorithm,
            )
        )
    return adjusted


def sample_action(actions: List[BanditAction]) -> BanditAction:
    """
    Stochastic selection proportional to propensities.
    If all propensities are zero we fall back to uniform random choice.
    """
    total = sum(a.propensity for a in actions)
    if total <= 0.0:
        return random.choice(actions)
    r = random.random() * total
    cum = 0.0
    for a in actions:
        cum += a.propensity
        if r <= cum:
            return a
    return actions[-1]  # fallback due to floating‑point rounding


def hybrid_cycle(
    text_a: str,
    text_b: str,
    actions: List[BanditAction],
    store: StoreState,
) -> Tuple[BanditAction, StoreState, List[BanditAction]]:
    """
    Execute one hybrid iteration:
    1. Derive inflow/outflow from the two texts.
    2. Update the store and obtain the dance signal.
    3. Rescale bandit propensities.
    4. Sample an action.
    Returns the sampled action, the updated store, and the rescaled actions.
    """
    inflow, outflow = compute_lsm_flow(text_a, text_b)
    dance = store_update_from_flow(store, inflow, outflow)
    new_actions = adjust_bandit_propensities(actions, dance)
    chosen = sample_action(new_actions)
    return chosen, store, new_actions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example texts
    txt1 = "I think that we should not go there because it is too far."
    txt2 = "You shouldn't go there; it's far away and not worth the effort."

    # Initialise a store
    my_store = StoreState(level=0.5, alpha=1.2, beta=0.8, dt=0.5, base=1.0)

    # Define a tiny action set
    raw_actions = [
        BanditAction(action_id="accept", propensity=0.6),
        BanditAction(action_id="reject", propensity=0.4),
    ]

    # Run a single hybrid iteration
    chosen_action, updated_store, rescaled_actions = hybrid_cycle(
        txt1, txt2, raw_actions, my_store
    )

    # Simple reporting
    print("Chosen action:", chosen_action.action_id)
    print("Store level:", updated_store.level)
    print("Dance signal (implicit):", math.tanh(updated_store.base * (updated_store.alpha * sum(compute_lsm_flow(txt1, txt2)[0]) - updated_store.beta * sum(compute_lsm_flow(txt1, txt2)[1]))))
    print("Rescaled propensities:", [(a.action_id, a.propensity) for a in rescaled_actions])