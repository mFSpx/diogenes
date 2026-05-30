# DARWIN HAMMER — match 2733, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py (gen2)
# parent_b: hard_truth_math.py (gen0)
# born: 2026-05-29T23:43:47Z

"""Hybrid Bandit‑Router & Stylometry Sketch Module

Parents:
- hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py (Bandit‑Router with Count‑Min sketch)
- hard_truth_math.py (Stylometry LSM vectors and scoring)

Mathematical bridge:
Both parents rely on *log‑count statistics*.  
The Bandit‑Router uses log‑frequency estimates from a Count‑Min sketch to drive
expected‑reward updates.  The Stylometry code aggregates token frequencies into
category‑level (LSM) vectors and scores them with a log‑scaled similarity measure.
The hybrid therefore:
1. Builds a Count‑Min sketch of the incoming token stream.
2. Retrieves *log‑estimated* frequencies for each token.
3. Aggregates those estimates into an approximate LSM vector.
4. Uses the LSM similarity (log‑scaled) as the *reward* for a chosen Bandit action.
5. Updates the Bandit policy with the reward, closing the loop between the two
   mathematical structures.

The core equations are:

- Sketch estimate: \(\hat f(x)=\min_{d}\;S_{d,h_d(x)}\)  
- Log‑count: \(c(x)=\log(\hat f(x)+1)\)  
- LSM component: \(v_{C}= \frac{\sum_{w\in C} c(w)}{\sum_{w} c(w)}\)  
- Reward (LSM similarity): \(r = \frac{1}{|C|}\sum_{C} \bigl[1-\frac{|a_C-b_C|}{a_C+b_C+\epsilon}\bigr]\)

These equations are fused in the functions below.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
import numpy as np
import re
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# Parent A – Bandit structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float = 0.0
    expected_reward: float = 0.0
    confidence_bound: float = 0.0
    algorithm: str = "hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float = 0.0


_POLICY: Dict[str, List[float]] = {}  # action_id -> [total_reward, count]


def reset_policy() -> None:
    _POLICY.clear()


def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incrementally update the global bandit statistics."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Parent A – Count‑Min sketch utilities
# ----------------------------------------------------------------------
def _hash(item: str, seed: int, width: int) -> int:
    """Simple deterministic hash for sketch rows."""
    h = hashlib.blake2b(digest_size=4, person=seed.to_bytes(4, "little"))
    h.update(item.encode("utf-8"))
    return int.from_bytes(h.digest(), "little") % width


def count_min_sketch(
    items: Iterable[str], width: int = 64, depth: int = 4
) -> List[List[int]]:
    """Count‑Min sketch of item frequencies."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = _hash(item, d, width)
            table[d][idx] += 1
    return table


def estimate_frequency(sketch: List[List[int]], item: str) -> int:
    """Return the minimum count over all hash rows (standard CM estimate)."""
    depth = len(sketch)
    width = len(sketch[0])
    mins = sys.maxsize
    for d in range(depth):
        idx = _hash(item, d, width)
        mins = min(mins, sketch[d][idx])
    return mins


# ----------------------------------------------------------------------
# Parent B – Stylometry utilities
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
        "no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def words(text: str) -> List[str]:
    """Tokenise lower‑cased alphabetic words."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Return average similarity across categories and per‑category detail."""
    detail: Dict[str, float] = {}
    scores: List[float] = []
    eps = 1e-6
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + eps))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
        scores.append(score)
    return sum(scores) / len(scores), detail


# ----------------------------------------------------------------------
# Hybrid core: sketch → LSM → reward → bandit
# ----------------------------------------------------------------------
def lsm_vector_from_sketch(
    sketch: List[List[int]],
    token_list: Iterable[str],
    width: int = 64,
    depth: int = 4,
) -> Dict[str, float]:
    """
    Approximate an LSM vector directly from a Count‑Min sketch.
    For each token we obtain a log‑estimated count, then aggregate per
    functional category.
    """
    # Gather log‑counts for all distinct tokens in the provided list
    log_counts: Dict[str, float] = {}
    total_log = 0.0
    for tok in token_list:
        freq = estimate_frequency(sketch, tok)
        lc = math.log(freq + 1)  # log‑count statistic
        log_counts[tok] = lc
        total_log += lc

    if total_log == 0.0:
        # empty or all zero – return uniform zero vector
        return {cat: 0.0 for cat in FUNCTION_CATS}

    # Aggregate into categories
    cat_vals: Dict[str, float] = {cat: 0.0 for cat in FUNCTION_CATS}
    for tok, lc in log_counts.items():
        for cat, vocab in FUNCTION_CATS.items():
            if tok in vocab:
                cat_vals[cat] += lc

    # Normalise by total log‑count
    return {cat: val / total_log for cat, val in cat_vals.items()}


def select_action_ucb(actions: List[BanditAction], alpha: float = 2.0) -> BanditAction:
    """
    Upper‑Confidence Bound (UCB) selector.
    Expected reward + alpha * sqrt(log(T)/n_i)
    where T is total pulls and n_i is pulls of action i.
    """
    total_pulls = sum(_count(a.action_id) for a in actions) + 1.0
    best = None
    best_score = -math.inf
    for a in actions:
        exp = _reward(a.action_id)
        n_i = _count(a.action_id) + 1e-6
        ucb = exp + alpha * math.sqrt(math.log(total_pulls) / n_i)
        if ucb > best_score:
            best_score = ucb
            best = a
    # Return a copy with populated fields for transparency
    return BanditAction(
        action_id=best.action_id,
        propensity=_count(best.action_id) / total_pulls,
        expected_reward=_reward(best.action_id),
        confidence_bound=best_score - _reward(best.action_id),
        algorithm="hybrid",
    )


def process_text(
    text: str,
    actions: List[BanditAction],
    target_lsm: Dict[str, float],
    width: int = 64,
    depth: int = 4,
) -> Tuple[BanditAction, BanditUpdate]:
    """
    End‑to‑end hybrid step:
    1. Sketch the token stream.
    2. Derive an approximate LSM vector.
    3. Compute similarity to a target LSM (reward).
    4. Choose an action via UCB.
    5. Return the chosen action and a BanditUpdate ready for policy update.
    """
    toks = words(text)
    sketch = count_min_sketch(toks, width=width, depth=depth)
    approx_lsm = lsm_vector_from_sketch(sketch, toks, width=width, depth=depth)
    reward, _detail = lsm_score(approx_lsm, target_lsm)

    chosen = select_action_ucb(actions)
    update = BanditUpdate(
        context_id=text[:30],
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
    )
    return chosen, update


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define three dummy actions
    actions = [
        BanditAction(action_id="reply_positive"),
        BanditAction(action_id="reply_neutral"),
        BanditAction(action_id="reply_negative"),
    ]

    # Target LSM: a stylometric fingerprint we wish to emulate
    target_text = "I think you are very clever and I love the way you solve problems."
    target_lsm = lsm_vector_from_sketch(
        count_min_sketch(words(target_text)), words(target_text)
    )

    # Simulated incoming messages
    messages = [
        "You are awesome! I really enjoy our chats.",
        "Well, that's not what I expected.",
        "I don't know if this works, but let's try.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for msg in messages:
        action, upd = process_text(msg, actions, target_lsm)
        print(f"Message: {msg!r}")
        print(f"  Chosen action: {action.action_id}")
        print(f"  Reward (LSM similarity): {upd.reward:.4f}")
        update_policy([upd])
        print("-" * 40)

    # Final policy snapshot
    print("Final policy statistics:")
    for aid, (total, cnt) in _POLICY.items():
        print(f"  {aid}: total_reward={total:.3f}, pulls={cnt:.0f}, avg={total/cnt:.3f}")