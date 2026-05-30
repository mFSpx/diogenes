# DARWIN HAMMER — match 2169, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hoeffding_tre_m16_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_endpoi_m403_s2.py (gen5)
# born: 2026-05-29T23:41:07Z

"""Hybrid Bandit‑Tree Algorithm
Parents:
- hybrid_hybrid_hybrid_sketch_hybrid_hoeffding_tre_m16_s2.py (count‑min sketch, Hoeffding bound, tropical max‑plus algebra)
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_endpoi_m403_s2.py (bandit router, propensity scores, policy updates)

Mathematical bridge:
The count‑min sketch provides a low‑dimensional, non‑negative matrix **S** that aggregates
(context, action) co‑occurrences.  Bandit reward estimates **rₐ** are scalar values stored per
action.  Using tropical (max‑plus) algebra we combine **S** and **rₐ** via the tropical
matrix‑vector product  

  **scoreₐ = max_i ( S_{i, h(a)} + rₐ )  

where *h(a)* is a hash‑based column index.  This yields a piecewise‑linear convex score that
captures both frequency (via the sketch) and expected reward (via the bandit).  The Hoeffding
bound supplies a statistical confidence interval on the gain gap between the best and second‑best
scores, enabling a principled split/selection decision analogous to the tree‑splitting criterion.

The module implements the fused operations, exposing three core functions:
1. `count_min_sketch` – builds the sketch matrix.
2. `tropical_score` – computes tropical max‑plus scores for actions.
3. `select_hybrid_action` – chooses an action using ε‑greedy exploration guided by Hoeffding‑based
   confidence on the tropical scores.

Policy updates and sketch maintenance are also provided.
"""

import hashlib
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Global policy store (mirrors parent B)
_POLICY: Dict[str, List[float]] = {}
# ----------------------------------------------------------------------


def reset_policy() -> None:
    """Clear all stored rewards and counts."""
    _POLICY.clear()


def update_policy(updates: List[Tuple[str, float]]) -> None:
    """
    Update the reward statistics for actions.

    Parameters
    ----------
    updates : List[Tuple[str, float]]
        Each tuple contains (action_id, reward_observed).
    """
    for action_id, reward in updates:
        stats = _POLICY.setdefault(action_id, [0.0, 0.0])  # [total_reward, count]
        stats[0] += float(reward)
        stats[1] += 1.0


def _reward(action_id: str) -> float:
    """Return the empirical mean reward for an action."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n > 0 else 0.0


# ----------------------------------------------------------------------
# Count‑Min Sketch (parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """
    Build a Count‑Min sketch matrix for a list of hashable items.

    Returns a depth×width integer matrix.
    """
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][h] += 1
    return table


# ----------------------------------------------------------------------
# Tropical (max‑plus) algebra utilities (parent A)
# ----------------------------------------------------------------------
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition: element‑wise maximum."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication: element‑wise sum."""
    return np.add(x, y)


def t_matmul(A: List[List[int]], v: List[float]) -> np.ndarray:
    """
    Tropical matrix‑vector product (max‑plus).

    For each row i of A, compute max_j (A[i][j] + v[j]).
    """
    A_arr = np.asarray(A, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    # Broadcast addition then max over columns
    return np.max(A_arr + v_arr, axis=1)


# ----------------------------------------------------------------------
# Hoeffding bound utilities (parent A)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """
    Compute Hoeffding bound for bounded random variable in [0, r].

    Parameters
    ----------
    r : float
        Range of the variable.
    delta : float
        Desired failure probability (0 < delta < 1).
    n : int
        Number of independent observations.

    Returns
    -------
    float
        The epsilon bound.
    """
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(best_gain: float, second_best_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> SplitDecision:
    """
    Decide whether the observed gain gap justifies a split (or, in the hybrid,
    a confident exploitation of the best action).

    Returns a SplitDecision dataclass.
    """
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def tropical_score(sketch: List[List[int]], actions: List[str]) -> Dict[str, float]:
    """
    Compute tropical max‑plus scores for a set of actions.

    For each action a, the column index c = hash(a) % width.
    The score is max_i ( sketch[i][c] + μₐ ), where μₐ is the empirical mean reward.

    Returns a mapping action → score.
    """
    width = len(sketch[0])
    depth = len(sketch)
    scores: Dict[str, float] = {}
    for a in actions:
        col = int(hashlib.sha256(a.encode()).hexdigest(), 16) % width
        col_vals = [sketch[i][col] for i in range(depth)]
        mu = _reward(a)
        scores[a] = max(v + mu for v in col_vals)
    return scores


def select_hybrid_action(context_items: List[str],
                         actions: List[str],
                         epsilon: float = 0.1,
                         algorithm: str = "epsilon_greedy",
                         delta: float = 0.05,
                         r: float = 1.0,
                         seed: int | str | None = 7) -> Tuple[str, float]:
    """
    Choose an action given a context.

    The procedure:
    1. Build a count‑min sketch of the context.
    2. Compute tropical scores using the sketch and current reward estimates.
    3. Apply Hoeffding‑based confidence to decide whether the best action is
       statistically superior; if not, fall back to the requested exploration
       strategy.

    Returns a tuple (chosen_action, score_of_chosen_action).
    """
    if not actions:
        raise ValueError("actions list must not be empty")

    rng = random.Random(seed)

    # 1. Sketch the context
    sketch = count_min_sketch(context_items)

    # 2. Tropical scores
    scores = tropical_score(sketch, actions)
    sorted_actions = sorted(actions, key=lambda a: scores[a], reverse=True)
    best, second = sorted_actions[0], sorted_actions[1] if len(sorted_actions) > 1 else None
    best_score = scores[best]
    second_score = scores[second] if second is not None else -math.inf

    # 3. Confidence check using Hoeffding bound
    decision = should_split(best_score, second_score, r, delta, n=len(context_items))
    if decision.should_split:
        # Confident enough to exploit the best action
        return best, best_score
    else:
        # Not confident: use exploration algorithm
        if algorithm == "epsilon_greedy" and rng.random() < epsilon:
            chosen = rng.choice(actions)
        else:
            # Default to greedy fallback when not exploring
            chosen = best
        return chosen, scores[chosen]


def hybrid_update(updates: List[Tuple[str, float]],
                  context_items: List[str],
                  action_id: str) -> None:
    """
    Perform a joint update:
    - Update the bandit reward statistics.
    - Increment the sketch counters for the (context_item, action) pair.

    Parameters
    ----------
    updates : List[Tuple[str, float]]
        Rewards to feed into the policy (action_id, reward) tuples.
    context_items : List[str]
        The raw context items that contributed to the decision.
    action_id : str
        The action that was taken (used for sketch hashing).
    """
    # Update policy statistics
    update_policy(updates)

    # Update sketch counts for each context item paired with the action
    # (We reuse the global _POLICY only for rewards; the sketch is transient per call,
    #  but we keep a lightweight persistent sketch for illustration.)
    # For simplicity, we store a global sketch matrix.
    global _GLOBAL_SKETCH
    if '_GLOBAL_SKETCH' not in globals():
        _GLOBAL_SKETCH = count_min_sketch([])  # empty initialization

    width = len(_GLOBAL_SKETCH[0])
    depth = len(_GLOBAL_SKETCH)
    for item in context_items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}:{action_id}".encode()).hexdigest(), 16) % width
            _GLOBAL_SKETCH[d][h] += 1


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic scenario
    context = ["user:alice", "device:mobile", "hour:14"]
    actions = ["click_ad", "skip_ad", "show_offer"]

    # Simulate a few rounds of interaction
    for round_idx in range(5):
        chosen, score = select_hybrid_action(
            context_items=context,
            actions=actions,
            epsilon=0.2,
            algorithm="epsilon_greedy",
            delta=0.1,
            r=1.0,
            seed=round_idx,
        )
        # Fake reward: +1 if chosen is "click_ad", else 0
        reward = 1.0 if chosen == "click_ad" else 0.0
        hybrid_update(
            updates=[(chosen, reward)],
            context_items=context,
            action_id=chosen,
        )
        print(f"Round {round_idx+1}: chosen={chosen}, score={score:.3f}, reward={reward}")

    # Show final policy estimates
    print("\nFinal empirical rewards:")
    for a in actions:
        print(f"  {a}: { _reward(a):.3f }")