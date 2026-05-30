# DARWIN HAMMER — match 5696, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_sketches_rlct_hybrid_hybrid_hybrid_m1211_s4.py (gen4)
# parent_b: hybrid_hybrid_bandit_router_hybrid_privacy_sketc_m275_s2.py (gen2)
# born: 2026-05-30T00:04:24Z

"""Hybrid RLCT‑Bandit Sketch Router
================================

Parents:
    - ``hybrid_hybrid_sketches_rlct_hybrid_hybrid_hybrid_m1211_s4.py`` (Algorithm A)
    - ``hybrid_hybrid_bandit_router_hybrid_privacy_sketc_m275_s2.py`` (Algorithm B)

Mathematical bridge
-------------------
Both parents rely on logarithmic scaling of observed quantities:

* **Algorithm A** estimates the Real Log Canonical Threshold (RLCT) by a linear
  regression of ``log(loss)`` on ``log(log(n))`` where *n* is the sample size.
* **Algorithm B** uses an Upper‑Confidence‑Bound (UCB) bandit where the confidence
  term is proportional to ``sqrt(log(total_counts) / action_counts)`` – again a
  log‑based scaling.

The hybrid therefore **shares the log‑space**: the RLCT estimate for each
action is injected as a *complexity penalty* into the UCB reward.  The reward
for an action *a* becomes


R_a = (1 - privacy_risk_a) - λ * RLCT_a_norm


where ``privacy_risk_a`` is derived from a Count‑Min Sketch (CMS) that
approximates the number of unique quasi‑identifiers, and ``RLCT_a_norm`` is the
RLCT value scaled to ``[0,1]`` across all actions.  The bandit then selects the
action with the highest ``UCB = R_a + sqrt(2*log(T)/N_a)`` where ``T`` is the
total number of selections and ``N_a`` the count for action *a*.

The code below fuses the core topologies: CMS handling, RLCT estimation, and
UCB bandit updates, providing three public functions that demonstrate the
hybrid operation.

"""

import hashlib
import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Sketch utilities (from Algorithm A)
# ----------------------------------------------------------------------
def count_min_sketch_update(sketch: List[List[int]], items: Iterable[str],
                            width: int = 64, depth: int = 4) -> None:
    """Increment CMS counters for each *item*."""
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16) % width
            sketch[d][idx] += 1


def count_min_sketch_init(width: int = 64, depth: int = 4) -> List[List[int]]:
    """Create an empty CMS."""
    return [[0] * width for _ in range(depth)]


def cms_estimate_cardinality(sketch: List[List[int]]) -> int:
    """Very crude cardinality estimate: number of non‑zero cells divided by depth."""
    nonzero = sum(1 for row in sketch for v in row if v > 0)
    return max(1, nonzero // len(sketch))


def cms_total_counts(sketch: List[List[int]]) -> int:
    """Total number of increments stored in the sketch."""
    return sum(sum(row) for row in sketch)


# ----------------------------------------------------------------------
# RLCT estimation (from Algorithm A)
# ----------------------------------------------------------------------
def estimate_rlct_from_losses(train_losses_per_n: List[float],
                              n_values: List[int]) -> float:
    """Linear regression of log(loss) on log(log(n))."""
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if np.any(ns <= math.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")

    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    var_x = (x_centered ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")

    slope = float((x_centered * y_centered).sum() / var_x)
    return slope  # the RLCT estimate


# ----------------------------------------------------------------------
# Bandit data structures (from Algorithm B, unified)
# ----------------------------------------------------------------------
@dataclass
class ActionState:
    """All mutable state kept per action."""
    sketch: List[List[int]] = field(default_factory=count_min_sketch_init)
    loss_history: List[Tuple[int, float]] = field(default_factory=list)  # (n, loss)
    cumulative_reward: float = 0.0
    count: int = 0
    expected_reward: float = 0.0  # running average of R_a
    rlct: float = 0.0            # latest RLCT estimate


# Global registry of actions
_ACTIONS: Dict[str, ActionState] = {}

# Global budget (privacy‑budget store) – simple scalar
_PRIVACY_BUDGET: float = 1.0


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def _ensure_action(action_id: str) -> ActionState:
    """Create a fresh ActionState if it does not exist."""
    if action_id not in _ACTIONS:
        _ACTIONS[action_id] = ActionState()
    return _ACTIONS[action_id]


def privacy_risk(action_id: str, total_records: int) -> float:
    """
    Approximate reconstruction‑risk score.
    Defined as the ratio of estimated unique quasi‑identifiers to the total records.
    """
    state = _ensure_action(action_id)
    uniq_est = cms_estimate_cardinality(state.sketch)
    return min(1.0, uniq_est / max(1, total_records))


def update_sketch_and_policy(action_id: str,
                             new_items: Iterable[str],
                             n: int,
                             loss: float,
                             lambda_rlct: float = 0.5) -> None:
    """
    Hybrid update routine:

    1. Update the action's Count‑Min Sketch with *new_items*.
    2. Append (n, loss) to the loss history and recompute the RLCT.
    3. Compute a privacy‑aware reward:
           r = (1 - privacy_risk) - λ * normalized_RLCT
    4. Update cumulative reward, count, and expected reward.
    """
    state = _ensure_action(action_id)

    # ---- Sketch update -------------------------------------------------
    count_min_sketch_update(state.sketch, new_items)

    # ---- Loss history & RLCT -------------------------------------------
    state.loss_history.append((n, loss))
    if len(state.loss_history) >= 2:
        ns, losses = zip(*state.loss_history)
        try:
            rlct_est = estimate_rlct_from_losses(list(losses), list(ns))
        except Exception:
            rlct_est = 0.0
    else:
        rlct_est = 0.0
    state.rlct = rlct_est

    # ---- Reward computation --------------------------------------------
    total_counts = cms_total_counts(state.sketch)
    risk = privacy_risk(action_id, total_counts)

    # Normalise RLCT across all actions (simple min‑max scaling)
    all_rlcts = [a.rlct for a in _ACTIONS.values()]
    rlct_min, rlct_max = (min(all_rlcts), max(all_rlcts)) if all_rlcts else (0.0, 1.0)
    if rlct_max - rlct_min > 1e-9:
        rlct_norm = (rlct_est - rlct_min) / (rlct_max - rlct_min)
    else:
        rlct_norm = 0.0

    reward = (1.0 - risk) - lambda_rlct * rlct_norm
    reward = max(0.0, min(1.0, reward))  # clamp to [0,1]

    # ---- Policy update -------------------------------------------------
    state.cumulative_reward += reward
    state.count += 1
    state.expected_reward = state.cumulative_reward / state.count


def select_action(context_id: str,
                  candidate_action_ids: List[str],
                  exploration_coef: float = 2.0) -> str:
    """
    Upper‑Confidence‑Bound (UCB) selection that incorporates the RLCT penalty.
    Returns the action_id with the highest UCB value.
    """
    total_selections = sum(_ensure_action(a).count for a in candidate_action_ids) + 1e-9
    best_action = None
    best_score = -math.inf

    for aid in candidate_action_ids:
        state = _ensure_action(aid)
        # Expected reward already contains privacy‑risk and RLCT penalty.
        avg = state.expected_reward
        # Classic UCB confidence term.
        confidence = math.sqrt(exploration_coef * math.log(total_selections) / (state.count + 1e-9))
        ucb = avg + confidence
        if ucb > best_score:
            best_score = ucb
            best_action = aid

    return best_action


def update_store(consumed: float) -> None:
    """
    Simple privacy‑budget store update.
    *consumed* is subtracted from the global budget; the budget is never allowed
    to become negative.
    """
    global _PRIVACY_BUDGET
    _PRIVACY_BUDGET = max(0.0, _PRIVACY_BUDGET - consumed)


def get_action_statistics(action_id: str) -> Dict[str, Any]:
    """
    Return a dictionary with the current statistics of *action_id*.
    Useful for debugging / monitoring.
    """
    state = _ensure_action(action_id)
    return {
        "count": state.count,
        "expected_reward": state.expected_reward,
        "rlct": state.rlct,
        "privacy_risk": privacy_risk(action_id, cms_total_counts(state.sketch)),
        "total_sketch_counts": cms_total_counts(state.sketch),
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny set of actions
    actions = ["A", "B", "C"]
    # Simulate 20 rounds of interaction
    for t in range(20):
        # Randomly generate a context (unused in this simple demo)
        ctx = f"ctx_{t}"
        # Choose action via hybrid UCB
        chosen = select_action(ctx, actions)
        # Simulate incoming identifiers (strings) and a synthetic loss
        identifiers = [f"user_{random.randint(0, 50)}" for _ in range(random.randint(5, 15))]
        n_samples = random.randint(50, 200)
        loss = random.random() * math.log(n_samples)  # arbitrary loss decreasing with n
        # Perform hybrid update
        update_sketch_and_policy(chosen, identifiers, n_samples, loss)
        # Consume a small privacy budget proportional to sketch size
        update_store(consumed=0.001)

    # Print final statistics
    for aid in actions:
        stats = get_action_statistics(aid)
        print(f"Action {aid}: {stats}")

    print(f"Remaining privacy budget: {_PRIVACY_BUDGET:.4f}")