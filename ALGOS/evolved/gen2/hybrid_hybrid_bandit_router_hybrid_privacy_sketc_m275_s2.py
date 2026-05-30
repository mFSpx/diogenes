# DARWIN HAMMER — match 275, survivor 2
# gen: 2
# parent_a: hybrid_bandit_router_honeybee_store_m9_s0.py (gen1)
# parent_b: hybrid_privacy_sketches_m15_s2.py (gen1)
# born: 2026-05-29T23:28:13Z

"""
Hybrid Bandit‑Sketch Privacy Store
=================================

This module fuses the core mathematics of the two parent algorithms:

* **Bandit‑Router / Honeybee‑Store** – a multi‑armed bandit that selects an
  action using an optimistic reward estimate and a store that evolves with
  inflow/outflow dynamics.
* **Privacy‑Sketches** – a Count‑Min Sketch (CMS) that estimates frequencies,
  a cardinality estimator derived from the CMS, and a reconstruction‑risk
  score that quantifies privacy exposure.

**Mathematical bridge**  
For every action we keep a dedicated CMS. The bandit’s *reward* is defined as
the privacy‑preserving utility of that action:


reward(action) = 1 - reconstruction_risk_score(
                     unique_quasi_identifiers(action),
                     total_records
                 )


`unique_quasi_identifiers` is approximated from the CMS by counting the
non‑zero cells (divided by sketch depth).  The bandit therefore prefers
actions whose sketches reveal fewer distinct identifiers (lower privacy risk).
The store’s inflow/outflow is interpreted as a privacy‑budget pool that is
consumed (outflow) when an action is executed and replenished (inflow) by a
global budget allocation.

The three public functions below demonstrate this hybrid operation:
`select_action`, `update_sketch_and_policy`, and `update_store`.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Iterable, Tuple

# ----------------------------------------------------------------------
# Data structures shared with the bandit side
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

# ----------------------------------------------------------------------
# Global policy storage (action_id -> [cumulative_reward, count])
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Erase all learned reward statistics."""
    _POLICY.clear()

def _policy_stats(action_id: str) -> Tuple[float, float]:
    """Return (total_reward, count) for the given action."""
    return tuple(_POLICY.get(action_id, [0.0, 0.0]))

def update_policy(updates: List[BanditUpdate]) -> None:
    """Incrementally incorporate reward observations."""
    for u in updates:
        total, cnt = _policy_stats(u.action_id)
        _POLICY[u.action_id] = [total + float(u.reward), cnt + 1.0]

def _reward(action_id: str) -> float:
    """Mean reward for an action (0 if never observed)."""
    total, cnt = _policy_stats(action_id)
    return total / cnt if cnt else 0.0

# ----------------------------------------------------------------------
# Count‑Min Sketch utilities (privacy side)
# ----------------------------------------------------------------------
def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Row‑wise column indices for a given item."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]

def count_min_sketch(items: Iterable[str],
                     width: int = 64,
                     depth: int = 4) -> np.ndarray:
    """Create a CMS matrix from an iterable of string items."""
    cms = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        cols = _cms_hash(item, depth, width)
        for d, c in enumerate(cols):
            cms[d, c] += 1
    return cms

def add_laplace_noise(cms: np.ndarray, epsilon: float, rng: random.Random) -> np.ndarray:
    """Add i.i.d. Laplace noise to each cell (zero‑mean, scale = 1/epsilon)."""
    scale = 1.0 / max(epsilon, 1e-12)
    noisy = cms.astype(np.float64)
    noise = rng.laplace(0.0, scale, size=cms.shape)  # type: ignore[attr-defined]
    noisy += noise
    return noisy

def _estimate_cardinality_from_cms(cms: np.ndarray) -> int:
    """Coarse cardinality: distinct (row, col) cells divided by depth."""
    nonzero = np.count_nonzero(cms)
    depth = cms.shape[0]
    return max(1, nonzero // depth)

def reconstruction_risk_score(unique_qi: int, total_records: int) -> float:
    """Privacy risk as ratio of quasi‑identifiers to total records, clipped."""
    if total_records <= 0:
        return 0.0
    ratio = unique_qi / total_records
    return max(0.0, min(1.0, ratio))

# ----------------------------------------------------------------------
# Hybrid bandit‑sketch functions
# ----------------------------------------------------------------------
def select_action(context: Dict[str, float],
                  actions: List[str],
                  sketches: Dict[str, np.ndarray],
                  total_records: int,
                  algorithm: str = 'linucb',
                  epsilon: float = 0.1,
                  seed: int | str | None = 7) -> BanditAction:
    """
    Choose an action using a bandit rule where the reward estimate is
    derived from the privacy‑risk of each action's sketch.

    Parameters
    ----------
    context: feature vector used only for scaling in the LinUCB branch.
    actions: list of candidate action identifiers.
    sketches: mapping action_id -> CMS matrix (may be empty for unseen actions).
    total_records: global record count used in the risk formula.
    algorithm: 'epsilon_greedy', 'thompson' or any other value for LinUCB‑like.
    epsilon: exploration probability for epsilon‑greedy.
    seed: random seed.

    Returns
    -------
    BanditAction describing the chosen arm and its statistics.
    """
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)

    # Compute privacy‑driven reward estimates on‑the‑fly
    for a in actions:
        cms = sketches.get(a)
        if cms is None:
            # unseen action → assume worst privacy (high risk → low reward)
            est_reward = 0.0
        else:
            uniq = _estimate_cardinality_from_cms(cms)
            risk = reconstruction_risk_score(uniq, total_records)
            est_reward = 1.0 - risk
        # sync with global policy (so future LinUCB uses observed means)
        total, cnt = _policy_stats(a)
        if cnt == 0:
            _POLICY[a] = [est_reward, 1.0]
        else:
            # simple exponential moving average could be used; here we just keep the mean
            _POLICY[a] = [total + est_reward, cnt + 1.0]

    # Bandit selection
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        # Beta posterior with successes = reward*count, failures = (1-reward)*count
        def sample(a):
            tot, cnt = _policy_stats(a)
            mean = tot / cnt if cnt else 0.0
            a_param = 1 + max(0, mean * cnt)
            b_param = 1 + max(0, (1 - mean) * cnt)
            return rng.betavariate(a_param, b_param)
        chosen = max(actions, key=sample)
    else:
        # LinUCB‑style optimism
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0
        def optimistic_value(a):
            mean = _reward(a)
            cnt = _POLICY.get(a, [0.0, 0.0])[1]
            return mean + 0.1 * scale / math.sqrt(1.0 + cnt)
        chosen = max(actions, key=optimistic_value)

    prop = 1.0 / len(actions)
    exp_rw = _reward(chosen)
    conf = 1.0 / math.sqrt(1.0 + _POLICY.get(chosen, [0.0, 0.0])[1])
    return BanditAction(chosen, prop, exp_rw, conf, algorithm)

def update_sketch_and_policy(action_id: str,
                             items: Iterable[str],
                             sketches: Dict[str, np.ndarray],
                             total_records: int,
                             epsilon: float = 1.0,
                             width: int = 64,
                             depth: int = 4,
                             rng_seed: int | str | None = None) -> None:
    """
    Incorporate new items into the CMS of ``action_id`` with differential‑privacy
    noise, then push a reward update derived from the resulting privacy risk.

    The function mutates ``sketches`` in place and records a ``BanditUpdate``
    in the global policy.
    """
    rng = random.Random(rng_seed)
    # Build raw sketch for the new items
    raw_cms = count_min_sketch(items, width=width, depth=depth)
    # Add Laplace noise for DP
    noisy_cms = add_laplace_noise(raw_cms, epsilon=epsilon, rng=rng)

    # Merge with existing sketch (if any)
    if action_id in sketches:
        sketches[action_id] = sketches[action_id] + noisy_cms
    else:
        sketches[action_id] = noisy_cms

    # Compute privacy‑risk based reward
    uniq = _estimate_cardinality_from_cms(sketches[action_id])
    risk = reconstruction_risk_score(uniq, total_records)
    reward = 1.0 - risk  # higher reward = lower risk

    # Record the observation
    update = BanditUpdate(context_id='global',
                          action_id=action_id,
                          reward=reward,
                          propensity=1.0)  # propensity is not used downstream
    update_policy([update])

def update_store(store: float,
                 inflow: List[float],
                 outflow: List[float],
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 dt: float = 1.0) -> Tuple[float, float]:
    """
    Classic honeybee‑store dynamics.

    Parameters
    ----------
    store: current resource (privacy‑budget) level.
    inflow: list of incoming budget contributions.
    outflow: list of consumed budget amounts.
    alpha, beta: scaling factors for inflow / outflow.
    dt: time step.

    Returns
    -------
    (new_store, delta) where ``delta`` is the net change before clipping.
    """
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

# ----------------------------------------------------------------------
# Demonstration helpers
# ----------------------------------------------------------------------
def hybrid_step(context: Dict[str, float],
                actions: List[str],
                sketches: Dict[str, np.ndarray],
                total_records: int,
                store: float,
                budget_per_step: float,
                rng_seed: int | None = 42) -> Tuple[BanditAction, float]:
    """
    Execute one hybrid iteration:
    1. Select an action via the privacy‑aware bandit.
    2. Simulate arrival of a few new items for that action.
    3. Update the sketch (with DP) and policy.
    4. Update the privacy‑budget store (consume ``budget_per_step``).

    Returns the chosen ``BanditAction`` and the updated store level.
    """
    # 1. selection
    chosen = select_action(context, actions, sketches, total_records,
                           algorithm='linucb', seed=rng_seed)

    # 2. generate dummy items (hashes of step + action)
    rng = random.Random(rng_seed)
    new_items = [f"{chosen.action_id}_item_{rng.randint(0, 1_000_000)}" for _ in range(5)]

    # 3. update sketch/policy (DP epsilon = 0.5 for demo)
    update_sketch_and_policy(chosen.action_id,
                             new_items,
                             sketches,
                             total_records,
                             epsilon=0.5,
                             rng_seed=rng_seed)

    # 4. budget dynamics
    inflow = []                     # no external funding in this demo
    outflow = [budget_per_step]    # consume a fixed amount per step
    new_store, _ = update_store(store, inflow, outflow, alpha=1.0, beta=1.0, dt=1.0)
    return chosen, new_store

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic run
    random.seed(0)
    reset_policy()

    actions = ["search", "recommend", "ads"]
    sketches: Dict[str, np.ndarray] = {}
    total_records = 10_000
    store = 100.0
    budget_per_step = 2.5
    context = {"user_age": 30, "session_length": 5.0}

    for step in range(5):
        chosen, store = hybrid_step(context,
                                    actions,
                                    sketches,
                                    total_records,
                                    store,
                                    budget_per_step,
                                    rng_seed=step)
        print(f"Step {step+1}: chosen={chosen.action_id}, "
              f"reward≈{chosen.expected_reward:.3f}, store={store:.2f}")

    # final policy snapshot
    print("\nFinal policy statistics:")
    for a in actions:
        total, cnt = _policy_stats(a)
        print(f"  {a}: avg_reward={total/cnt if cnt else 0.0:.3f}, count={int(cnt)}")