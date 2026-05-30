# DARWIN HAMMER — match 275, survivor 3
# gen: 2
# parent_a: hybrid_bandit_router_honeybee_store_m9_s0.py (gen1)
# parent_b: hybrid_privacy_sketches_m15_s2.py (gen1)
# born: 2026-05-29T23:28:13Z

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
# Global policy storage (action_id -> [cumulative_reward, count, total_propensity])
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Erase all learned reward statistics."""
    _POLICY.clear()

def _policy_stats(action_id: str) -> Tuple[float, float, float]:
    """Return (total_reward, count, total_propensity) for the given action."""
    return tuple(_POLICY.get(action_id, [0.0, 0.0, 0.0]))

def update_policy(updates: List[BanditUpdate]) -> None:
    """Incrementally incorporate reward observations."""
    for u in updates:
        total, cnt, total_prop = _policy_stats(u.action_id)
        _POLICY[u.action_id] = [total + float(u.reward), cnt + 1.0, total_prop + u.propensity]

def _reward(action_id: str) -> float:
    """Mean reward for an action (0 if never observed)."""
    total, cnt, _ = _policy_stats(action_id)
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
    noise = rng.laplace(0.0, scale, size=cms.shape)  
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
        total, cnt, _ = _policy_stats(a)
        if cnt == 0:
            _POLICY[a] = [est_reward, 1.0, 1.0 / len(actions)]
        else:
            # simple exponential moving average could be used; here we just keep the mean
            _POLICY[a] = [total + est_reward, cnt + 1.0, _policy_stats(a)[2] + 1.0 / len(actions)]

    # Bandit selection
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        # Beta posterior with successes = reward*count, failures = (1-reward)*count
        def sample(a):
            tot, cnt, _ = _policy_stats(a)
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
            cnt = _POLICY.get(a, [0.0, 0.0, 0.0])[1]
            return mean + 0.1 * scale / math.sqrt(1.0 + cnt)
        chosen = max(actions, key=optimistic_value)

    prop = 1.0 / len(actions)
    exp_rw = _reward(chosen)
    conf = 1.0 / math.sqrt(1.0 + _POLICY.get(chosen, [0.0, 0.0, 0.0])[1])
    return BanditAction(chosen, prop, exp_rw, conf, algorithm)

def update_sketch_and_policy(action_id: str,
                             items: Iterable[str],
                             sketches: Dict[str, np.ndarray],
                             total_records: int,
                             epsilon: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Update the Count-Min Sketch for the given action and return the updated sketch.

    Parameters
    ----------
    action_id: identifier of the action.
    items: iterable of string items to add to the sketch.
    sketches: mapping action_id -> CMS matrix.
    total_records: global record count used in the risk formula.
    epsilon: privacy budget.

    Returns
    -------
    Updated CMS matrix and the estimated privacy risk.
    """
    cms = count_min_sketch(items)
    noisy_cms = add_laplace_noise(cms, epsilon, random.Random())
    sketches[action_id] = noisy_cms
    uniq = _estimate_cardinality_from_cms(noisy_cms)
    risk = reconstruction_risk_score(uniq, total_records)
    return noisy_cms, risk

def update_store(action_id: str,
                store: Dict[str, float],
                inflow: float) -> None:
    """
    Update the store with the given inflow.

    Parameters
    ----------
    action_id: identifier of the action.
    store: mapping action_id -> store value.
    inflow: inflow value.
    """
    if action_id not in store:
        store[action_id] = 0.0
    store[action_id] += inflow