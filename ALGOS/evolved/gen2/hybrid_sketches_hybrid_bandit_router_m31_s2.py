# DARWIN HAMMER — match 31, survivor 2
# gen: 2
# parent_a: sketches.py (gen0)
# parent_b: hybrid_bandit_router_honeybee_store_m9_s0.py (gen1)
# born: 2026-05-29T23:23:11Z

"""Hybrid Sketch-Bandit Algorithm
Parent A: sketches.py – count‑min sketch, hyperloglog cardinality, minhash LSH.
Parent B: hybrid_bandit_router_honeybee_store_m9_s0.py – contextual bandit action
selection with LinUCB/Epsilon‑Greedy/Thompson, and honeybee store dynamics.

Mathematical Bridge:
- The high‑dimensional context (a multiset of strings) is compressed by a
  count‑min sketch C∈ℕ^{depth×width}.  The ℓ₂‑norm of the sketch vector
  √∑_{i,j}C_{i,j}² provides a scale factor σ that replaces the ad‑hoc
  Euclidean norm used in the original bandit selector.
- The hyperloglog cardinality estimate |U| of the context set is used to
  adapt the exploration parameter ε←ε·log(1+|U|), linking the probabilistic
  set‑size estimator to the bandit’s exploration‑exploitation trade‑off.
- Actions are treated as “documents” of shingles; a minhash LSH bucket
  B(action) groups similar actions.  Policy statistics are stored per bucket
  rather than per raw action, fusing the LSH topology into the bandit’s
  reward‑propensity matrix.

The resulting hybrid system selects actions using a sketch‑derived scale,
updates a honeybee‑style store, and propagates rewards through LSH buckets.
"""

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Dict, Set, Tuple

import numpy as np

# ---------- Parent A components ----------
def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a depth×width count‑min sketch matrix for the given items."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table

def hyperloglog_cardinality(items: Iterable[str]) -> int:
    """Placeholder HLL: exact distinct count (used as a deterministic proxy)."""
    return len(set(items))

def minhash_lsh_index(docs: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Bucket documents by the minimum 6‑hex‑digit SHA‑1 hash of their shingles."""
    buckets = defaultdict(list)
    for doc_id, shingles in docs.items():
        key = min(
            (hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles),
            default="empty",
        )
        buckets[key].append(doc_id)
    return dict(buckets)

# ---------- Parent B components ----------
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

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])  # [cumulative reward, count]
        stats[0] += float(u.reward)
        stats[1] += 1.0

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def select_action(
    context_vec: List[float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """Select an action using the adapted LinUCB/Epsilon‑Greedy/Thompson rule."""
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Scale derived from the sketch norm
    scale = math.sqrt(sum(v * v for v in context_vec)) if context_vec else 1.0

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0, _reward(a)), 1 + max(0, 1 - _reward(a))
            ),
        )
    else:  # LinUCB‑style
        chosen = max(
            actions,
            key=lambda a: _reward(a)
            + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
        )
    return BanditAction(
        action_id=chosen,
        propensity=1.0 / len(actions),
        expected_reward=_reward(chosen),
        confidence_bound=1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1]),
        algorithm=algorithm,
    )

def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    """Honeybee store dynamics."""
    delta = alpha * sum(inflow) - beta * sum(outflow)
    return max(0.0, store + dt * delta), delta

def dance_duration(delta_store: float, base: float = 1.0, gain: float = 1.0, limit: float = 10.0) -> float:
    return max(0.0, min(limit, base + gain * delta_store))

# ---------- Hybrid Functions ----------
def hybrid_context_vector(
    items: Iterable[str], width: int = 64, depth: int = 4
) -> List[float]:
    """
    Produce a fixed‑size numeric vector from a multiset of strings.
    The vector is the flattened count‑min sketch (depth·width entries) cast to float.
    """
    sketch = count_min_sketch(items, width, depth)
    return [float(c) for row in sketch for c in row]

def hybrid_select_action(
    context_items: Iterable[str],
    actions: List[str],
    store: float,
    inflow: List[float],
    outflow: List[float],
    width: int = 64,
    depth: int = 4,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> Tuple[BanditAction, float]:
    """
    1. Compress the raw context via count‑min sketch → vector.
    2. Adjust exploration epsilon using hyperloglog cardinality.
    3. Select action with the adapted bandit rule.
    4. Update the honeybee store.
    Returns the chosen BanditAction and the new store level.
    """
    ctx_vec = hybrid_context_vector(context_items, width, depth)
    cardinality = hyperloglog_cardinality(context_items)
    adj_epsilon = epsilon * math.log(1 + cardinality)

    action = select_action(ctx_vec, actions, algorithm, adj_epsilon, seed)
    new_store, _ = update_store(store, inflow, outflow)
    return action, new_store

def hybrid_update_policy(
    updates: List[BanditUpdate],
    action_shingles: Dict[str, Set[str]],
) -> None:
    """
    Update the policy using LSH buckets:
    - Compute LSH buckets for actions.
    - Aggregate updates per bucket before feeding into the global policy.
    """
    buckets = minhash_lsh_index(action_shingles)  # bucket_key -> [action_ids]
    # Map each action to its bucket key for fast lookup
    action_to_bucket = {}
    for bucket_key, acts in buckets.items():
        for act in acts:
            action_to_bucket[act] = bucket_key

    # Aggregate rewards per bucket
    bucket_rewards: Dict[str, List[BanditUpdate]] = defaultdict(list)
    for u in updates:
        bucket = action_to_bucket.get(u.action_id, None)
        if bucket is not None:
            bucket_rewards[bucket].append(u)

    # Apply aggregated updates: treat bucket key as a pseudo‑action identifier
    for bucket_key, ups in bucket_rewards.items():
        # Summarize reward as average; propagate to each action in the bucket
        avg_reward = sum(u.reward for u in ups) / len(ups)
        for act in buckets[bucket_key]:
            update_policy([BanditUpdate(u.context_id, act, avg_reward, u.propensity) for u in ups])

def hybrid_optimize(
    context_items: Iterable[str],
    actions: List[str],
    store: float,
    inflow: List[float],
    outflow: List[float],
    action_shingles: Dict[str, Set[str]],
    width: int = 64,
    depth: int = 4,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> float:
    """
    Full optimization loop:
    - Select action with hybrid_select_action.
    - Simulate a reward (here deterministic: reward = 1.0 if action id ends with '1').
    - Update policy via LSH‑aware hybrid_update_policy.
    - Return the dance duration based on store change.
    """
    action, new_store = hybrid_select_action(
        context_items,
        actions,
        store,
        inflow,
        outflow,
        width,
        depth,
        algorithm,
        epsilon,
        seed,
    )

    # Simulated reward logic (placeholder)
    reward = 1.0 if action.action_id.endswith("1") else 0.0
    upd = BanditUpdate(
        context_id="ctx",
        action_id=action.action_id,
        reward=reward,
        propensity=action.propensity,
    )
    hybrid_update_policy([upd], action_shingles)

    delta = new_store - store
    return dance_duration(delta)

# ---------- Smoke Test ----------
if __name__ == "__main__":
    reset_policy()

    # Example context: a bag of words
    context_items = ["apple", "banana", "apple", "cherry", "date", "banana"]
    actions = ["action1", "action2", "action3"]
    store = 10.0
    inflow = [2.0, 1.5]
    outflow = [0.5, 0.2]

    # Shingles per action (simulating document fingerprints)
    action_shingles = {
        "action1": {"apple", "banana"},
        "action2": {"cherry", "date"},
        "action3": {"banana", "date"},
    }

    # Run a single optimization step
    result = hybrid_optimize(
        context_items,
        actions,
        store,
        inflow,
        outflow,
        action_shingles,
        width=32,
        depth=3,
        algorithm="linucb",
        epsilon=0.05,
        seed=42,
    )
    print(f"Hybrid dance duration: {result:.3f}")