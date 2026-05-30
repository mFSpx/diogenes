# DARWIN HAMMER — match 4675, survivor 5
# gen: 3
# parent_a: hybrid_regret_engine_hybrid_doomsday_cale_m19_s4.py (gen2)
# parent_b: sketches.py (gen0)
# born: 2026-05-29T23:57:21Z

"""Hybrid Regret‑Weighted Sketch Engine with Gini Fairness Metric
Parent A: regret_engine (probability distribution via regret‑adjusted softmax)
Parent B: sketches (count‑min sketch, HyperLogLog cardinality, MinHash LSH)

Mathematical Bridge
-------------------
* The regret engine consumes a vector **v** of action‑wise expected utilities and
  produces a normalized probability vector **p** = softmax(v̂) where v̂ incorporates
  counterfactual adjustments.
* The Gini coefficient **G(p)** is a scalar functional that quantifies inequality
  of any non‑negative distribution that sums to one.
* Sketches provide *approximate* frequencies **f̂** for a multiset of items.
  By treating **f̂** (or a monotone transform of it) as the raw expected value
  for each action, the sketch output becomes the input **v** to the regret engine.
* HyperLogLog supplies a cardinality estimate **C** that can be used as a scaling
  factor for the regret scores, while MinHash LSH groups actions with similar
  token sets, allowing per‑group Gini evaluation.

The code below fuses these components:
1. Approximate frequencies from a count‑min sketch are injected into the regret
   computation.
2. The resulting probability distribution is fed to the Gini calculator.
3. Groups derived from MinHash LSH receive their own regret‑weighted strategy
   and Gini score, optionally scaled by the HyperLogLog cardinality estimate."""

from __future__ import annotations

import hashlib
import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Dict, Set

import numpy as np

# ----------------------------------------------------------------------
# Data structures – from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """An action with an expected value and optional cost/risk penalties."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """A counterfactual adjustment for a specific action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Sketch utilities – from Parent B
# ----------------------------------------------------------------------
def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a depth×width count‑min sketch table for the given items."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16)
            col = h % width
            table[d][col] += 1
    return table


def hyperloglog_cardinality(items: Iterable[str]) -> int:
    """Very lightweight cardinality estimator (exact for small sets)."""
    return len(set(items))


def minhash_lsh_index(docs: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """
    Build a simple LSH index: each document is hashed to the minimum 6‑hex‑digit
    SHA‑1 prefix of its shingles; documents sharing the same prefix fall into the
    same bucket.
    """
    buckets: defaultdict[str, List[str]] = defaultdict(list)
    for doc_id, shingles in docs.items():
        key = min(
            (hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles),
            default="empty",
        )
        buckets[key].append(doc_id)
    return dict(buckets)


# ----------------------------------------------------------------------
# Core regret engine – distilled from Parent A
# ----------------------------------------------------------------------
def _regret_score(action: MathAction, counterfactuals: List[MathCounterfactual]) -> float:
    """Base regret score = expected - cost - risk + cf‑adjustments."""
    base = action.expected_value - action.cost - action.risk
    cf_adj = sum(
        cf.outcome_value * cf.probability
        for cf in counterfactuals
        if cf.action_id == action.id
    )
    return base + cf_adj


def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """
    Compute a softmax‑like probability distribution over actions where the logits
    are regret‑adjusted scores.
    """
    scores = np.array([_regret_score(a, counterfactuals) for a in actions], dtype=float)
    # Numerical stability
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    probs = exp_scores / exp_scores.sum()
    return {a.id: float(p) for a, p in zip(actions, probs)}


# ----------------------------------------------------------------------
# Gini coefficient – functional over a probability vector
# ----------------------------------------------------------------------
def gini_coefficient(probabilities: Dict[str, float]) -> float:
    """
    Compute the Gini coefficient of a discrete distribution.
    The input must sum to (approximately) 1.
    """
    values = np.array(list(probabilities.values()), dtype=float)
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)  # ascending
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    # Gini formula for discrete non‑negative values
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


# ----------------------------------------------------------------------
# Hybrid functions – fusion of regret engine + sketches
# ----------------------------------------------------------------------
def sketch_based_regret_strategy(
    actions: List[MathAction],
    items: Iterable[str],
    counterfactuals: List[MathCounterfactual],
    width: int = 64,
    depth: int = 4,
) -> Dict[str, float]:
    """
    1. Build a count‑min sketch from *items*.
    2. Estimate a frequency for each action.id by querying the sketch.
    3. Use the estimated frequency as an additive term to the action's
       expected_value before feeding it to the regret engine.
    Returns the regret‑weighted probability distribution.
    """
    sketch = count_min_sketch(items, width=width, depth=depth)

    # Helper to query sketch for a given key
    def query(key: str) -> int:
        cols = [
            sketch[d][int(hashlib.sha256(f"{d}:{key}".encode()).hexdigest(), 16) % width]
            for d in range(depth)
        ]
        return min(cols)  # count‑min estimate

    # Adjust expected values with sketch frequencies
    adjusted_actions = [
        MathAction(
            id=a.id,
            expected_value=a.expected_value + query(a.id),
            cost=a.cost,
            risk=a.risk,
        )
        for a in actions
    ]

    return compute_regret_weighted_strategy(adjusted_actions, counterfactuals)


def gini_of_sketch_regret(
    actions: List[MathAction],
    items: Iterable[str],
    counterfactuals: List[MathCounterfactual],
) -> float:
    """
    Compute Gini coefficient of the regret‑weighted distribution obtained via
    sketch‑based expected values.
    """
    probs = sketch_based_regret_strategy(actions, items, counterfactuals)
    return gini_coefficient(probs)


def grouped_regret_gini_via_minhash(
    actions: List[MathAction],
    docs: Dict[str, Set[str]],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """
    1. Bucket actions using MinHash LSH on the provided document shingles.
    2. For each bucket, compute a regret‑weighted strategy where the expected
       value of each action is scaled by the HyperLogLog cardinality of the
       bucket's items (a proxy for diversity).
    3. Return the Gini coefficient per bucket.
    """
    # Build LSH index
    buckets = minhash_lsh_index(docs)

    # Map action.id -> MathAction for quick lookup
    action_map = {a.id: a for a in actions}

    bucket_ginis: Dict[str, float] = {}

    for bucket_key, doc_ids in buckets.items():
        # Items belonging to this bucket (used for cardinality scaling)
        bucket_items = [doc_id for doc_id in doc_ids]

        # Cardinality estimate
        C = hyperloglog_cardinality(bucket_items) or 1  # avoid division by zero

        # Build adjusted actions for this bucket
        bucket_actions = [
            MathAction(
                id=aid,
                expected_value=action_map[aid].expected_value * C,
                cost=action_map[aid].cost,
                risk=action_map[aid].risk,
            )
            for aid in doc_ids
            if aid in action_map
        ]

        if not bucket_actions:
            continue

        probs = compute_regret_weighted_strategy(bucket_actions, counterfactuals)
        bucket_ginis[bucket_key] = gini_coefficient(probs)

    return bucket_ginis


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        MathAction(id="apple", expected_value=10.0, cost=2.0),
        MathAction(id="banana", expected_value=8.0, cost=1.0),
        MathAction(id="cherry", expected_value=6.0, cost=0.5),
        MathAction(id="date", expected_value=4.0, cost=0.2),
    ]

    # Counterfactuals (optional)
    counterfactuals = [
        MathCounterfactual(action_id="banana", outcome_value=1.5, probability=0.7),
        MathCounterfactual(action_id="date", outcome_value=-0.3, probability=0.9),
    ]

    # Synthetic stream of items (could be transaction logs, etc.)
    items = [
        "apple", "apple", "banana", "banana", "banana",
        "cherry", "date", "date", "date", "date", "date",
    ]

    # 1. Regret strategy using raw actions
    raw_probs = compute_regret_weighted_strategy(actions, counterfactuals)
    print("Raw regret probabilities:", raw_probs)

    # 2. Sketch‑augmented strategy + Gini
    sketch_probs = sketch_based_regret_strategy(actions, items, counterfactuals)
    print("Sketch‑based regret probabilities:", sketch_probs)
    print("Gini of sketch‑based distribution:", gini_of_sketch_regret(actions, items, counterfactuals))

    # 3. MinHash LSH grouping (documents = actions with token sets)
    docs = {
        "apple": {"fruit", "red", "sweet"},
        "banana": {"fruit", "yellow", "sweet"},
        "cherry": {"fruit", "red", "tart"},
        "date": {"fruit", "brown", "sweet"},
    }
    bucket_ginis = grouped_regret_gini_via_minhash(actions, docs, counterfactuals)
    print("Bucket‑wise Gini coefficients:", bucket_ginis)

    sys.exit(0)