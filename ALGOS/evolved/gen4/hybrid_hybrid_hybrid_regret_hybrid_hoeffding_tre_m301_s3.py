# DARWIN HAMMER — match 301, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s2.py (gen3)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s2.py (gen1)
# born: 2026-05-29T23:28:10Z

"""
Hybrid Regret‑Weighted Hoeffding‑Gini Engine

Parents:
- hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s2.py (Regret‑Weighted strategy + MinHash ternary vector)
- hybrid_hoeffding_tree_gini_coefficient_m13_s2.py (Hoeffding bound + Gini‑weighted split decision)

Mathematical bridge:
The regret of each action forms a non‑negative distribution.  The Gini coefficient of this
distribution quantifies inequality among regrets.  By feeding the Gini‑scaled regret
vector into the Hoeffding bound we obtain a statistically sound split criterion that
operates on the MinHash signature space used by the ternary‑lens component.  Thus the
signature similarity guides the construction of candidate splits while the Gini‑weighted
regret supplies the confidence term for Hoeffding‑based decisions.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Dict
import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ----------------------------------------------------------------------
# Helper functions (Parent A)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity on MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

# ----------------------------------------------------------------------
# Hoeffding & Gini utilities (Parent B)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_regret_weights(actions: List[MathAction],
                           counterfactuals: List[MathCounterfactual]) -> Dict[str, float]:
    """
    Regret‑weighted probability for each action.
    Regret = expected_value - Σ outcome_value * probability over counterfactuals.
    The raw regrets are passed through a sigmoid to obtain a probability‑like weight.
    """
    # Aggregate counterfactual contributions per action
    cf_map: Dict[str, float] = {}
    for cf in counterfactuals:
        cf_map.setdefault(cf.action_id, 0.0)
        cf_map[cf.action_id] += cf.outcome_value * cf.probability

    regrets = {}
    for act in actions:
        cf_sum = cf_map.get(act.id, 0.0)
        regret = act.expected_value - cf_sum - act.cost - act.risk
        regrets[act.id] = max(regret, 0.0)  # non‑negative regret

    # Convert regrets to a probability distribution via sigmoid
    regret_array = np.array(list(regrets.values()))
    weights = sigmoid(regret_array)
    # Normalize to sum to 1 (optional but convenient)
    if weights.sum() == 0:
        normalized = {aid: 0.0 for aid in regrets}
    else:
        normalized = {aid: w / weights.sum() for aid, w in zip(regrets.keys(), weights)}
    return normalized

def gini_weighted_regret_signature(actions: List[MathAction],
                                   counterfactuals: List[MathCounterfactual],
                                   k: int = 64) -> List[int]:
    """
    Combine regret weighting with MinHash signature.
    The action ids are hashed; the resulting signature is then modulated by the Gini
    coefficient of the regret distribution, yielding a signature that reflects both
    content similarity and inequality of regrets.
    """
    # 1. Compute regret weights (probability‑like)
    weights = compute_regret_weights(actions, counterfactuals)
    # 2. Build a token list where each action id appears proportionally to its weight
    #    (rounded to nearest integer, at least one copy to keep token alive)
    tokens = []
    for act in actions:
        multiplicity = max(1, int(round(weights[act.id] * 10)))  # scale factor 10 for demo
        tokens.extend([act.id] * multiplicity)

    # 3. MinHash signature on the weighted token list
    raw_sig = signature(tokens, k=k)

    # 4. Compute Gini on the raw regret values (not the sigmoided weights)
    regret_vals = [act.expected_value - sum(
        cf.outcome_value * cf.probability
        for cf in counterfactuals if cf.action_id == act.id) - act.cost - act.risk
                   for act in actions]
    gini = gini_coefficient([max(v, 0.0) for v in regret_vals])

    # 5. Modulate each component of the signature by (1‑gini) to shrink the
    #    influence of highly unequal regret distributions.
    modulated = [int(val * (1.0 - gini)) for val in raw_sig]
    return modulated

def hoeffding_gini_split_decision(regret_weights: Dict[str, float],
                                  r: float = 1.0,
                                  delta: float = 0.05,
                                  n: int = 100,
                                  tie_threshold: float = 0.05) -> SplitDecision:
    """
    Use Hoeffding bound together with the Gini coefficient of the regret weight
    distribution to decide whether a split (e.g., a decision node in a tree of actions)
    is statistically justified.

    best_gain and second_best_gain are approximated by the two largest regret weights.
    """
    if not regret_weights:
        return SplitDecision(False, 0.0, 0.0, "no_data")

    sorted_weights = sorted(regret_weights.values(), reverse=True)
    best_gain = sorted_weights[0]
    second_best_gain = sorted_weights[1] if len(sorted_weights) > 1 else 0.0

    gini = gini_coefficient(list(regret_weights.values()))
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    gini_weighted_gap = gap * (1 - gini)

    split = gini_weighted_gap > eps or eps < tie_threshold
    reason = ("gini_weighted_gap_exceeds_bound"
              if gini_weighted_gap > eps else
              ("tie_threshold" if eps < tie_threshold else "await_more_data"))
    return SplitDecision(split, eps, gini_weighted_gap, reason)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic scenario
    actions = [
        MathAction(id="A", expected_value=10.0, cost=1.0, risk=0.5),
        MathAction(id="B", expected_value=8.0, cost=0.5, risk=0.2),
        MathAction(id="C", expected_value=5.0, cost=0.2, risk=0.1),
    ]

    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=7.0, probability=0.6),
        MathCounterfactual(action_id="A", outcome_value=6.0, probability=0.4),
        MathCounterfactual(action_id="B", outcome_value=5.0, probability=0.7),
        MathCounterfactual(action_id="C", outcome_value=4.0, probability=0.9),
    ]

    # 1. Regret weights
    rw = compute_regret_weights(actions, counterfactuals)
    print("Regret weights:", rw)

    # 2. Gini‑weighted signature
    sig = gini_weighted_regret_signature(actions, counterfactuals, k=32)
    print("Gini‑weighted MinHash signature (len=32):", sig[:8], "...")

    # 3. Hoeffding‑Gini split decision
    decision = hoeffding_gini_split_decision(rw, r=1.0, delta=0.05, n=200)
    print("Split decision:", decision)
    sys.exit(0)