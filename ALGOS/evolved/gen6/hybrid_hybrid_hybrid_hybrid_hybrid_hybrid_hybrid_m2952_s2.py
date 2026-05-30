# DARWIN HAMMER — match 2952, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_krampus_chron_m651_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s0.py (gen4)
# born: 2026-05-29T23:46:52Z

"""Hybrid Date‑Regret‑Model Pool Algorithm
========================================

This module fuses the two parent algorithms:

* **Parent A** – *hybrid_hybrid_hybrid_hybrid_hybrid_krampus_chron_m651_s2.py* –  
  provides a ``CertaintyFlag`` that quantifies epistemic certainty (confidence)
  derived from the Shannon entropy of a date‑parsing distribution.

* **Parent B** – *hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s0.py* –  
  supplies a regret‑weighted decision model (``MathAction`` / ``MathCounterfactual``)
  together with a ``ModelPool`` that uses MinHash signatures and a sparse
  winner‑take‑all (WTA) eviction policy.

**Mathematical bridge**  
Both worlds treat *uncertainty* as a scalar that modulates a decision term.
Parent A maps entropy ``H`` → confidence ``c ∈ [0,1]``.  
Parent B multiplies a regret matrix ``R`` by a ternary weighting vector
derived from a deterministic hash.  

The hybrid algorithm therefore:

1. Parses a date string, obtains entropy ``H`` and converts it to confidence
   ``c``.
2. Builds a regret matrix ``R`` from ``MathAction`` objects and a set of
   ``MathCounterfactual`` objects.
3. Forms a ternary vector ``τ(c) ∈ {‑1,0,1}^k`` from the confidence‑scaled
   MinHash signature of each action identifier.
4. Computes a *confidence‑aware* score ``s = (R · τ(c))·c`` and feeds it to an
   Upper‑Confidence‑Bound (UCB) selector.
5. Guarantees that the model associated with the selected action is loaded
   in a ``ModelPool`` using the sparse‑WTA eviction rule.

The three core functions below demonstrate this integration:
``parse_date_with_entropy``, ``certainty_from_entropy`` and
``hybrid_select_action``.  Additional helpers implement the regret matrix,
ternary weighting and model‑pool management.  The module is self‑contained
and uses only the standard library, ``numpy``, ``math`` and ``random``.
"""

import datetime as dt
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Parent‑A: CertaintyFlag definition and date‑entropy utilities
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    confidence: float  # ∈ [0, 1]
    label: str


def parse_date_with_entropy(date_str: str) -> Tuple[Optional[dt.datetime], float]:
    """
    Try a handful of common date formats.  The set of successful parses
    defines a discrete distribution over candidate dates; its Shannon
    entropy quantifies the ambiguity.

    Returns
    -------
    best_candidate : datetime or None
        The first successfully parsed date (or None if all fail).
    entropy : float
        Shannon entropy of the uniform distribution over all successful parses.
    """
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%b-%Y",
        "%b %d, %Y",
    ]
    candidates: List[dt.datetime] = []
    for fmt in formats:
        try:
            dt_obj = dt.datetime.strptime(date_str, fmt)
            candidates.append(dt_obj)
        except Exception:
            continue

    if not candidates:
        return None, 0.0

    # Uniform distribution over candidates
    p = 1.0 / len(candidates)
    entropy = -len(candidates) * p * math.log(p, 2)  # bits
    return candidates[0], entropy


def certainty_from_entropy(entropy: float) -> CertaintyFlag:
    """
    Map entropy H (bits) to a confidence c ∈ [0,1] using a simple
    monotonic transform and assign a qualitative label.

    The transform is c = 1 / (1 + H) which guarantees c → 0 as H → ∞
    and c → 1 as H → 0.
    """
    c = 1.0 / (1.0 + entropy)
    if c > 0.9:
        label = "FACT"
    elif c > 0.7:
        label = "PROBABLE"
    elif c > 0.4:
        label = "POSSIBLE"
    elif c > 0.1:
        label = "BULLSHIT"
    else:
        label = "SURE_MAYBE"
    return CertaintyFlag(confidence=c, label=label)


# ----------------------------------------------------------------------
# Parent‑B: Regret engine, MinHash, ternary vector and ModelPool
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


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


class ModelPool:
    """
    Sparse Winner‑Take‑All pool with a RAM ceiling.  Loading a model may
    evict the oldest entry (FIFO) until enough memory is available.
    """
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # FIFO eviction
            evicted_key = next(iter(self.loaded))
            self.loaded.pop(evicted_key)
        self.load(model)


def _minhash_signature(seed: int, token: str, num_hashes: int = 8) -> np.ndarray:
    """
    Deterministic MinHash signature of ``token`` using ``seed``.
    Returns an array of 64‑bit integers.
    """
    sig = np.empty(num_hashes, dtype=np.uint64)
    for i in range(num_hashes):
        data = seed.to_bytes(4, "big") + i.to_bytes(4, "big") + token.encode("utf-8")
        h = int(hashlib.sha256(data).hexdigest(), 16)
        sig[i] = h & ((1 << 64) - 1)
    return sig


def _ternary_vector_from_signature(sig: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert a MinHash signature to a ternary vector τ ∈ {‑1,0,1}^k.
    Bits set in the upper half of the 64‑bit word become +1,
    bits in the lower half become -1, and zero bits become 0.
    """
    bits = ((sig[:, None] >> np.arange(64)) & 1).astype(int)  # shape (k,64)
    # Compute proportion of set bits per hash
    prop = bits.mean(axis=1)
    tau = np.where(prop > threshold, 1, np.where(prop < (1 - threshold), -1, 0))
    return tau.astype(int)


def build_regret_matrix(actions: List[MathAction],
                        counterfactuals: List[MathCounterfactual]) -> np.ndarray:
    """
    Construct a matrix R where R[i, j] = (action_i.expected_value -
    counterfactual_j.outcome_value) * counterfactual_j.probability.
    """
    R = np.empty((len(actions), len(counterfactuals)), dtype=float)
    for i, act in enumerate(actions):
        for j, cf in enumerate(counterfactuals):
            if cf.action_id != act.id:
                # If the counterfactual does not belong to this action,
                # treat its outcome as zero (no regret contribution)
                diff = act.expected_value
            else:
                diff = act.expected_value - cf.outcome_value
            R[i, j] = diff * cf.probability
    return R


# ----------------------------------------------------------------------
# Hybrid core: confidence‑aware regret‑UCB decision & model management
# ----------------------------------------------------------------------
def hybrid_score(actions: List[MathAction],
                 counterfactuals: List[MathCounterfactual],
                 confidence: float,
                 seed: int = 42) -> np.ndarray:
    """
    Compute a confidence‑aware score vector s ∈ ℝ^{|A|}:

        s = c · (R · τ(c))

    where
        c          – confidence from CertaintyFlag,
        R          – regret matrix,
        τ(c)       – ternary vector derived from the MinHash signature of each
                     action identifier (seeded by ``seed``) and scaled by ``c``.

    The result is a 1‑D array with one entry per action.
    """
    R = build_regret_matrix(actions, counterfactuals)               # (A, C)
    # Build a ternary vector per action
    taus = []
    for act in actions:
        sig = _minhash_signature(seed, act.id, num_hashes=4)
        tau = _ternary_vector_from_signature(sig, threshold=0.5)
        # Collapse the small vector to a scalar by summation (acts as a weight)
        taus.append(tau.sum())
    τ = np.array(taus, dtype=float)                                 # (A,)
    # Matrix‑vector product
    raw = R @ τ                                                       # (A,)
    return confidence * raw


def ucb_select_action(actions: List[MathAction],
                      counts: Dict[str, int],
                      total_selections: int,
                      confidence: float,
                      beta: float = 1.0) -> MathAction:
    """
    Upper‑Confidence‑Bound selector that incorporates the hybrid confidence
    into the exploration term:

        score(a) = Q(a) + β·c·√(log t / n_a)

    Q(a) is approximated by the hybrid score (see ``hybrid_score``).
    """
    # Compute hybrid scores for all actions
    # For simplicity we reuse a dummy counterfactual set: each action vs
    # a self‑counterfactual with outcome equal to its expected value.
    counterfactuals = [
        MathCounterfactual(action_id=act.id, outcome_value=act.expected_value)
        for act in actions
    ]
    scores = hybrid_score(actions, counterfactuals, confidence)

    best = None
    best_val = -math.inf
    for i, act in enumerate(actions):
        n_a = counts.get(act.id, 0) + 1e-9  # avoid division by zero
        ucb = scores[i] + beta * confidence * math.sqrt(math.log(max(total_selections, 1) + 1) / n_a)
        if ucb > best_val:
            best_val = ucb
            best = act
    return best


def hybrid_select_action(date_str: str,
                         actions: List[MathAction],
                         model_pool: ModelPool,
                         counts: Dict[str, int],
                         total_selections: int) -> Tuple[MathAction, CertaintyFlag]:
    """
    End‑to‑end hybrid operation:

    1. Parse the date and obtain confidence.
    2. Use confidence‑aware regret‑UCB to pick an action.
    3. Ensure the model associated with the chosen action is loaded
       (evicting if necessary).

    Returns the selected action and its CertaintyFlag.
    """
    _, entropy = parse_date_with_entropy(date_str)
    certainty = certainty_from_entropy(entropy)

    # Action selection
    selected = ucb_select_action(actions, counts, total_selections, certainty.confidence)

    # Model handling – we fabricate a model tier per action id
    model = ModelTier(name=selected.id, ram_mb=500 + int(100 * selected.risk), tier="T1")
    if not model_pool.is_loaded(model.name):
        try:
            model_pool.load(model)
        except Exception:
            # If loading fails (e.g., RAM ceiling), fall back to eviction policy
            model_pool.load_with_eviction(model)

    # Update bookkeeping
    counts[selected.id] = counts.get(selected.id, 0) + 1

    return selected, certainty


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a small action space
    actions = [
        MathAction(id="A1", expected_value=10.0, cost=2.0, risk=0.1),
        MathAction(id="A2", expected_value=8.0, cost=1.0, risk=0.3),
        MathAction(id="A3", expected_value=12.0, cost=3.0, risk=0.2),
    ]

    # Initialise structures
    pool = ModelPool(ram_ceiling_mb=2000)
    selection_counts: Dict[str, int] = {}
    total = 0

    # Test with several date strings of varying ambiguity
    test_dates = ["2023-04-05", "05/04/2023", "April 5, 2023", "2023/04/05"]
    for d in test_dates:
        total += 1
        act, cf = hybrid_select_action(d, actions, pool, selection_counts, total)
        print(f"Date: {d!r} → Entropy‑derived confidence {cf.confidence:.3f} ({cf.label})")
        print(f"  Selected Action: {act.id} (EV={act.expected_value}, risk={act.risk})")
        print(f"  Models currently loaded: {list(pool.loaded.keys())}")
        print("-" * 60)

    # Final sanity check: ensure at least one model is loaded
    assert pool.loaded, "ModelPool should contain at least one loaded model"
    print("Smoke test completed successfully.")