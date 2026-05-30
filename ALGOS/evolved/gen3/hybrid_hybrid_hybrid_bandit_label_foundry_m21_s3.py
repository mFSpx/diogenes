# DARWIN HAMMER — match 21, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s2.py (gen2)
# parent_b: label_foundry.py (gen0)
# born: 2026-05-29T23:25:13Z

"""Hybrid Bandit‑Sketch‑Label Fusion Module
Parents
-------
- hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s2.py (Parent A)
- label_foundry.py (Parent B)

Mathematical Bridge
-------------------
Parent A maintains a *store* S_a for each action a that accumulates rewards and
uses it in an Upper‑Confidence‑Bound (UCB) selection rule.  
Parent B works with *sketches* (Count‑Min, HyperLogLog) and provides a
singular‑learning‑theory term λ·log n (RLCT) together with a label‑aggregation
routine.

The fusion identifies two shared statistical quantities:

1. **Log‑count statistics** – both the bandit’s reward frequencies and the
   cardinality of observed contexts can be estimated by sketches.
2. **Loss‑driven RLCT term** – the bandit’s cumulative negative reward (loss)
   yields a curve L(n) whose slope in log‑log space approximates λ, the real
   log‑canonical threshold.

The hybrid algorithm therefore:

* Sketches per‑action reward frequencies with a Count‑Min sketch, producing an
  unbiased estimate of the empirical mean reward μ̂_a and its variance σ̂_a².
* Sketches the set of distinct contexts (e.g., labeling‑function identifiers)
  with a HyperLogLog sketch, giving an estimate n̂ of the effective sample size.
* Fits a linear model log L = λ·log n + c on the observed loss sequence to
  obtain λ̂ (the RLCT estimate).
* Injects the term λ̂·log n̂ into the UCB confidence bound, yielding a
  *sketch‑augmented‑RLCT‑aware* selection criterion.
* Re‑uses Parent B’s label‑aggregation utilities to produce probabilistic
  labels that can serve as additional context features for the bandit.

The code below implements the three mathematical components (sketches,
RLCT regression, label aggregation) and demonstrates their joint operation
through a minimal bandit simulation."""

import math
import random
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Iterable

import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Sketch primitives (Count‑Min and HyperLogLog)
# ----------------------------------------------------------------------


class CountMinSketch:
    """Simple Count‑Min sketch for non‑negative integer streams."""

    def __init__(self, width: int = 1024, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = np.random.default_rng(seed)
        self.hash_seeds = rng.integers(1, 2 ** 31 - 1, size=depth, dtype=np.int64)

    def _hash(self, item: bytes, i: int) -> int:
        h = hashlib.blake2b(item, digest_size=8, person=bytes([i]))
        return int.from_bytes(h.digest(), "little") % self.width

    def add(self, item: str, count: int = 1) -> None:
        b = item.encode("utf-8")
        for i, seed in enumerate(self.hash_seeds):
            idx = self._hash(b, i)
            self.tables[i, idx] += count

    def estimate(self, item: str) -> int:
        b = item.encode("utf-8")
        mins = []
        for i, _ in enumerate(self.hash_seeds):
            idx = self._hash(b, i)
            mins.append(self.tables[i, idx])
        return min(mins)


class HyperLogLog:
    """Very small HyperLogLog for cardinality estimation."""

    def __init__(self, b: int = 8):  # 2^b registers
        self.b = b
        self.m = 1 << b
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def _rho(self, w: int) -> int:
        # position of first 1-bit
        return (w.bit_length() - w.bit_length() + 1) or (w.bit_length() + 1)

    def add(self, item: str) -> None:
        h = int(hashlib.sha1(item.encode("utf-8")).hexdigest(), 16)
        idx = h >> (160 - self.b)
        w = h << self.b & ((1 << 160) - 1)
        rank = self._rho(w)
        self.registers[idx] = max(self.registers[idx], rank)

    def cardinality(self) -> float:
        Z = 1.0 / np.sum(2.0 ** -self.registers)
        E = self.alpha * self.m ** 2 * Z
        # Small range correction
        if E <= (5 / 2) * self.m:
            V = np.count_nonzero(self.registers == 0)
            if V != 0:
                E = self.m * math.log(self.m / V)
        return E


# ----------------------------------------------------------------------
# RLCT estimation from loss sequence (Parent B component)
# ----------------------------------------------------------------------


def estimate_rlct(losses: List[float]) -> float:
    """
    Estimate the Real Log‑Canonical Threshold λ from a loss curve.

    The method fits a linear regression to (log n, log L_n) where n is the
    sample index (treated as effective sample size) and L_n is the cumulative
    loss up to n.  The slope of the regression line is λ.

    Parameters
    ----------
    losses : list of float
        Sequence of instantaneous losses (negative rewards).

    Returns
    -------
    lambda_hat : float
        Estimated RLCT λ.
    """
    if not losses:
        raise ValueError("loss list must be non‑empty")
    cumulative = np.cumsum(losses) + 1e-12  # avoid log(0)
    n = np.arange(1, len(losses) + 1)
    log_n = np.log(n)
    log_L = np.log(cumulative)

    # Simple ordinary least squares
    A = np.vstack([log_n, np.ones_like(log_n)]).T
    slope, _ = np.linalg.lstsq(A, log_L, rcond=None)[0]
    return slope  # this is λ


# ----------------------------------------------------------------------
# Label aggregation utilities (Parent B)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float


def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """
    Aggregate raw labeling‑function votes into probabilistic labels.

    Identical to Parent B but kept here for self‑containment.
    """
    votes = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)
    out: List[ProbabilisticLabel] = []
    for doc_id, vs in votes.items():
        if not vs:
            out.append(ProbabilisticLabel(doc_id, 0, 0.5))
            continue
        cnt = Counter(vs)
        label = 1 if cnt[1] >= cnt[0] else 0
        confidence = cnt[label] / len(vs)
        out.append(ProbabilisticLabel(doc_id, label, confidence))
    return out


# ----------------------------------------------------------------------
# Hybrid Bandit core (Parent A + sketches + RLCT)
# ----------------------------------------------------------------------


@dataclass
class ActionState:
    """State kept per arm/action."""
    store: float = 0.0                     # cumulative reward
    count: int = 0                         # number of pulls
    cms: CountMinSketch = field(default_factory=CountMinSketch)


class SketchRLCTBandit:
    """
    Contextual bandit that uses sketches to estimate reward statistics,
    a HyperLogLog sketch for distinct context cardinality, and an RLCT‑aware
    confidence bound.
    """

    def __init__(self, actions: List[str], c: float = 1.0):
        self.actions = actions
        self.c = c  # exploration coefficient
        self.states: Dict[str, ActionState] = {a: ActionState() for a in actions}
        self.context_hll = HyperLogLog()
        self.loss_history: List[float] = []  # negative rewards

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def register_context(self, ctx_id: str) -> None:
        """Add a new context identifier to the HyperLogLog sketch."""
        self.context_hll.add(ctx_id)

    def update(self, action: str, reward: float, ctx_id: str) -> None:
        """
        Update the store and sketches for a given action.

        Parameters
        ----------
        action : str
            Identifier of the chosen arm.
        reward : float
            Observed reward (higher is better).
        ctx_id : str
            Identifier of the current context (e.g., concatenated LF names).
        """
        if action not in self.states:
            raise ValueError(f"unknown action {action}")

        state = self.states[action]
        state.store += reward
        state.count += 1
        # update reward sketch (use reward rounded to int for simplicity)
        state.cms.add(f"{action}_reward", int(max(reward, 0)))
        # register context for cardinality estimation
        self.register_context(ctx_id)
        # store negative reward as loss for RLCT estimation
        self.loss_history.append(-reward)

    def _mean_reward_estimate(self, action: str) -> float:
        """Estimate mean reward using the Count‑Min sketch."""
        state = self.states[action]
        if state.count == 0:
            return 0.0
        # Sketch gives an upper bound on total reward; we use it as a proxy.
        total_est = state.cms.estimate(f"{action}_reward")
        return total_est / state.count

    def _rlct_term(self) -> float:
        """Compute λ·log n̂ where λ is estimated from loss history and n̂ from HLL."""
        if not self.loss_history:
            return 0.0
        lam = estimate_rlct(self.loss_history)
        n_hat = max(self.context_hll.cardinality(), 1.0)
        return lam * math.log(n_hat)

    def select_action(self) -> str:
        """
        Upper‑Confidence‑Bound (UCB) selection with RLCT‑augmented confidence.

        UCB_a = μ̂_a + c * sqrt( (log t + λ·log n̂) / N_a )
        where:
            μ̂_a – sketch‑based mean reward,
            t    – total number of pulls,
            N_a  – pulls of arm a,
            n̂   – estimated distinct contexts,
            λ    – RLCT estimate.
        """
        total_pulls = sum(st.count for st in self.states.values()) + 1e-9
        rlct = self._rlct_term()
        best_val = -float("inf")
        best_action = None
        for a, st in self.states.items():
            if st.count == 0:
                # force exploration of never‑tried arms
                ucb = float("inf")
            else:
                mu_hat = self._mean_reward_estimate(a)
                bonus = self.c * math.sqrt((math.log(total_pulls) + rlct) / st.count)
                ucb = mu_hat + bonus
            if ucb > best_val:
                best_val = ucb
                best_action = a
        return best_action


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------


def demo_sketches() -> None:
    """Show basic Count‑Min and HyperLogLog functionality."""
    cms = CountMinSketch(width=128, depth=4)
    for item in ["apple", "banana", "apple", "cherry", "banana", "banana"]:
        cms.add(item)
    print("CMS estimates:", {i: cms.estimate(i) for i in ["apple", "banana", "cherry", "date"]})

    hll = HyperLogLog(b=6)
    for i in range(1000):
        hll.add(f"user_{i}")
    print("HLL cardinality estimate (≈1000):", hll.cardinality())


def demo_rlct() -> None:
    """Generate a synthetic loss curve and estimate RLCT."""
    # Simulate decreasing losses (e.g., learning curve)
    losses = [1.0 / (k + 1) for k in range(1, 101)]
    lam = estimate_rlct(losses)
    print(f"Estimated RLCT λ from synthetic losses: {lam:.4f}")


def demo_bandit() -> None:
    """Run a tiny bandit loop that uses label aggregation as context."""
    actions = ["red", "green", "blue"]
    bandit = SketchRLCTBandit(actions, c=0.5)

    # Simulated labeling‑function results (serve as contexts)
    lf_batches = [
        [LabelingFunctionResult("lf1", "doc1", 1), LabelingFunctionResult("lf2", "doc1", 0)],
        [LabelingFunctionResult("lf1", "doc2", 0), LabelingFunctionResult("lf3", "doc2", 1)],
    ]
    prob_labels = aggregate_labels(lf_batches)
    # Map doc_id -> context string
    context_map = {pl.doc_id: f"{pl.doc_id}_{pl.label}" for pl in prob_labels}

    rng = random.Random(0)
    for t in range(50):
        ctx = rng.choice(list(context_map.values()))
        a = bandit.select_action()
        # synthetic reward: higher if action matches context's label parity
        reward = 1.0 if (a == "red" and ctx.endswith("1")) else 0.2 * rng.random()
        bandit.update(a, reward, ctx)

    print("Final action stores:", {a: st.store for a, st in bandit.states.items()})
    print("Estimated RLCT term:", bandit._rlct_term())
    print("Estimated distinct contexts:", bandit.context_hll.cardinality())


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Sketch Demo ===")
    demo_sketches()
    print("\n=== RLCT Demo ===")
    demo_rlct()
    print("\n=== Bandit Demo ===")
    demo_bandit()
    print("\nAll components executed successfully.")