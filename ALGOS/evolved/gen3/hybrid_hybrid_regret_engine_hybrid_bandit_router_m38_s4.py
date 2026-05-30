# DARWIN HAMMER — match 38, survivor 4
# gen: 3
# parent_a: hybrid_regret_engine_hybrid_liquid_time_c_m13_s1.py (gen2)
# parent_b: hybrid_bandit_router_honeybee_store_m9_s5.py (gen1)
# born: 2026-05-29T23:25:25Z

"""Hybrid Regret‑Weighted Bandit with Honeybee Store and MinHash Bridge

This module fuses the core mathematics of two parent algorithms:

* **Parent A – Regret‑Weighted Strategy with MinHash**  
  The regret‑weighted value of each action is computed from expected value,
  cost, risk and counterfactual outcomes.  A MinHash signature of the
  action’s token set provides a similarity metric that can modulate those
  values.

* **Parent B – Hybrid Bandit Router with Honeybee Store**  
  A contextual bandit selects actions using propensity scores (soft‑max
  of a utility) while a “store” dynamics (the honeybee dance) turns the
  net inflow/outflow of resources into a bounded control signal.

**Mathematical bridge** – the MinHash similarity between an action and a
reference set is used as a *multiplicative similarity factor* that scales
the regret‑weighted utility before it enters the bandit’s soft‑max.  The
resulting utility drives both the action selection (propensity) and the
store update (inflow proportional to the chosen action’s propensity)."""

import sys
import pathlib
import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Data structures (union of both parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Action definition used by the regret‑weighted component."""
    id: str
    tokens: Tuple[str, ...]          # token set for MinHash
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection performed by the bandit."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBandit"


@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the policy (not used directly in the demo)."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee‑style store that converts net flow into a bounded “dance”."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0  # internal cache for ``dance``

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the most recent Δ."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


# ----------------------------------------------------------------------
# MinHash utilities (from Parent A)
# ----------------------------------------------------------------------


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token iterable."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity based on equality of MinHash components."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Regret‑weighted core (from Parent A)
# ----------------------------------------------------------------------


def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """Base regret‑weighted value for each action."""
    if not actions:
        return {}
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions
    }
    return vals


# ----------------------------------------------------------------------
# Hybrid mathematics
# ----------------------------------------------------------------------


def hybrid_scores(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    reference_actions: List[MathAction],
    k: int = 128,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute a similarity‑modulated regret‑weighted utility for every action.

    For each action *a* we compute

        u_a = R_a * (1 + S(a, R))

    where
        R_a – regret‑weighted value from ``compute_regret_weighted_strategy``.
        S(a, R) – MinHash similarity between *a* and the union of tokens of
                  all reference actions.

    The additive 1 ensures the similarity factor stays ≥1.
    """
    # 1. Regret‑weighted base values
    base_vals = compute_regret_weighted_strategy(actions, counterfactuals)

    # 2. Reference signature (union of tokens of all reference actions)
    ref_tokens = set()
    for ra in reference_actions:
        ref_tokens.update(ra.tokens)
    ref_sig = signature(ref_tokens, k=k)

    # 3. Modulate each action
    scores: Dict[str, float] = {}
    for a in actions:
        act_sig = signature(a.tokens, k=k)
        sim = similarity(act_sig, ref_sig)
        scores[a.id] = base_vals.get(a.id, 0.0) * (1.0 + sim)
    return scores


def softmax_propensities(scores: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """
    Convert raw scores into a probability distribution using a temperature‑scaled soft‑max.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids = list(scores.keys())
    raw = np.array([scores[i] for i in ids], dtype=float)
    # Numerical stability
    raw = raw - np.max(raw)
    exp_vals = np.exp(raw / temperature)
    probs = exp_vals / np.sum(exp_vals)
    return dict(zip(ids, probs))


def select_hybrid_action(
    scores: Dict[str, float],
    temperature: float = 0.5,
) -> BanditAction:
    """
    Choose an action according to soft‑max propensities and attach a simple confidence bound.
    """
    propensities = softmax_propensities(scores, temperature=temperature)
    chosen_id = random.choices(list(propensities.keys()), weights=propensities.values(), k=1)[0]
    p = propensities[chosen_id]
    # Confidence bound: inverse‑sqrt of effective samples (here a proxy)
    cb = 1.0 / math.sqrt(p + 1e-12)
    expected_reward = scores[chosen_id]
    return BanditAction(
        action_id=chosen_id,
        propensity=p,
        expected_reward=expected_reward,
        confidence_bound=cb,
    )


# ----------------------------------------------------------------------
# Hybrid system that glues bandit + store + MinHash
# ----------------------------------------------------------------------


class HybridRegretBanditStore:
    """
    End‑to‑end hybrid that

    1. Computes similarity‑modulated regret utilities.
    2. Selects an action via a soft‑max bandit.
    3. Updates a honeybee store where the inflow is proportional to the
       chosen action's propensity and the outflow is a constant leakage.
    """

    def __init__(self, store: StoreState | None = None):
        self.store = store if store is not None else StoreState()

    def step(
        self,
        actions: List[MathAction],
        counterfactuals: List[MathCounterfactual],
        reference_actions: List[MathAction],
        leakage: float = 0.1,
    ) -> Tuple[BanditAction, float]:
        """
        Perform a single decision‑update cycle.

        Returns
        -------
        chosen_action : BanditAction
            The action selected by the bandit.
        dance_signal : float
            The current store's bounded control signal.
        """
        # 1. Compute hybrid scores
        scores = hybrid_scores(actions, counterfactuals, reference_actions)

        # 2. Bandit selection
        chosen = select_hybrid_action(scores)

        # 3. Store dynamics: inflow proportional to propensity, outflow = leakage
        inflow = [chosen.propensity]
        outflow = [leakage]
        _, _ = self.store.update(inflow, outflow)

        return chosen, self.store.dance


# ----------------------------------------------------------------------
# Demonstration functions (required ≥3)
# ----------------------------------------------------------------------


def demo_signatures():
    """Show MinHash signatures for two simple token sets."""
    sig1 = signature(["apple", "banana", "cherry"], k=32)
    sig2 = signature(["banana", "date"], k=32)
    print("Signature 1 (first 5 values):", sig1[:5])
    print("Signature 2 (first 5 values):", sig2[:5])
    print("Similarity:", similarity(sig1, sig2))


def demo_hybrid_scores():
    """Compute and print hybrid scores for a tiny toy problem."""
    actions = [
        MathAction(id="a1", tokens=("red", "circle"), expected_value=5.0, cost=1.0),
        MathAction(id="a2", tokens=("blue", "square"), expected_value=4.0, cost=0.5),
    ]
    counterfactuals = [MathCounterfactual(action_id="a2", outcome_value=2.0, probability=0.8)]
    reference = [MathAction(id="ref", tokens=("red", "square"), expected_value=0.0)]
    scores = hybrid_scores(actions, counterfactuals, reference, k=64)
    print("Hybrid scores:", scores)


def demo_full_cycle():
    """Run a full decision‑update loop using the hybrid system."""
    actions = [
        MathAction(id="a1", tokens=("alpha", "beta"), expected_value=3.0, cost=0.2),
        MathAction(id="a2", tokens=("gamma", "delta"), expected_value=2.5, cost=0.1),
        MathAction(id="a3", tokens=("epsilon", "zeta"), expected_value=4.0, cost=0.3),
    ]
    counterfactuals = [
        MathCounterfactual(action_id="a1", outcome_value=1.0, probability=0.9),
        MathCounterfactual(action_id="a3", outcome_value=0.5, probability=0.5),
    ]
    reference = [
        MathAction(id="ref1", tokens=("alpha", "delta"), expected_value=0.0),
        MathAction(id="ref2", tokens=("gamma", "zeta"), expected_value=0.0),
    ]

    hybrid_system = HybridRegretBanditStore()
    for step in range(5):
        chosen, dance = hybrid_system.step(actions, counterfactuals, reference, leakage=0.05)
        print(
            f"Step {step+1}: chosen={chosen.action_id}, "
            f"propensity={chosen.propensity:.3f}, "
            f"dance={dance:.3f}"
        )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    print("=== Demo: MinHash signatures ===")
    demo_signatures()
    print("\n=== Demo: Hybrid scores ===")
    demo_hybrid_scores()
    print("\n=== Demo: Full hybrid cycle ===")
    demo_full_cycle()