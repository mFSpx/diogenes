# DARWIN HAMMER — match 38, survivor 5
# gen: 3
# parent_a: hybrid_regret_engine_hybrid_liquid_time_c_m13_s1.py (gen2)
# parent_b: hybrid_bandit_router_honeybee_store_m9_s5.py (gen1)
# born: 2026-05-29T23:25:25Z

"""HybridRegretBanditStore
Integrates:
- Parent A: Regret‑Weighted Strategy with MinHash similarity (regret_engine + hybrid_liquid_time_constant_minhash)
- Parent B: Bandit router with Honeybee store dynamics (bandit_router + honeybee_store)

Mathematical bridge:
The hidden state of the regret‑weighted strategy (the scalar “raw value” of each action) is projected
into a MinHash signature.  The Jaccard‑like similarity between an action’s signature and a
reference signature (e.g. recent high‑reward actions) is used as a multiplicative factor that
modulates the LinUCB confidence bound produced by the bandit router.  Simultaneously the
Honeybee store’s “dance” signal scales the overall regret‑weighting term, providing a liquid
time‑constant that smoothly adapts the influence of past regret.  The resulting hybrid score
for action *i* is

    S_i = g(R_i) · (1 + sim(sig_i, sig_ref)) · dance

where
    R_i = expected_value_i – cost_i – risk_i + counterfactual_i
    g(·) = sigmoid (regret‑weighting)
    sim(·,·) = MinHash Jaccard similarity
    dance = StoreState.dance (bounded control signal)

The policy selects actions proportionally to softmax(S_i) and attaches a LinUCB‑style
confidence bound that is also inflated by the similarity term.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – MinHash utilities and regret weighting
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


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ----------------------------------------------------------------------
# Parent B – Honeybee store dynamics
# ----------------------------------------------------------------------


@dataclass
class StoreState:
    """Liquid‑time‑constant store used as a global gain."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._store_last_delta(delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the most recent Δ."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

    def _store_last_delta(self, delta: float) -> None:
        self._last_delta = delta


# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of a hybrid action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBanditStore"


@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the hybrid policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Global LinUCB‑style state (per‑action)
# ----------------------------------------------------------------------


class LinUCBState:
    """Very lightweight LinUCB bookkeeping using scalar features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        # For each action we keep A (scalar) and b (scalar) because we use a 1‑D context.
        self.A: Dict[str, float] = {}
        self.b: Dict[str, float] = {}

    def ensure(self, aid: str) -> None:
        if aid not in self.A:
            self.A[aid] = 1.0  # regularisation term
            self.b[aid] = 0.0

    def update(self, aid: str, reward: float, x: float = 1.0) -> None:
        self.ensure(aid)
        self.A[aid] += x * x
        self.b[aid] += reward * x

    def theta(self, aid: str) -> float:
        self.ensure(aid)
        return self.b[aid] / self.A[aid]

    def confidence(self, aid: str, x: float = 1.0) -> float:
        self.ensure(aid)
        return math.sqrt(self.alpha * x * x / self.A[aid])


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def compute_hybrid_scores(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    store: StoreState,
    reference_tokens: Iterable[str],
    k: int = 128,
) -> Dict[str, float]:
    """
    Compute the hybrid score S_i for every action.

    The score combines regret‑weighted value, MinHash similarity to a reference set,
    and the store's dance signal.
    """
    # 1️⃣ Build reference signature once.
    ref_sig = signature(reference_tokens, k=k)

    # 2️⃣ Build a lookup for counterfactual contributions.
    cf_lookup = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}

    scores: Dict[str, float] = {}
    for a in actions:
        # raw regret‑weighted value
        raw = a.expected_value - a.cost - a.risk + cf_lookup.get(a.id, 0.0)

        # sigmoid gating
        gated = sigmoid(np.array([raw]))[0]

        # MinHash similarity between action id tokens and reference
        act_sig = signature([a.id], k=k)
        sim = similarity(act_sig, ref_sig)

        # Final hybrid score
        score = gated * (1.0 + sim) * store.dance
        scores[a.id] = score
    return scores


def select_hybrid_action(
    actions: List[MathAction],
    scores: Dict[str, float],
    linucb: LinUCBState,
) -> BanditAction:
    """
    Choose an action using a softmax over hybrid scores.
    Attach a LinUCB confidence bound that is inflated by the same similarity factor.
    """
    # Softmax for propensities
    raw_vals = np.array([scores[a.id] for a in actions])
    max_raw = raw_vals.max() if raw_vals.size else 0.0
    exp_vals = np.exp(raw_vals - max_raw)  # numerical stability
    probs = exp_vals / exp_vals.sum() if exp_vals.sum() else np.ones_like(exp_vals) / len(exp_vals)

    # Sample according to probabilities
    chosen_idx = int(np.random.choice(len(actions), p=probs))
    chosen = actions[chosen_idx]

    # Confidence bound (LinUCB) – we reuse the similarity factor as extra gain
    # Re‑compute similarity with reference (using a dummy reference set of recent ids)
    # For simplicity we reuse the same reference signature from the score computation:
    # the similarity factor is embedded in the score, we can extract it via:
    #   sim_factor = scores[chosen.id] / (sigmoid(raw) * store.dance) - 1
    # but to keep it robust we recompute directly.
    # Here we approximate by using the same sigmoid‑gated raw value.
    raw = chosen.expected_value - chosen.cost - chosen.risk
    gated = sigmoid(np.array([raw]))[0]
    # approximate similarity factor from score
    sim_factor = scores[chosen.id] / (gated * store.dance) - 1.0 if gated * store.dance != 0 else 0.0
    sim_factor = max(0.0, sim_factor)  # guard against negative noise

    base_conf = linucb.confidence(chosen.id)
    confidence = base_conf * (1.0 + sim_factor)

    return BanditAction(
        action_id=chosen.id,
        propensity=probs[chosen_idx],
        expected_reward=chosen.expected_value,
        confidence_bound=confidence,
        algorithm="HybridRegretBanditStore",
    )


def update_hybrid_policy(
    update: BanditUpdate,
    store: StoreState,
    linucb: LinUCBState,
    inflow_factor: float = 1.0,
    outflow_factor: float = 0.5,
) -> None:
    """
    Update the Honeybee store and the LinUCB parameters using the observed reward.
    """
    # Store dynamics: reward is treated as inflow, (expected_reward - reward) as outflow.
    inflow = [inflow_factor * update.reward]
    outflow = [outflow_factor * (update.reward - update.propensity)]
    store.update(inflow, outflow)

    # LinUCB update – we use a unit context vector.
    linucb.update(update.action_id, update.reward, x=1.0)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny problem
    actions = [
        MathAction(id="A", expected_value=5.0, cost=0.5, risk=0.2),
        MathAction(id="B", expected_value=3.0, cost=0.2, risk=0.1),
        MathAction(id="C", expected_value=4.0, cost=0.3, risk=0.3),
    ]

    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=0.5),
        MathCounterfactual(action_id="C", outcome_value=-0.2),
    ]

    # Reference set – imagine recent high‑reward actions
    reference = ["A", "C"]

    store = StoreState(alpha=0.8, beta=0.5, dt=0.5, base=1.0, gain=2.0, limit=5.0)

    linucb = LinUCBState(alpha=1.0)

    # 1️⃣ Compute hybrid scores
    scores = compute_hybrid_scores(actions, counterfactuals, store, reference)

    # 2️⃣ Select an action
    chosen = select_hybrid_action(actions, scores, linucb)
    print(f"Chosen action: {chosen}")

    # 3️⃣ Simulate a reward and update policy
    simulated_reward = random.uniform(0, 1) * chosen.expected_reward
    upd = BanditUpdate(
        context_id="ctx1",
        action_id=chosen.action_id,
        reward=simulated_reward,
        propensity=chosen.propensity,
    )
    update_hybrid_policy(upd, store, linucb)

    # Verify that store and LinUCB have changed
    print(f"Store level after update: {store.level:.3f}, dance: {store.dance:.3f}")
    print(f"LinUCB theta for {chosen.action_id}: {linucb.theta(chosen.action_id):.3f}")
    print("Smoke test completed without errors.")