# DARWIN HAMMER — match 4507, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_minimum_cost__m994_s1.py (gen4)
# born: 2026-05-29T23:56:14Z

"""HybridRegretBanditTreeEngine
Combines:
- Parent A: Regret‑weighted scoring with MinHash similarity and a “dance” signal.
- Parent B: Global bandit statistics (empirical rewards, counts), tree‑cost evaluation,
  and the Schoolfield temperature‑dependent developmental rate.

Mathematical bridge
------------------
Parent A produces a scalar score  

    S_i = σ(R_i) · (1 + J(sig_i, sig_ref)) · D(T)

where  

* σ(R) = 1/(1+exp(‑R)) is the sigmoid regret‑weighting,  
* R_i = E_i – C_i – K_i + Φ_i aggregates expected value, cost, risk and an optional
  counterfactual term,  
* J(sig_i, sig_ref) is the MinHash Jaccard‑like similarity between the action’s
  signature and a reference signature,  
* D(T) = developmental_rate(T) (Schoolfield model, Parent B) plays the role of the
  “dance” signal, smoothly scaling with temperature T.

Parent B supplies a classic UCB‑type confidence bound  

    U_i = √( 2·log(N) / n_i )

with N the total number of selections and n_i the count of action i.
The hybrid policy finally chooses the action that maximises  

    H_i = S_i + U_i

and updates the global statistics with the observed reward.

The code below implements this fused system, exposing three core functions:
`compute_hybrid_score`, `select_hybrid_action`, and `update_hybrid_policy`.  A
light‑weight smoke test demonstrates end‑to‑end execution."""

import hashlib
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Action definition used by the regret engine."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0
    tokens: Tuple[str, ...] = field(default_factory=tuple)  # for MinHash


@dataclass(frozen=True)
class Counterfactual:
    """Optional counterfactual contribution."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the bandit statistics."""
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0


@dataclass(frozen=True)
class BanditAction:
    """Result of a hybrid selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    hybrid_score: float
    algorithm: str = "HybridRegretBanditTreeEngine"


# ----------------------------------------------------------------------
# Parent‑A utilities (MinHash & regret)
# ----------------------------------------------------------------------


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash for a (seed, token) pair."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set (length k)."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        # Empty token set → signature of infinities (no collisions)
        return [sys.maxsize] * k
    mins = [sys.maxsize] * k
    for seed in range(k):
        for t in toks:
            h = _hash(seed, t)
            if h < mins[seed]:
                mins[seed] = h
    return mins


def minhash_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must have equal length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def regret_term(action: MathAction, counterfactuals: Dict[str, Counterfactual] = None) -> float:
    """Compute the raw regret aggregation R_i."""
    cf = 0.0
    if counterfactuals and action.id in counterfactuals:
        cf_obj = counterfactuals[action.id]
        cf = cf_obj.outcome_value * cf_obj.probability
    return action.expected_value - action.cost - action.risk + cf


def sigmoid(x: float) -> float:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------------------------------------------------
# Parent‑B utilities (bandit stats, tree cost, Schoolfield rate)
# ----------------------------------------------------------------------


_POLICY: Dict[str, List[float]] = {}  # action_id → [total_reward, count]


def reset_policy() -> None:
    """Erase all learned statistics."""
    _POLICY.clear()


def _reward(action_id: str) -> float:
    """Empirical mean reward for *action_id* (0 if never observed)."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action_id: str) -> float:
    """Number of times *action_id* has been selected."""
    return _POLICY.get(action_id, [0.0, 0.0])[1]


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate a batch of observations into the global policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def total_selections() -> float:
    """Total number of action selections across all actions."""
    return sum(stats[1] for stats in _POLICY.values())


def ucb_confidence(action_id: str, alpha: float = 1.0) -> float:
    """UCB‑type confidence bound √(2·log(N)/n_i)."""
    n_i = _count(action_id)
    N = total_selections()
    if n_i == 0:
        # Encourage exploration of unseen actions
        return float("inf")
    return alpha * math.sqrt(2.0 * math.log(max(N, 1.0)) / n_i)


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters for the Schoolfield temperature model."""
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal·mol⁻¹·K⁻¹


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield model – returns a temperature‑scaled rate (dimensionless)."""
    # Simplified version: Arrhenius‑like term with low/high cut‑offs
    if temp_k <= params.t_low or temp_k >= params.t_high:
        return 0.0
    term_act = math.exp(-params.delta_h_activation / (params.r_cal * temp_k))
    term_low = math.exp(-params.delta_h_low / (params.r_cal * temp_k))
    term_high = math.exp(-params.delta_h_high / (params.r_cal * temp_k))
    rate = params.rho_25 * term_act / (1.0 + term_low + term_high)
    # Normalise to (0,1) for use as a scaling factor
    return max(0.0, min(1.0, rate))


def tree_cost(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]]) -> float:
    """Placeholder minimum‑spanning‑tree cost (sum of Euclidean edge lengths)."""
    total = 0.0
    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        total += math.hypot(xa - xb, ya - yb)
    return total


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


# Global signature cache: action_id → signature list
_SIGNATURE_CACHE: Dict[str, List[int]] = {}


def compute_hybrid_score(
    action: MathAction,
    reference_signature: List[int],
    temperature_k: float,
    counterfactuals: Dict[str, Counterfactual] = None,
) -> float:
    """Calculate the hybrid deterministic part S_i = σ(R_i)·(1+J)·D(T)."""
    # 1. Regret term and sigmoid weighting
    R_i = regret_term(action, counterfactuals)
    sigma = sigmoid(R_i)

    # 2. MinHash signature (cached)
    if action.id not in _SIGNATURE_CACHE:
        _SIGNATURE_CACHE[action.id] = signature(action.tokens)
    sig_i = _SIGNATURE_CACHE[action.id]

    # 3. Similarity with reference
    sim = minhash_jaccard(sig_i, reference_signature)

    # 4. Dance factor via developmental rate
    dance = developmental_rate(temperature_k)

    return sigma * (1.0 + sim) * dance


def select_hybrid_action(
    actions: List[MathAction],
    reference_tokens: Iterable[str],
    temperature_k: float,
    counterfactuals: Dict[str, Counterfactual] = None,
) -> BanditAction:
    """Select an action using hybrid score + UCB confidence."""
    if not actions:
        raise ValueError("No actions to select from")

    # Reference signature (e.g., recent high‑reward actions)
    ref_sig = signature(reference_tokens)

    best = None
    best_value = -float("inf")

    for act in actions:
        # Deterministic hybrid component
        s = compute_hybrid_score(act, ref_sig, temperature_k, counterfactuals)

        # Stochastic exploration term (UCB)
        ub = ucb_confidence(act.id)

        total = s + ub

        if total > best_value:
            best_value = total
            best = act
            best_score = s
            best_conf = ub

    # Propensity is a softmax over raw hybrid scores (optional)
    scores = np.array([compute_hybrid_score(a, ref_sig, temperature_k, counterfactuals) for a in actions])
    exp_scores = np.exp(scores - np.max(scores))
    propensities = exp_scores / exp_scores.sum()
    propensity = float(propensities[actions.index(best)])

    return BanditAction(
        action_id=best.id,
        propensity=propensity,
        expected_reward=_reward(best.id),
        confidence_bound=best_conf,
        hybrid_score=best_score,
    )


def update_hybrid_policy(updates: List[BanditUpdate]) -> None:
    """Update global bandit statistics and refresh cached signatures if needed."""
    update_policy(updates)
    # Optionally prune stale signatures (not required for correctness)
    # Here we simply keep them – they are cheap memory‑wise.


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Reset any prior state
    reset_policy()
    _SIGNATURE_CACHE.clear()

    # Define a tiny action set
    actions = [
        MathAction(id="a", expected_value=5.0, cost=1.0, risk=0.5, tokens=("alpha", "beta")),
        MathAction(id="b", expected_value=3.0, cost=0.5, risk=0.2, tokens=("gamma", "delta")),
        MathAction(id="c", expected_value=4.5, cost=0.8, risk=0.3, tokens=("alpha", "delta")),
    ]

    # Simulate a reference token set (e.g., tokens of recent successful actions)
    reference = ("alpha", "epsilon")

    # Temperature in Kelvin (mid‑range of Schoolfield model)
    temp_k = 295.0

    # No counterfactuals for this demo
    cf_dict = {}

    # Perform a selection
    chosen = select_hybrid_action(actions, reference, temp_k, cf_dict)
    print(f"Chosen action: {chosen}")

    # Simulate receiving a reward and update the policy
    reward = random.uniform(0, 10)  # mock reward
    upd = BanditUpdate(context_id="ctx1", action_id=chosen.action_id, reward=reward)
    update_hybrid_policy([upd])

    # Show updated statistics
    print(f"Updated mean reward for {chosen.action_id}: {_reward(chosen.action_id):.3f}")
    print(f"Total selections so far: {total_selections():.0f}")