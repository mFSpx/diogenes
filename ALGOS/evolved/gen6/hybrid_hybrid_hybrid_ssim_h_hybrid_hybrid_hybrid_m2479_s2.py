# DARWIN HAMMER — match 2479, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_ssim_hybrid_h_hybrid_sketches_rlct_m1064_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s4.py (gen3)
# born: 2026-05-29T23:42:38Z

"""Hybrid Multivector‑Bandit Algorithm

This module fuses the two parent algorithms:

* **Parent A** – builds Euclidean Clifford multivectors from statistical moments and
  evaluates similarity via the scalar part of the geometric product.
* **Parent B** – a contextual bandit with virtual “store” dynamics, confidence‑bound
  exploration and reward bookkeeping.

The mathematical bridge is the **scalar part of the geometric product** between a
multivector representing the current context and a multivector representing an
action’s reward statistics.  This scalar is interpreted as a *propensity* (inflow)
for the bandit store and also as an *estimated reward* in the selection rule.
Thus the two topologies are merged: the Clifford‑algebra similarity feeds directly
into the bandit’s stochastic control equations.

The resulting system provides:
* `stats_to_multivector` – common conversion used by both sides.
* `geometric_bandit_score` – similarity → propensity/reward estimate.
* `select_action_hybrid` – ε‑greedy selection using the geometric score,
  confidence bounds and virtual‑store dynamics.
* `update_hybrid` – policy and store update after observing a reward.

All code runs with only the Python standard library and NumPy.
"""

import math
import random
import sys
from pathlib import Path
from typing import Sequence, List, Dict, Tuple, FrozenSet, Any
import numpy as np

# ----------------------------------------------------------------------
# Multivector core (from Parent A)
# ----------------------------------------------------------------------
class Multivector:
    """Simple Euclidean Clifford algebra up to grade 2."""
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # prune near‑zero components
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }
        self.n = int(n)  # dimension of the underlying vector space

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, value in other.components.items():
            result[blade] = result.get(blade, 0.0) + value
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (outer product only, no metric sign changes)."""
        result: Dict[FrozenSet[int], float] = {}
        for b1, v1 in self.components.items():
            for b2, v2 in other.components.items():
                new_blade = frozenset(b1.union(b2))
                result[new_blade] = result.get(new_blade, 0.0) + v1 * v2
        return Multivector(result, self.n)

def stats_to_multivector(seq: Sequence[float]) -> Multivector:
    """
    Convert a 1‑D numeric sequence into a multivector of moments.
    Grade‑0 : mean
    Grade‑1 : variance (stored on basis {0})
    Grade‑2 : covariance placeholder (basis {0,1})
    """
    arr = np.asarray(seq, dtype=float)
    mean = float(np.mean(arr)) if arr.size else 0.0
    var = float(np.var(arr)) if arr.size else 0.0
    # Covariance placeholder – zero because we have a single sequence
    cov = 0.0
    comps: Dict[FrozenSet[int], float] = {
        frozenset(): mean,
        frozenset([0]): var,
        frozenset([0, 1]): cov,
    }
    return Multivector(comps, 2)

def geometric_ssim(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Hybrid similarity: scalar part of the geometric product of moment multivectors.
    """
    mx = stats_to_multivector(x)
    my = stats_to_multivector(y)
    return (mx * my).scalar_part()

# ----------------------------------------------------------------------
# Bandit core (from Parent B)
# ----------------------------------------------------------------------
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
ETA0 = 0.01          # base learning rate for reward updates
HOEFFDING_DELTA = 0.05
CLAMP_LO = -5.0
CLAMP_HI = 5.0

class BanditAction:
    """Immutable record returned by the selector."""
    __slots__ = ("action_id", "propensity", "expected_reward",
                 "confidence_bound", "algorithm")
    def __init__(self, action_id: str, propensity: float,
                 expected_reward: float, confidence_bound: float,
                 algorithm: str):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

    def __repr__(self) -> str:
        return (f"BanditAction(action_id={self.action_id!r}, propensity={self.propensity:.3f}, "
                f"expected_reward={self.expected_reward:.3f}, "
                f"confidence_bound={self.confidence_bound:.3f}, algorithm={self.algorithm!r})")

# Global mutable state (policy & virtual store)
_POLICY: Dict[str, Tuple[float, float, int]] = {}   # action_id -> (total_reward, total_sq_reward, count)
_STORE: Dict[str, float] = {}                      # virtual VRAM store per action

def reset_policy() -> None:
    """Clear learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()

def _reward_estimate(action_id: str) -> Tuple[float, float]:
    """Return (mean, variance) for an action; zero if unseen."""
    total, total_sq, n = _POLICY.get(action_id, (0.0, 0.0, 0))
    if n == 0:
        return 0.0, 0.0
    mean = total / n
    var = max(0.0, total_sq / n - mean * mean)
    return mean, var

def _confidence_bound(action_id: str) -> float:
    """Hoeffding‑type confidence bound for the given action."""
    _, _, n = _POLICY.get(action_id, (0.0, 0.0, 0))
    if n == 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / HOEFFDING_DELTA) / (2.0 * n))

# ----------------------------------------------------------------------
# Hybrid bridge utilities
# ----------------------------------------------------------------------
def context_to_multivector(context: Dict[str, float]) -> Multivector:
    """Encode a numeric context dict as a multivector of its values."""
    values = list(context.values())
    return stats_to_multivector(values)

def action_to_multivector(action_id: str) -> Multivector:
    """Encode an action's reward statistics as a multivector."""
    mean, var = _reward_estimate(action_id)
    # Use a dummy two‑element sequence to feed the same moment extractor.
    seq = [mean, mean + math.sqrt(var) if var > 0 else mean]
    return stats_to_multivector(seq)

def geometric_bandit_score(context: Dict[str, float], action_id: str) -> float:
    """
    Compute the scalar part of the geometric product between the context
    multivector and the action‑reward multivector.  The result is interpreted
    as a propensity (inflow) for the virtual store and as an estimated reward.
    """
    ctx_mv = context_to_multivector(context)
    act_mv = action_to_multivector(action_id)
    return (ctx_mv * act_mv).scalar_part()

# ----------------------------------------------------------------------
# Hybrid bandit selector
# ----------------------------------------------------------------------
def select_action_hybrid(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "geometric_ucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    ε‑greedy selection where the exploitation score for each action is:

        score = geometric_bandit_score(context, a) + confidence_bound(a) - store[a]

    The store term implements the virtual‑store dynamics (outflow reduces the score).
    """
    if seed is not None:
        random.seed(seed)

    if not actions:
        raise ValueError("action list cannot be empty")

    # Exploration branch
    if random.random() < epsilon:
        chosen = random.choice(actions)
        prop = geometric_bandit_score(context, chosen)
        exp_reward, _ = _reward_estimate(chosen)
        conf = _confidence_bound(chosen)
        return BanditAction(chosen, prop, exp_reward, conf, algorithm)

    # Exploitation branch: compute scores for all actions
    best_score = -float("inf")
    best_action = actions[0]

    for a in actions:
        prop = geometric_bandit_score(context, a)
        conf = _confidence_bound(a)
        store_val = _STORE.get(a, 0.0)
        score = prop + conf - store_val
        if score > best_score:
            best_score = score
            best_action = a

    prop = geometric_bandit_score(context, best_action)
    exp_reward, _ = _reward_estimate(best_action)
    conf = _confidence_bound(best_action)
    return BanditAction(best_action, prop, exp_reward, conf, algorithm)

# ----------------------------------------------------------------------
# Hybrid update routine
# ----------------------------------------------------------------------
def update_hybrid(
    action: BanditAction,
    reward: float,
    propensity: float | None = None,
) -> None:
    """
    Update policy statistics, the virtual store, and optionally apply a learning‑rate
    correction to the reward estimate.
    """
    aid = action.action_id
    # ---- policy statistics ----
    total, total_sq, n = _POLICY.get(aid, (0.0, 0.0, 0))
    total += reward
    total_sq += reward * reward
    n += 1
    _POLICY[aid] = (total, total_sq, n)

    # ---- virtual store dynamics ----
    inflow = propensity if propensity is not None else action.propensity
    outflow = action.confidence_bound
    prev = _STORE.get(aid, 0.0)
    new_val = prev + DT * (ALPHA * inflow - BETA * outflow)
    # Clamp to avoid divergence
    _STORE[aid] = max(CLAMP_LO, min(CLAMP_HI, new_val))

# ----------------------------------------------------------------------
# Demonstration functions (three required)
# ----------------------------------------------------------------------
def compute_similarity_matrix(
    contexts: List[Dict[str, float]],
    actions: List[str],
) -> np.ndarray:
    """
    Return a matrix S where S[i, j] = geometric_bandit_score(context_i, action_j).
    """
    m = len(contexts)
    n = len(actions)
    S = np.zeros((m, n), dtype=float)
    for i, ctx in enumerate(contexts):
        for j, act in enumerate(actions):
            S[i, j] = geometric_bandit_score(ctx, act)
    return S

def simulate_one_step(
    context: Dict[str, float],
    actions: List[str],
    true_reward_func: Any,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> Tuple[BanditAction, float]:
    """
    Run a single selection‑update cycle.
    `true_reward_func` is a callable(action_id) → float giving the stochastic reward.
    """
    act = select_action_hybrid(context, actions, epsilon=epsilon, seed=seed)
    reward = true_reward_func(act.action_id)
    update_hybrid(act, reward)
    return act, reward

def dump_state() -> Dict[str, Any]:
    """Utility to expose internal state for debugging / testing."""
    return {
        "policy": {k: {"total": v[0], "total_sq": v[1], "count": v[2]} for k, v in _POLICY.items()},
        "store": dict(_STORE),
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic reward model for testing
    def reward_model(aid: str) -> float:
        base = {"a": 1.0, "b": 0.5, "c": -0.2}.get(aid, 0.0)
        noise = random.gauss(0, 0.1)
        return base + noise

    reset_policy()
    actions = ["a", "b", "c"]
    # Generate three synthetic contexts
    contexts = [
        {"x1": 0.2, "x2": 1.1},
        {"x1": -0.5, "x2": 0.3},
        {"x1": 0.7, "x2": -0.8},
    ]

    print("=== Similarity matrix ===")
    print(compute_similarity_matrix(contexts, actions))

    # Run a few interaction steps
    for step in range(5):
        ctx = random.choice(contexts)
        act, rew = simulate_one_step(ctx, actions, reward_model, epsilon=0.2, seed=step)
        print(f"Step {step+1}: chosen {act.action_id!r}, reward={rew:.3f}, propensity={act.propensity:.3f}")

    print("\n=== Final internal state ===")
    print(dump_state())