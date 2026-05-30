# DARWIN HAMMER — match 4895, survivor 2
# gen: 6
# parent_a: hybrid_bandit_router_hybrid_hybrid_hybrid_m2649_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m2337_s1.py (gen5)
# born: 2026-05-29T23:58:35Z

"""Hybrid Bandit‑Regret RBF Router
=================================

This module fuses the two parent algorithms:

* **Parent A** – a multi‑armed bandit that mixes LinUCB / Thompson /
  ε‑greedy and learns a Radial‑Basis‑Function (RBF) surrogate model
  (`RBFSurrogate`) to predict context‑dependent expected rewards.

* **Parent B** – a regret‑weighted strategy that, given a set of
  `MathAction`s and optional counter‑factual outcomes, builds a
  probability distribution proportional to `exp(value‑best)` and
  provides entropy / Gini diagnostics.

**Mathematical bridge**

The bridge is the *expected reward* that the bandit supplies for a
particular `(context, action)` pair.  In the hybrid we let the RBF
surrogate supply a **contextual estimate** `\hat r(c,a)`.  This estimate
is fed directly into the regret‑weighted value
`v_a = \hat r(c,a) - cost_a - risk_a + cf_a`.  The resulting
probabilities are then *modulated* by the original bandit propensity
(`propensity_a`) to preserve exploration information while biasing the
selection toward low‑regret actions.

Formally, for each action `a`:


\hat r_a   = RBFSurrogate.predict(c, a)          # parent A
v_a       = \hat r_a - cost_a - risk_a + cf_a    # parent B
w_a       = exp(v_a - max_b v_b)                 # regret weighting
p_regret  = w_a / Σ_b w_b
p_final   = (propensity_a * p_regret) / Σ_b (propensity_b * p_regret)


The hybrid therefore unifies the *surrogate‑driven reward estimation* of
Parent A with the *regret‑weighted probability computation* of Parent B
into a single coherent decision rule.

The implementation below provides the core data structures, the RBF
surrogate, the regret‑weighted engine, and three public functions that
demonstrate the hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Sequence, Iterable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Bandit core (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # exploration propensity from the bandit policy
    expected_reward: float     # raw empirical or surrogate estimate
    confidence_bound: float    # width of confidence interval (unused in hybrid)
    algorithm: str             # name of the underlying bandit algorithm

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# Global stores – mimicking the original parent
_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, count]
_STORE: Dict[str, float] = {}                 # placeholder VRAM store (unused)

# ----------------------------------------------------------------------
# RBF surrogate (Parent B – core component)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel  exp( -(ε·r)² )."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

class RBFSurrogate:
    """Simple RBF interpolator for (context, action) → reward."""
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.centers: List[Tuple[Vector, str]] = []   # (context vector, action_id)
        self.weights: List[float] = []               # observed reward at each centre

    def add_sample(self, context: Vector, action_id: str, reward: float) -> None:
        self.centers.append((list(context), action_id))
        self.weights.append(reward)

    def predict(self, context: Vector, action_id: str) -> float:
        """RBF prediction for a given (context, action)."""
        if not self.centers:
            return 0.0
        # Compute kernel values only for matching action_id to keep dimensionality sensible
        ks = []
        ws = []
        for (c, aid), w in zip(self.centers, self.weights):
            if aid != action_id:
                continue
            dist = euclidean(c, context)
            k = gaussian(dist, self.epsilon)
            ks.append(k)
            ws.append(w)
        if not ks:
            # No samples for this action – fall back to global average
            return 0.0
        total = sum(k * w for k, w in zip(ks, ws))
        norm = sum(ks)
        return total / norm if norm != 0 else 0.0

# Global surrogate instance
_SURROGATE = RBFSurrogate(epsilon=1.0)

def reset_policy() -> None:
    """Clear all learned statistics and the surrogate model."""
    _POLICY.clear()
    _STORE.clear()
    global _SURROGATE
    _SURROGATE = RBFSurrogate(epsilon=1.0)

def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

# ----------------------------------------------------------------------
# Regret‑Weighted engine (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float          # will be filled with surrogate estimate
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: Optional[List[MathCounterfactual]] = None
) -> Dict[str, float]:
    """Return a probability distribution over actions using regret weighting."""
    if not actions:
        return {}
    cf: Dict[str, float] = {}
    if counterfactuals:
        cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    # Value = expected - cost - risk + counterfactual contribution
    vals = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0)
        for a in actions
    }
    best = max(vals.values())
    # Exponential weighting (numerically stable)
    exp_weights = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(exp_weights.values()) or 1.0
    return {k: v / total for k, v in exp_weights.items()}

def shannon_entropy(probabilities: Iterable[float]) -> float:
    """Binary Shannon entropy (base‑2)."""
    probs = [p for p in probabilities if p > 0]
    return -sum(p * math.log(p, 2) for p in probs)

def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient for a non‑negative value list."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cum = 0.0
    for i, x in enumerate(xs, 1):
        cum += (2 * i - n - 1) * x
    return cum / (n * sum(xs))

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def update_policy(update: BanditUpdate, context_vector: Vector) -> None:
    """
    Incorporate a new observation into the empirical policy and the RBF surrogate.
    """
    # Update empirical totals
    total, count = _POLICY.get(update.action_id, [0.0, 0.0])
    _POLICY[update.action_id] = [total + update.reward, count + 1]

    # Store the sample in the surrogate model
    _SURROGATE.add_sample(context_vector, update.action_id, update.reward)

def _surrogate_estimate(context: Vector, action_id: str) -> float:
    """
    Return a context‑dependent reward estimate using the RBF surrogate.
    Falls back to the empirical mean if the surrogate has no data.
    """
    est = _SURROGATE.predict(context, action_id)
    if est == 0.0:
        # No surrogate data – use empirical average
        est = _empirical_reward(action_id)
    return est

def hybrid_select_action(
    context_id: str,
    context_vector: Vector,
    available_action_ids: List[str],
    counterfactuals: Optional[List[MathCounterfactual]] = None,
    exploration_epsilon: float = 0.1
) -> BanditAction:
    """
    Perform a single hybrid decision step:
    1. Query the RBF surrogate for each action → expected reward.
    2. Build MathAction objects (cost/risk are set to zero for simplicity).
    3. Compute regret‑weighted probabilities.
    4. Blend with the bandit propensity (ε‑greedy style) and renormalise.
    5. Return a BanditAction describing the chosen arm.
    """
    # 1️⃣ Surrogate‑driven expected rewards
    math_actions = []
    for aid in available_action_ids:
        est = _surrogate_estimate(context_vector, aid)
        # Propensity from a simple ε‑greedy policy (parent A style)
        propensity = exploration_epsilon / len(available_action_ids)
        math_actions.append(MathAction(id=aid, expected_value=est))

    # 2️⃣ Regret‑weighted distribution
    regret_probs = compute_regret_weighted_strategy(math_actions, counterfactuals)

    # 3️⃣ Bandit propensity (ε‑greedy baseline)
    # For demonstration we reuse the same ε‑greedy propensity as above.
    bandit_props = {
        aid: exploration_epsilon / len(available_action_ids) for aid in available_action_ids
    }

    # 4️⃣ Hybrid blending: element‑wise product then renormalisation
    blended = {
        aid: bandit_props[aid] * regret_probs.get(aid, 0.0)
        for aid in available_action_ids
    }
    total_blend = sum(blended.values()) or 1.0
    final_probs = {aid: p / total_blend for aid, p in blended.items()}

    # 5️⃣ Sample an action according to the final distribution
    chosen_id = random.choices(
        population=list(final_probs.keys()),
        weights=list(final_probs.values()),
        k=1
    )[0]

    # Build the BanditAction result (confidence bound is a placeholder)
    chosen_action = BanditAction(
        action_id=chosen_id,
        propensity=final_probs[chosen_id],
        expected_reward=_surrogate_estimate(context_vector, chosen_id),
        confidence_bound=0.0,               # could be derived from variance of kernel weights
        algorithm="HybridBanditRegretRBF"
    )
    return chosen_action

def diagnostic_report(action_ids: List[str]) -> Tuple[float, float]:
    """
    Compute entropy and Gini coefficient of the current empirical reward distribution
    across the supplied action identifiers.  This function showcases the
    statistical tools inherited from Parent B.
    """
    rewards = [_empirical_reward(aid) for aid in action_ids]
    probs = [r / sum(rewards) if sum(rewards) > 0 else 0.0 for r in rewards]
    entropy = shannon_entropy(probs)
    gini = gini_coefficient(rewards)
    return entropy, gini

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset any prior state
    reset_policy()

    # Define a tiny synthetic environment
    actions = ["A", "B", "C"]
    contexts = {
        "c1": np.array([0.1, 0.2]),
        "c2": np.array([0.4, 0.5]),
        "c3": np.array([0.9, 0.1]),
    }

    # Simulate a few interactions
    for step in range(15):
        ctx_id = random.choice(list(contexts.keys()))
        ctx_vec = contexts[ctx_id]
        # Hybrid decision
        chosen = hybrid_select_action(
            context_id=ctx_id,
            context_vector=ctx_vec,
            available_action_ids=actions,
            counterfactuals=None,
            exploration_epsilon=0.2
        )
        # Simulated stochastic reward (just a random function of context)
        reward = float(np.dot(ctx_vec, np.random.rand(2))) + random.random()
        # Update policy with the observed reward
        update = BanditUpdate(
            context_id=ctx_id,
            action_id=chosen.action_id,
            reward=reward,
            propensity=chosen.propensity
        )
        update_policy(update, ctx_vec)

        if step % 5 == 0:
            ent, gin = diagnostic_report(actions)
            print(f"Step {step:02d} – chosen {chosen.action_id} (p={chosen.propensity:.3f})")
            print(f"    Entropy={ent:.3f}, Gini={gin:.3f}")

    print("\nFinal empirical rewards:")
    for aid in actions:
        print(f"  {aid}: total={_POLICY.get(aid, [0,0])[0]:.2f}, count={_POLICY.get(aid, [0,0])[1]}")
    print("\nHybrid test completed without errors.")