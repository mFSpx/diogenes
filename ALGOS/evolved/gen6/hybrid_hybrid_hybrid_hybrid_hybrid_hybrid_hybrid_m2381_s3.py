# DARWIN HAMMER — match 2381, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_fold_c_hybrid_hybrid_hybrid_m653_s0.py (gen5)
# born: 2026-05-29T23:42:12Z

"""Hybrid Geometric‑Product LTC + Fold‑Change Bandit Fusion
========================================================

Parents
-------
* **Parent A** – *Hybrid Geometric Product‑LTC*  
  Provides a Clifford‑algebra multivector representation of a weight matrix
  `W` and a geometric product based update rule.

* **Parent B** – *Hybrid Fold‑Change Detection & Pheromone Infotaxis*  
  Supplies a contextual bandit with counts, log‑count ratios and a fold‑change
  detector for scalar signals.

Mathematical Bridge
-------------------
Both parents manipulate a *matrix‑like* object:

* In **A**, `W` is a multivector `𝑊 = Σ c_I e_I` (sum of basis blades) that is
  updated by a geometric product `𝑊 ← 𝑊 ⊙ X` where `X` is the input multivector.

* In **B**, the update strength is driven by a *log‑count ratio*  
  `λ = log(count_a / count_b)` multiplied by a *fold‑change* term
  `φ = log(x/ε)`.

The fusion therefore replaces the scalar learning‑rate `η` of the LTC with the
product `λ·φ`.  The resulting hybrid update is


𝑊_{t+1} = (λ·φ) · (𝑊_t ⊙ X_t)


where `⊙` denotes the Clifford geometric product.  This couples the
information‑theoretic bandit statistics directly to the algebraic weight
evolution, yielding a single unified system.

The module below implements this hybrid dynamics, exposing three core
functions:

1. `geometric_product(a, b)` – Clifford product of two `Multivector`s.
2. `hybrid_update(W, X, action_id)` – Performs the fused LTC‑bandit update.
3. `select_action(context_id, actions)` – Bandit‑style action selection that
   feeds back into the update rule.

A minimal smoke‑test demonstrates a complete cycle without external
dependencies.
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Clifford algebra utilities (Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return a sorted blade and its sign after bubble‑sorting.
    Duplicate indices cancel (e ↔ e = 0)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate basis vectors
                lst.pop(j)
                lst.pop(j)  # second element now at position j
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Simple multivector for Cl(n,0)."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # store only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}
        self.n = int(n)

    def __add__(self, other: "Multivector") -> "Multivector":
        comp = self.components.copy()
        for k, v in other.components.items():
            comp[k] = comp.get(k, 0.0) + v
        return Multivector(comp, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({k: v * scalar for k, v in self.components.items()}, self.n)

    __rmul__ = __mul__

    def copy(self) -> "Multivector":
        return Multivector(self.components.copy(), self.n)

    def __repr__(self) -> str:
        terms = [f"{v:.3g}*e{sorted(list(k))}" if k else f"{v:.3g}" for k, v in self.components.items()]
        return " + ".join(terms) if terms else "0"


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Full geometric product a ⊙ b."""
    result: Dict[FrozenSet[int], float] = defaultdict(float)
    for blade_a, coeff_a in a.components.items():
        for blade_b, coeff_b in b.components.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            result[blade_res] += coeff_a * coeff_b * sign
    return Multivector(dict(result), a.n)


# ----------------------------------------------------------------------
# Bandit utilities (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridLTCBandit"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, List[float]] = {}  # action_id → [cumulative_reward, count]


def reset_policy() -> None:
    """Clear the internal bandit statistics."""
    _POLICY.clear()


def _record(action: str, reward: float) -> None:
    total, cnt = _POLICY.get(action, [0.0, 0.0])
    _POLICY[action] = [total + reward, cnt + 1.0]


def _reward(action: str) -> float:
    """Mean reward for an action."""
    total, cnt = _POLICY.get(action, [0.0, 0.0])
    return total / cnt if cnt else 0.0


def _count(action: str) -> float:
    """Number of times an action has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]


def _log_count_ratio(action_a: str, action_b: str) -> float:
    """log(count_a / (count_b + 1)) with a small epsilon for stability."""
    cnt_a = _count(action_a) + 1e-9
    cnt_b = _count(action_b) + 1e-9
    return math.log(cnt_a / cnt_b)


def _fold_change_detection(x: float, eps: float = 1e-9) -> float:
    """Fold‑change detector used as a scalar learning‑rate surrogate."""
    return math.log(abs(x) / max(abs(x), eps) + eps)


def _hybrid_store_factor(action_id: str, reference_action: str) -> float:
    """Hybrid scaling factor λ = log‑count‑ratio * fold‑change."""
    lc_ratio = _log_count_ratio(action_id, reference_action)
    fc = _fold_change_detection(_reward(action_id))
    return lc_ratio * fc


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_update(
    W: Multivector,
    X: Multivector,
    action_id: str,
    reference_action: str = "baseline",
) -> Multivector:
    """
    Perform one hybrid LTC‑bandit update.

    1. Compute geometric product `P = W ⊙ X`.
    2. Determine hybrid scaling factor `λ = log_count_ratio * fold_change`.
    3. Return the scaled product `W' = λ·P`.

    The function does **not** modify `W` in‑place; a new multivector is
    returned.
    """
    # 1. geometric product
    P = geometric_product(W, X)

    # 2. hybrid scaling factor
    λ = _hybrid_store_factor(action_id, reference_action)

    # 3. scaled update
    W_next = P * λ
    return W_next


def select_action(context_id: str, candidate_ids: Iterable[str]) -> BanditAction:
    """
    Choose an action using an Upper‑Confidence‑Bound (UCB) style rule.
    The returned `BanditAction` contains the propensity (selection probability)
    and the current expected reward.
    """
    # Ensure every candidate appears in the policy
    for aid in candidate_ids:
        if aid not in _POLICY:
            _POLICY[aid] = [0.0, 0.0]

    # Compute UCB scores
    total_counts = sum(_count(a) for a in candidate_ids) + 1e-9
    scores: List[Tuple[float, str]] = []
    for aid in candidate_ids:
        avg = _reward(aid)
        cnt = _count(aid)
        # exploration term scales with sqrt(log(N)/n)
        explore = math.sqrt(math.log(total_counts) / (cnt + 1e-9))
        ucb = avg + explore
        scores.append((ucb, aid))

    # Pick the best
    best_score, best_id = max(scores, key=lambda t: t[0])
    propensity = best_score / sum(s for s, _ in scores)  # normalized
    return BanditAction(
        action_id=best_id,
        propensity=propensity,
        expected_reward=_reward(best_id),
        confidence_bound=best_score,
    )


def apply_bandit_feedback(update: BanditUpdate) -> None:
    """
    Record the observed reward and update internal statistics.
    """
    _record(update.action_id, update.reward)


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a simple 2‑dimensional Clifford algebra (e1, e2)
    n_dim = 2
    # Weight multivector W = 1 + 0.5 e1 + 0.3 e12
    W = Multivector(
        {
            frozenset(): 1.0,
            frozenset({1}): 0.5,
            frozenset({1, 2}): 0.3,
        },
        n=n_dim,
    )
    # Input multivector X = 0.2 e2 + 0.7 e12
    X = Multivector(
        {
            frozenset({2}): 0.2,
            frozenset({1, 2}): 0.7,
        },
        n=n_dim,
    )

    # Initialise a tiny bandit policy with two actions
    reset_policy()
    _record("explore", 0.1)
    _record("exploit", 0.4)

    # Choose an action for the current context
    action = select_action(context_id="ctx1", candidate_ids=["explore", "exploit"])
    print("Selected action:", action)

    # Perform hybrid update using the selected action
    W_new = hybrid_update(W, X, action_id=action.action_id, reference_action="exploit")
    print("Old W :", W)
    print("Input X:", X)
    print("Updated W:", W_new)

    # Simulate receiving a reward and feed it back
    simulated_reward = random.random()  # placeholder reward
    feedback = BanditUpdate(
        context_id="ctx1",
        action_id=action.action_id,
        reward=simulated_reward,
        propensity=action.propensity,
    )
    apply_bandit_feedback(feedback)

    # Show updated policy statistics
    print("Policy after feedback:")
    for aid, (tot, cnt) in _POLICY.items():
        print(f"  {aid}: total_reward={tot:.3f}, count={cnt}")

    sys.exit(0)