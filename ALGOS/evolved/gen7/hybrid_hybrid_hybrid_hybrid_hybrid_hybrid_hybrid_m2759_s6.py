# DARWIN HAMMER — match 2759, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_regret_m746_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m2009_s0.py (gen6)
# born: 2026-05-29T23:45:35Z

"""
Hybrid Algorithm: Fusion of
- Parent A (hybrid_hybrid_hybrid_bandit_hybrid_hybrid_regret_m746_s0): regret‑weighted
  ternary decision analysis with bandit propensity selection.
- Parent B (hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m2009_s0): temperature‑dependent
  learning‑rate curvature proxy together with a Count‑Min Sketch (CMS) estimator.

Mathematical Bridge
-------------------
The CMS provides a compact estimate of the frequency (propensity) of each
action and of the ratio ρ = (unique‑estimated actions) / (total updates).  
We use ρ as a *temperature* τ = 1 + ρ in the curvature computation
`learning_rate = α·τ`.  The same τ scales the regret‑weighted probability
distribution:

    w_i = exp( τ · (R_max – R_i) ) · p̂_i

where `R_i` is the empirical reward of action i, `R_max` the best observed
reward, and `p̂_i` the CMS‑estimated propensity.  The hybrid selector draws
an action proportionally to w_i, thus unifying the bandit core of Parent A
with the temperature‑modulated curvature of Parent B.

The module implements:
1. A lightweight Count‑Min Sketch.
2. Curvature computation that consumes the CMS‑derived temperature.
3. A hybrid action selector that merges regret weighting and CMS propensity.
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Simple Count‑Min Sketch (CMS) – parent B component
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    A deterministic Count‑Min Sketch with d hash functions and width w.
    Only integer counts are stored; collisions are handled by taking the
    minimum across rows.
    """
    def __init__(self, depth: int = 5, width: int = 2000, seed: int = 42):
        self.depth = depth
        self.width = width
        self.tables = [defaultdict(int) for _ in range(depth)]
        rng = random.Random(seed)
        self.seeds = [rng.randint(0, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: str, i: int) -> int:
        # Simple mixed‑multiply hash; deterministic for given seeds
        h = hash((item, self.seeds[i]))
        return h % self.width

    def add(self, item: str, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.tables[i][idx] += increment

    def estimate(self, item: str) -> int:
        """Return the minimum count across hash rows."""
        return min(self.tables[i][self._hash(item, i)] for i in range(self.depth))

    def total_count(self) -> int:
        """Approximate total number of updates (sum of first row)."""
        return sum(self.tables[0].values())

    def unique_estimate(self) -> int:
        """Upper‑bound estimate of distinct items using the CMS trick."""
        # Use the harmonic mean of row occupancies as a crude estimator.
        occupancies = [len(row) for row in self.tables]
        if not occupancies:
            return 0
        return int(self.depth / sum(1.0 / o if o else 0 for o in occupancies))

# ----------------------------------------------------------------------
# Parent A – Regret‑Weighted Bandit Core (trimmed)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _update_policy(action: str, reward: float) -> None:
    total, n = _POLICY.get(action, [0.0, 0.0])
    _POLICY[action] = [total + reward, n + 1]

# ----------------------------------------------------------------------
# Parent B – Temperature‑Dependent Curvature (trimmed)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class KrampusParams:
    base_alpha: float = 0.1   # base learning rate

def compute_curvature(adj_matrix: np.ndarray, temperature: float) -> np.ndarray:
    """
    Ollivier‑Ricci curvature proxy where the learning rate is scaled by
    temperature τ.  For each node i the curvature is the sum of incident
    edge weights multiplied by α·τ.
    """
    n = adj_matrix.shape[0]
    alpha = KrampusParams().base_alpha * temperature
    curvature = np.zeros(n)
    for i in range(n):
        curvature[i] = alpha * np.sum(adj_matrix[i, :])
    return curvature

# ----------------------------------------------------------------------
# Hybrid Functions (the required three+ functions)
# ----------------------------------------------------------------------
def estimate_temperature(sketch: CountMinSketch) -> float:
    """
    Derive a temperature τ from the sketch:
        τ = 1 + (unique_estimate / total_updates)
    This yields τ∈[1,2] for typical workloads and smoothly modulates
    both curvature learning‑rate and regret weighting.
    """
    total = sketch.total_count()
    if total == 0:
        return 1.0
    unique = sketch.unique_estimate()
    return 1.0 + (unique / total)

def hybrid_curvature(adj_matrix: np.ndarray, sketch: CountMinSketch) -> np.ndarray:
    """
    Compute curvature using a temperature derived from the sketch.
    """
    τ = estimate_temperature(sketch)
    return compute_curvature(adj_matrix, τ)

def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    adj_matrix: np.ndarray,
    sketch: CountMinSketch,
    algorithm: str = "epsilon_greedy",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Hybrid action selector.

    1. Update the sketch with each candidate action (simulating a prior
       observation count).  This yields CMS‑based propensities.
    2. Compute temperature τ from the sketch and curvature (optional use).
    3. Form regret‑weighted probabilities:
           w_i = exp( τ·(R_max – R_i) ) * p̂_i
       where p̂_i = sketch.estimate(action_i) / sketch.total_count().
    4. Apply the chosen bandit algorithm (epsilon‑greedy or Thompson).
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # ---- Step 1: update sketch (pretend we observed each action once) ----
    for a in actions:
        sketch.add(a, increment=1)

    total_updates = sketch.total_count()
    propensities = {
        a: sketch.estimate(a) / total_updates if total_updates else 1.0 / len(actions)
        for a in actions
    }

    # ---- Step 2: temperature from sketch ----
    τ = estimate_temperature(sketch)

    # ---- Step 3: regret‑weighted distribution ----
    rewards = {a: _reward(a) for a in actions}
    R_max = max(rewards.values()) if rewards else 0.0
    weights = {
        a: math.exp(τ * (R_max - rewards[a])) * propensities[a]
        for a in actions
    }
    weight_sum = sum(weights.values())
    probabilities = {a: w / weight_sum for a, w in weights.items()}

    # ---- Step 4: bandit algorithm choice ----
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen_id = rng.choice(actions)
    elif algorithm == "thompson":
        # Sample from Beta using estimated reward as success count
        chosen_id = max(
            actions,
            key=lambda a: rng.betavariate(1 + max(0.0, rewards[a]),
                                          1 + max(0.0, 1.0 - rewards[a]))
        )
    else:
        # Default: sample according to the hybrid probabilities
        r = rng.random()
        cumulative = 0.0
        chosen_id = actions[-1]  # fallback
        for a in actions:
            cumulative += probabilities[a]
            if r <= cumulative:
                chosen_id = a
                break

    # Build BanditAction result
    return BanditAction(
        action_id=chosen_id,
        propensity=propensities[chosen_id],
        expected_reward=_reward(chosen_id),
        confidence_bound=τ,          # repurposed as a proxy confidence metric
        algorithm=algorithm,
    )

def hybrid_update(
    update: BanditUpdate,
    sketch: CountMinSketch,
    adj_matrix: np.ndarray,
) -> None:
    """
    Perform a policy update with a new reward and simultaneously adjust the
    curvature learning rate via the sketch‑derived temperature.
    """
    # Update policy statistics
    _update_policy(update.action_id, update.reward)

    # Update sketch with the observed action (reinforce its propensity)
    sketch.add(update.action_id, increment=1)

    # Optionally recompute curvature (side‑effect for external callers)
    _ = hybrid_curvature(adj_matrix, sketch)  # result can be cached elsewhere

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph (3 nodes, fully connected with weight 1)
    adj = np.array([[0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]], dtype=float)

    actions = ["A", "B", "C"]
    context = {"feature1": 0.5, "feature2": -0.2}

    cms = CountMinSketch(depth=4, width=1024, seed=123)

    # Perform a few updates to seed the sketch and policy
    for _ in range(5):
        ba = hybrid_select_action(
            context=context,
            actions=actions,
            adj_matrix=adj,
            sketch=cms,
            algorithm="epsilon_greedy",
            epsilon=0.2,
            seed=42,
        )
        print(f"Selected: {ba}")

        # Simulate a stochastic reward
        reward = random.random()
        upd = BanditUpdate(
            context_id="ctx1",
            action_id=ba.action_id,
            reward=reward,
            propensity=ba.propensity,
        )
        hybrid_update(upd, cms, adj)

    # Final curvature view
    curv = hybrid_curvature(adj, cms)
    print("Curvature vector:", curv)