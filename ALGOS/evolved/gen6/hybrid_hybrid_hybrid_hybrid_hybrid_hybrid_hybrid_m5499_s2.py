# DARWIN HAMMER — match 5499, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2050_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m1193_s0.py (gen5)
# born: 2026-05-30T00:02:20Z

"""
Hybrid Algorithm: Fusion of hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2050_s0.py (A)
and hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m1193_s0.py (B).

Mathematical Bridge
-------------------
Both parents expose a scalar “store” variable that can be used as a global
modulator:

* A – ``StoreState.level`` controls the width of a Gaussian beam and the
  risk‑score scaling.
* B – the same scalar is used to modulate the pruning probability
  ``p_i(t)`` in the Bayesian update of edge weights.

The hybrid therefore treats ``S = StoreState.level`` as a *bridge*:
the Gaussian width σ = base / (1 + gain·S) feeds the bandit propensity,
while the Bayesian pruning factor is multiplied by a monotone function of
S (here ``exp(-λ·S)``).  The MinHash signature provides a discrete
probability distribution that biases the bandit’s propensity and also
weights the Bayesian update of the minimum‑cost tree.

The resulting system:
1. Update the store (A).
2. Use the updated store to scale a Bayesian edge‑weight update (B).
3. Use the same store together with a MinHash‑derived bucket frequency to
   compute bandit propensities (A) → final action selection.

The three core functions below demonstrate this integrated workflow.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (shared)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "Hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Store dynamics (from Parent A)
# ----------------------------------------------------------------------


@dataclass
class StoreState:
    """Encapsulates the honeybee‑style store and its derived control signal."""
    level: float = 0.0      # scalar store level S
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply a simple leaky‑integrator store equation."""
        net = sum(inflow) - sum(outflow)
        new_level = self.level + self.dt * (self.alpha * net - self.beta * self.level)
        new_level = max(0.0, min(new_level, self.limit))
        self.level = new_level
        # Return (new_level, net) for convenience
        return new_level, net

    def gaussian_width(self) -> float:
        """Width σ of the Gaussian beam, inversely proportional to store level."""
        return self.base / (1.0 + self.gain * self.level)


# ----------------------------------------------------------------------
# Bayesian utilities (from Parent B)
# ----------------------------------------------------------------------


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Minimum‑cost tree representation (simplified)
# ----------------------------------------------------------------------


Edge = Tuple[str, str]          # (node_u, node_v)
Weight = float


@dataclass
class CostTree:
    """Undirected graph with edge weights representing a minimum‑cost tree."""
    edges: List[Edge] = field(default_factory=list)
    weights: Dict[Edge, Weight] = field(default_factory=dict)

    def add_edge(self, u: str, v: str, w: float) -> None:
        e = tuple(sorted((u, v)))
        if e not in self.weights:
            self.edges.append(e)
        self.weights[e] = w

    def total_cost(self) -> float:
        return sum(self.weights[e] for e in self.edges)


# ----------------------------------------------------------------------
# MinHash signature (discrete probability distribution)
# ----------------------------------------------------------------------


def simple_minhash(tokens: Iterable[str], buckets: int = 64) -> Dict[int, float]:
    """
    Produce a MinHash‑like frequency distribution over ``buckets``.
    For each token we compute ``hash(token) % buckets`` and count occurrences.
    The returned dict maps bucket index -> normalized frequency.
    """
    counts = np.zeros(buckets, dtype=float)
    for t in tokens:
        idx = hash(t) % buckets
        counts[idx] += 1.0
    total = counts.sum()
    if total == 0:
        return {i: 0.0 for i in range(buckets)}
    return {i: counts[i] / total for i in range(buckets)}


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def compute_gaussian_beam(state: StoreState, x: float) -> float:
    """
    Gaussian beam value at position ``x`` with width σ modulated by the store.
    """
    sigma = state.gaussian_width()
    return math.exp(-0.5 * (x / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def bayesian_edge_update(
    tree: CostTree,
    prior: float,
    likelihood: float,
    false_positive: float,
    state: StoreState,
    lam: float = 0.1,
) -> None:
    """
    Update every edge weight using a Bayesian rule whose pruning probability
    is scaled by ``exp(-lam * S)`` where ``S`` is the store level.
    The updated weight is ``w * posterior`` to favour edges with higher posterior.
    """
    scaling = math.exp(-lam * state.level)  # bridge factor
    for e in tree.edges:
        prior_e = prior
        marginal = bayes_marginal(prior_e, likelihood, false_positive)
        posterior = bayes_update(prior_e, likelihood, marginal)
        # Apply bridge scaling
        posterior_scaled = posterior * scaling
        # Ensure weight stays positive
        tree.weights[e] = max(1e-6, tree.weights[e] * posterior_scaled)


def hybrid_select_action(
    state: StoreState,
    actions: List[str],
    tokens: List[str],
    base_reward: Callable[[str], float],
    exploration_coef: float = 1.0,
) -> BanditAction:
    """
    Select an action using a bandit‑style propensity that blends:
    * Gaussian beam value (depends on store level)
    * MinHash bucket frequency of the token set (bias)
    * Upper‑confidence bound (UCB) style confidence term.
    """
    # 1. MinHash distribution over a small bucket set
    signature = simple_minhash(tokens, buckets=32)
    # 2. Compute raw scores for each action
    scores = {}
    for a in actions:
        # Position x is derived from a hash to map into the Gaussian domain
        x = (hash(a) % 1000) / 100.0  # arbitrary scaling
        beam = compute_gaussian_beam(state, x)
        # Bias from MinHash: pick bucket based on action hash
        bucket = hash(a) % 32
        bias = signature.get(bucket, 0.0)
        # Expected reward from external oracle
        exp_r = base_reward(a)
        # Confidence bound (UCB) using store level as pseudo‑count
        confidence = exploration_coef * math.sqrt(math.log(1 + state.level) / (1 + bias + 1e-9))
        # Propensity proportional to beam * (exp_r + bias)
        propensity = beam * (exp_r + bias)
        scores[a] = (propensity, exp_r, confidence)

    # 3. Choose action with maximal propensity + confidence (simple policy)
    best_action = max(scores.items(), key=lambda kv: kv[1][0] + kv[1][2])[0]
    prop, exp_r, conf = scores[best_action]

    return BanditAction(
        action_id=best_action,
        propensity=prop,
        expected_reward=exp_r,
        confidence_bound=conf,
        algorithm="Hybrid",
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # 1. Initialise store and perform an update
    store = StoreState(level=2.0, alpha=0.8, beta=0.2, dt=0.5, base=1.0, gain=0.5, limit=5.0)
    inflow = [0.3, 0.4]
    outflow = [0.1]
    new_level, net = store.update(inflow, outflow)
    print(f"Store updated: level={new_level:.3f}, net={net:.3f}")

    # 2. Build a tiny cost tree and apply Bayesian edge update
    tree = CostTree()
    tree.add_edge("A", "B", 1.0)
    tree.add_edge("B", "C", 2.0)
    tree.add_edge("C", "A", 3.0)
    print(f"Initial total cost: {tree.total_cost():.3f}")
    bayesian_edge_update(
        tree,
        prior=0.6,
        likelihood=0.7,
        false_positive=0.1,
        state=store,
        lam=0.05,
    )
    print(f"Post‑update total cost: {tree.total_cost():.3f}")

    # 3. Define actions and tokens, then select an action
    actions = ["move_left", "move_right", "stay", "jump"]
    tokens = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]

    def dummy_reward(action_id: str) -> float:
        # Simple deterministic reward mapping for testing
        return {"move_left": 0.2, "move_right": 0.5, "stay": 0.1, "jump": 0.8}.get(action_id, 0.0)

    selected = hybrid_select_action(
        state=store,
        actions=actions,
        tokens=tokens,
        base_reward=dummy_reward,
        exploration_coef=0.5,
    )
    print(f"Selected action: {selected.action_id}")
    print(f"Propensity: {selected.propensity:.6f}, Expected Reward: {selected.expected_reward:.3f}, Confidence: {selected.confidence_bound:.3f}")

    # Ensure no unhandled exceptions
    sys.exit(0)