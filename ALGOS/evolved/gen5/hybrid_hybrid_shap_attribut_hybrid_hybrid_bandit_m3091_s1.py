# DARWIN HAMMER — match 3091, survivor 1
# gen: 5
# parent_a: hybrid_shap_attribution_hybrid_hybrid_pherom_m70_s1.py (gen4)
# parent_b: hybrid_hybrid_bandit_router_hybrid_hybrid_endpoi_m134_s3.py (gen3)
# born: 2026-05-29T23:47:45Z

"""Hybrid Shapley‑Bandit Algorithm
=================================

This module fuses the two parent algorithms:

* **Parent A** – ``hybrid_shap_attribution_hybrid_hybrid_pherom_m70_s1.py``  
  Provides Shapley value computation, hash‑based similarity (``compute_phash``,
  ``compute_dhash``) and a broadcast‑probability driven leader election.

* **Parent B** – ``hybrid_hybrid_bandit_router_hybrid_hybrid_endpoi_m134_s3.py``  
  Supplies a contextual multi‑armed bandit framework, a temperature‑performance
  (Schoolfield) curve and a circuit‑breaker for unreliable nodes.

**Mathematical bridge**  
The Shapley value of a feature (or graph node) is an *expected marginal
contribution* – exactly the quantity needed as the ``expected_reward`` of a
bandit arm.  Conversely, the bandit’s confidence bound can be expressed through
the Hamming distance between the node’s ``phash`` and ``dhash`` signatures,
providing a data‑driven uncertainty estimate.  The Schoolfield curve,
originally modelling temperature‑dependent rates, is repurposed to *modulate*
the broadcast probability of the leader‑election phase, turning environmental
temperature into a soft‑capacity control.

The resulting hybrid algorithm:

1. Computes Shapley values for all nodes (features) → bandit ``expected_reward``.
2. Derives hash signatures → Hamming distance → bandit ``confidence_bound``.
3. Scales the broadcast probability with the Schoolfield rate → temperature‑aware
   leader election.
4. Uses a per‑node ``EndpointCircuitBreaker`` to discard nodes that repeatedly
   fail the election.

The public API consists of three core functions that showcase this integration.
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types shared by both parents
# ----------------------------------------------------------------------
Node = int
Graph = Dict[int, Set[int]]
Model = Dict[int, float]

# ----------------------------------------------------------------------
# Parent A utilities (shapley, hashes, broadcast probability)
# ----------------------------------------------------------------------
def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Weight factor of the Shapley kernel."""
    return (
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / math.factorial(feature_count)
    )


def compute_phash(values: List[float]) -> int:
    """Binary hash based on the mean of the values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def compute_dhash(values: List[float]) -> int:
    """Directional hash: 1 if successive value increases, else 0."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer bit‑patterns."""
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    """Base probability used in the original leader election."""
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


# ----------------------------------------------------------------------
# Parent B utilities (bandit structures, Schoolfield curve, circuit breaker)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit arm (here a graph node)."""
    node: Node
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridShapleyBandit"


@dataclass(frozen=True)
class BanditUpdate:
    """Record of a single interaction with a node."""
    node: Node
    reward: float
    propensity: float = 1.0  # default uniform propensity


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Schoolfield temperature‑performance curve."""
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal·mol⁻¹·K⁻¹ (≈8.314 J·mol⁻¹·K⁻¹)


@dataclass
class EndpointCircuitBreaker:
    """Tracks consecutive election failures for a node."""
    failure_threshold: int = 3
    failures: int = 0

    def record_failure(self) -> None:
        self.failures = min(self.failures + 1, self.failure_threshold)

    def reset(self) -> None:
        self.failures = 0

    @property
    def is_tripped(self) -> bool:
        return self.failures >= self.failure_threshold


# ----------------------------------------------------------------------
# Global bandit policy store (mirrors Parent B)
# ----------------------------------------------------------------------
_POLICY: Dict[Node, List[float]] = defaultdict(lambda: [0.0, 0.0])  # [cumulative_reward, pulls]


def reset_policy() -> None:
    """Clear all learned statistics."""
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """In‑place update of the global policy using a batch of observations."""
    for u in updates:
        stats = _POLICY[u.node]
        stats[0] += float(u.reward)
        stats[1] += 1.0


def estimate_reward(node: Node) -> float:
    """Return the empirical mean reward for a node (0 if never pulled)."""
    cum, pulls = _POLICY[node]
    return cum / pulls if pulls > 0 else 0.0


def confidence_bound(node: Node, total_pulls: int) -> float:
    """UCB‑style confidence bound (sqrt term)."""
    _, pulls = _POLICY[node]
    if pulls == 0:
        return float("inf")
    return math.sqrt(2 * math.log(total_pulls + 1) / pulls)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_shapley_all(
    feature_count: int, value_fn: Callable[[frozenset[int]], float]
) -> Dict[int, float]:
    """
    Compute Shapley values for every feature index 0 … feature_count‑1.

    The implementation follows Parent A's ``shap_value`` but returns the full
    vector in a single pass.
    """
    shapley: Dict[int, float] = {i: 0.0 for i in range(feature_count)}
    # Enumerate all subsets once and distribute marginal contributions.
    for k in range(feature_count + 1):
        weight = shapley_kernel_weight(k, feature_count)
        for subset in combinations(range(feature_count), k):
            s = frozenset(subset)
            base = value_fn(s)
            for i in range(feature_count):
                if i not in s:
                    marginal = value_fn(s | {i}) - base
                    shapley[i] += weight * marginal
    return shapley


def temperature_weighted_probability(
    phase: int,
    step: int,
    temperature_K: float,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> float:
    """
    Combine Parent A's ``broadcast_probability`` with the Schoolfield
    temperature curve from Parent B.

    The raw probability is multiplied by a normalized rate ``r(T) / r(25°C)``.
    """
    base = broadcast_probability(phase, step)
    # Schoolfield rate (relative to 25 °C)
    def rate(T: float) -> float:
        term1 = params.rho_25 * math.exp(
            -params.delta_h_activation / (params.r_cal * (T - 273.15))
        )
        term2 = 1 + math.exp(
            (params.delta_h_low / params.r_cal) * (1 / params.t_low - 1 / T)
        )
        term3 = 1 + math.exp(
            (params.delta_h_high / params.r_cal) * (1 / T - 1 / params.t_high)
        )
        return term1 / (term2 * term3)

    r25 = rate(298.15)  # 25 °C in Kelvin
    rT = rate(temperature_K)
    temp_factor = rT / r25 if r25 != 0 else 1.0
    return min(1.0, base * temp_factor)


def hybrid_leader_bandit_election(
    graph: Graph,
    node_values: List[float],
    temperature_K: float,
    rng_seed: int | str | None = None,
) -> Set[Node]:
    """
    Perform a leader election that is guided by a contextual bandit whose
    expected rewards are Shapley values.  The election proceeds in phases; in
    each phase a subset of undecided nodes broadcast with a temperature‑aware
    probability.  Nodes that broadcast without neighbour interference become
    leaders.  Nodes that repeatedly fail are disabled by an ``EndpointCircuitBreaker``.

    Returns the set of elected leader nodes.
    """
    rng = random.Random(rng_seed)
    node_list = list(graph.keys())
    # 1️⃣ Shapley values → expected rewards for bandit arms
    def value_fn(subset: frozenset[int]) -> float:
        # Simple additive model: sum of node values in the subset
        return sum(node_values[i] for i in subset)

    shapley_vals = compute_shapley_all(len(node_list), value_fn)

    # Initialise per‑node structures
    breakers: Dict[Node, EndpointCircuitBreaker] = {
        n: EndpointCircuitBreaker() for n in node_list
    }
    leaders: Set[Node] = set()
    undecided: Set[Node] = set(node_list)

    total_pulls = 0  # for confidence bound scaling

    for phase in range(1, 9):  # mirrors Parent A's 8 phases
        if not undecided:
            break

        prob = temperature_weighted_probability(
            phase=8, step=phase, temperature_K=temperature_K
        )
        broadcasts = {n for n in undecided if rng.random() < prob}
        if not broadcasts:
            continue

        # Build BanditAction objects for the broadcasting nodes
        actions: List[BanditAction] = []
        for n in broadcasts:
            total_pulls += 1
            exp_r = shapley_vals[node_list.index(n)]
            # Confidence bound derived from hash similarity
            sub_vals = [node_values[node_list.index(m)] for m in undecided]
            ph = compute_phash(sub_vals)
            dh = compute_dhash(sub_vals)
            cb = hamming_distance(ph, dh) / 64.0  # normalised distance as uncertainty
            actions.append(
                BanditAction(node=n, expected_reward=exp_r, confidence_bound=cb)
            )

        # Choose the action with highest Upper Confidence Estimate (UCE)
        def uce(act: BanditAction) -> float:
            return act.expected_reward + act.confidence_bound

        chosen = max(actions, key=uce)

        # Verify that the chosen node has no neighbour also broadcasting
        if not (graph.get(chosen.node, set()) & broadcasts):
            leaders.add(chosen.node)
            undecided.remove(chosen.node)
            # Successful election → reset its breaker
            breakers[chosen.node].reset()
            # Record a positive reward in the bandit policy
            update_policy([BanditUpdate(node=chosen.node, reward=chosen.expected_reward)])
        else:
            # Failure: record a zero reward and increment breaker
            update_policy([BanditUpdate(node=chosen.node, reward=0.0)])
            breakers[chosen.node].record_failure()
            if breakers[chosen.node].is_tripped:
                # Node is permanently excluded
                undecided.remove(chosen.node)

    return leaders


# ----------------------------------------------------------------------
# Additional demonstration function
# ----------------------------------------------------------------------
def simulate_hybrid_process(
    graph: Graph,
    node_values: List[float],
    temperature_series: Iterable[float],
    rng_seed: int | str | None = None,
) -> List[Set[Node]]:
    """
    Run ``hybrid_leader_bandit_election`` for a sequence of temperatures,
    returning the list of leader sets per step.  This illustrates how the
    temperature‑dependent broadcast probability shapes the dynamics.
    """
    results: List[Set[Node]] = []
    for temp in temperature_series:
        leaders = hybrid_leader_bandit_election(
            graph, node_values, temperature_K=temp, rng_seed=rng_seed
        )
        results.append(leaders)
        # Reset global policy between steps to keep each simulation independent
        _POLICY.clear()
    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple undirected graph of 5 nodes
    sample_graph: Graph = {
        0: {1, 2},
        1: {0, 3},
        2: {0, 4},
        3: {1},
        4: {2},
    }

    # Arbitrary node values (could be model coefficients, feature importances, …)
    node_vals = [0.8, 0.3, 0.5, 0.9, 0.2]

    # Temperature sweep from 15 °C to 35 °C (in Kelvin)
    temps = np.linspace(288.15, 308.15, num=5)

    leaders_per_temp = simulate_hybrid_process(
        sample_graph, node_vals, temps, rng_seed=42
    )

    for t, leaders in zip(temps, leaders_per_temp):
        print(f"Temp {t - 273.15:5.1f} °C → Leaders: {sorted(leaders)}")