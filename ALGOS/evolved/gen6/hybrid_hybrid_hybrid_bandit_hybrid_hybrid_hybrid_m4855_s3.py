# DARWIN HAMMER — match 4855, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_tri_algo_cond_m1011_s1.py (gen5)
# born: 2026-05-29T23:58:25Z

"""
Hybrid Bandit‑Sketch RLCT Module
================================

This module fuses the two parent algorithms:

* **Parent A** – a contextual multi‑armed bandit that maintains a *store* (cumulative
  reward) and uses a confidence bound for action selection.
* **Parent B** – sketch primitives (Count‑Min, HyperLogLog) together with a
  singular‑learning‑theory (RLCT) estimator
  ``hybrid_hybrid_sketches_rlct_cockpit_estimate`` that combines a sketch‑derived
  log‑likelihood, an Ollivier‑Ricci curvature term from feature statistics and
  a Shannon‑entropy penalty.

The mathematical bridge is the **log‑count statistics** that appear in both
families:

1. The bandit’s reward estimator ``μ̂`` is replaced by the **mean derived from a
   Count‑Min sketch** of the reward stream per arm.
2. The effective sample size ``N`` in the UCB confidence term ``log(N)`` is
   estimated by a **HyperLogLog sketch** of the distinct contexts observed.
3. The exploration coefficient ``α`` is no longer a fixed constant but is set
   to the **RLCT‑derived scalar λ** produced by the Parent B estimator.

Consequently the Upper‑Confidence‑Bound for arm *a* becomes  


UCB_a = μ̂_a  +  λ * sqrt( log( N̂ ) / ( n_a + 1 ) )


where  

* ``μ̂_a`` – mean reward from the Count‑Min sketch of arm *a*,  
* ``N̂``   – cardinality estimate from the HyperLogLog sketch of contexts,  
* ``n_a``  – number of times arm *a* has been pulled,  
* ``λ``    – RLCT estimate returned by ``rlct_estimate``.

The code below implements this hybrid system, exposing three core functions
demonstrating the integration and a simple smoke‑test when run as a script.
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Sketch primitives (from Parent B)
# ----------------------------------------------------------------------


def count_min_sketch(
    items: Iterable[str], width: int = 128, depth: int = 4
) -> List[List[int]]:
    """Return a Count‑Min sketch matrix for the given items."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        item_bytes = str(item).encode("utf-8")
        for d in range(depth):
            # deterministic hash per depth
            h = hashlib.sha256(item_bytes + bytes([d])).hexdigest()
            index = int(h, 16) % width
            table[d][index] += 1
    return table


def count_min_query(sketch: List[List[int]], item: str) -> int:
    """Estimate the frequency of *item* from a Count‑Min sketch."""
    width = len(sketch[0])
    depth = len(sketch)
    item_bytes = str(item).encode("utf-8")
    estimates = []
    for d in range(depth):
        h = hashlib.sha256(item_bytes + bytes([d])).hexdigest()
        index = int(h, 16) % width
        estimates.append(sketch[d][index])
    return min(estimates)


def hyperloglog_add(sketch: List[int], item: str, p: int = 10) -> None:
    """Add *item* to a HyperLogLog sketch (register array)."""
    # p bits for bucket index, remaining bits for rank
    hash_bytes = hashlib.sha256(str(item).encode("utf-8")).digest()
    x = int.from_bytes(hash_bytes, "big")
    bucket = x >> (256 - p)
    w = (x << p) & ((1 << 256) - 1)  # shift out bucket bits
    rank = 1
    while rank <= 256 - p and (w >> (256 - p - rank)) & 1 == 0:
        rank += 1
    sketch[bucket] = max(sketch[bucket], rank)


def hyperloglog_estimate(sketch: List[int], p: int = 10) -> float:
    """Return cardinality estimate from HyperLogLog register array."""
    m = 1 << p
    alpha_m = 0.7213 / (1 + 1.079 / m)
    Z = 1.0 / sum([2.0 ** -v for v in sketch])
    E = alpha_m * m * m * Z
    # Small range correction
    if E <= (5 / 2) * m:
        V = sketch.count(0)
        if V != 0:
            return m * math.log(m / V)
    # Large range correction
    if E > (1 / 30) * (1 << 32):
        return - (1 << 32) * math.log(1 - E / (1 << 32))
    return E


def shannon_entropy(sequence: str) -> float:
    """Binary Shannon entropy of a string of characters."""
    seq = [ord(c) for c in sequence]
    n = len(seq)
    if n == 0:
        return 0.0
    freq = defaultdict(int)
    for c in seq:
        freq[c] += 1
    ent = 0.0
    for count in freq.values():
        p = count / n
        ent -= p * math.log(p, 2)
    return ent


def rlct_estimate(
    sketch: List[List[int]],
    features: Dict[str, float],
    signal: float,
    noise: float,
) -> float:
    """
    RLCT‑style scalar λ from Parent B.

    λ = (∑ sketch) * (∑_i f_i log f_i) * exp( - H(signal) )
    where H is Shannon entropy of the signal.
    """
    log_likelihood_sum = sum(sum(row) for row in sketch)
    curvature = sum(v * math.log(v) for v in features.values() if v > 0)
    entropy = shannon_entropy(str(signal))
    return log_likelihood_sum * curvature * math.exp(-entropy)


# ----------------------------------------------------------------------
# Hybrid Bandit core (integrates Parent A logic)
# ----------------------------------------------------------------------


@dataclass
class HybridBandit:
    actions: List[str]
    width: int = 128
    depth: int = 4
    p_hll: int = 10
    # per‑action sketches
    reward_sketches: Dict[str, List[List[int]]] = field(default_factory=dict)
    pulls: Dict[str, int] = field(default_factory=dict)
    # global context sketch
    context_sketch: List[int] = field(default_factory=list)
    # feature vector for RLCT (static for demo)
    features: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reward_sketches = {a: [[0] * self.width for _ in range(self.depth)] for a in self.actions}
        self.pulls = {a: 0 for a in self.actions}
        self.context_sketch = [0] * (1 << self.p_hll)
        # generate a dummy feature dict (could be learned from data)
        self.features = {
            "operator_visceral_ratio": random.random(),
            "operator_tech_ratio": random.random(),
            "psyche_forensic_shield_ratio": random.random(),
            "resilience_swarm_orchestration_density": random.random(),
        }

    # ------------------------------------------------------------------
    # Sketch handling
    # ------------------------------------------------------------------

    def update_context(self, context_id: str) -> None:
        """Incorporate a new context identifier into the HyperLogLog sketch."""
        hyperloglog_add(self.context_sketch, context_id, self.p_hll)

    def update_reward(self, action: str, reward: float) -> None:
        """Insert the observed reward (as a string) into the action's Count‑Min sketch."""
        # Use the string representation of the reward for hashing
        count_min_sketch([str(reward)], self.width, self.depth)
        # Increment counts manually to avoid rebuilding the whole sketch
        item = str(reward)
        for d in range(self.depth):
            h = hashlib.sha256(item.encode("utf-8") + bytes([d])).hexdigest()
            idx = int(h, 16) % self.width
            self.reward_sketches[action][d][idx] += 1
        self.pulls[action] += 1

    # ------------------------------------------------------------------
    # Estimators
    # ------------------------------------------------------------------

    def mean_reward_estimate(self, action: str) -> float:
        """Estimate the mean reward of *action* using its Count‑Min sketch."""
        # Approximate total count and sum of rewards.
        # For the demo we treat the sketch count as proxy for sum.
        sketch = self.reward_sketches[action]
        total_counts = sum(sum(row) for row in sketch)
        pulls = self.pulls[action] if self.pulls[action] > 0 else 1
        return total_counts / pulls

    def distinct_context_estimate(self) -> float:
        """Estimate number of distinct contexts via HyperLogLog."""
        return hyperloglog_estimate(self.context_sketch, self.p_hll)

    def lambda_rlct(self, signal: float = 1.0, noise: float = 0.1) -> float:
        """
        Compute the RLCT‑derived exploration coefficient λ.
        Uses a merged sketch of all actions (concatenated rows).
        """
        # Merge all reward sketches into a single sketch for the RLCT estimate
        merged = [[0] * self.width for _ in range(self.depth)]
        for sketch in self.reward_sketches.values():
            for d in range(self.depth):
                for i in range(self.width):
                    merged[d][i] += sketch[d][i]
        return rlct_estimate(merged, self.features, signal, noise)

    # ------------------------------------------------------------------
    # Decision rule
    # ------------------------------------------------------------------

    def select_action(self) -> str:
        """
        Upper‑Confidence‑Bound selection with RLCT‑scaled exploration term.
        """
        N_hat = max(self.distinct_context_estimate(), 1.0)
        lam = self.lambda_rlct()
        ucb_values = {}
        for a in self.actions:
            mu_hat = self.mean_reward_estimate(a)
            n_a = self.pulls[a] if self.pulls[a] > 0 else 1
            exploration = lam * math.sqrt(math.log(N_hat) / n_a)
            ucb_values[a] = mu_hat + exploration
        # Return action with maximal UCB
        return max(ucb_values, key=ucb_values.get)


# ----------------------------------------------------------------------
# Demonstration functions (fulfil the “at least 3 functions” requirement)
# ----------------------------------------------------------------------


def simulate_bandit_step(bandit: HybridBandit, context_id: str) -> Tuple[str, float]:
    """
    Perform a single interaction:
    1. Register the context.
    2. Choose an action via UCB.
    3. Sample a stochastic reward (Gaussian for demo).
    4. Update sketches.
    Returns the selected action and observed reward.
    """
    bandit.update_context(context_id)
    action = bandit.select_action()
    # Synthetic reward: mean depends on action name hash for variability
    base_mean = (int(hashlib.sha256(action.encode()).hexdigest(), 16) % 100) / 10.0
    reward = random.gauss(base_mean, 1.0)
    bandit.update_reward(action, reward)
    return action, reward


def run_simulation(num_steps: int = 1000, num_actions: int = 5) -> HybridBandit:
    """Run a full simulation and return the trained HybridBandit instance."""
    actions = [f"arm_{i}" for i in range(num_actions)]
    bandit = HybridBandit(actions=actions)
    for t in range(num_steps):
        ctx = f"context_{random.randint(0, 5000)}"
        simulate_bandit_step(bandit, ctx)
    return bandit


def report_bandit_stats(bandit: HybridBandit) -> None:
    """Print a concise summary of the learned statistics."""
    print("=== Hybrid Bandit Summary ===")
    print(f"Distinct contexts (HLL estimate): {bandit.distinct_context_estimate():.0f}")
    print(f"RLCT λ coefficient: {bandit.lambda_rlct():.4f}")
    for a in bandit.actions:
        mu = bandit.mean_reward_estimate(a)
        pulls = bandit.pulls[a]
        print(f"Action {a}: pulls={pulls}, μ̂≈{mu:.3f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    # Run a short simulation
    demo_bandit = run_simulation(num_steps=500, num_actions=3)
    report_bandit_stats(demo_bandit)

    # Verify that select_action returns a valid arm after training
    chosen = demo_bandit.select_action()
    assert chosen in demo_bandit.actions, "Selected action not in action set"
    print(f"Final selected action after training: {chosen}")