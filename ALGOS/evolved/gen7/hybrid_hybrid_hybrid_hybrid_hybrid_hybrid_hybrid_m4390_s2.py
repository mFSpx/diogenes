# DARWIN HAMMER — match 4390, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_hybrid_m1866_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m149_s2.py (gen4)
# born: 2026-05-29T23:55:19Z

"""Hybrid Algorithm: Fusion of Count‑Min Sketch / RLCT (Parent A) with Fractional Pheromone / Bandit (Parent B)

Mathematical Bridge
-------------------
* The Fisher information scalar derived from the reward distribution of the
  bandit subsystem (Parent B) is used as a weighting factor for the
  Real Log‑Canonical‑Threshold (RLCT) estimator of training losses (Parent A).
* The same Fisher scalar also modulates the Caputo fractional decay kernel that
  governs pheromone signal attenuation, thereby linking statistical curvature
  (A) with temporal decay (B).
* Count‑Min Sketches compress both incoming items and decayed pheromone
  signals, providing a unified low‑dimensional representation that feeds into
  the RLCT computation and the Hoeffding‑bound based leader election.

The resulting module offers a cohesive hybrid pipeline:
1. Sketch incoming identifiers.
2. Estimate a Fisher‑weighted RLCT.
3. Decay pheromones with a Fisher‑scaled fractional kernel.
4. Elect a leader among bandit actions using Hoeffding bounds on the sketch.

All components are implemented using only the Python standard library and
NumPy.
"""

import hashlib
import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def count_min_sketch(items: List[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Classic Count‑Min Sketch."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table

def estimate_rlct(train_losses: List[float], n_values: List[int], fisher_weight: float) -> float:
    """RLCT estimator weighted by a Fisher‑information scalar."""
    losses = np.asarray(train_losses, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= math.e):
        raise ValueError("All n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses and n_values must have the same length")
    mean_loss = np.mean(losses)
    rlct = math.log(math.log(np.mean(ns))) + fisher_weight * mean_loss
    return rlct

# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
def fractional_decay(alpha: float, t: float) -> float:
    """
    Caputo‑type fractional decay kernel:
        K(t) = t^{-alpha} / Gamma(1 - alpha)
    """
    if t <= 0:
        raise ValueError("time t must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1) for a proper fractional decay")
    return (t ** (-alpha)) / math.gamma(1.0 - alpha)

def compute_fisher_information(rewards: List[float]) -> float:
    """
    Simple Fisher information estimate for a Gaussian‑like reward distribution:
        I = 1 / variance
    """
    if len(rewards) < 2:
        return 1.0  # default when variance cannot be estimated
    var = np.var(rewards, ddof=1)
    return 1.0 / var if var > 0 else 1e6

# ----------------------------------------------------------------------
# Data structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBandit"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = field(default=0.0, init=False, repr=False)

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """A bounded signal derived from the last delta."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))

# ----------------------------------------------------------------------
# Hybrid system combining both parents
# ----------------------------------------------------------------------
class HybridPheromoneBanditSystem:
    """
    Holds pheromone traces, a Count‑Min sketch, and a StoreState.
    Provides methods that simultaneously apply fractional decay, Fisher‑scaled
    RLCT estimation, and sketch‑based leader election.
    """

    def __init__(self, sketch_width: int = 64, sketch_depth: int = 4):
        self.pheromones: Dict[str, List[float]] = defaultdict(list)
        self.sketch_width = sketch_width
        self.sketch_depth = sketch_depth
        self.sketch_table: List[List[int]] = [[0] * sketch_width for _ in range(sketch_depth)]
        self.store_state = StoreState()

    # ------------------------------------------------------------------
    # 1. Sketch insertion (Parent A)
    # ------------------------------------------------------------------
    def ingest_items(self, items: List[Any]) -> None:
        """Insert items into the internal Count‑Min sketch."""
        for item in items:
            for d in range(self.sketch_depth):
                idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % self.sketch_width
                self.sketch_table[d][idx] += 1

    # ------------------------------------------------------------------
    # 2. Pheromone update with Fisher‑scaled fractional decay (Parent B + bridge)
    # ------------------------------------------------------------------
    def update_pheromone(self,
                         surface_key: str,
                         raw_signal: float,
                         alpha: float,
                         current_time: float,
                         fisher_score: float) -> float:
        """
        Apply a fractional decay kernel multiplied by a Fisher weight,
        store the decayed signal, and also sketch the signal magnitude.
        """
        decay = fractional_decay(alpha, current_time)
        weighted_signal = raw_signal * decay * fisher_score
        self.pheromones[surface_key].append(weighted_signal)

        # Sketch the magnitude (rounded to int) to keep the sketch aware of pheromone activity
        magnitude = int(abs(weighted_signal))
        for d in range(self.sketch_depth):
            idx = int(hashlib.sha256(f"{d}:{magnitude}".encode()).hexdigest(), 16) % self.sketch_width
            self.sketch_table[d][idx] += 1

        return weighted_signal

    # ------------------------------------------------------------------
    # 3. RLCT estimation using Fisher‑weighted losses (Parent A + bridge)
    # ------------------------------------------------------------------
    def rlct_from_training(self,
                           train_losses: List[float],
                           n_values: List[int],
                           rewards: List[float]) -> float:
        """
        Compute Fisher information from rewards and feed it as a weight to the RLCT estimator.
        """
        fisher = compute_fisher_information(rewards)
        return estimate_rlct(train_losses, n_values, fisher)

    # ------------------------------------------------------------------
    # 4. Leader election using Hoeffding bound on sketch estimates (Parent A)
    # ------------------------------------------------------------------
    def elect_leader(self,
                     actions: List[BanditAction],
                     confidence: float = 0.95) -> BanditAction:
        """
        Estimate the frequency of each action_id via the sketch,
        then apply a Hoeffding bound to select the action with the highest
        upper confidence estimate.
        """
        # Helper to query sketch frequency for a given key
        def sketch_estimate(key: str) -> int:
            return min(
                self.sketch_table[d][int(hashlib.sha256(f"{d}:{key}".encode()).hexdigest(), 16) % self.sketch_width]
                for d in range(self.sketch_depth)
            )

        best_action = None
        best_ucb = -math.inf
        total_counts = sum(
            sum(row) for row in self.sketch_table
        ) or 1  # avoid division by zero

        for act in actions:
            freq = sketch_estimate(act.action_id)
            empirical = freq / total_counts
            # Hoeffding bound for Bernoulli variables
            bound = math.sqrt(math.log(2 / (1 - confidence)) / (2 * total_counts))
            ucb = empirical + bound
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = act
        return best_action

# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def demo_sketch_and_decay() -> None:
    """Show sketch ingestion and Fisher‑scaled pheromone decay."""
    system = HybridPheromoneBanditSystem()
    items = [f"user_{i}" for i in range(100)]
    system.ingest_items(items)

    # Simulate a pheromone signal
    fisher = 2.5
    decayed = system.update_pheromone(
        surface_key="node_A",
        raw_signal=5.0,
        alpha=0.4,
        current_time=10.0,
        fisher_score=fisher,
    )
    print(f"Decayed pheromone signal (Fisher‑scaled): {decayed:.4f}")

def demo_rlct_estimation() -> None:
    """Compute a Fisher‑weighted RLCT from synthetic losses and rewards."""
    system = HybridPheromoneBanditSystem()
    losses = [random.uniform(0.1, 1.0) for _ in range(20)]
    ns = list(range(50, 70))
    rewards = [random.gauss(0.5, 0.1) for _ in range(30)]
    rlct = system.rlct_from_training(losses, ns, rewards)
    print(f"Fisher‑weighted RLCT estimate: {rlct:.6f}")

def demo_leader_election() -> None:
    """Run leader election on a set of bandit actions after sketching identifiers."""
    system = HybridPheromoneBanditSystem()
    # Ingest some identifiers that overlap with action ids
    ids = ["act_1", "act_2", "act_3", "act_2", "act_1", "act_1"]
    system.ingest_items(ids)

    actions = [
        BanditAction(action_id="act_1", propensity=0.3, expected_reward=0.6, confidence_bound=0.1),
        BanditAction(action_id="act_2", propensity=0.5, expected_reward=0.4, confidence_bound=0.2),
        BanditAction(action_id="act_3", propensity=0.2, expected_reward=0.7, confidence_bound=0.15),
    ]
    leader = system.elect_leader(actions, confidence=0.99)
    print(f"Elected leader action: {leader.action_id}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Hybrid System Demo ===")
    demo_sketch_and_decay()
    demo_rlct_estimation()
    demo_leader_election()
    print("All demos executed successfully.")