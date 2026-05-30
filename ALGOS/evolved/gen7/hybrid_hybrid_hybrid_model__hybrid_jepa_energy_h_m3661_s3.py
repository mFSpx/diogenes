# DARWIN HAMMER — match 3661, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_hybrid_hybrid_m1453_s4.py (gen6)
# parent_b: hybrid_jepa_energy_hybrid_sparse_wta_hy_m181_s0.py (gen3)
# born: 2026-05-29T23:51:06Z

"""Hybrid Algorithm integrating VRAM‑Bandit Scheduler (Parent A) and JEPA‑Energy Model Pool (Parent B).

Mathematical Bridge:
- Parent A provides a *store equation* where each BanditAction has an inflow rate (propensity) and an outflow rate
  (confidence_bound). The net flow `f = propensity – confidence_bound` quantifies temporal relevance.
- Parent B manages a model pool whose free‑energy is altered by loading/eviction events. Model loading decisions are
  driven by a variational free‑energy term.
- The fusion introduces an *information‑theoretic metric* that combines the Kullback‑Leibler (KL) divergence
  between the normalized bandit inflow distribution **p** and a pheromone‑decay distribution **q** (derived from a
  pheromone state vector) with the net flow `f`. Formally:

      H_hybrid = KL(p‖q) + α·f

  where `α` balances information‑theoretic surprise and resource‑flow urgency.

  This hybrid metric guides model‑pool actions: low `H_hybrid` encourages loading a model (reducing free‑energy),
  while high `H_hybrid` may trigger eviction to respect memory constraints.

The module implements the combined dynamics and provides three core functions demonstrating the hybrid operation.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Iterable, Tuple

import numpy as np

# ---------------------------- Data structures (from Parent A) ----------------------------

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # inflow rate
    expected_reward: float
    confidence_bound: float    # outflow rate
    algorithm: str

# ---------------------------- Data structures (from Parent B) ----------------------------

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

class ModelPool:
    """Manages a pool of models with a free‑energy bookkeeping."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def add_model(self, model: ModelTier) -> None:
        # Penalties for tier conflicts or memory overflow
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy += 1e6
        self.loaded[model.name] = model

    def load(self, model: ModelTier) -> None:
        self._energy -= 1e4   # reward for loading
        self.add_model(model)

    def load_with_eviction(self, model: ModelTier) -> None:
        self._energy -= 1e3   # reward for evicting
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # Evict the oldest entry (FIFO)
            evicted_name = next(iter(self.loaded))
            del self.loaded[evicted_name]
        self.load(model)

    def free_energy(self) -> float:
        return self._energy

    def memory_usage(self) -> Tuple[int, int]:
        """Returns (used_mb, ceiling_mb)."""
        return self._used(), self.ram_ceiling_mb

# ---------------------------- Core Hybrid Functions ----------------------------

def net_flow(bandit: BanditAction) -> float:
    """Compute the net flow f = propensity – confidence_bound for a single bandit."""
    return bandit.propensity - bandit.confidence_bound


def pheromone_decay(state: np.ndarray, decay_rate: float) -> np.ndarray:
    """
    Apply exponential decay to a pheromone state vector.
    state: non‑negative vector representing pheromone concentrations.
    decay_rate: positive scalar controlling decay speed.
    Returns the decayed state, renormalized to sum to 1.
    """
    if decay_rate < 0:
        raise ValueError("decay_rate must be non‑negative")
    decayed = state * np.exp(-decay_rate)
    total = decayed.sum()
    return decayed / total if total > 0 else decayed


def hybrid_information_metric(
    bandits: List[BanditAction],
    pheromone_state: np.ndarray,
    alpha: float = 1.0,
    epsilon: float = 1e-12
) -> float:
    """
    Compute the hybrid metric H_hybrid = KL(p‖q) + α·⟨f⟩,
    where p is the normalized propensity distribution of bandits,
    q is the normalized pheromone distribution, and ⟨f⟩ is the average net flow.
    """
    if not bandits:
        raise ValueError("bandits list cannot be empty")
    # Build propensity vector and normalize to a probability distribution p
    propensities = np.array([b.propensity for b in bandits], dtype=float)
    p = propensities + epsilon
    p /= p.sum()
    # Ensure pheromone_state is a probability distribution q
    q = pheromone_state + epsilon
    q /= q.sum()
    # KL divergence KL(p‖q)
    kl = np.sum(p * np.log(p / q))
    # Average net flow
    avg_flow = np.mean([net_flow(b) for b in bandits])
    return kl + alpha * avg_flow


def hybrid_load_decision(
    pool: ModelPool,
    model: ModelTier,
    bandits: List[BanditAction],
    pheromone_state: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 1.0
) -> None:
    """
    Decide whether to load a model into the pool based on the hybrid metric.
    If H_hybrid < threshold → load (or load with eviction if needed).
    Otherwise, do nothing (model stays unloaded).
    """
    metric = hybrid_information_metric(bandits, pheromone_state, alpha=alpha)
    if metric < threshold:
        if model.ram_mb + pool._used() > pool.ram_ceiling_mb:
            pool.load_with_eviction(model)
        else:
            pool.load(model)
    # No explicit else: the pool remains unchanged.


def simulate_hybrid_step(
    pool: ModelPool,
    models: List[ModelTier],
    bandits: List[BanditAction],
    pheromone_state: np.ndarray,
    decay_rate: float = 0.1,
    alpha: float = 1.0,
    threshold: float = 0.5
) -> None:
    """
    Perform a single simulation step:
    1. Decay pheromone state.
    2. For each candidate model, evaluate hybrid_load_decision.
    3. Update free‑energy accordingly.
    """
    # Step 1: decay pheromones
    pheromone_state[:] = pheromone_decay(pheromone_state, decay_rate)

    # Step 2: iterate over models
    for model in models:
        hybrid_load_decision(
            pool=pool,
            model=model,
            bandits=bandits,
            pheromone_state=pheromone_state,
            threshold=threshold,
            alpha=alpha
        )
    # No explicit return; pool state is mutated.


# ---------------------------- Utility Functions ----------------------------

def random_bandits(num: int) -> List[BanditAction]:
    """Generate a list of random BanditAction objects for testing."""
    actions = []
    for i in range(num):
        prop = random.uniform(0.1, 5.0)
        conf = random.uniform(0.0, prop)  # ensure net flow can be positive or negative
        actions.append(
            BanditAction(
                action_id=f"bandit_{i}",
                propensity=prop,
                expected_reward=random.uniform(0, 1),
                confidence_bound=conf,
                algorithm="HybridBandit"
            )
        )
    return actions


def random_models(num: int) -> List[ModelTier]:
    """Generate a list of random ModelTier objects for testing."""
    tiers = ["T1", "T2", "T3"]
    models = []
    for i in range(num):
        models.append(
            ModelTier(
                name=f"model_{i}",
                ram_mb=random.randint(200, 1500),
                tier=random.choice(tiers)
            )
        )
    return models


# ---------------------------- Smoke test ----------------------------

if __name__ == "__main__":
    # Initialise components
    pool = ModelPool(ram_ceiling_mb=6000)
    bandits = random_bandits(5)
    models = random_models(8)

    # Initialise a pheromone vector (size matches number of bandits)
    pheromone_state = np.random.rand(len(bandits))
    pheromone_state /= pheromone_state.sum()  # normalize

    # Run a few hybrid steps
    for step in range(3):
        simulate_hybrid_step(
            pool=pool,
            models=models,
            bandits=bandits,
            pheromone_state=pheromone_state,
            decay_rate=0.05,
            alpha=0.8,
            threshold=0.7
        )
        used, ceiling = pool.memory_usage()
        print(f"Step {step+1}: Used {used} MB / {ceiling} MB, Free energy = {pool.free_energy():.2f}")

    # Final sanity check
    assert used <= ceiling, "Memory usage exceeded ceiling"
    print("Hybrid simulation completed successfully.")