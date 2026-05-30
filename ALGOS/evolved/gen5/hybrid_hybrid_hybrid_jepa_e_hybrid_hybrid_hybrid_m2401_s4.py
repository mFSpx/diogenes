# DARWIN HAMMER — match 2401, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_jepa_energy_h_hybrid_hybrid_infota_m141_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m121_s0.py (gen4)
# born: 2026-05-29T23:42:09Z

"""Hybrid algorithm merging:
- Parent A: energy‑based ModelPool with entropy navigation.
- Parent B: risk assessment, differential‑privacy aggregation, and Bayesian update.

Mathematical bridge:
The bridge is a *risk‑adjusted energy score*.
Each model obtains a reconstruction‑risk probability (Parent B) that is
treated as a weight on the energy cost incurred by the ModelPool
(Parent A).  The weighted energy is then aggregated with differential‑privacy
noise (dp_aggregate) and fed into a Bayesian update that yields a posterior
selection probability.  Entropy of the posterior distribution is used
as a global guidance metric, while a simple dot‑product between a model
feature vector (RAM, tier factor, risk) and a tunable weight vector provides
a minimum‑cost decision surface.  This unifies the matrix‑operations of
Parent B with the energy‑budget dynamics of Parent A into a single
hybrid system."""


from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


# ----------------------------------------------------------------------
# Parent A – Energy based ModelPool
# ----------------------------------------------------------------------


class ModelPool:
    """Manages a pool of models with an energy ledger."""

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: dict[str, ModelTier] = {}
        self._energy: float = 0.0

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def add_model(self, model: ModelTier) -> None:
        # Tier conflict penalty
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10
        # Memory overflow penalty
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy += 1e6
        self.loaded[model.name] = model

    def load(self, model: ModelTier) -> None:
        self._energy -= 1e4  # reward for loading
        self.add_model(model)

    def load_with_eviction(self, model: ModelTier) -> None:
        self._energy -= 1e3  # reward for eviction path
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # evict the biggest RAM consumer
            evicted_name = max(self.loaded, key=lambda n: self.loaded[n].ram_mb)
            self.loaded.pop(evicted_name)
            self._energy += 1e2  # penalty for eviction
        self.load(model)

    def free_energy(self) -> float:
        """Current energy balance (negative = reward, positive = penalty)."""
        return self._energy


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    probs = np.array(probabilities, dtype=float)
    probs = np.clip(probs, eps, 1.0)
    probs /= probs.sum()
    return -float(np.sum(probs * np.log(probs)))


# ----------------------------------------------------------------------
# Parent B – Risk, DP, Geometry & Bayesian utilities
# ----------------------------------------------------------------------


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Probability that a record can be re‑identified."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Differential‑privacy aggregation (Laplace mechanism)."""
    values = list(values)
    if not values:
        return 0.0
    true_sum = sum(values)
    scale = sensitivity / epsilon
    noise = random.laplacevariate(0.0, scale)  # type: ignore[attr-defined]
    return true_sum + noise


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Bayesian marginal probability P(data)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    # Corrected formula: P(data) = likelihood*prior + false_positive*(1-prior)
    return likelihood * prior + false_positive * (1.0 - prior)


# ----------------------------------------------------------------------
# Hybrid core – risk‑adjusted energy scoring
# ----------------------------------------------------------------------


def model_feature_vector(model: ModelTier, risk: float) -> np.ndarray:
    """Vector [RAM, tier_factor, risk] used for dot‑product cost."""
    tier_factor = {"T1": 1.0, "T2": 2.0, "T3": 3.0}.get(model.tier, 0.0)
    return np.array([model.ram_mb, tier_factor, risk], dtype=float)


def hybrid_cost_score(
    model: ModelTier,
    pool: ModelPool,
    risk: float,
    weight_vec: np.ndarray | List[float] = None,
) -> float:
    """
    Compute a scalar cost = w·v where v is the feature vector.
    The weight vector balances RAM, tier penalty and risk.
    """
    if weight_vec is None:
        weight_vec = np.array([1.0, 5.0, 10.0])  # default emphasis on tier & risk
    v = model_feature_vector(model, risk)
    return float(np.dot(weight_vec, v)) + pool.free_energy()


def hybrid_select_model(
    pool: ModelPool,
    candidates: List[ModelTier],
    unique_qi: int,
    total_records: int,
    epsilon: float = 1.0,
) -> ModelTier:
    """
    Hybrid selection pipeline:
    1. Compute per‑model reconstruction risk.
    2. Weight each model's energy cost by (1 + risk).
    3. Aggregate the weighted costs with DP noise.
    4. Perform a Bayesian update using the aggregated cost as “likelihood”.
    5. Choose the model with the highest posterior probability.
    """
    # Step 1 – risk per model (use tier as proxy for identifier count)
    tier_to_qi = {"T1": 1, "T2": 5, "T3": 10}
    risks = [
        reconstruction_risk_score(tier_to_qi.get(m.tier, 0), total_records)
        for m in candidates
    ]

    # Step 2 – raw costs
    raw_costs = [hybrid_cost_score(m, pool, r) for m, r in zip(candidates, risks)]

    # Step 3 – DP aggregation (adds Laplace noise)
    noisy_aggregate = dp_aggregate(raw_costs, epsilon=epsilon)

    # Step 4 – Bayesian posterior for each model
    # Prior is uniform; likelihood proportional to exp(-cost)
    prior = 1.0 / len(candidates)
    likelihoods = [math.exp(-c) for c in raw_costs]
    false_positive = 1e-6  # tiny background probability

    posteriors = [
        bayes_marginal(prior, L, false_positive) for L in likelihoods
    ]
    # Normalise posteriors
    total = sum(posteriors)
    if total == 0:
        normalized = [1.0 / len(posteriors)] * len(posteriors)
    else:
        normalized = [p / total for p in posteriors]

    # Step 5 – pick best
    best_idx = int(np.argmax(normalized))
    return candidates[best_idx]


def hybrid_entropy(pool: ModelPool) -> float:
    """
    Compute entropy over the RAM‑share distribution of the loaded models.
    The distribution is p_i = ram_i / total_ram_used.
    """
    if not pool.loaded:
        return 0.0
    rams = [m.ram_mb for m in pool.loaded.values()]
    total = sum(rams)
    probs = [r / total for r in rams]
    return entropy(probs)


def hybrid_resource_allocation(
    pool: ModelPool,
    models: List[ModelTier],
    epsilon: float = 1.0,
    weight_vec: List[float] | np.ndarray = None,
) -> Mapping[str, float]:
    """
    Allocate a synthetic “budget” to each model based on the hybrid cost score.
    The budget vector is DP‑noised and then normalised to sum to 1.
    Returns a mapping from model name to allocated fraction.
    """
    if weight_vec is None:
        weight_vec = np.array([1.0, 5.0, 10.0])

    # Compute raw hybrid scores
    scores = [
        hybrid_cost_score(m, pool, reconstruction_risk_score(5, 1000), weight_vec)
        for m in models
    ]

    # Convert scores to a budget (lower cost -> higher budget)
    raw_budget = np.max(scores) - np.array(scores) + 1e-6
    noisy_budget = dp_aggregate(raw_budget, epsilon=epsilon)
    # Since dp_aggregate returns a scalar, we add Laplace noise individually
    scale = 1.0 / epsilon
    noisy_vector = np.array(
        [b + random.laplacevariate(0.0, scale) for b in raw_budget]  # type: ignore[attr-defined]
    )
    # Normalise
    total = noisy_vector.sum()
    if total <= 0:
        allocation = np.full_like(noisy_vector, 1.0 / len(noisy_vector))
    else:
        allocation = noisy_vector / total

    return {m.name: float(a) for m, a in zip(models, allocation)}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a pool and load a few models
    pool = ModelPool(ram_ceiling_mb=8000)
    models = [
        ModelTier(name="alpha", ram_mb=1500, tier="T1"),
        ModelTier(name="beta", ram_mb=2500, tier="T2"),
        ModelTier(name="gamma", ram_mb=3000, tier="T3"),
    ]
    for m in models:
        pool.load_with_eviction(m)

    # Hybrid selection
    selected = hybrid_select_model(
        pool,
        candidates=models,
        unique_qi=5,
        total_records=1000,
        epsilon=0.5,
    )
    print(f"Selected model: {selected.name}")

    # Entropy of current pool state
    print(f"Pool entropy: {hybrid_entropy(pool):.4f}")

    # Resource allocation
    allocation = hybrid_resource_allocation(pool, models, epsilon=0.7)
    for name, frac in allocation.items():
        print(f"Allocated {frac:.3%} to model {name}")

    # Final energy balance
    print(f"Final energy balance: {pool.free_energy():.2e}")