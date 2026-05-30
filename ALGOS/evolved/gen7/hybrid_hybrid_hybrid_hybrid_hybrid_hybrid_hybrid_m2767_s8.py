# DARWIN HAMMER — match 2767, survivor 8
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1968_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1175_s0.py (gen5)
# born: 2026-05-29T23:45:52Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, replace
from typing import List, Tuple, Dict, Callable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A – probabilistic & Bayesian utilities
# ----------------------------------------------------------------------
def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Metropolis‑style acceptance probability, never zero."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prob = math.exp(-delta_energy / temperature)
    return max(prob, 1e-12)


def hoeffding_decision(num_samples: int, epsilon: float, delta: float = 0.05) -> bool:
    """Return True if the Hoeffding bound is tighter than epsilon."""
    if num_samples <= 0:
        return False
    bound = math.sqrt(math.log(2.0 / delta) / (2.0 * num_samples))
    return bound < epsilon


@dataclass(frozen=True)
class EdgeBetaPrior:
    """Beta prior for a Bernoulli edge reliability."""
    alpha: float = 1.0
    beta: float = 1.0


def bayesian_edge_update(
    prior: EdgeBetaPrior,
    successes: int,
    failures: int,
) -> Tuple[float, float, EdgeBetaPrior]:
    """
    Return posterior mean, variance and updated prior.
    The variance serves as a confidence indicator (lower ⇒ higher trust).
    """
    alpha_post = prior.alpha + successes
    beta_post = prior.beta + failures
    mean = alpha_post / (alpha_post + beta_post)
    var = (alpha_post * beta_post) / ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))
    updated = EdgeBetaPrior(alpha=alpha_post, beta=beta_post)
    return mean, var, updated


# ----------------------------------------------------------------------
# Parent B – model pool infrastructure
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    text: str  # textual description / metadata
    embedding: Optional[np.ndarray] = None  # optional vector for similarity


class ModelPool:
    """RAM‑aware container with Bayesian‑trust‑weighted eviction."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        # Edge statistics stored as (successes, failures, prior)
        self.edge_stats: Dict[Tuple[str, str], Tuple[int, int, EdgeBetaPrior]] = {}

    # ------------------------------------------------------------------
    # Basic RAM bookkeeping
    # ------------------------------------------------------------------
    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    # ------------------------------------------------------------------
    # Edge‑reliability bookkeeping
    # ------------------------------------------------------------------
    def _edge_key(self, a: str, b: str) -> Tuple[str, str]:
        return tuple(sorted((a, b)))

    def record_interaction(self, a: str, b: str, success: bool) -> None:
        """Update Beta prior for the edge (a,b) based on a binary outcome."""
        key = self._edge_key(a, b)
        successes, failures, prior = self.edge_stats.get(key, (0, 0, EdgeBetaPrior()))
        if success:
            successes += 1
        else:
            failures += 1
        mean, var, updated = bayesian_edge_update(prior, successes, failures)
        self.edge_stats[key] = (successes, failures, updated)

    def edge_trust(self, a: str, b: str) -> float:
        """Posterior mean trust for edge (a,b); defaults to 0.5 if unseen."""
        key = self._edge_key(a, b)
        if key not in self.edge_stats:
            return 0.5
        successes, failures, prior = self.edge_stats[key]
        mean, _, _ = bayesian_edge_update(prior, successes, failures)
        return mean

    # ------------------------------------------------------------------
    # RBF similarity utilities (the bridge)
    # ------------------------------------------------------------------
    @staticmethod
    def _rbf(vec1: np.ndarray, vec2: np.ndarray, sigma: float) -> float:
        diff = vec1 - vec2
        return math.exp(-np.dot(diff, diff) / (2 * sigma ** 2))

    def trust_weighted_similarity(
        self,
        candidate: ModelTier,
        sigma: float = 1.0,
    ) -> float:
        """
        Compute the aggregated trust‑weighted RBF similarity between *candidate*
        and all currently loaded models.
        """
        if not self.loaded:
            return 0.0
        sims = []
        for loaded in self.loaded.values():
            # If either model lacks an embedding, fall back to 0 similarity.
            if candidate.embedding is None or loaded.embedding is None:
                rbf = 0.0
            else:
                rbf = self._rbf(candidate.embedding, loaded.embedding, sigma)
            trust = self.edge_trust(candidate.name, loaded.name)
            sims.append(trust * rbf)
        return sum(sims) / len(sims)

    # ------------------------------------------------------------------
    # Loading / eviction governed by hybrid acceptance
    # ------------------------------------------------------------------
    def _select_eviction_target(self, sigma: float) -> Optional[str]:
        if not self.loaded:
            return None
        min_trust = float('inf')
        target = None
        for model in self.loaded.values():
            trust = self.trust_weighted_similarity(model, sigma)
            if trust < min_trust:
                min_trust = trust
                target = model.name
        return target

    def load_with_hybrid_decision(
        self,
        candidate: ModelTier,
        temperature: float,
        sigma: float = 1.0,
        epsilon: float = 0.05,
    ) -> bool:
        """
        Attempt to load *candidate*.
        - Compute ΔE = - similarity (more similar ⇒ lower energy).
        - Accept with Metropolis probability.
        - If not accepted, optionally evict the least‑trusted model and retry.
        Returns True if the candidate ends up loaded.
        """
        # 1. Check RAM constraint; if exceeded we may need eviction.
        if candidate.ram_mb + self._used() > self.ram_ceiling_mb:
            # Evict the model with the smallest trust‑weighted similarity to others.
            evict_target = self._select_eviction_target(sigma)
            if evict_target:
                del self.loaded[evict_target]
            else:
                # No viable eviction → reject outright.
                return False

        # 2. Compute energy change ΔE for the candidate.
        sims = self.trust_weighted_similarity(candidate, sigma)
        delta_energy = -sims

        # 3. Accept or reject according to Metropolis criterion.
        prob = acceptance_probability(delta_energy, temperature)
        if random.random() < prob:
            self.loaded[candidate.name] = candidate
            return True

        return False

    def ensure_loaded(self, candidate: ModelTier, temperature: float, sigma: float = 1.0) -> None:
        if not self.is_loaded(candidate.name):
            self.load_with_hybrid_decision(candidate, temperature, sigma)

# Usage
if __name__ == "__main__":
    pool = ModelPool()
    model_tier = ModelTier("example", 100, "test", "example model")
    model_tier.embedding = np.array([1.0, 2.0, 3.0])
    pool.ensure_loaded(model_tier, 1.0)