# DARWIN HAMMER — match 2767, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1968_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1175_s0.py (gen5)
# born: 2026-05-29T23:45:52Z

"""
Hybrid Algorithm: fusion_hybrid_rbf_bayes_modelpool.py

Parents:
- Parent A (hybrid_hybrid_hybrid_hammer_bridgerbf_m1195_s4_b_m409_s1): provides
  * Metropolis‑style acceptance probability,
  * Hoeffding bound decision,
  * Bayesian edge‑reliability updates (Beta prior/posterior).
- Parent B (hybrid_hybrid_hybrid_cockpi_m1175_s0): provides a ModelPool with
  RAM‑aware loading/eviction and simple model descriptors.

Mathematical Bridge:
We treat each loaded model as a node in a graph.  Pairwise similarity between
models is computed with a Radial Basis Function (RBF) kernel on an optional
embedding vector.  The raw RBF similarity ϕ(i,j) is then *modulated* by the
posterior mean μ_ij of a Bayesian edge‑reliability estimate (Beta posterior)
derived from observed successes/failures of interactions between the two
models.  The resulting trust‑weighted similarity

    S_ij = μ_ij * exp( -||x_i - x_j||² / (2σ²) )

forms a symmetric matrix that drives model‑pool decisions:
* When a new model is proposed, an “energy” ΔE is defined as the negative sum of
  its trust‑weighted similarities to currently loaded models.
* The Metropolis acceptance probability P = exp( -ΔE / T ) (with a temperature
  schedule T) decides whether the model is admitted.
* Hoeffding’s bound is used as an auxiliary stop‑criterion on the number of
  similarity samples collected for a given edge.

The three core functions below illustrate this hybrid operation.
"""

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
                self.loaded.pop(evict_target)
            else:
                # No viable eviction → reject outright.
                return False

        # 2. Compute energy based on trust‑weighted similarity.
        similarity = self.trust_weighted_similarity(candidate, sigma)
        delta_energy = -similarity  # more similarity ⇒ lower energy

        # 3. Acceptance test.
        prob = acceptance_probability(delta_energy, temperature)
        if random.random() < prob:
            # Accepted
            self.loaded[candidate.name] = candidate
            return True

        # 4. If not accepted, use Hoeffding bound on the number of similarity samples.
        num_samples = len(self.loaded)  # one sample per loaded model
        if hoeffding_decision(num_samples, epsilon):
            # Bound is tight → force acceptance to avoid stagnation.
            self.loaded[candidate.name] = candidate
            return True

        # 5. Final fallback: reject.
        return False

    def _select_eviction_target(self, sigma: float) -> Optional[str]:
        """
        Choose the loaded model with the lowest average trust‑weighted similarity
        to all other loaded models. Returns the model name or None if pool is empty.
        """
        if not self.loaded:
            return None
        avg_sims = {}
        names = list(self.loaded.keys())
        for i, name_i in enumerate(names):
            sims = []
            for j, name_j in enumerate(names):
                if i == j:
                    continue
                model_i = self.loaded[name_i]
                model_j = self.loaded[name_j]
                if model_i.embedding is None or model_j.embedding is None:
                    rbf = 0.0
                else:
                    rbf = self._rbf(model_i.embedding, model_j.embedding, sigma)
                trust = self.edge_trust(name_i, name_j)
                sims.append(trust * rbf)
            avg_sims[name_i] = sum(sims) / len(sims) if sims else 0.0
        # Evict the model with minimal average similarity (i.e., least trusted).
        evict_name = min(avg_sims, key=avg_sims.get)
        return evict_name


# ----------------------------------------------------------------------
# Demonstration functions (the required three)
# ----------------------------------------------------------------------
def generate_random_embedding(dim: int = 8) -> np.ndarray:
    """Create a normalized random embedding vector."""
    vec = np.random.randn(dim)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def simulate_interactions(pool: ModelPool, steps: int = 20) -> None:
    """
    Randomly pick pairs of loaded models and record synthetic successes/failures.
    Success probability is proportional to their current trust‑weighted similarity.
    """
    names = list(pool.loaded.keys())
    if len(names) < 2:
        return
    for _ in range(steps):
        a, b = random.sample(names, 2)
        # Compute current trust‑weighted similarity as a proxy for true success prob.
        sim = pool.trust_weighted_similarity(pool.loaded[a])  # using a as candidate against others
        success_prob = 0.5 + 0.5 * sim  # map [0,1] → [0.5,1]
        success = random.random() < success_prob
        pool.record_interaction(a, b, success)


def hybrid_demo() -> None:
    """
    End‑to‑end demo:
    1. Initialise a pool.
    2. Create a few candidate models with random embeddings.
    3. Attempt to load them using the hybrid decision rule.
    4. Run a few interaction simulations to evolve edge trusts.
    5. Print final pool status.
    """
    pool = ModelPool(ram_ceiling_mb=2000)

    # Create 5 candidate models.
    candidates = [
        ModelTier(
            name=f"model_{i}",
            ram_mb=random.randint(200, 600),
            tier=random.choice(["T1", "T2", "T3"]),
            text=f"Demo model {i}",
            embedding=generate_random_embedding(),
        )
        for i in range(5)
    ]

    temperature = 1.0
    sigma = 0.8

    for cand in candidates:
        loaded = pool.load_with_hybrid_decision(cand, temperature, sigma)
        print(f"Attempted to load {cand.name} (RAM {cand.ram_mb} MB): {'Loaded' if loaded else 'Rejected'}")

    # Simulate interactions to update Bayesian trusts.
    simulate_interactions(pool, steps=30)

    # Show final trust matrix.
    print("\nFinal trust‑weighted similarity matrix:")
    names = list(pool.loaded.keys())
    for i, name_i in enumerate(names):
        row = []
        for j, name_j in enumerate(names):
            if i == j:
                row.append("1.00")
            else:
                trust = pool.edge_trust(name_i, name_j)
                row.append(f"{trust:.2f}")
        print(f"{name_i}: {' '.join(row)}")


if __name__ == "__main__":
    # Simple smoke test – should run without raising exceptions.
    hybrid_demo()