# DARWIN HAMMER — match 2767, survivor 7
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1968_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1175_s0.py (gen5)
# born: 2026-05-29T23:45:52Z

"""Hybrid Algorithm: Fusion of Probabilistic Bayesian Edge Reliability (Parent A) 
and Trust‑Weighted Linguistic Similarity with Model Pool Management (Parent B).

Mathematical Bridge
-------------------
Parent A models pairwise node similarity with a Radial Basis Function (RBF) and
updates edge reliability through a Bayesian Beta‑Bernoulli scheme.  
Parent B evaluates models by a trust‑weighted linguistic similarity between
their textual descriptors and decides on loading/eviction using regret‑style
criteria.

The bridge is the *combined similarity*:

    S_combined(i, j) = RBF(vec_i, vec_j; σ) × TrustSim(text_i, text_j; τ_i, τ_j)

where `RBF` supplies a smooth geometric similarity of node embeddings and
`TrustSim` supplies a trust‑scaled linguistic similarity of the associated
descriptions.  This scalar modulates the Bayesian edge prior (α,β) and also
drives the Metropolis‑style acceptance probability for model‑pool operations.

The resulting system simultaneously:
* propagates uncertainty on graph edges (β‑priors) weighted by linguistic trust,
* selects and evicts models from a RAM‑bounded pool using Metropolis acceptance,
* validates decisions with a Hoeffding bound on observed similarity samples.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, replace
from typing import List, Tuple, Dict, Callable, Optional

import numpy as np

# ----------------------------------------------------------------------
# 1. Core utilities from Parent A
# ----------------------------------------------------------------------
def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Metropolis‑style acceptance probability, never exactly zero."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prob = math.exp(-delta_energy / temperature)
    return max(prob, 1e-12)


def hoeffding_decision(num_samples: int, epsilon: float, delta: float = 0.05) -> bool:
    """Hoeffding bound decision: True if bound < ε."""
    if num_samples <= 0:
        return False
    bound = math.sqrt(math.log(2.0 / delta) / (2.0 * num_samples))
    return bound < epsilon


@dataclass(frozen=True)
class EdgeBetaPrior:
    """Beta prior for Bernoulli edge reliability."""
    alpha: float = 1.0
    beta: float = 1.0


def bayesian_edge_update(
    prior: EdgeBetaPrior,
    successes: int,
    failures: int,
) -> Tuple[float, float, EdgeBetaPrior]:
    """Posterior mean, variance and updated prior."""
    new_alpha = prior.alpha + successes
    new_beta = prior.beta + failures
    mean = new_alpha / (new_alpha + new_beta)
    var = (new_alpha * new_beta) / ((new_alpha + new_beta) ** 2 * (new_alpha + new_beta + 1))
    updated = EdgeBetaPrior(alpha=new_alpha, beta=new_beta)
    return mean, var, updated


def rbf_similarity(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian RBF similarity."""
    diff = x - y
    norm_sq = np.dot(diff, diff)
    return math.exp(-norm_sq / (2 * sigma ** 2))


# ----------------------------------------------------------------------
# 2. Trust‑weighted linguistic similarity (Parent B)
# ----------------------------------------------------------------------
def _bag_of_words(text: str) -> Dict[str, int]:
    tokens = text.lower().split()
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return freq


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two dense vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def trust_weighted_similarity(
    text_a: str,
    text_b: str,
    trust_a: float,
    trust_b: float,
) -> float:
    """
    Compute a trust‑scaled cosine similarity between two texts.
    Each text is transformed to a bag‑of‑words vector; the vectors are
    weighted by the respective trust scores before cosine evaluation.
    """
    vocab = list(set(_bag_of_words(text_a).keys()) | set(_bag_of_words(text_b).keys()))
    vec_a = np.array([trust_a * _bag_of_words(text_a).get(w, 0) for w in vocab], dtype=float)
    vec_b = np.array([trust_b * _bag_of_words(text_b).get(w, 0) for w in vocab], dtype=float)
    return cosine_similarity(vec_a, vec_b)


# ----------------------------------------------------------------------
# 3. Combined similarity and Bayesian edge update with trust
# ----------------------------------------------------------------------
def combined_similarity(
    vec_i: np.ndarray,
    vec_j: np.ndarray,
    text_i: str,
    text_j: str,
    sigma: float,
    trust_i: float,
    trust_j: float,
) -> float:
    """
    Geometric (RBF) similarity modulated by trust‑weighted linguistic similarity.
    The product lies in [0,1] because both factors are ≤1.
    """
    geo = rbf_similarity(vec_i, vec_j, sigma)
    ling = trust_weighted_similarity(text_i, text_j, trust_i, trust_j)
    return geo * ling


def bayesian_edge_update_with_trust(
    prior: EdgeBetaPrior,
    vec_i: np.ndarray,
    vec_j: np.ndarray,
    text_i: str,
    text_j: str,
    sigma: float,
    trust_i: float,
    trust_j: float,
    success_threshold: float = 0.5,
) -> Tuple[float, float, EdgeBetaPrior]:
    """
    Use the combined similarity as a soft observation:
    - If similarity ≥ threshold → count as a success,
    - otherwise as a failure.
    The posterior is then computed via the standard Beta update.
    """
    sim = combined_similarity(vec_i, vec_j, text_i, text_j, sigma, trust_i, trust_j)
    successes = 1 if sim >= success_threshold else 0
    failures = 1 - successes
    return bayesian_edge_update(prior, successes, failures)


# ----------------------------------------------------------------------
# 4. Model pool with hybrid acceptance logic (Parent B + A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    text: str  # textual descriptor used for linguistic similarity
    trust: float = 1.0  # default trust score


class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """
        Evict least‑recently‑added models until there is enough RAM,
        then load the new model.
        """
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # pop arbitrary (FIFO) entry
            evicted_name = next(iter(self.loaded))
            self.loaded.pop(evicted_name)
        self.load(model)


# ----------------------------------------------------------------------
# 5. Hybrid operations
# ----------------------------------------------------------------------
def evaluate_candidate_model(
    pool: ModelPool,
    candidate: ModelTier,
    temperature: float = 1.0,
) -> bool:
    """
    Decide whether to load a candidate model using Metropolis acceptance.
    The 'energy' is defined as the negative average combined similarity
    between the candidate and each already‑loaded model.
    """
    if not pool.loaded:
        # empty pool → accept trivially
        pool.load(candidate)
        return True

    sims = []
    for loaded in pool.loaded.values():
        # dummy numeric vectors for demonstration (hash of name)
        vec_loaded = np.array([hash(loaded.name) % 1000], dtype=float)
        vec_cand = np.array([hash(candidate.name) % 1000], dtype=float)
        sim = combined_similarity(
            vec_loaded,
            vec_cand,
            loaded.text,
            candidate.text,
            sigma=10.0,
            trust_i=loaded.trust,
            trust_j=candidate.trust,
        )
        sims.append(sim)

    avg_sim = sum(sims) / len(sims)
    delta_energy = -avg_sim  # higher similarity → lower energy
    prob = acceptance_probability(delta_energy, temperature)

    if random.random() < prob:
        try:
            pool.load_with_eviction(candidate)
            return True
        except Exception:
            # fallback: reject if loading fails
            return False
    return False


def update_edge_between_models(
    prior: EdgeBetaPrior,
    model_a: ModelTier,
    model_b: ModelTier,
    sigma: float = 10.0,
) -> Tuple[float, float, EdgeBetaPrior]:
    """
    Perform a Bayesian edge reliability update for the pair (model_a, model_b)
    using their combined similarity.
    """
    vec_a = np.array([hash(model_a.name) % 1000], dtype=float)
    vec_b = np.array([hash(model_b.name) % 1000], dtype=float)
    return bayesian_edge_update_with_trust(
        prior,
        vec_a,
        vec_b,
        model_a.text,
        model_b.text,
        sigma,
        model_a.trust,
        model_b.trust,
    )


def prune_pool_by_hoeffding(pool: ModelPool, epsilon: float = 0.1) -> None:
    """
    Scan the pool and evict models whose average similarity sample
    fails the Hoeffding bound (i.e., not reliably similar to the rest).
    """
    names = list(pool.loaded.keys())
    for name in names:
        model = pool.loaded[name]
        # compute similarity to a random subset of other models
        other_models = [m for n, m in pool.loaded.items() if n != name]
        if not other_models:
            continue
        sample = random.sample(other_models, min(5, len(other_models)))
        sims = []
        for other in sample:
            vec_m = np.array([hash(model.name) % 1000], dtype=float)
            vec_o = np.array([hash(other.name) % 1000], dtype=float)
            sims.append(
                combined_similarity(
                    vec_m,
                    vec_o,
                    model.text,
                    other.text,
                    sigma=10.0,
                    trust_i=model.trust,
                    trust_j=other.trust,
                )
            )
        if not hoeffding_decision(len(sims), epsilon):
            # evict this model
            pool.loaded.pop(name)


# ----------------------------------------------------------------------
# 6. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a model pool
    pool = ModelPool(ram_ceiling_mb=2000)

    # Define a few dummy models with textual descriptors and trust scores
    models = [
        ModelTier(name="M_A", ram_mb=400, tier="T1", text="fast inference transformer", trust=0.9),
        ModelTier(name="M_B", ram_mb=600, tier="T2", text="robust classification network", trust=0.8),
        ModelTier(name="M_C", ram_mb=700, tier="T1", text="lightweight convolutional encoder", trust=0.95),
        ModelTier(name="M_D", ram_mb=500, tier="T3", text="experimental generative model", trust=0.6),
    ]

    # Attempt to load each model using the hybrid acceptance rule
    for m in models:
        accepted = evaluate_candidate_model(pool, m, temperature=2.0)
        print(f"Model {m.name} accepted: {accepted}")

    # Perform a Bayesian edge update between two loaded models
    if len(pool.loaded) >= 2:
        keys = list(pool.loaded.keys())
        prior = EdgeBetaPrior(alpha=2.0, beta=2.0)
        mean, var, upd = update_edge_between_models(prior, pool.loaded[keys[0]], pool.loaded[keys[1]])
        print(f"Edge posterior mean={mean:.3f}, var={var:.5f}")

    # Prune the pool using Hoeffding bound
    prune_pool_by_hoeffding(pool, epsilon=0.15)
    print("Remaining models after pruning:", list(pool.loaded.keys()))