# DARWIN HAMMER — match 4427, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_hybrid_hybrid_m1806_s4.py (gen6)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_bayes_update__m1099_s1.py (gen3)
# born: 2026-05-29T23:55:42Z

"""Hybrid Pheromone‑Semantic‑Morphology Bayesian Engine
====================================================

Parents
-------
* **Parent A** – *hybrid_pheromone_inf_hybrid_hybrid_hybrid_m1806_s4.py*  
  Provides a pheromone signal model with exponential decay and a Bayesian
  Beta‑prior edge reliability estimator.

* **Parent B** – *hybrid_hybrid_semantic_neig_hybrid_bayes_update__m1099_s1.py*  
  Supplies morphological indices (sphericity, flatness, righting‑time,
  recovery priority) and a cosine‑similarity based semantic‑neighbor routine.

Mathematical Bridge
-------------------
Both parents operate on *probabilistic weights* that can be interpreted as
likelihoods for a Bayesian update:

* In **Parent A** the decayed pheromone value `p` is a *prior weight* for an
  edge.
* In **Parent B** the semantic cosine similarity `s ∈ [0,1]` and the recovery
  priority `r ∈ [0,1]` together form a *likelihood* that a connection is
  reliable.

The hybrid algorithm treats the product `ℓ = s·r` as the number of
*pseudo‑successes* and `(1‑ℓ)` as pseudo‑failures, and feeds them into the
Beta‑posterior together with the pheromone‑derived prior count
`p·K` (where `K` is a scaling constant).  The resulting posterior mean is a
single scalar that fuses pheromone strength, semantic similarity and
morphological recovery into a unified reliability estimate.

The code below implements this fusion while preserving the core
topologies of both parents.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Pheromone system and Bayesian edge update
# ----------------------------------------------------------------------
class PheromoneSystem:
    """Stores and decays pheromone signals per surface and signal kind."""
    def __init__(self):
        self.pheromone_signals: dict[str, dict[str, float]] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        """Return the decayed signal after one time step (Δt = 1 s)."""
        if surface_key not in self.pheromone_signals:
            self.pheromone_signals[surface_key] = {}
        # initialise if missing
        if signal_kind not in self.pheromone_signals[surface_key]:
            self.pheromone_signals[surface_key][signal_kind] = signal_value
        # exponential decay: v * (1/2)^{Δt/τ}
        elapsed = 1.0
        current = self.pheromone_signals[surface_key][signal_kind]
        decayed = current * math.pow(0.5, elapsed / half_life_seconds)
        self.pheromone_signals[surface_key][signal_kind] = decayed
        return decayed

    def update_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> None:
        """Overwrite the stored value (used for injection of fresh pheromone)."""
        if surface_key not in self.pheromone_signals:
            self.pheromone_signals[surface_key] = {}
        self.pheromone_signals[surface_key][signal_kind] = signal_value

def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Metropolis acceptance used by the original parent (kept for completeness)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return math.exp(-delta_energy / temperature)

@dataclass(frozen=True)
class EdgeBetaPrior:
    """Beta prior for edge reliability."""
    alpha: float = 1.0
    beta: float = 1.0

def bayesian_edge_update(
    prior: EdgeBetaPrior,
    successes: float,
    failures: float,
) -> tuple[float, EdgeBetaPrior]:
    """Return posterior mean and new prior after observing successes/failures."""
    new_alpha = prior.alpha + successes
    new_beta = prior.beta + failures
    posterior_mean = new_alpha / (new_alpha + new_beta)
    return posterior_mean, EdgeBetaPrior(new_alpha, new_beta)

# ----------------------------------------------------------------------
# Parent B – Morphology, semantic similarity and neighbor utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimension‑based sphericity (unit‑less)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness ratio used in righting‑time calculation."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(
    m: Morphology,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    """Physical model of how quickly an organism can right itself."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority ∈[0,1] derived from righting‑time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def _cos(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if den == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / den

def semantic_neighbors(
    doc_id: str,
    vector: list[float],
    k: int = 5,
) -> list[tuple[str, float]]:
    """
    Return the top‑k nearest neighbours (including a synthetic reference set)
    based on cosine similarity.
    """
    candidates = [("doc" + str(i), np.random.rand(len(vector)).tolist())
                  for i in range(1, k * 2)]  # oversample, then cut
    sims = [(d, _cos(vector, v)) for d, v in candidates if d != doc_id]
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[:k]

def extract_full_features(text: str) -> dict[str, float]:
    """
    Very lightweight placeholder that extracts a few numeric features from
    raw text.  In a real system this would be far richer.
    """
    length = len(text)
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / max(1, len(words))
    vowel_ratio = sum(c.lower() in "aeiou" for c in text) / max(1, length)
    return {"length": float(length), "avg_word_len": avg_word_len, "vowel_ratio": vowel_ratio}

# ----------------------------------------------------------------------
# Hybrid core – three public functions demonstrating the fused operation
# ----------------------------------------------------------------------
def hybrid_pheromone_decay(
    pheromone_system: PheromoneSystem,
    surface_key: str,
    signal_kind: str,
    signal_value: float,
    half_life_seconds: float,
) -> float:
    """
    Decay (or initialise) a pheromone signal and return the current value.
    This value will later be interpreted as a *prior count* for the Bayesian
    update (scaled by `PHEROMONE_SCALE`).
    """
    return pheromone_system.calculate_pheromone_signal(
        surface_key, signal_kind, signal_value, half_life_seconds
    )

def hybrid_semantic_likelihood(
    query_vector: list[float],
    reference_vector: list[float],
    morphology: Morphology,
) -> float:
    """
    Compute a likelihood term ℓ ∈[0,1] that fuses semantic similarity
    (cosine) with morphological recovery priority.
    """
    similarity = _cos(query_vector, reference_vector)  # ∈[0,1]
    recovery = recovery_priority(morphology)          # ∈[0,1]
    return similarity * recovery

def hybrid_pheromone_bayesian_reliability(
    pheromone_system: PheromoneSystem,
    surface_key: str,
    signal_kind: str,
    signal_value: float,
    half_life_seconds: float,
    query_vector: list[float],
    reference_vector: list[float],
    morphology: Morphology,
    prior: EdgeBetaPrior,
    pheromone_scale: float = 10.0,
) -> tuple[float, EdgeBetaPrior]:
    """
    Full hybrid update:
      1. Decay pheromone → prior count `p = decay * pheromone_scale`.
      2. Compute likelihood ℓ from semantic similarity & morphology.
      3. Treat ℓ as pseudo‑success probability for `p` virtual trials.
      4. Perform Beta‑posterior update.

    Returns the posterior mean reliability and the updated prior.
    """
    # 1. Pheromone decay (prior weight)
    decayed = hybrid_pheromone_decay(
        pheromone_system,
        surface_key,
        signal_kind,
        signal_value,
        half_life_seconds,
    )
    virtual_trials = max(1.0, decayed * pheromone_scale)

    # 2. Likelihood from semantics + morphology
    ell = hybrid_semantic_likelihood(query_vector, reference_vector, morphology)

    # 3. Convert to pseudo‑counts
    successes = ell * virtual_trials
    failures = (1.0 - ell) * virtual_trials

    # 4. Bayesian Beta update
    posterior_mean, new_prior = bayesian_edge_update(prior, successes, failures)
    return posterior_mean, new_prior

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    pheromone = PheromoneSystem()
    prior = EdgeBetaPrior(alpha=2.0, beta=2.0)

    # Example morphology (arbitrary but plausible)
    morph = Morphology(length=1.2, width=0.8, height=0.5, mass=3.0)

    # Random vectors to simulate semantic embeddings
    np.random.seed(42)
    query_vec = np.random.rand(5).tolist()
    reference_vec = np.random.rand(5).tolist()

    # Perform hybrid update
    reliability, updated_prior = hybrid_pheromone_bayesian_reliability(
        pheromone_system=pheromone,
        surface_key="surface_01",
        signal_kind="trail",
        signal_value=1.0,
        half_life_seconds=30.0,
        query_vector=query_vec,
        reference_vector=reference_vec,
        morphology=morph,
        prior=prior,
        pheromone_scale=8.0,
    )

    print(f"Posterior reliability: {reliability:.4f}")
    print(f"Updated prior: α={updated_prior.alpha:.2f}, β={updated_prior.beta:.2f}")

    # Verify that semantic_neighbors runs without error
    neighbours = semantic_neighbors("doc0", query_vec, k=3)
    print("Top semantic neighbours (id, similarity):")
    for nid, sim in neighbours:
        print(f"  {nid}: {sim:.3f}")

    sys.exit(0)