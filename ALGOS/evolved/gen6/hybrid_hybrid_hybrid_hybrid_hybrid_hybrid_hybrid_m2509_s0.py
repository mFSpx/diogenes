# DARWIN HAMMER — match 2509, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m560_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m413_s3.py (gen4)
# born: 2026-05-29T23:42:43Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Parents A and B.

Parent A contributes:
- Stylometric feature extraction producing a categorical count matrix.
- PheromoneEntry with exponential decay representing dynamic signal strength.

Parent B contributes:
- Bayesian update functions (bayes_marginal, bayes_update).
- Semantic neighbor distances used as likelihoods.
- Entropy‑driven action selection.

Mathematical Bridge:
The normalized stylometric feature vector of a text serves as the *prior* probability
distribution over linguistic categories. The similarity scores returned by
`semantic_neighbors` are transformed into likelihoods (via a soft‑exponential map)
and combined with the priors through the Bayesian marginal and update formulas.
The resulting posterior updates the `signal_value` of each `PheromoneEntry`.  An
expected‑entropy calculation on the posterior distribution selects the token that
minimises the anticipated information loss, thus fusing the decay dynamics of
Parent A with the Bayesian‑entropy decision logic of Parent B into a single
coherent system.
"""

import math
import random
import sys
import pathlib
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# Feature extraction (Parent A)
# ----------------------------------------------------------------------
def stylometric_feature_extraction(texts: list[str]) -> np.ndarray:
    """Return a matrix (len(texts) × categories) of stylometric counts."""
    FUNCTION_CATS = {
        "pronoun": {
            "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
            "he", "him", "his", "she", "her", "hers", "they", "them", "their",
            "theirs", "we", "us", "our", "ours"
        },
        "article": {"a", "an", "the"},
        "preposition": {
            "about", "above", "after", "against", "around", "as", "at", "before",
            "behind", "below", "between", "by", "during", "for", "from", "in",
            "into", "of", "off", "on", "onto", "over", "through", "to", "under",
            "with", "without"
        },
        "auxiliary": {
            "am", "are", "be", "been", "being", "can", "could", "did", "do",
            "does", "had", "has", "have", "is", "may", "might"
        }
    }
    categories = list(FUNCTION_CATS.keys())
    matrix = np.zeros((len(texts), len(categories)), dtype=float)

    for idx, text in enumerate(texts):
        lower = text.lower()
        for cat_idx, (cat, words) in enumerate(FUNCTION_CATS.items()):
            count = sum(lower.count(word) for word in words)
            matrix[idx, cat_idx] = count
    return matrix


# ----------------------------------------------------------------------
# Functions from Parent B
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


def label_score(text: str, label: str) -> float:
    """Simple literal fallback score: count occurrences of the label."""
    return float(text.lower().count(label.lower()))


def semantic_neighbors(doc_id: str, k: int = 5) -> list[tuple[str, float]]:
    """Mocked semantic neighbor generator: returns k (doc_id, similarity) pairs."""
    # Similarity in [0,1]; higher means more similar.
    return [(f"{doc_id}_nbr_{i}", random.random()) for i in range(k)]


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Row‑wise L1 normalisation, safe for zero rows."""
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


def hybrid_feature_prior(texts: list[str]) -> np.ndarray:
    """
    Convert stylometric counts into a prior probability distribution.
    Each row corresponds to a text and sums to 1.
    """
    counts = stylometric_feature_extraction(texts)
    priors = _normalize_rows(counts)
    return priors


def _likelihood_from_neighbors(neighbors: list[tuple[str, float]]) -> np.ndarray:
    """
    Transform raw similarity scores into likelihoods via an exponential decay
    and normalise to a proper probability vector.
    """
    sims = np.array([sim for _, sim in neighbors], dtype=float)
    # Convert similarity to distance‑like quantity and apply softmax‑like mapping
    raw = np.exp(-sims)               # higher similarity → larger likelihood
    if raw.sum() == 0:
        return np.full_like(raw, 1.0 / raw.size)
    return raw / raw.sum()


def hybrid_bayesian_pheromone_update(
    pheromones: list[PheromoneEntry],
    priors: np.ndarray,
    doc_id: str,
    false_positive: float = 0.01
) -> None:
    """
    For each pheromone entry, treat its current signal_value as the prior
    probability of the associated linguistic category.  Use semantic neighbor
    similarities as likelihoods, perform a Bayesian update, and finally apply
    exponential decay.
    """
    # Assume the number of categories equals the number of pheromones
    neighbors = semantic_neighbors(doc_id, k=len(pheromones))
    likelihoods = _likelihood_from_neighbors(neighbors)

    # Align priors (matrix) with pheromones: take the mean prior across texts
    mean_prior = priors.mean(axis=0)  # shape: (categories,)

    for idx, pher in enumerate(pheromones):
        prior = mean_prior[idx] if idx < mean_prior.size else 0.0
        likelihood = likelihoods[idx] if idx < likelihoods.size else 0.0
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        # Blend the posterior into the pheromone's signal value
        pher.signal_value = (pher.signal_value + posterior) / 2.0
        pher.apply_decay()


def expected_entropy(prob_dist: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    eps = np.finfo(float).eps
    p = np.clip(prob_dist, eps, 1.0)
    return -np.sum(p * np.log(p))


def hybrid_entropy_action_selection(
    tokens: list[str],
    prob_dists: list[np.ndarray]
) -> str:
    """
    Given candidate tokens and their associated posterior probability distributions,
    select the token that yields the minimal expected entropy after the update.
    """
    if len(tokens) != len(prob_dists):
        raise ValueError("tokens and prob_dists must have the same length")
    entropies = [expected_entropy(dist) for dist in prob_dists]
    min_idx = int(np.argmin(entropies))
    return tokens[min_idx]


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample texts
    sample_texts = [
        "I think therefore I am. The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore. You are the sunshine of my life."
    ]

    # 1. Build prior from stylometric features
    priors = hybrid_feature_prior(sample_texts)
    print("Prior matrix (rows=texts, cols=categories):")
    print(priors)

    # 2. Initialise pheromones – one per stylometric category
    categories = ["pronoun", "article", "preposition", "auxiliary"]
    pheromones = [
        PheromoneEntry(surface_key=cat,
                       signal_kind="syntactic",
                       signal_value=random.random(),
                       half_life_seconds=30)
        for cat in categories
    ]

    # 3. Update pheromones using Bayesian fusion with semantic neighbors
    hybrid_bayesian_pheromone_update(pheromones, priors, doc_id="doc_42")
    print("\nPheromone values after Bayesian update and decay:")
    for p in pheromones:
        print(f"{p.surface_key}: {p.signal_value:.4f}")

    # 4. Prepare candidate tokens and their posterior distributions
    candidate_tokens = ["if", "while", "for"]
    # For illustration, create mock posterior distributions by perturbing priors
    posterior_dists = []
    for _ in candidate_tokens:
        noise = np.random.rand(priors.shape[1]) * 0.05
        dist = _normalize_rows(priors.mean(axis=0).reshape(1, -1) + noise)
        posterior_dists.append(dist.ravel())

    # 5. Select the action (token) with minimal expected entropy
    chosen = hybrid_entropy_action_selection(candidate_tokens, posterior_dists)
    print(f"\nChosen token (min expected entropy): {chosen}")