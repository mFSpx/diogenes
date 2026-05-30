# DARWIN HAMMER — match 2509, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m560_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m413_s3.py (gen4)
# born: 2026-05-29T23:42:43Z

"""
Hybrid Algorithm Fusion of:
- Parent A: stylometric feature extraction & pheromone decay (DARWIN HAMMER match 560)
- Parent B: semantic neighbor Bayesian update & entropy‑driven decision (DARWIN HAMMER match 413)

Mathematical Bridge
-------------------
We treat the *pheromone signal* associated with a label as the **prior** 𝑃(L).
The *semantic‑neighbor similarity* (1‑distance) supplies the **likelihood** 𝑃(E|L).
A lightweight stylometric fingerprint of the input text provides a **false‑positive**
rate 𝑃(E|¬L) derived from the relative frequency of functional word categories.
Using the Bayesian formulas from Parent B we obtain the posterior
𝑃(L|E)=𝑃(L)·𝑃(E|L) / 𝑃(E) where 𝑃(E)=𝑃(E|L)𝑃(L)+𝑃(E|¬L)(1‑𝑃(L)).
Finally, an entropy‑based selector (also from Parent B) chooses the label (or
action) that minimises the expected post‑update entropy, thus fusing both
topologies into a single decision engine.
"""

import math
import random
import sys
import pathlib
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

# ----------------------------------------------------------------------
# Parent A components
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
    """Decay‑aware pheromone container (Parent A)."""

    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
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
        self.signal_value *= self.decay_factor()
        self.last_decay = datetime.now(timezone.utc)


def stylometric_feature_extraction(texts: list[str]) -> np.ndarray:
    """
    Extract a simple stylometric fingerprint based on functional‑word categories.
    Returns an (n_texts × n_categories) matrix of normalized counts.
    """
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
        },
    }

    n_cats = len(FUNCTION_CATS)
    feats = np.zeros((len(texts), n_cats), dtype=float)

    for idx, txt in enumerate(texts):
        tokens = txt.lower().split()
        total = len(tokens) if tokens else 1
        for cat_idx, (_, word_set) in enumerate(FUNCTION_CATS.items()):
            count = sum(tok in word_set for tok in tokens)
            feats[idx, cat_idx] = count / total  # normalised frequency

    return feats


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------


Point = tuple[float, float]
Edge = tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|L)·P(L) + P(E|¬L)·(1‑P(L))."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(L|E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def label_score(text: str, label: str) -> float:
    """Literal fallback scoring: raw occurrence count."""
    return float(text.lower().count(label.lower()))


def semantic_neighbors(doc_id: str, k: int = 5) -> list[tuple[str, float]]:
    """
    Dummy neighbour generator.
    Returns (neighbor_id, distance) where distance ∈ (0,1].
    """
    random.seed(hash(doc_id))  # deterministic per doc_id
    return [(f"doc_{i}", random.random()) for i in range(k)]


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


# ----------------------------------------------------------------------
# Fusion core
# ----------------------------------------------------------------------


def aggregate_pheromone_prior(label: str, pheromones: list[PheromoneEntry]) -> float:
    """
    Combine all pheromone entries that match *label* into a prior probability.
    The raw sum is normalised to the interval [0,1] using a simple sigmoid.
    """
    total = sum(p.signal_value for p in pheromones if p.signal_kind == label)
    # sigmoid to squash arbitrary sums into [0,1]
    return 1.0 / (1.0 + math.exp(-total))


def stylometric_false_positive(text: str, label: str) -> float:
    """
    Derive a false‑positive rate from stylometric features.
    We map the pronoun frequency (as a proxy for personal style) to a probability.
    """
    feats = stylometric_feature_extraction([text])[0]  # shape (n_cats,)
    # Use the pronoun column (index 0) as the basis; clamp to [0,1]
    pronoun_freq = feats[0]
    # Simple linear mapping with a small offset to avoid zero
    return min(1.0, max(0.0, pronoun_freq * 0.8 + 0.1))


def semantic_likelihood(label: str, neighbors: list[tuple[str, float]]) -> float:
    """
    Transform neighbour distances into a likelihood for *label*.
    We treat (1‑distance) as similarity; the maximum similarity among neighbours
    that contain the label substring is taken as the likelihood.
    """
    sims = [(1.0 - d) for _, d in neighbors if label.lower() in _.lower()]
    if not sims:
        # fallback to a small uniform likelihood
        return 0.05
    return max(sims)


def compute_label_posterior(
    text: str,
    label: str,
    pheromones: list[PheromoneEntry],
    doc_id: str,
) -> float:
    """
    Full Bayesian fusion:
        prior      ← aggregated pheromone signal
        likelihood ← semantic‑neighbor similarity
        fp_rate    ← stylometric‑derived false‑positive
        posterior  ← bayes_update(prior, likelihood, marginal)
    """
    prior = aggregate_pheromone_prior(label, pheromones)
    neighbors = semantic_neighbors(doc_id)
    likelihood = semantic_likelihood(label, neighbors)
    false_pos = stylometric_false_positive(text, label)

    marginal = bayes_marginal(prior, likelihood, false_pos)
    posterior = bayes_update(prior, likelihood, marginal)
    return posterior


def expected_entropy(posteriors: dict[str, float]) -> float:
    """
    Compute Shannon entropy of the posterior distribution over labels.
    """
    eps = 1e-12
    ent = 0.0
    for p in posteriors.values():
        p = max(eps, min(1.0 - eps, p))
        ent -= p * math.log(p, 2)
    return ent


def select_min_entropy_label(
    text: str,
    candidate_labels: list[str],
    pheromones: list[PheromoneEntry],
    doc_id: str,
) -> tuple[str, float]:
    """
    Evaluate each candidate label, compute its posterior, and return the label
    that yields the minimal expected entropy of the overall distribution.
    """
    posteriors = {}
    for lab in candidate_labels:
        posteriors[lab] = compute_label_posterior(text, lab, pheromones, doc_id)

    # Normalise to ensure they sum to 1 (required for proper entropy)
    total = sum(posteriors.values()) or 1.0
    for k in posteriors:
        posteriors[k] /= total

    best_label = min(posteriors, key=lambda k: expected_entropy({k: posteriors[k]}))
    return best_label, posteriors[best_label]


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _demo():
    # Sample texts and labels
    texts = [
        "I think we should consider the impact of the new policy on our community.",
        "The quick brown fox jumps over the lazy dog.",
        "Data science combines statistics, programming, and domain knowledge."
    ]
    labels = ["policy", "animal", "technology"]

    # Create a few pheromone entries with varying half‑life
    pheromones = [
        PheromoneEntry(surface_key="doc_0", signal_kind="policy", signal_value=1.2, half_life_seconds=3600),
        PheromoneEntry(surface_key="doc_0", signal_kind="animal", signal_value=0.4, half_life_seconds=1800),
        PheromoneEntry(surface_key="doc_0", signal_kind="technology", signal_value=0.9, half_life_seconds=7200),
    ]

    # Apply decay once to simulate time passing
    for p in pheromones:
        p.apply_decay()

    for idx, txt in enumerate(texts):
        doc_id = f"doc_{idx}"
        chosen, prob = select_min_entropy_label(txt, labels, pheromones, doc_id)
        print(f"Document {doc_id!r} => chosen label: {chosen!r} (posterior={prob:.4f})")


if __name__ == "__main__":
    _demo()