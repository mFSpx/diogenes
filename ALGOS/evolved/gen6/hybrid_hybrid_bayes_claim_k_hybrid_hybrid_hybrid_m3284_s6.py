# DARWIN HAMMER — match 3284, survivor 6
# gen: 6
# parent_a: hybrid_bayes_claim_kernel_hybrid_ternary_lens__m26_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m560_s3.py (gen5)
# born: 2026-05-29T23:48:59Z

"""Hybrid Bayesian-Pheromone Algorithm
Parent A: hybrid_bayes_claim_kernel_hybrid_ternary_lens__m26_s0.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m560_s3.py

Mathematical Bridge
-------------------
Both parents employ exponential decay mechanisms:
* Parent A prunes evidence with a decreasing‑rate schedule  
  p_prune(t)=λ·exp(−α t).
* Parent B decays pheromone signal values with a half‑life model  
  f_decay(t)=0.5^{t/τ}.

The hybrid treats each piece of evidence as a “pheromone” whose
signal value is multiplied by the pruning probability, yielding an
effective weight  
 w(t)=signal·p_prune(t).  
During Bayesian updating the odds are multiplied by the product of the
likelihood ratio and this weight, thus unifying the two decay
formulations in a single probabilistic update.

The module implements:
* Pheromone‑aware Bayesian hypothesis update.
* Vectorised batch update using NumPy.
* Decay and pruning utilities that share the same exponential kernel.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import uuid
import numpy as np
from typing import List, Tuple, Dict, Iterable

# ----------------------------------------------------------------------
# Data structures (from Parent A and Parent B)
# ----------------------------------------------------------------------
class MathClaim:
    def __init__(self, id: str):
        self.id = id


class MathEvidence:
    def __init__(self, id: str):
        self.id = id


class MathHypothesis:
    def __init__(self, id: str, prior: float, posterior: float, evidence_ids: List[str]):
        self.id = id
        self.prior = prior          # prior before the last update
        self.posterior = posterior  # current belief
        self.evidence_ids = evidence_ids


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind",
                 "signal_value", "half_life_seconds",
                 "created_at", "last_decay")

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
        """Half‑life exponential decay factor."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        """Apply decay in‑place."""
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Exponential pruning probability from Parent A."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def weighted_likelihood(likelihood_ratio: float, pheromone: PheromoneEntry,
                        t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """
    Combine a raw likelihood ratio with pheromone decay and pruning probability.
    The effective weight is:
        w = likelihood_ratio * pheromone.signal_value * prune_probability(t)
    """
    if likelihood_ratio < 0:
        raise ValueError("likelihood_ratio must be non‑negative")
    # decay the pheromone to the current time
    pheromone.apply_decay()
    weight = likelihood_ratio * pheromone.signal_value * prune_probability(t, lam, alpha)
    return weight


def update_hypothesis(hypothesis: MathHypothesis,
                     evidence: MathEvidence,
                     weighted_lr: float) -> MathHypothesis:
    """
    Bayesian odds update using the weighted likelihood ratio.
    Mirrors Parent A's update but accepts a pre‑weighted ratio.
    """
    p = max(0.0, min(1.0, hypothesis.posterior))
    if p <= 0.0 or weighted_lr == 0.0:
        posterior = 0.0
    elif p >= 1.0:
        posterior = 1.0
    else:
        odds = p / (1.0 - p)
        new_odds = odds * weighted_lr
        posterior = new_odds / (1.0 + new_odds)

    posterior = max(0.0, min(1.0, posterior))
    ids = hypothesis.evidence_ids + [evidence.id]
    return MathHypothesis(id=hypothesis.id,
                          prior=hypothesis.posterior,
                          posterior=posterior,
                          evidence_ids=ids)


# ----------------------------------------------------------------------
# Hybrid operations (three required functions)
# ----------------------------------------------------------------------
def hybrid_update_one(hypothesis: MathHypothesis,
                      evidence: MathEvidence,
                      pheromone: PheromoneEntry,
                      t: float,
                      base_lr: float,
                      lam: float = 1.0,
                      alpha: float = 0.2) -> MathHypothesis:
    """
    Update a single hypothesis with a single evidence item,
    using the pheromone‑aware weighted likelihood.
    """
    w_lr = weighted_likelihood(base_lr, pheromone, t, lam, alpha)
    return update_hypothesis(hypothesis, evidence, w_lr)


def hybrid_batch_update(hypotheses: List[MathHypothesis],
                        evidences: List[MathEvidence],
                        pheromones: Dict[str, PheromoneEntry],
                        t: float,
                        base_lrs: np.ndarray,
                        lam: float = 1.0,
                        alpha: float = 0.2) -> List[MathHypothesis]:
    """
    Vectorised batch update for many hypotheses against many evidences.
    * hypotheses  – list of length H
    * evidences   – list of length E
    * base_lrs    – H×E numpy array of raw likelihood ratios
    Returns a new list of hypotheses with updated posteriors.
    """
    H = len(hypotheses)
    E = len(evidences)
    if base_lrs.shape != (H, E):
        raise ValueError("base_lrs must have shape (len(hypotheses), len(evidences))")

    # Compute pruning probability once (scalar for the whole batch)
    p_prune = prune_probability(t, lam, alpha)

    # Prepare an array of current posterior odds
    post = np.array([max(0.0, min(1.0, h.posterior)) for h in hypotheses])
    odds = np.where(post == 1.0, np.inf,
                    np.where(post == 0.0, 0.0, post / (1.0 - post)))

    # Apply pheromone decay and combine with pruning probability
    decay_factors = np.empty((H, E))
    for i, h in enumerate(hypotheses):
        for j, ev in enumerate(evidences):
            ph = pheromones.get(ev.id)
            if ph is None:
                decay = 1.0  # no pheromone → neutral weight
            else:
                ph.apply_decay()
                decay = ph.signal_value
            decay_factors[i, j] = decay

    weighted = base_lrs * decay_factors * p_prune
    new_odds = odds[:, None] * weighted
    # Convert back to probabilities safely
    new_post = np.where(np.isinf(new_odds), 1.0,
                        new_odds / (1.0 + new_odds))
    new_post = np.clip(new_post, 0.0, 1.0)

    # Build updated hypothesis objects
    updated = []
    for i, h in enumerate(hypotheses):
        # accumulate evidence ids from all evidences (order preserved)
        new_ids = h.evidence_ids + [ev.id for ev in evidences]
        updated.append(MathHypothesis(id=h.id,
                                      prior=h.posterior,
                                      posterior=float(new_post[i]),
                                      evidence_ids=new_ids))
    return updated


def stylometric_feature_extraction(texts: List[str]) -> np.ndarray:
    """
    Simple stylometric extractor (inspired by Parent B).
    Returns an N×C matrix where C is the number of function‑word categories.
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
        }
    }
    cat_names = list(FUNCTION_CATS.keys())
    C = len(cat_names)
    N = len(texts)
    matrix = np.zeros((N, C), dtype=float)

    for i, txt in enumerate(texts):
        lowered = txt.lower()
        for j, cat in enumerate(cat_names):
            words = FUNCTION_CATS[cat]
            count = sum(lowered.count(w) for w in words)
            matrix[i, j] = count
        # normalise by length to obtain frequencies
        length = max(1, len(txt.split()))
        matrix[i] /= length
    return matrix


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a single hypothesis
    hyp = MathHypothesis(id="H1", prior=0.5, posterior=0.5, evidence_ids=[])

    # Two pieces of evidence
    ev1 = MathEvidence(id="E1")
    ev2 = MathEvidence(id="E2")
    evidences = [ev1, ev2]

    # Pheromones attached to evidence
    ph1 = PheromoneEntry(surface_key="E1", signal_kind="typeA", signal_value=0.9, half_life_seconds=30)
    ph2 = PheromoneEntry(surface_key="E2", signal_kind="typeB", signal_value=0.4, half_life_seconds=30)
    pheromones = {"E1": ph1, "E2": ph2}

    # Base likelihood ratios (H×E matrix)
    base_lr = np.array([[2.0, 0.5]])  # single hypothesis vs two evidences

    # Perform batch update at time t=5
    updated = hybrid_batch_update([hyp], evidences, pheromones, t=5.0, base_lrs=base_lr)

    print("Updated hypothesis posterior:", updated[0].posterior)
    print("Evidence chain:", updated[0].evidence_ids)

    # Demonstrate stylometric extraction
    texts = ["I have a dream.", "She walks through the garden."]
    feats = stylometric_feature_extraction(texts)
    print("Stylometric features:\n", feats)