# DARWIN HAMMER — match 3420, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m442_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_model__hybrid_hard_truth_ma_m23_s0.py (gen3)
# born: 2026-05-29T23:49:58Z

"""Hybrid Fusion Module
=====================

This module fuses the core mathematical structures of two parent algorithms:

* **Parent A** – provides epistemic certainty flags and Bayesian probability
  updates (`bayes_marginal`, `bayes_update`).  
* **Parent B** – defines a weight matrix `W` that is updated via gradient descent
  using stylometry feature vectors derived from text.

**Mathematical bridge**

The bridge is built on the observation that epistemic certainty can be treated as
a probability distribution over *feature importance*.  We therefore map the
epistemic flags to a probability vector `p_epistemic`.  This vector scales the
gradient used to update the weight matrix `W`.  The scaling factor itself is
adapted by a Bayesian posterior computed from a prior, a likelihood and a false‑positive
rate.  In this way the Bayesian reasoning of Parent A directly controls the
gradient‑descent dynamics of Parent B, yielding a single unified update rule:


posterior = bayes_update(prior, likelihood, bayes_marginal(prior, likelihood, fp))
W ← W − η·posterior·(features ⊗ p_epistemic)


where `⊗` denotes the outer product, `features` is the stylometry feature vector
and `η` is a base learning rate.

The functions below implement this hybrid operation and expose three
demonstration utilities.
"""

import re
import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – epistemic flags and Bayesian utilities
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_EPISTEMIC_MAP: dict[str, float] = {
    "FACT": 0.99,
    "PROBABLE": 0.80,
    "POSSIBLE": 0.50,
    "SURE_MAYBE": 0.60,
    "BULLSHIT": 0.00,
}

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for a Bayesian update."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probability arguments must be in [0, 1].")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform a Bayesian update returning the posterior probability."""
    if marginal == 0.0:
        return 0.0
    return (likelihood * prior) / marginal

def epistemic_vector(flags: Iterable[str]) -> np.ndarray:
    """
    Convert an iterable of epistemic flag strings to a normalized probability vector.
    Unknown flags receive a probability of 0.0.
    """
    probs = [_EPISTEMIC_MAP.get(flag.upper(), 0.0) for flag in flags]
    total = sum(probs)
    if total == 0.0:
        # avoid division by zero – return a uniform low‑confidence vector
        return np.full(len(EPISTEMIC_FLAGS), 1.0 / len(EPISTEMIC_FLAGS))
    return np.array(probs) / total

# ----------------------------------------------------------------------
# Parent B – stylometry feature extraction and weight‑matrix handling
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def _tokenize(text: str) -> List[str]:
    """Very light tokenizer: lower‑case, strip punctuation, split on whitespace."""
    cleaned = text.lower()
    for ch in PUNCT:
        cleaned = cleaned.replace(ch, " ")
    return [tok for tok in cleaned.split() if tok]

def stylometry_feature_vector(text: str) -> np.ndarray:
    """
    Produce a numeric feature vector where each component is the count of words
    belonging to a functional category defined in ``FUNCTION_CATS``.
    The order of components follows the sorted keys of ``FUNCTION_CATS``.
    """
    tokens = _tokenize(text)
    counts = []
    for cat in sorted(FUNCTION_CATS.keys()):
        cat_set = FUNCTION_CATS[cat]
        counts.append(sum(1 for tok in tokens if tok in cat_set))
    return np.array(counts, dtype=float)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_gradient(features: np.ndarray, epistemic: np.ndarray) -> np.ndarray:
    """
    Compute the outer product between a stylometry feature vector and an
    epistemic probability vector, yielding a gradient matrix compatible with the
    weight matrix ``W``.
    """
    if features.ndim != 1 or epistemic.ndim != 1:
        raise ValueError("Both inputs must be one‑dimensional arrays.")
    return np.outer(features, epistemic)

def hybrid_update_weights(
    W: np.ndarray,
    text: str,
    epistemic_flags: Iterable[str],
    prior: float,
    likelihood: float,
    false_positive: float,
    base_lr: float = 0.01,
) -> np.ndarray:
    """
    Perform a single hybrid update of the weight matrix ``W``:

    1. Extract stylometry features from ``text``.
    2. Convert ``epistemic_flags`` to a probability vector.
    3. Compute a Bayesian posterior using ``prior``, ``likelihood`` and ``false_positive``.
    4. Scale the outer‑product gradient by ``base_lr * posterior`` and subtract it from ``W``.
    """
    # Step 1 – feature extraction
    feat_vec = stylometry_feature_vector(text)

    # Step 2 – epistemic vector
    epi_vec = epistemic_vector(epistemic_flags)

    # Step 3 – Bayesian posterior
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    # Step 4 – gradient and update
    grad = hybrid_gradient(feat_vec, epi_vec)
    update = base_lr * posterior * grad
    return W - update

def hybrid_predict(
    W: np.ndarray,
    text: str,
    epistemic_flags: Iterable[str],
) -> np.ndarray:
    """
    Produce a prediction vector by multiplying the weight matrix ``W`` with the
    epistemic‑scaled feature vector.  The result can be interpreted as a
    confidence‑weighted score for each stylometric category.
    """
    feat_vec = stylometry_feature_vector(text)
    epi_vec = epistemic_vector(epistemic_flags)
    # Combine feature importance with epistemic confidence via element‑wise product
    combined = feat_vec * np.mean(epi_vec)  # simple scalar weighting
    return W @ combined  # matrix‑vector product

def hybrid_initialize_weights(num_features: int, num_epistemic: int, seed: int | None = None) -> np.ndarray:
    """
    Initialise a weight matrix with small random values.  ``num_features`` is the
    length of the stylometry feature vector; ``num_epistemic`` is the number of
    epistemic flags (normally ``len(EPISTEMIC_FLAGS)``).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.1, size=(num_features, num_epistemic))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal demonstration that runs without external data.
    sample_text = (
        "I think that the quick brown fox jumps over the lazy dog while we observe "
        "the results of a test. It is possible that the outcome is correct."
    )
    flags = ["PROBABLE", "SURE_MAYBE", "FACT", "BULLSHIT"]
    prior = 0.3
    likelihood = 0.7
    false_positive = 0.1

    # Initialise weight matrix
    W = hybrid_initialize_weights(num_features=len(FUNCTION_CATS), num_epistemic=len(EPISTEMIC_FLAGS), seed=42)

    # Perform an update
    W_new = hybrid_update_weights(
        W,
        sample_text,
        flags,
        prior,
        likelihood,
        false_positive,
        base_lr=0.05,
    )

    # Predict using the updated matrix
    prediction = hybrid_predict(W_new, sample_text, flags)

    # Simple sanity prints
    print("Original weight matrix (first row):", W[0, :5])
    print("Updated weight matrix (first row):", W_new[0, :5])
    print("Prediction vector:", prediction)
    print("Done.")