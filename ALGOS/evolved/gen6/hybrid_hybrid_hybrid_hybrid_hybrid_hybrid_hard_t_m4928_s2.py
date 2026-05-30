# DARWIN HAMMER — match 4928, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_indy_learning_m952_s1.py (gen4)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_minhas_m1874_s1.py (gen5)
# born: 2026-05-29T23:58:56Z

"""Hybrid algorithm combining stylometry matrix dynamics (Parent A) with
categorical LSM vectors and MinHash‑based edge weighting (Parent B).

Mathematical bridge:
- From Parent A we obtain a categorical co‑occurrence (impedance) matrix ζ
  where ζ₍i,j₎ counts how often function‑category i and j appear together.
- From Parent B we compute a categorical frequency vector **c** (LSM) and a
  MinHash signature vector **h**.
- The hybrid edge weight for each matrix entry follows Parent B’s formula

      w₍i,j₎ = ζ₍i,j₎ · (1 + α·mean(c)) · (1 + β·mean(h)/2⁶⁴)

  which injects the matrix ζ into the vector‑based scaling.
- A Bayesian update normalises these weights to a posterior edge‑cost
  distribution.

The module provides functions to build the vectors, the impedance matrix,
the hybrid weight matrix and the posterior distribution, demonstrating a
mathematical fusion of both parent topologies.
"""

import math
import random
import sys
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Shared linguistic resources
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
_CAT_LIST: List[str] = list(FUNCTION_CATS.keys())
_N_CAT: int = len(_CAT_LIST)

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    """Simple word tokeniser, lower‑cases and strips punctuation."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [tok for tok in cleaned.split() if tok]

def _sentence_split(text: str) -> List[str]:
    """Very light sentence splitter based on punctuation."""
    return [s.strip() for s in re.split(r"[.!?;\n]+", text) if s.strip()]

def _map_token_to_cats(token: str) -> List[int]:
    """Return list of category indices a token belongs to."""
    idxs = []
    for i, cat in enumerate(_CAT_LIST):
        if token in FUNCTION_CATS[cat]:
            idxs.append(i)
    return idxs

# ----------------------------------------------------------------------
# Core hybrid components
# ----------------------------------------------------------------------
def categorical_frequency_vector(text: str) -> np.ndarray:
    """
    Compute the LSM‑style categorical frequency vector **c**.
    Length = number of function categories.
    """
    vec = np.zeros(_N_CAT, dtype=np.float64)
    for token in _tokenize(text):
        for idx in _map_token_to_cats(token):
            vec[idx] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total  # normalise to a probability distribution
    return vec


def minhash_signature(text: str, num_perm: int = 64) -> np.ndarray:
    """
    Compute a simple MinHash signature **h** of length ``num_perm``.
    Each permutation is simulated by hashing the token concatenated with a
    seed and taking the minimum 64‑bit integer.
    """
    tokens = _tokenize(text)
    if not tokens:
        return np.full(num_perm, np.uint64(2**64 - 1), dtype=np.uint64)

    signature = np.full(num_perm, np.uint64(2**64 - 1), dtype=np.uint64)
    for seed in range(num_perm):
        min_hash = np.uint64(2**64 - 1)
        for tok in tokens:
            # deterministic hash mixing token with seed
            h = hashlib.blake2b(digest_size=8, person=seed.to_bytes(1, "little"))
            h.update(tok.encode("utf-8"))
            val = np.frombuffer(h.digest(), dtype=np.uint64)[0]
            if val < min_hash:
                min_hash = val
        signature[seed] = min_hash
    return signature


def impedance_matrix(text: str) -> np.ndarray:
    """
    Build the categorical co‑occurrence (impedance) matrix ζ.
    ζ₍i,j₎ counts how many sentences contain at least one token from
    category i and at least one token from category j.
    The matrix is symmetric with zeros on the diagonal.
    """
    ζ = np.zeros((_N_CAT, _N_CAT), dtype=np.float64)
    sentences = _sentence_split(text)

    for sent in sentences:
        present = set()
        for token in _tokenize(sent):
            present.update(_map_token_to_cats(token))
        for i in present:
            for j in present:
                if i != j:
                    ζ[i, j] += 1.0
    # Symmetrise
    ζ = (ζ + ζ.T) / 2.0
    return ζ


def hybrid_weight_matrix(
    text: str, alpha: float = 0.5, beta: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the hybrid edge‑weight matrix **W** and its Bayesian posterior
    distribution **P**.

    Steps:
    1. ζ ← impedance_matrix(text)
    2. c ← categorical_frequency_vector(text)
    3. h ← minhash_signature(text)
    4. s_c = 1 + α·mean(c)
    5. s_h = 1 + β·mean(h) / 2⁶⁴
    6. W = ζ * s_c * s_h               (element‑wise scaling)
    7. π = W / Σ_e W                    (prior over edges)
    8. P = (W * π) / Σ_e (W * π)        (Bayesian posterior, element‑wise)

    Returns
    -------
    W : np.ndarray
        Hybrid weight matrix.
    P : np.ndarray
        Posterior edge‑cost matrix, rows sum to 1.
    """
    ζ = impedance_matrix(text)
    c = categorical_frequency_vector(text)
    h = minhash_signature(text)

    s_c = 1.0 + alpha * float(c.mean())
    s_h = 1.0 + beta * (float(h.mean()) / float(2**64))

    W = ζ * s_c * s_h

    # Prior over edges (flattened then reshaped)
    total_w = W.sum()
    if total_w == 0:
        prior = np.full_like(W, 1.0 / W.size)
    else:
        prior = W / total_w

    # Bayesian update (element‑wise multiplication then normalise)
    unnorm_posterior = W * prior
    sum_post = unnorm_posterior.sum()
    if sum_post == 0:
        posterior = np.full_like(W, 1.0 / W.size)
    else:
        posterior = unnorm_posterior / sum_post

    return W, posterior


def update_state_matrix(
    state: np.ndarray,
    text: str,
    damping: float = 0.1,
) -> np.ndarray:
    """
    Evolve a dynamic state matrix using the hybrid weight matrix as a
    driving term.

    new_state = (1 - damping) * state + damping * W

    Parameters
    ----------
    state : np.ndarray
        Current square state matrix (must be N_CAT × N_CAT).
    text : str
        New textual evidence.
    damping : float
        Mixing factor between 0 and 1.

    Returns
    -------
    np.ndarray
        Updated state matrix.
    """
    if state.shape != (_N_CAT, _N_CAT):
        raise ValueError("state matrix must be of shape (N_CAT, N_CAT)")

    W, _ = hybrid_weight_matrix(text)
    new_state = (1.0 - damping) * state + damping * W
    return new_state


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "I think that you and I can solve this problem, but it might not be easy. "
        "However, the truth is that the solution is simple if we work together."
    )

    # 1. categorical vector
    c_vec = categorical_frequency_vector(sample)
    print("Categorical frequency vector (c):")
    for cat, val in zip(_CAT_LIST, c_vec):
        print(f"  {cat:12s}: {val:.4f}")

    # 2. MinHash signature statistics
    h_vec = minhash_signature(sample, num_perm=32)
    print("\nMinHash signature (first 8 values):", h_vec[:8])
    print("Mean of MinHash signature:", float(h_vec.mean()))

    # 3. Hybrid weight and posterior matrices
    W, P = hybrid_weight_matrix(sample, alpha=0.8, beta=0.6)
    print("\nHybrid weight matrix (W) – sample slice:")
    print(W[:5, :5])
    print("\nPosterior matrix (P) – rows sum to 1?")
    print("Row sums:", P.sum(axis=1)[:5])

    # 4. State matrix evolution
    init_state = np.zeros((_N_CAT, _N_CAT), dtype=np.float64)
    updated_state = update_state_matrix(init_state, sample, damping=0.3)
    print("\nUpdated state matrix (first 5×5 block):")
    print(updated_state[:5, :5])

    print("\nSmoke test completed without errors.")