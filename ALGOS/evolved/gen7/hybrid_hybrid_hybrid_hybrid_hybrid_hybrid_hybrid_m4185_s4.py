# DARWIN HAMMER — match 4185, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rectif_hybrid_hybrid_hybrid_m1689_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_gini_c_hybrid_hybrid_pherom_m1505_s0.py (gen6)
# born: 2026-05-29T23:54:07Z

"""Hybrid Textual‑Regret‑Gini‑Pheromone Engine
Parents:
- hybrid_hybrid_hybrid_rectif_hybrid_hybrid_hybrid_m1689_s2.py (LSM vectors, regret‑epistemic weighting)
- hybrid_hybrid_hybrid_gini_c_hybrid_hybrid_pherom_m1505_s0.py (Gini‑guided tropical matrix multiplication, pheromone hash & drag integration)

Mathematical Bridge
------------------
1. A raw text is transformed into an LSM (least‑squares‑style) frequency vector **v**.
2. The Gini coefficient *G(v)* quantifies the inequality of term frequencies.
3. *G(v)* scales a tropical (max‑plus) matrix multiplication  
   **y = (G·A) ⊗ v**, where “⊗” denotes max‑plus multiplication and *A* is a learned
   base matrix. This yields a morphology‑aware signal **y**.
4. The same text‑derived vector supplies epistemic certainty flags; a simple
   variance‑based measure **ε** drives a regret‑weighting of a prior weight vector **w₀**:
   **w = w₀ ⊙ (1 + ε)** (⊙ element‑wise product).
5. A pheromone force series **f(t)** is hashed (perceptual hash) to obtain a
   64‑bit identifier. The Hamming weight of this identifier modulates the quadratic
   drag coefficient *c* in a discrete kinematic integrator, producing a velocity
   profile **v(t)**.
6. The final posterior weight combines the regret‑weighted prior **w** with the
   tropical output **y** (element‑wise multiplication) and normalises the result.

The following implementation realises this pipeline in a self‑contained,
executable Python module."""
import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A utilities (text → LSM vector, epistemic flags, regret weighting)
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


def words(text: str) -> List[str]:
    """Split text into lower‑case alphabetic tokens."""
    return [word for word in (text or "").lower().split() if word.isalpha()]


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Least‑Squares‑style (LSM) term‑frequency vector.
    Returns a dictionary mapping each token to its normalised count.
    """
    ws = words(text)
    total = max(1, len(ws))
    freq = Counter(ws)
    return {token: count / total for token, count in freq.items()}


def epistemic_flags(vec: Dict[str, float]) -> float:
    """
    Simple epistemic certainty proxy: inverse of the coefficient of variation.
    Higher variance → lower certainty → higher regret weight.
    """
    values = list(vec.values())
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    cv = math.sqrt(variance) / mean
    # Convert to a regret factor in [0, 1]
    return min(1.0, cv)


def regret_weighted(prior: np.ndarray, regret_factor: float) -> np.ndarray:
    """
    Apply regret weighting: w_i = prior_i * (1 + regret_factor).
    The regret factor is the same for all dimensions for simplicity.
    """
    return prior * (1.0 + regret_factor)


# ----------------------------------------------------------------------
# Parent‑B utilities (Gini, tropical max‑plus, pheromone hashing & drag)
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Compute Gini coefficient for a list of non‑negative values."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def tropical_max_plus_mul(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Max‑plus (tropical) multiplication: (A ⊗ x)_i = max_j (A_ij + x_j)
    """
    result = np.full(matrix.shape[0], -np.inf)
    for i in range(matrix.shape[0]):
        result[i] = np.max(matrix[i, :] + vector)
    return result


def gini_guided_tropical_mul(
    base_matrix: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    """
    Scale the base matrix by the Gini coefficient of the input vector
    before performing tropical multiplication.
    """
    g = gini_coefficient(vector)
    scaled_matrix = base_matrix * (1.0 + g)  # modest scaling
    return tropical_max_plus_mul(scaled_matrix, vector)


def compute_phash(values: List[float]) -> int:
    """Compute a 64‑bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_weight(x: int) -> int:
    """Count the number of set bits in an integer."""
    return bin(x).count("1")


def integrate_pheromone_drag(
    forces: Sequence[float],
    mass: float = 1.0,
    base_drag: float = 0.1,
    dt: float = 0.01,
) -> List[float]:
    """
    Discrete integration of 1‑D motion under a force series with quadratic drag.
    Drag coefficient is modulated by the Hamming weight of the pheromone hash.
    Returns the velocity trajectory.
    """
    # Modulate drag using a hash derived from the force series
    phash = compute_phash(list(forces))
    drag_mod = 1.0 + hamming_weight(phash) / 64.0
    c = base_drag * drag_mod

    v = 0.0
    velocities = []
    for f in forces:
        # dv/dt = (f - c * v^2) / m
        a = (f - c * v * v) / mass
        v += a * dt
        velocities.append(v)
    return velocities


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def text_to_numeric_vector(text: str, vocab: List[str]) -> np.ndarray:
    """
    Convert LSM dict to a fixed‑size numeric vector aligned with *vocab*.
    Missing terms receive a zero entry.
    """
    lsm = lsm_vector(text)
    return np.array([lsm.get(tok, 0.0) for tok in vocab], dtype=float)


def hybrid_posterior(
    text: str,
    force_series: Sequence[float],
    prior_weights: np.ndarray,
    base_matrix: np.ndarray,
    vocab: List[str],
) -> np.ndarray:
    """
    Full hybrid update:
    1. LSM vector → numeric *v*.
    2. Regret weighting of *prior_weights* using epistemic flags.
    3. Gini‑guided tropical multiplication producing *y*.
    4. Pheromone drag integration (velocity profile) → scalar *d*.
    5. Combine: posterior ∝ (regret_weighted_prior ⊙ y) * (1 + d_mean).
    """
    # Step 1 – textual embedding
    v = text_to_numeric_vector(text, vocab)

    # Step 2 – regret weighting
    regret = epistemic_flags(lsm_vector(text))
    w_regret = regret_weighted(prior_weights, regret)

    # Step 3 – morphology‑aware tropical signal
    y = gini_guided_tropical_mul(base_matrix, v)

    # Step 4 – pheromone‑drag derived scalar
    velocities = integrate_pheromone_drag(force_series)
    d_mean = np.mean(np.abs(velocities))  # magnitude as a confidence boost

    # Step 5 – fuse components
    combined = w_regret * y
    posterior_unnorm = combined * (1.0 + d_mean)

    # Normalise to a probability‑like distribution
    if posterior_unnorm.sum() == 0:
        return posterior_unnorm
    return posterior_unnorm / posterior_unnorm.sum()


def generate_random_base_matrix(dim: int, seed: int = 42) -> np.ndarray:
    """Utility to create a reproducible tropical base matrix."""
    rng = np.random.default_rng(seed)
    # Values are drawn from a modest range to keep tropical scores stable
    return rng.uniform(-1.0, 1.0, size=(dim, dim))


def demo_hybrid():
    """Run a small demonstration of the hybrid pipeline."""
    text = "The quick brown fox jumps over the lazy dog while the observer watches."
    force_series = [random.uniform(-1, 1) for _ in range(200)]

    # Vocabulary derived from the text (could be larger in practice)
    vocab = sorted(set(words(text)))
    dim = len(vocab) if len(vocab) > 0 else 1

    prior = np.ones(dim) / dim  # uniform prior
    base_mat = generate_random_base_matrix(dim)

    posterior = hybrid_posterior(
        text=text,
        force_series=force_series,
        prior_weights=prior,
        base_matrix=base_mat,
        vocab=vocab,
    )
    print("Vocabulary:", vocab)
    print("Posterior weights:", posterior)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid()