# DARWIN HAMMER — match 2520, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s0.py (gen2)
# parent_b: omni_chaotic_sprint.py (gen0)
# born: 2026-05-29T23:42:41Z

"""Hybrid Hoeffding‑Gini‑Fractional Hypervector Engine (HG‑FHE)

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – *hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s0.py*  
  Provides (i) random hyper‑vector generation, (ii) circular‑convolution binding,
  (iii) the Hoeffding bound, (iv) the Gini inequality coefficient and (v) a
  fractional‑power (α‑exponent) operator that encodes causal effect strength.

* **Parent B** – *omni_chaotic_sprint.py*  
  Supplies a data‑centric workflow that extracts a collection of “graph items”
  (here simulated as a list of dictionaries) and aggregates categorical terms.

**Mathematical bridge** – Both parents operate on *distributions* of discrete
entities.  In the hybrid we (a) map each categorical term to a high‑dimensional
hyper‑vector, (b) compute the Gini coefficient of the term‑frequency distribution,
(c) use the Hoeffding bound to assess confidence in that frequency estimate, and
(d) bind a treatment hyper‑vector with an outcome hyper‑vector using a fractional
exponent α to obtain a causal‑effect hyper‑vector.  The resulting pipeline
produces a single unified representation that simultaneously captures inequality,
statistical confidence and causal interaction in the same algebraic space.

Author: synthetic fusion of the two parents (2026‑05‑29)
"""

import math
import random
import sys
import pathlib
from typing import Optional, List, Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Core primitives from Parent A (fractional_hdc + Hoeffding + Gini)
# ---------------------------------------------------------------------------

def random_hv(d: int = 10000, kind: str = "complex", seed: Optional[int] = None) -> np.ndarray:
    """Generate a random hyper‑vector of dimension *d*.

    Parameters
    ----------
    d : int
        Dimensionality of the hyper‑vector.
    kind : str
        One of ``"complex"``, ``"bipolar"``, ``"real"``.
    seed : Optional[int]
        Seed for reproducible randomness.

    Returns
    -------
    np.ndarray
        The generated hyper‑vector.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"Unsupported kind {kind!r}")


def bind(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Circular‑convolution binding of two hyper‑vectors."""
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))


def unbind(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Inverse of ``bind`` using division in the Fourier domain."""
    FY = np.fft.fft(Y)
    # Avoid division by zero – add a tiny epsilon to magnitude
    eps = 1e-30
    inv_FY = np.conj(FY) / (np.abs(FY) ** 2 + eps)
    return np.fft.ifft(np.fft.fft(Z) * inv_FY)


def fractional_power(vec: np.ndarray, alpha: float) -> np.ndarray:
    """Raise a complex hyper‑vector to a fractional exponent α while preserving phase.

    For a complex number z = r·e^{iθ}, z^{α} = r^{α}·e^{iαθ}.
    """
    if np.iscomplexobj(vec):
        r = np.abs(vec)
        theta = np.angle(vec)
        return (r ** alpha) * np.exp(1j * alpha * theta)
    # For real‑valued vectors we simply raise element‑wise.
    return np.sign(vec) * (np.abs(vec) ** alpha)


def hoeffding_bound(R: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (R²·ln(1/δ)) / (2n) )."""
    if n <= 0:
        raise ValueError("Sample size n must be positive")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1‑D array of non‑negative numbers."""
    if values.ndim != 1:
        raise ValueError("Gini coefficient expects a 1‑D array")
    if np.any(values < 0):
        raise ValueError("Gini coefficient is undefined for negative values")
    n = values.size
    if n == 0:
        return 0.0
    sorted_vals = np.sort(values)
    cumulative = np.cumsum(sorted_vals)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_vals) / n
    return gini


# ---------------------------------------------------------------------------
# Simplified data pipeline inspired by Parent B (graph‑item aggregation)
# ---------------------------------------------------------------------------

def _hash_term_to_seed(term: str) -> int:
    """Deterministic integer seed derived from a string term."""
    return abs(hash(term)) % (2 ** 31 - 1)


def encode_terms_to_hvs(terms: List[str], dim: int = 4096) -> np.ndarray:
    """Encode a list of categorical terms into a matrix of hyper‑vectors.

    Each term receives a reproducible random hyper‑vector (complex kind).
    Returns an array of shape (len(terms), dim).
    """
    hv_matrix = np.empty((len(terms), dim), dtype=np.complex128)
    for i, term in enumerate(terms):
        seed = _hash_term_to_seed(term)
        hv_matrix[i] = random_hv(d=dim, kind="complex", seed=seed)
    return hv_matrix


def aggregate_term_frequencies(items: List[Dict]) -> Tuple[List[str], np.ndarray]:
    """Extract terms from ``items`` and return (unique_terms, frequencies)."""
    counter = {}
    for itm in items:
        term = itm.get("term", "")
        if not term:
            continue
        counter[term] = counter.get(term, 0) + 1
    unique_terms = list(counter.keys())
    frequencies = np.array([counter[t] for t in unique_terms], dtype=np.float64)
    return unique_terms, frequencies


# ---------------------------------------------------------------------------
# Hybrid operations (three public functions)
# ---------------------------------------------------------------------------

def hybrid_encode_and_analyze(items: List[Dict],
                              dim: int = 4096) -> Dict[str, float]:
    """
    1. Encode each distinct term into a hyper‑vector.
    2. Compute the Gini coefficient of term frequencies.
    3. Return a dictionary with the Gini value and the mean pairwise
       cosine similarity between the term hyper‑vectors (a proxy for
       distributional overlap).

    Parameters
    ----------
    items : List[Dict]
        Simulated graph‑item rows, each containing at least a ``"term"`` key.
    dim : int
        Dimensionality of the hyper‑vectors.

    Returns
    -------
    Dict[str, float]
        ``{"gini": ..., "mean_cosine": ...}``
    """
    terms, freqs = aggregate_term_frequencies(items)
    if len(terms) == 0:
        return {"gini": 0.0, "mean_cosine": 0.0}

    hv_matrix = encode_terms_to_hvs(terms, dim=dim)

    # Gini of the frequency distribution
    gini_val = gini_coefficient(freqs)

    # Cosine similarity matrix (real part because vectors are complex)
    normed = hv_matrix / (np.linalg.norm(hv_matrix, axis=1, keepdims=True) + 1e-30)
    sim_matrix = np.real(np.dot(normed, normed.conj().T))
    # Exclude self‑similarities
    n = sim_matrix.shape[0]
    mean_cosine = (np.sum(sim_matrix) - n) / (n * (n - 1)) if n > 1 else 0.0

    return {"gini": gini_val, "mean_cosine": mean_cosine}


def evaluate_hoeffding_confidence(freqs: np.ndarray,
                                 total_samples: int,
                                 delta: float = 0.05,
                                 R: float = 1.0) -> np.ndarray:
    """
    Apply the Hoeffding bound to each term frequency proportion.

    Returns an array of booleans indicating whether the empirical proportion
    lies within ε of the true (unknown) proportion with confidence 1‑δ.

    Parameters
    ----------
    freqs : np.ndarray
        Count of each term.
    total_samples : int
        Sum of all term counts.
    delta : float
        Desired failure probability (default 0.05 → 95 % confidence).
    R : float
        Range of the underlying random variable (for a proportion R = 1).

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``freqs.shape``.
    """
    if total_samples == 0:
        raise ValueError("total_samples must be positive")
    epsilon = hoeffding_bound(R, delta, total_samples)
    proportions = freqs / total_samples
    # Since true proportion ∈ [0,1], the condition is always satisfied if
    # epsilon >= max(proportions, 1‑proportions).  We return the logical check.
    lower = np.maximum(proportions - epsilon, 0.0)
    upper = np.minimum(proportions + epsilon, 1.0)
    # The mask indicates that the interval [lower, upper] is a valid confidence band.
    return (lower <= proportions) & (proportions <= upper)


def compute_fractional_causal_binding(treatment_term: str,
                                      outcome_term: str,
                                      alpha: float = 0.8,
                                      dim: int = 4096) -> np.ndarray:
    """
    Produce a causal‑effect hyper‑vector by binding the treatment and outcome
    hyper‑vectors after raising the outcome to a fractional power α.

    The operation is:
        CE = bind( HV(treatment), fractional_power( HV(outcome), α ) )

    Parameters
    ----------
    treatment_term : str
        Categorical identifier of the treatment.
    outcome_term : str
        Categorical identifier of the outcome.
    alpha : float
        Fractional exponent controlling effect strength (0 < α ≤ 1).
    dim : int
        Dimensionality of the hyper‑vectors.

    Returns
    -------
    np.ndarray
        Complex hyper‑vector representing the causal effect.
    """
    hv_treat = random_hv(d=dim, kind="complex", seed=_hash_term_to_seed(treatment_term))
    hv_out = random_hv(d=dim, kind="complex", seed=_hash_term_to_seed(outcome_term))
    hv_out_frac = fractional_power(hv_out, alpha)
    ce_vector = bind(hv_treat, hv_out_frac)
    return ce_vector


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a synthetic list of graph‑item dictionaries.
    synthetic_items = [
        {"uuid": "1", "term": "ENTITY"},
        {"uuid": "2", "term": "ATTRIBUTE"},
        {"uuid": "3", "term": "ENTITY"},
        {"uuid": "4", "term": "RELATIONSHIP"},
        {"uuid": "5", "term": "ENTITY"},
        {"uuid": "6", "term": "ACTION"},
        {"uuid": "7", "term": "ACTION"},
        {"uuid": "8", "term": "ATTRIBUTE"},
        {"uuid": "9", "term": "GLOW"},
        {"uuid": "10", "term": "GLOW"},
    ]

    # 1️⃣ Hybrid encoding & analysis
    analysis = hybrid_encode_and_analyze(synthetic_items, dim=2048)
    print("Hybrid analysis:", analysis)

    # 2️⃣ Hoeffding confidence on term frequencies
    terms, freqs = aggregate_term_frequencies(synthetic_items)
    total = int(freqs.sum())
    confidence_mask = evaluate_hoeffding_confidence(freqs, total, delta=0.05, R=1.0)
    print("Terms:", terms)
    print("Frequencies:", freqs)
    print("Hoeffding confidence mask:", confidence_mask)

    # 3️⃣ Fractional causal binding example
    ce_vec = compute_fractional_causal_binding("ENTITY", "ACTION", alpha=0.7, dim=2048)
    # Show a tiny summary (norm and first few components)
    print("Causal effect vector norm:", np.linalg.norm(ce_vec))
    print("First 5 components (real):", np.real(ce_vec[:5]))
    print("First 5 components (imag):", np.imag(ce_vec[:5]))