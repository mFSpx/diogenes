# DARWIN HAMMER — match 2007, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1012_s1.py (gen4)
# parent_b: hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_worksh_m162_s0.py (gen4)
# born: 2026-05-29T23:40:22Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Parents A and B

Parent A (hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1012_s1) contributes:
- Probabilistic reasoning via Bayesian marginalisation and update.
- Lightweight feature extraction from text (9‑dimensional count vector).
- A simple binary hash (phash) and Hamming distance utilities.

Parent B (hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_worksh_m162_s0) contributes:
- Hyperdimensional computing (HDC) primitives: random bipolar vectors,
  deterministic symbol vectors, and similarity via normalized dot product.
- A deterministic pseudo‑random generator seeded from arbitrary strings.

**Mathematical Bridge**

We treat the Bayesian posterior *p* as a scalar confidence and the HDC
similarity *s* ∈ [‑1, 1] as a high‑dimensional contextual affinity.
The fused decision metric is defined as

    score = p * (1 + s) / 2

where *s* is linearly mapped to [0, 1] and thus modulates the Bayesian
confidence by the contextual similarity of the feature‑derived HDC vector
to a reference vector.  The feature‑derived HDC vector itself is built by
super‑position of deterministic symbol vectors weighted by the extracted
feature counts, yielding a single bipolar vector that lives in the same
space as the random reference vector.

The module implements this fusion through three core functions:
`bayesian_hdc_posterior`, `generate_feature_hdc_vector`, and
`hybrid_decision_score`.  Additional utility functions are retained from
the parents for completeness.
"""

import math
import random
import sys
from pathlib import Path
import hashlib
import numpy as np
from typing import List, Tuple, Iterable

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Compute a 64‑bit binary hash based on the average of *values*."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit‑strings."""
    return (a ^ b).bit_count()

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)"""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)P(H) / P(E)"""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

def extract_features(text: str) -> np.ndarray:
    """Count occurrences of predefined regex patterns; returns a 9‑dim int array."""
    import re
    counts = []
    FEATURE_REGEXES = [
        ("evidence", r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b"),
        ("planning", r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b"),
        ("delay", r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|b)\b"),
        ("quality", r"\b(?:quality|high|low|grade|rating)\b"),
        ("security", r"\b(?:security|secure|vulnerability|exploit)\b"),
        ("performance", r"\b(?:performance|fast|slow|latency)\b"),
        ("compliance", r"\b(?:compliance|regulation|standard)\b"),
        ("cost", r"\b(?:cost|price|budget|expense)\b"),
        ("generic", r"\b\w{7,}\b"),
    ]
    for _, pattern in FEATURE_REGEXES:
        matches = re.findall(pattern, text, re.I)
        counts.append(len(matches))
    return np.array(counts, dtype=int)

# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: str | int | None = None) -> np.ndarray:
    """Generate a bipolar random vector (+1 / -1) of length *dim*."""
    rng = random.Random(seed)
    return np.array([1 if rng.getrandbits(1) else -1 for _ in range(dim)], dtype=int)

def symbol_vector(symbol: str, dim: int = 10000) -> np.ndarray:
    """Deterministic bipolar vector derived from a SHA‑256 hash of *symbol*."""
    seed = int.from_bytes(hashlib.sha256(symbol.encode('utf-8')).digest()[:8], 'big')
    return random_vector(dim, seed)

def _rng_from_text(text: str) -> random.Random:
    """Create a deterministic RNG seeded from the SHA‑256 of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)

def extract_full_features(text: str) -> dict:
    """Generate a deterministic pseudo‑random feature dictionary from *text*."""
    rnd = _rng_from_text(text)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "psyche_wrath_velocity",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index",
    ]
    # Produce a float in [0,1) for each key using the RNG.
    return {k: rnd.random() for k in keys}

def hdc_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Normalized dot‑product similarity in [-1, 1] for bipolar vectors."""
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape")
    return float(np.dot(v1, v2) / v1.size)

# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def generate_feature_hdc_vector(
    feature_counts: np.ndarray,
    dim: int = 1000,
    weight_scale: float = 1.0
) -> np.ndarray:
    """
    Build a bipolar HDC vector from the 9‑dimensional *feature_counts*.
    Each feature index i contributes a deterministic symbol vector
    `symbol_vector(f"f{i}")` weighted by the count (scaled by *weight_scale*).
    The super‑position is summed and then binarised to +/-1.
    """
    if feature_counts.shape != (9,):
        raise ValueError("feature_counts must be a length‑9 vector")
    accumulator = np.zeros(dim, dtype=float)
    for i, cnt in enumerate(feature_counts):
        if cnt == 0:
            continue
        sym_vec = symbol_vector(f"f{i}", dim).astype(float)
        accumulator += weight_scale * cnt * sym_vec
    # Binarise: positive → +1, zero/negative → -1
    return np.where(accumulator >= 0, 1, -1).astype(int)

def bayesian_hdc_posterior(
    prior: float,
    likelihood: float,
    false_positive: float,
    hdc_sim: float
) -> float:
    """
    Compute a Bayesian posterior and modulate it with the HDC similarity.
    The similarity *hdc_sim* ∈ [-1, 1] is first mapped to [0, 1] and then
    used as a multiplicative factor.
    """
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    # Map similarity from [-1,1] → [0,1]
    sim_factor = (hdc_sim + 1.0) / 2.0
    return posterior * sim_factor

def hybrid_decision_score(
    text: str,
    prior: float = 0.5,
    likelihood: float = 0.9,
    false_positive: float = 0.1,
    dim: int = 1000,
    weight_scale: float = 0.1
) -> float:
    """
    End‑to‑end hybrid decision metric for *text*.
    1. Extract 9‑dimensional count features.
    2. Build a feature‑derived HDC vector.
    3. Compare it to a deterministic reference vector derived from the text.
    4. Combine the resulting similarity with a Bayesian posterior.
    Returns a score in [0, 1].
    """
    # Step 1: feature extraction (Parent A)
    counts = extract_features(text)

    # Step 2: HDC vector generation (Parent B)
    feature_vec = generate_feature_hdc_vector(counts, dim=dim, weight_scale=weight_scale)

    # Step 3: Reference vector – deterministic based on the whole text
    reference_vec = symbol_vector(text, dim=dim)

    # Step 4: Similarity
    sim = hdc_similarity(feature_vec, reference_vec)

    # Step 5: Bayesian + similarity fusion
    return bayesian_hdc_posterior(prior, likelihood, false_positive, sim)

# ----------------------------------------------------------------------
# Additional Helper Demonstrating Cross‑Interaction
# ----------------------------------------------------------------------
def phash_hdc_fingerprint(text: str, dim: int = 1024) -> Tuple[int, np.ndarray]:
    """
    Produce a tuple (phash, hdc_vector) where:
    - *phash* is a 64‑bit hash of the raw feature counts.
    - *hdc_vector* is the feature‑derived HDC vector (bipolar).
    This showcases how the binary hash from Parent A can be paired with
    the high‑dimensional representation from Parent B.
    """
    counts = extract_features(text)
    ph = compute_phash(counts.tolist())
    hdc_vec = generate_feature_hdc_vector(counts, dim=dim)
    return ph, hdc_vec

# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The evidence confirms the security breach. "
        "We need a plan and a timeline. "
        "Performance metrics show latency spikes. "
        "Budget constraints may cause delay."
    )
    score = hybrid_decision_score(sample_text)
    print(f"Hybrid decision score: {score:.6f}")

    ph, vec = phash_hdc_fingerprint(sample_text)
    print(f"Phash: {ph:#018x}")
    print(f"HDC vector norm (should be sqrt(dim)): {np.linalg.norm(vec):.2f}")
    # Verify that the similarity between the vector and its own reference is 1.0
    ref = symbol_vector(sample_text, dim=vec.size)
    sim_self = hdc_similarity(vec, ref)
    print(f"Self‑similarity (vec vs reference): {sim_self:.6f}")