# DARWIN HAMMER — match 1012, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s5.py (gen3)
# born: 2026-05-29T23:32:25Z

import math
import random
import sys
from pathlib import Path
from typing import List, Iterable, Tuple

import numpy as np


def compute_phash(values: List[float]) -> int:
    """
    Compute a 64‑bit perceptual hash from a list of floats.

    The median of the values is used as a threshold; each of the first
    64 values contributes one bit (1 if the value is >= median, else 0).
    If fewer than 64 values are supplied the remaining bits are set to 0.
    """
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for i in range(64):
        v = values[i] if i < len(values) else 0.0
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Return the Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Bayesian marginal probability."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Bayesian posterior update."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def extract_features(text: str) -> np.ndarray:
    """Extract a fixed‑length feature vector from a piece of text."""
    import re

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

    counts = [len(re.findall(pattern, text, flags=re.I)) for _, pattern in FEATURE_REGEXES]
    return np.array(counts, dtype=int)


def _sigmoid(x: float) -> float:
    """Logistic sigmoid mapping ℝ → (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def hybrid_decision_hygiene_score(
    features: np.ndarray,
    w_pos: np.ndarray,
    w_neg: np.ndarray,
    evidence_hash: int,
) -> float:
    """
    Produce a hygiene score that blends Bayesian reasoning with a
    perceptual‑hash modulation.

    * A net weighted sum `s = (w_pos - w_neg)·features` is turned into a
      likelihood via a sigmoid.
    * The prior is fixed at 0.5 and the false‑positive rate at 0.1.
    * The marginal is computed, then the posterior is obtained.
    * Finally the posterior is scaled by a hash‑based factor that depends
      on the proportion of set bits in the hash (i.e. its Hamming weight).
    """
    if features.shape != (9,) or w_pos.shape != (9,) or w_neg.shape != (9,):
        raise ValueError("All vectors must have shape (9,)")

    net = (w_pos - w_neg).dot(features)
    likelihood = _sigmoid(net)                     # maps net to [0,1]
    marginal = bayes_marginal(prior=0.5, likelihood=likelihood, false_positive=0.1)
    posterior = bayes_update(prior=0.5, likelihood=likelihood, marginal=marginal)

    # Hash modulation: proportion of 1‑bits in the 64‑bit hash
    bit_density = evidence_hash.bit_count() / 64.0
    hash_factor = 1.0 + 0.2 * (bit_density - 0.5)   # modest scaling around 1.0

    return posterior * hash_factor


def integrate_strike(
    force_series: Iterable[float],
    dt: float,
    m_head: float,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
    v0: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Numerically integrate a 1‑D motion model with quadratic drag.

    Returns the final velocity, total displacement, and peak velocity.
    """
    if dt <= 0 or m_head <= 0 or drag_cd < 0 or fluid_density < 0 or area < 0:
        raise ValueError("invalid strike parameters")

    v = v0
    x = 0.0
    peak = v0

    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v += acc * dt
        x += v * dt
        peak = max(peak, v)

    return v, x, peak


def hybrid_hybrid_algorithm(
    features: np.ndarray,
    force_series: Iterable[float],
    dt: float,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Full hybrid pipeline:

    1. Compute a perceptual hash of the feature vector.
    2. Derive a drag‑coefficient modulation from the Hamming distance to a
       fixed target hash (normalized to the range [0.5, 1.5]).
    3. Run the physics integration with the modulated drag.
    4. Compute a deep hygiene score using the original feature vector,
       positive/negative weight vectors and the hash.
    5. Return the feature vector, final velocity, displacement, peak velocity,
       and hygiene score.
    """
    # 1. Hash of the evidence (feature) vector
    evidence_hash = compute_phash(features.tolist())

    # 2. Drag modulation based on Hamming distance to a target hash
    target_hash = 0x1234567890ABCDEF
    ham_dist = hamming_distance(evidence_hash, target_hash)
    # Normalise distance to [0, 1] (max possible distance for 64‑bit values is 64)
    norm_dist = ham_dist / 64.0
    # Map to a multiplicative factor roughly in [0.5, 1.5]
    drag_modulation = 0.5 + norm_dist

    # 3. Physics integration with modulated drag coefficient
    base_cd = 0.3
    v_final, displacement, v_peak = integrate_strike(
        force_series,
        dt,
        m_head=1.0,
        drag_cd=base_cd * drag_modulation,
        fluid_density=1000.0,
        area=0.001,
        v0=0.0,
    )

    # 4. Hygiene score using the original feature vector
    w_pos = np.array([0.5, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    w_neg = np.array([0.1, 0.1, 0.1, 0.1, 0.4, 0.5, 0.1, 0.1, 0.1])
    hygiene_score = hybrid_decision_hygiene_score(features, w_pos, w_neg, evidence_hash)

    return features, v_final, displacement, v_peak, hygiene_score


if __name__ == "__main__":
    # Simple smoke test
    sample_text = (
        "The evidence was verified and the plan includes a checklist. "
        "Performance is high but we must consider security and cost."
    )
    feature_vec = extract_features(sample_text)

    force_series = [1.0, 2.0, 3.0, 4.0, 5.0]
    dt = 0.01

    result = hybrid_hybrid_algorithm(feature_vec, force_series, dt)
    print("Feature vector :", result[0])
    print("Final velocity :", result[1])
    print("Displacement   :", result[2])
    print("Peak velocity  :", result[3])
    print("Hygiene score  :", result[4])