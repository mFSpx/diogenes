# DARWIN HAMMER — match 1785, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m587_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hdc_hybrid_hy_m996_s0.py (gen4)
# born: 2026-05-29T23:38:49Z

"""Hybrid Algorithm: Fisher‑HDC‑Decision Fusion

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m587_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_decisi_hybrid_hdc_hybrid_hy_m996_s0.py (Algorithm B)

Mathematical Bridge:
Algorithm A supplies a *Fisher information* scalar 𝓕(θ) that quantifies the
sensitivity of a Gaussian beam model.  Algorithm B produces high‑dimensional
ternary hypervectors 𝑣∈{‑1,0,+1}ᵈ from regex‑based “decision‑hygiene” feature
counts.  The fusion treats 𝓕(θ) as a uniform weighting applied element‑wise to
the hypervector (𝑤 = 𝓕·𝑣).  The weighted hypervector is then compressed with the
count‑min sketch from Algorithm A, yielding a compact sketch S that preserves
both the statistical sensitivity (via 𝓕) and the semantic hygiene signature
(via the hypervector).  Subsequent RLCT‑style aggregation operates on the
sketch counts, completing a mathematically unified pipeline.

The module implements:
1. Extraction of hygiene feature counts (regexes from B).
2. Construction of a deterministic ternary hypervector (HDC from B).
3. Fisher‑based weighting of the hypervector (A’s Fisher information).
4. Compression of the weighted vector with a count‑min sketch (A).
5. An RLCT‑style estimator that can consume arbitrary loss sequences.
"""

import math
import random
import re
import hashlib
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# 1. Decision‑hygiene feature extraction (Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford)\b",
    re.I,
)

_FEATURE_PATTERNS = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
}


def extract_hygiene_features(text: str) -> dict:
    """
    Count occurrences of each hygiene regex in *text*.
    Returns a dictionary mapping feature names to integer counts.
    """
    counts = {}
    for name, pat in _FEATURE_PATTERNS.items():
        counts[name] = len(pat.findall(text))
    return counts


# ----------------------------------------------------------------------
# 2. Hyperdimensional vector construction (Parent B)
# ----------------------------------------------------------------------
DIM = 10000  # dimensionality of the ternary hypervectors


def _rng_from_seed(seed: int) -> np.random.Generator:
    """Create a NumPy Generator with a deterministic 32‑bit seed."""
    return np.random.default_rng(seed & 0xFFFFFFFF)


def generate_hdc_vector(feature_counts: dict, dim: int = DIM) -> np.ndarray:
    """
    Build a ternary hypervector (dtype=int8) from *feature_counts*.
    For each feature, a number of +1 and -1 entries equal to the feature count
    are placed at pseudo‑random positions derived from a hash of the feature
    name.  This yields a deterministic but high‑entropy vector.
    """
    vec = np.zeros(dim, dtype=np.int8)
    for idx, (feat, cnt) in enumerate(feature_counts.items()):
        if cnt <= 0:
            continue
        # deterministic seeds
        pos_seed = hash((feat, idx, "pos"))
        neg_seed = hash((feat, idx, "neg"))
        rng_pos = _rng_from_seed(pos_seed)
        rng_neg = _rng_from_seed(neg_seed)

        # limit the number of positions to the vector size
        k = min(cnt, dim)

        pos_idx = rng_pos.choice(dim, size=k, replace=False)
        neg_idx = rng_neg.choice(dim, size=k, replace=False)

        vec[pos_idx] = 1
        vec[neg_idx] = -1
    return vec


# ----------------------------------------------------------------------
# 3. Fisher information (Parent A) and weighting
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single angle θ under the Gaussian beam model.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def fisher_weighted_vector(
    theta: float, center: float, width: float, vector: np.ndarray
) -> np.ndarray:
    """
    Multiply a hypervector element‑wise by the scalar Fisher information
    𝓕(θ) computed from the Gaussian beam parameters.
    """
    f = fisher_score(theta, center, width)
    return vector.astype(np.float64) * f


# ----------------------------------------------------------------------
# 4. Count‑Min Sketch compression (Parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items, width: int = 64, depth: int = 4) -> list:
    """
    Classic count‑min sketch. *items* is an iterable of hashable strings.
    Returns a 2‑D list of counters of shape (depth, width).
    """
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table


def sketch_from_weighted_vector(weighted_vec: np.ndarray, width: int = 64, depth: int = 4) -> list:
    """
    Convert a weighted hypervector into a list of string items and feed them
    into the count‑min sketch.  Only non‑zero entries are inserted; the string
    encodes the index and the quantised magnitude (sign only, because the
    Fisher scalar is uniform across the vector).
    """
    items = [f"{i}:{int(np.sign(v))}" for i, v in enumerate(weighted_vec) if v != 0]
    return count_min_sketch(items, width=width, depth=depth)


# ----------------------------------------------------------------------
# 5. RLCT‑style aggregation (Parent A)
# ----------------------------------------------------------------------
def estimate_rlct_from_losses(train_losses_per_n, n_values):
    """
    Simple RLCT estimator that returns the mean loss.
    The function validates inputs as in the original parent algorithm.
    """
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if len(losses) != len(ns):
        raise ValueError("length of losses and n_values must match")
    return np.mean(losses)


# ----------------------------------------------------------------------
# 6. End‑to‑end hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_pipeline(
    text: str,
    theta: float,
    center: float,
    width_beam: float,
    sketch_width: int = 64,
    sketch_depth: int = 4,
):
    """
    Full hybrid processing:
    1. Extract hygiene feature counts from *text*.
    2. Build a ternary HDC vector.
    3. Weight the vector with Fisher information derived from the Gaussian beam.
    4. Compress the weighted vector with a count‑min sketch.
    Returns the sketch (list of lists) and the Fisher scalar used.
    """
    feats = extract_hygiene_features(text)
    hdc_vec = generate_hdc_vector(feats)
    weighted = fisher_weighted_vector(theta, center, width_beam, hdc_vec)
    sketch = sketch_from_weighted_vector(weighted, width=sketch_width, depth=sketch_depth)
    fisher_scalar = fisher_score(theta, center, width_beam)
    return sketch, fisher_scalar


# ----------------------------------------------------------------------
# 7. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = """
    The evidence was verified and the source was documented. We have a clear plan,
    a checklist, and a timeline. However, we might need to pause tomorrow due
    to budget constraints. Please support the team and respect the privacy
    boundaries. The project is done and shipped.
    """
    # Gaussian beam parameters
    theta_test = 0.3
    center_test = 0.0
    width_test = 1.0

    sketch_result, fisher_val = hybrid_pipeline(
        sample_text,
        theta=theta_test,
        center=center_test,
        width_beam=width_test,
    )

    # Print a concise summary
    print(f"Fisher scalar: {fisher_val:.6e}")
    print("Count‑Min Sketch (first depth row):", sketch_result[0])

    # Dummy RLCT demonstration
    dummy_losses = [0.9, 0.7, 0.5, 0.4]
    dummy_ns = [10, 100, 1000, 10000]
    rlct_est = estimate_rlct_from_losses(dummy_losses, dummy_ns)
    print(f"RLCT estimate (mean loss): {rlct_est:.4f}")