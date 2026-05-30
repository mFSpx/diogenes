# DARWIN HAMMER — match 1571, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m565_s0.py (gen5)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_ternary_lens__m71_s1.py (gen3)
# born: 2026-05-29T23:37:30Z

"""Hybrid Algorithm Fusion of Parent A and Parent B

Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m565_s0) provides
- Count‑Min Sketch (CMS) with optional propensity scaling
- HyperLogLog‑style cardinality estimator
- Hoeffding confidence bound

Parent B (hybrid_hybrid_fisher_locali_hybrid_ternary_lens__m71_s1) provides
- Gaussian beam intensity profile and its Fisher information (confidence)
- Structural Similarity Index (SSIM) for one‑dimensional signals
- A regex‑driven ternary feature extractor

**Mathematical Bridge**

1. *Propensity ↔ Gaussian Beam*: The Gaussian intensity `g(θ)` is used as a
   continuous propensity factor that modulates the increment added to each CMS
   cell. Thus the CMS update becomes a weighted count where the weight is the
   beam intensity evaluated at a pseudo‑angle derived from the item hash.

2. *Confidence Bound ↔ Fisher Information*: The Fisher information of the
   Gaussian beam at the same angle supplies a data‑driven confidence bound
   `β`. This replaces the static `confidence_bound` used in the original
   bandit‑style hybrid and feeds the Hoeffding bound `ε = sqrt( r²·log(1/δ) / (2·n) )`
   with `r = β`.

3. *Signal‑to‑Noise ↔ SSIM*: The signal‑to‑noise gap `γ = β / |U|` (where `|U|`
   is the estimated cardinality) is compared against a reference signal using
   SSIM, yielding a similarity score that can be interpreted as a normalized
   gap metric.

The following module implements three core hybrid functions that embody this
fusion, together with lightweight ternary regex features that can be merged
into the sketch representation if desired.
"""

import hashlib
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks (adapted)
# ----------------------------------------------------------------------
def count_min_sketch(items, width=64, depth=4):
    """Standard Count‑Min Sketch with unit increments."""
    table = [[0.0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1.0
    return table


def propensity_modulated_count_min_sketch(items, width=64, depth=4,
                                         propensity_func=lambda _: 1.0):
    """CMS where each update is scaled by a user‑provided propensity function."""
    table = [[0.0] * width for _ in range(depth)]
    for item in items:
        # map item to a pseudo‑angle in [0, 2π) via its hash
        angle = (int(hashlib.sha256(f"angle:{item}".encode()).hexdigest(), 16)
                 % 10_000) / 10_000 * 2 * math.pi
        prop = propensity_func(angle)
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += prop
    return table


def hyperloglog_cardinality(items):
    """Placeholder for HLL – here we simply return the exact distinct count."""
    return len(set(items))


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt(r²·log(1/δ) / (2·n))."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("Invalid arguments for Hoeffding bound")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def t_add(x, y):
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication (addition)."""
    return np.add(x, y)


# ----------------------------------------------------------------------
# Parent‑B building blocks (adapted)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """One‑dimensional Structural Similarity Index."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    cov = np.cov(x, y, ddof=0)[0, 1]
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Ternary regex feature extractor (Parent‑B component)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12

_REGEX_CATALOG = [
    re.compile(r"\b(evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I),  # 0
    re.compile(r"\b(plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I),  # 1
    re.compile(r"\b(pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|first|after|review)\b", re.I),  # 2
    re.compile(r"\b(ask|call|text|friend|colleague|team|support|help|consult)\b", re.I),  # 3
    re.compile(r"\b(error|fail|exception|crash|bug|incorrect|wrong|mismatch|invalid)\b", re.I),  # 4
    re.compile(r"\b(success|pass|ok|complete|finished|done|resolved)\b", re.I),  # 5
    re.compile(r"\b(temperature|temp|heat|cold|cool|warm)\b", re.I),  # 6
    re.compile(r"\b(speed|fast|slow|latency|throughput|bandwidth)\b", re.I),  # 7
    re.compile(r"\b(memory|ram|storage|disk|space|capacity)\b", re.I),  # 8
    re.compile(r"\b(user|client|server|node|process|thread)\b", re.I),  # 9
    re.compile(r"\b(encrypt|decryp|hash|sign|verify|checksum)\b", re.I),  #10
    re.compile(r"\b(update|upgrade|patch|release|version)\b", re.I),  #11
]


def ternary_feature_vector(texts):
    """Return a (len(texts), TERNARY_DIMS) array with counts of regex matches."""
    vectors = np.zeros((len(texts), TERNARY_DIMS), dtype=np.float32)
    for i, txt in enumerate(texts):
        for dim, regex in enumerate(_REGEX_CATALOG):
            vectors[i, dim] = len(regex.findall(txt))
    return vectors


# ----------------------------------------------------------------------
# Hybrid Functions (the novel fused core)
# ----------------------------------------------------------------------
def hybrid_weighted_sketch(items, width=64, depth=4,
                           beam_center=math.pi, beam_width=1.0):
    """
    Build a Count‑Min Sketch where each item's contribution is weighted by the
    Gaussian beam intensity evaluated at a hash‑derived angle.
    The same angle is also used to compute a Fisher‑information‑based confidence
    bound for later statistical guarantees.
    Returns:
        sketch_table   – weighted CMS (list of lists)
        confidence    – Fisher information at the average angle of the stream
    """
    # Propensity function derived from Gaussian beam
    propensity_func = lambda theta: gaussian_beam(theta, beam_center, beam_width)

    sketch = propensity_modulated_count_min_sketch(items, width, depth,
                                                   propensity_func)

    # Compute average angle for the confidence estimate
    angles = [(int(hashlib.sha256(f"angle:{it}".encode()).hexdigest(), 16) % 10_000) / 10_000 * 2 * math.pi
              for it in items]
    avg_theta = sum(angles) / len(angles) if angles else beam_center
    confidence = fisher_score(avg_theta, beam_center, beam_width)

    return sketch, confidence


def hybrid_signal_noise_ssim(sketch_table, items,
                             reference_signal,
                             delta=0.05):
    """
    Combine the signal‑to‑noise gap derived from the sketch with SSIM.
    Steps:
        1. Estimate cardinality via hyperloglog_cardinality.
        2. Use the Fisher‑derived confidence (from the last call) as r.
        3. Compute Hoeffding bound ε.
        4. Gap γ = r / |U|.
        5. Flatten the sketch counts and compute SSIM against a reference.
    Returns a dictionary with all intermediate metrics.
    """
    # Flatten sketch to a 1‑D float array
    flat_counts = np.array([c for row in sketch_table for c in row], dtype=np.float64)

    # Cardinality estimate
    U = hyperloglog_cardinality(items)
    if U == 0:
        raise ValueError("Empty item set leads to zero cardinality")

    # Use the mean of the flattened counts as a surrogate for r (Fisher info)
    r = np.mean(flat_counts)

    # Hoeffding bound
    n = len(items)
    epsilon = hoeffding_bound(r, delta, n)

    # Signal‑to‑noise gap
    gamma = r / U

    # Ensure reference signal matches shape
    if reference_signal.shape != flat_counts.shape:
        # Broadcast or truncate to match
        min_len = min(reference_signal.size, flat_counts.size)
        ref = reference_signal[:min_len]
        cur = flat_counts[:min_len]
    else:
        ref = reference_signal
        cur = flat_counts

    similarity = ssim(cur, ref)

    return {
        "cardinality": U,
        "mean_weight": r,
        "hoeffding_epsilon": epsilon,
        "signal_to_noise_gap": gamma,
        "ssim_similarity": similarity
    }


def hybrid_ternary_enhanced_sketch(items, texts,
                                   width=64, depth=4,
                                   beam_center=math.pi, beam_width=1.0):
    """
    Extend the weighted sketch with ternary regex features.
    The ternary vector for each text is added (via tropical addition) to the
    corresponding CMS row, creating a richer representation that captures
    semantic cues alongside frequency information.
    Returns the enhanced sketch (numpy array of shape (depth, width)).
    """
    # Base weighted sketch and confidence
    sketch, confidence = hybrid_weighted_sketch(items, width, depth,
                                                beam_center, beam_width)

    # Convert sketch to numpy for tropical ops
    sketch_np = np.array(sketch, dtype=np.float64)

    # Extract ternary features and reduce to (depth, width) by simple aggregation
    ternary_vec = ternary_feature_vector(texts)  # shape (len(texts), TERNARY_DIMS)

    # Aggregate ternary counts per depth bucket (simple modulo mapping)
    agg = np.zeros((depth, width), dtype=np.float64)
    for i, vec in enumerate(ternary_vec):
        d = i % depth
        # Distribute the TERNARY_DIMS values across the width using a hash spread
        for dim, val in enumerate(vec):
            idx = (hash((i, dim)) % width)
            agg[d, idx] += val

    # Fuse using tropical addition (max) to keep dominant signals
    enhanced = t_add(sketch_np, agg)

    return enhanced, confidence


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic item stream
    items = [f"item_{i}" for i in range(200)]

    # Synthetic textual payloads (same length as items for simplicity)
    texts = [
        "The verification of the plan was successful and logged.",
        "Error detected in the memory module, needs patch.",
        "User requested a fast update, checksum verified.",
        "Pause the deployment until tomorrow.",
    ] * 50  # repeat to match 200 items

    # Reference signal for SSIM – use a uniform low‑value array
    reference = np.full(64 * 4, 0.5, dtype=np.float64)

    # Run hybrid weighted sketch
    sketch, conf = hybrid_weighted_sketch(items)
    print(f"Weighted sketch built. Confidence (Fisher): {conf:.6f}")

    # Evaluate signal‑to‑noise and SSIM
    metrics = hybrid_signal_noise_ssim(sketch, items, reference)
    print("Metrics:", metrics)

    # Build ternary‑enhanced sketch
    enhanced_sketch, conf2 = hybrid_ternary_enhanced_sketch(items, texts)
    print(f"Enhanced sketch shape: {enhanced_sketch.shape}, Confidence: {conf2:.6f}")

    sys.exit(0)