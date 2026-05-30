# DARWIN HAMMER — match 2756, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_geomet_m2314_s0.py (gen5)
# parent_b: hybrid_hybrid_hdc_hybrid_hy_hybrid_hoeffding_tre_m2618_s3.py (gen4)
# born: 2026-05-29T23:45:48Z

"""
Hybrid Fusion Module
--------------------

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – weighted‑entropy driven allocation together with the
  geometric product (Clifford algebra) used for distances in a Voronoi‑type
  setting.
* **Parent B** – hyperdimensional (HD) computing primitives (random vectors,
  binding, bundling, similarity) together with statistical tools
  (Hoeffding bound and Gini coefficient).

**Mathematical Bridge**

The bridge is the observation that a *weighted entropy* vector
`w = (w₁,…,w_k)` can be interpreted as a *scale* applied to the components
of an HD vector.  Binding in HD computing (`a ⊗ b = a·b` element‑wise) is
mathematically analogous to the *geometric product* of two multivectors,
which also consists of an inner (scalar) part and an outer (wedge) part.
By scaling the binding with the entropy weights we obtain a *weighted
geometric product*:


GP_w(a,b) = (w ∘ (a ⊗ b))          # element‑wise product of a and b,
                                   # then scaled by the entropy weights.


The resulting vector can be compared with another using the standard HD
similarity (normalized dot product).  The statistical side‑effects are
controlled by the Hoeffding bound (confidence on the entropy estimate) and
the Gini coefficient (inequality of the pattern‑count distribution).

The three public functions below demonstrate this hybrid operation:
`weighted_entropy`, `text_to_hd_vector`, and `hybrid_similarity`.
"""

import hashlib
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex patterns and weighted entropy utilities
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
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit)\b",
    re.I,
)

PATTERN_MAP = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
}


def count_pattern_matches(text: str) -> Dict[str, int]:
    """Return a dictionary mapping pattern names to the number of matches."""
    counts = {}
    for name, regex in PATTERN_MAP.items():
        counts[name] = len(regex.findall(text))
    return counts


def weighted_entropy(counts: Dict[str, int], weights: Dict[str, float] | None = None) -> Tuple[float, np.ndarray]:
    """
    Compute the Shannon entropy of the normalized counts, optionally weighted.

    Returns
    -------
    entropy : float
        Weighted entropy value.
    weight_vec : np.ndarray
        Vector of the same length as the number of patterns, containing the
        per‑pattern weights (default weight = 1.0).
    """
    total = sum(counts.values())
    if total == 0:
        # No evidence → entropy is zero, weights are uniform.
        probs = np.zeros(len(counts))
    else:
        probs = np.array([c / total for c in counts.values()], dtype=float)

    # Base entropy
    with np.errstate(divide="ignore", invalid="ignore"):
        base_entropy = -np.nansum(probs * np.log2(probs))

    # Apply external weights if supplied, otherwise use ones.
    if weights is None:
        weight_vec = np.ones_like(probs)
    else:
        weight_vec = np.array([weights.get(k, 1.0) for k in counts.keys()], dtype=float)

    weighted_entropy = float(base_entropy * np.mean(weight_vec))
    return weighted_entropy, weight_vec


# ----------------------------------------------------------------------
# Parent B – hyperdimensional computing primitives
# ----------------------------------------------------------------------
Vector = List[int]


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (geometric product’s inner part)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[Vector]) -> Vector:
    """Superpose a list of vectors via majority vote."""
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("vectors must have equal length")
    sums = np.sum(np.array(vectors, dtype=int), axis=0)
    return [1 if x >= 0 else -1 for x in sums]


def similarity(a: Vector, b: Vector) -> float:
    """Normalized dot product."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    if not a:
        raise ValueError("vectors must not be empty")
    return sum(x * y for x, y in zip(a, b)) / len(a)


# ----------------------------------------------------------------------
# Statistical tools from Parent B
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a bounded random variable."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: List[float]) -> float:
    """Gini coefficient for a non‑negative list."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def weighted_binding(a: Vector, b: Vector, weight_vec: np.ndarray) -> Vector:
    """
    Perform a weighted geometric product:
        GP_w(a,b) = (w ∘ (a ⊗ b))

    where `w` is broadcast over the vector components.
    """
    if len(a) != len(b) or len(a) != len(weight_vec):
        raise ValueError("All inputs must share the same dimensionality")
    bound = bind(a, b)
    # Convert to float for scaling, then back to bipolar {-1,1}
    scaled = np.multiply(bound, weight_vec)
    return [1 if x >= 0 else -1 for x in scaled]


def text_to_hd_vector(text: str, dim: int = 4096) -> Tuple[Vector, np.ndarray]:
    """
    Convert a piece of text into a hyperdimensional vector.

    The vector is a bundled collection of symbol vectors for each token.
    The bundling is weighted by the entropy‑derived weight vector obtained
    from pattern counts.
    """
    # Tokenise (simple whitespace split)
    tokens = text.split()
    symbol_vecs = [symbol_vector(tok, dim) for tok in tokens]

    # Pattern based weighting
    counts = count_pattern_matches(text)
    _, weight_vec = weighted_entropy(counts)

    # Ensure weight vector length matches dimensionality
    if len(weight_vec) != dim:
        # Broadcast / repeat the short weight vector to full size
        repeats = int(np.ceil(dim / len(weight_vec)))
        weight_vec = np.tile(weight_vec, repeats)[:dim]

    # Apply weights during bundling: each symbol vector is scaled before majority vote
    weighted_symbols = [
        [1 if x * w >= 0 else -1 for x, w in zip(vec, weight_vec)]
        for vec in symbol_vecs
    ]
    hd_vec = bundle(weighted_symbols)
    return hd_vec, weight_vec


def hybrid_similarity(text_a: str, text_b: str, dim: int = 4096) -> float:
    """
    Compute similarity between two texts using the weighted geometric product
    as the bridge between entropy (Parent A) and hyperdimensional similarity
    (Parent B).
    """
    vec_a, weight_a = text_to_hd_vector(text_a, dim)
    vec_b, weight_b = text_to_hd_vector(text_b, dim)

    # Use the average of the two weight vectors as the scaling factor
    weight_vec = (weight_a + weight_b) / 2.0

    gp_vec = weighted_binding(vec_a, vec_b, weight_vec)
    return similarity(gp_vec, vec_a)  # similarity of product with one operand


def allocation_decision(text: str, delta: float = 0.05, n_samples: int = 200) -> Dict[str, float]:
    """
    Decide whether to allocate a work‑unit based on entropy confidence
    (Hoeffding bound) and inequality of pattern distribution (Gini).

    Returns a dictionary with the computed metrics and a boolean decision.
    """
    counts = count_pattern_matches(text)
    entropy, _ = weighted_entropy(counts)

    # Maximum possible entropy for k patterns is log2(k)
    k = len(PATTERN_MAP)
    r = math.log2(k) if k > 0 else 1.0
    bound = hoeffding_bound(r, delta, n_samples)

    gini = gini_coefficient(list(counts.values()))

    # Simple heuristic: allocate if entropy is well above bound and distribution is not too skewed.
    allocate = (entropy - bound) > 0.1 and gini < 0.6

    return {
        "entropy": entropy,
        "hoeffding_bound": bound,
        "gini_coefficient": gini,
        "allocate": float(allocate),  # 1.0 for True, 0.0 for False
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_a = "We need to plan the next phase and verify all evidence before release."
    sample_b = "The outcome was successful, the team shipped the product and all logs are verified."

    sim = hybrid_similarity(sample_a, sample_b)
    decision = allocation_decision(sample_a)

    print(f"Hybrid similarity: {sim:.4f}")
    print("Allocation decision metrics:")
    for k, v in decision.items():
        print(f"  {k}: {v}")
    # Ensure no exception occurs
    sys.exit(0)