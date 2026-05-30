# DARWIN HAMMER — match 2756, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_geomet_m2314_s0.py (gen5)
# parent_b: hybrid_hybrid_hdc_hybrid_hy_hybrid_hoeffding_tre_m2618_s3.py (gen4)
# born: 2026-05-29T23:45:48Z

"""Hybrid Algorithm integrating Parent A (weighted entropy & regex categorization) and
Parent B (hyperdimensional computing, Hoeffding bound & Gini coefficient).

Mathematical Bridge
-------------------
Parent A provides a *weighted entropy* derived from categorical token counts
(evidence, planning, delay, …).  Those scalar weights are used to scale the
hyper‑dimensional (HD) vectors generated in Parent B.  The scaling is performed
by a *geometric product*‑like operation: element‑wise multiplication of the HD
vector with a scalar weight (the geometric product of a grade‑0 multivector
with a grade‑1 multivector reduces to this scaling).  The resulting weighted
vectors are then *bundled* (majority vote) to obtain a single HD representation
for a piece of text.

Similarity between two texts is computed as the HD cosine‑like similarity,
but the raw similarity is wrapped by a Hoeffding confidence bound that uses
the Gini coefficient of the underlying weight distribution as the range
parameter *r*.  This yields a statistically‑grounded similarity measure that
honours both the information‑theoretic (entropy) and geometric (product) aspects
of the two parent algorithms.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import re

# ----------------------------------------------------------------------
# Parent A – Regex categories and weighted entropy
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

CATEGORY_REGEX = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
}


def count_categories(text: str) -> Dict[str, int]:
    """Return a dictionary mapping each category to the number of matches in *text*."""
    counts = {}
    for name, regex in CATEGORY_REGEX.items():
        matches = regex.findall(text)
        counts[name] = len(matches)
    return counts


def weighted_entropy(counts: Dict[str, int]) -> float:
    """Compute Shannon entropy weighted by the relative frequencies of the categories."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    entropy = -sum(p * math.log(p, 2) for p in probs)
    # Weight by the uniformity of the distribution (higher entropy → higher weight)
    return entropy


# ----------------------------------------------------------------------
# Parent B – Hyperdimensional Computing primitives
# ----------------------------------------------------------------------
Vector = List[int]


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Binding (geometric product for grade‑1 vectors) – element‑wise multiplication."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[Vector]) -> Vector:
    """Majority‑vote bundling of a list of hypervectors."""
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("vectors must have equal length")
    sums = [0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    return [1 if s >= 0 else -1 for s in sums]


def similarity(a: Vector, b: Vector) -> float:
    """Normalized dot‑product similarity (range [-1, 1])."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return sum(x * y for x, y in zip(a, b)) / len(a)


# ----------------------------------------------------------------------
# Hoeffding bound & Gini coefficient (Parent B)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a bounded random variable in [0, r]."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: List[float]) -> float:
    """Return the Gini coefficient of a non‑negative value list."""
    arr = np.array([float(v) for v in values])
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("values must be non‑negative")
    if arr.sum() == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(sorted_arr)
    gini = (n + 1 - 2 * np.sum(cumulative) / arr.sum()) / n
    return gini


# ----------------------------------------------------------------------
# Hybrid Functions – Fusion of both parents
# ----------------------------------------------------------------------
def hd_vector_from_text(
    text: str, dim: int = 4096, entropy_weight: float | None = None
) -> Vector:
    """
    Create a hyperdimensional representation of *text*.

    1. Count category occurrences (Parent A).
    2. For each category, generate a symbol hypervector (Parent B).
    3. Scale each symbol vector by the category count **and** optionally by a
       global entropy weight using a geometric‑product‑like scaling.
    4. Bundle all scaled vectors into a single HD vector.
    """
    counts = count_categories(text)
    total = sum(counts.values()) or 1
    # Global weight: either supplied or derived from weighted entropy
    if entropy_weight is None:
        entropy_weight = weighted_entropy(counts) / math.log2(len(CATEGORY_REGEX) or 1)
    # Build scaled vectors
    scaled_vectors: List[Vector] = []
    for cat, cnt in counts.items():
        if cnt == 0:
            continue
        base_vec = symbol_vector(cat, dim)
        # Scale by count (binding with a scalar represented as a vector of 1s/-1s)
        scalar_vec = [1 if random.random() < 0.5 else -1 for _ in range(dim)]
        # Approximate scalar multiplication via binding with a random sign vector;
        # then apply the entropy weight as a geometric product (element‑wise scaling).
        bound = bind(base_vec, scalar_vec)  # introduces variability proportional to count
        weighted = [int(entropy_weight * x) if x * entropy_weight >= 0 else -int(-entropy_weight * x) for x in bound]
        # Repeat the vector `cnt` times before bundling to reflect frequency
        scaled_vectors.extend([weighted] * cnt)
    if not scaled_vectors:
        # Fallback: a random vector so that downstream ops never see an empty list
        return random_vector(dim, seed="fallback")
    return bundle(scaled_vectors)


def hybrid_similarity(
    text_a: str,
    text_b: str,
    dim: int = 4096,
    delta: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute similarity between two texts with statistical confidence.

    Returns a tuple ``(sim, bound)`` where ``sim`` is the HD similarity and
    ``bound`` is the Hoeffding confidence interval (half‑width) around ``sim``.
    The bound uses the Gini coefficient of the combined category weight
    distribution as the range *r*.
    """
    vec_a = hd_vector_from_text(text_a, dim)
    vec_b = hd_vector_from_text(text_b, dim)
    sim = similarity(vec_a, vec_b)

    # Build weight distribution for Hoeffding bound
    counts_a = count_categories(text_a)
    counts_b = count_categories(text_b)
    combined_weights = list(counts_a.values()) + list(counts_b.values())
    r = gini_coefficient(combined_weights)  # range of the similarity estimator
    n = dim  # number of independent dimensions acts as sample size
    bound = hoeffding_bound(r, delta, n)
    return sim, bound


def allocate_work_units(day_of_week: str, text: str, max_units: int = 100) -> int:
    """
    Allocate a number of work units based on the day of the week and the
    weighted entropy of *text*.

    The day influences a base factor (Monday → 1.0, … Sunday → 0.5).  The final
    allocation is ``int(base_factor * entropy * max_units)`` clipped to
    ``[0, max_units]``.
    """
    day_factors = {
        "monday": 1.0,
        "tuesday": 0.9,
        "wednesday": 0.8,
        "thursday": 0.7,
        "friday": 0.6,
        "saturday": 0.5,
        "sunday": 0.5,
    }
    base = day_factors.get(day_of_week.lower(), 0.6)
    entropy = weighted_entropy(count_categories(text))
    allocation = int(base * entropy * max_units)
    return max(0, min(allocation, max_units))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_a = (
        "We need to verify the source and plan the next steps. "
        "The deadline is tomorrow, but we have a backup plan."
    )
    sample_b = (
        "Confirmed evidence was logged. The schedule is set for next week, "
        "however we might need to pause due to resource constraints."
    )
    sim, bound = hybrid_similarity(sample_a, sample_b)
    print(f"Hybrid similarity: {sim:.4f} ± {bound:.4f}")

    units = allocate_work_units("Wednesday", sample_a, max_units=50)
    print(f"Allocated work units (Wednesday): {units}")

    # Ensure that the HD vector creation does not raise.
    vec = hd_vector_from_text(sample_a)
    print(f"HD vector length: {len(vec)} (should be 4096)")