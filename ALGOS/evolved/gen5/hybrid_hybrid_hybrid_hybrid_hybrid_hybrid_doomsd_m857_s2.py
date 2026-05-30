# DARWIN HAMMER — match 857, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s2.py (gen4)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hdc_serpentin_m313_s2.py (gen2)
# born: 2026-05-29T23:31:15Z

"""Hybrid Stylometry‑HDC Temporal Inequality Algorithm
Parents:
- hybrid_hybrid_hard_t_hybrid_hybrid_bayes__m19_s2.py (stylometry with hash‑seeded PRNG)
- hybrid_hybrid_doomsday_cale_hybrid_hdc_serpentin_m313_s2.py (hyperdimensional vectors, binding, Gini of weekday distribution)

Mathematical Bridge:
Both parents rely on a deterministic hash to seed pseudo‑random hypervectors.
We exploit this by (1) hashing the input text to seed category‑specific random vectors,
(2) weighting each category’s symbol‑vector by its stylometric proportion, (3) bundling the
resulting bound vectors into a single hypervector, and (4) modulating that hypervector with
the Gini coefficient of a weekday distribution computed via the Doomsday calendar.  The
Gini coefficient, a scalar measure of inequality, is turned into a bipolar hypervector and
bound to the stylometric hypervector, yielding a unified representation that captures
lexical style, temporal pattern, and distributional inequality.
"""

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import datetime as dt

# ----------------------------------------------------------------------
# Shared hyperdimensional primitives (adapted from Parent B)
# ----------------------------------------------------------------------
Vector = List[int]


def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    """Generate a bipolar (+1 / -1) hypervector of length *dim* seeded deterministically."""
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Hash a symbolic name into a deterministic seed and produce its hypervector."""
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (binding) of two hypervectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Superposition (addition) of hypervectors, followed by sign‑binarisation."""
    it = iter(vectors)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("no vectors to bundle")
    result = [float(x) for x in first]
    for vec in it:
        for i, val in enumerate(vec):
            result[i] += val
    # Binarise to bipolar representation
    return [1 if v >= 0 else -1 for v in result]


def gini_coefficient(values: Iterable[float]) -> float:
    """Calculate the Gini coefficient of a non‑negative value set."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cum = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cum / (n * sum(xs))


# ----------------------------------------------------------------------
# Stylometry utilities (adapted from Parent A)
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
        "no not never none neither cannot cant wont dont didnt isnt arent wasnt werent".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def _tokenize(text: str) -> List[str]:
    """Very light tokenisation – split on whitespace and strip punctuation."""
    punct = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*"
    return [
        word.strip(punct).lower()
        for word in text.split()
        if word.strip(punct)
    ]


def stylometry_proportions(text: str) -> Dict[str, float]:
    """
    Return the proportion of tokens belonging to each FUNCTION_CAT.
    Categories with zero count are omitted.
    """
    tokens = _tokenize(text)
    total = len(tokens) or 1
    counts: Dict[str, int] = {}
    for cat, vocab in FUNCTION_CATS.items():
        cnt = sum(1 for t in tokens if t in vocab)
        if cnt:
            counts[cat] = cnt
    return {cat: cnt / total for cat, cnt in counts.items()}


def _text_hash_seed(text: str) -> int:
    """Deterministic integer seed derived from SHA‑256 of the full text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big")


# ----------------------------------------------------------------------
# Temporal utilities (Do​msday + Gini) – inspired by Parent B
# ----------------------------------------------------------------------
def weekday_of(date: dt.date) -> int:
    """Return ISO weekday (Monday=0 … Sunday=6) using the Doomsday algorithm."""
    # For simplicity and correctness we delegate to datetime.weekday()
    return date.weekday()


def simulate_weekday_distribution(
    start: dt.date, num_days: int = 30
) -> List[int]:
    """
    Produce a list of weekday counts for *num_days* starting at *start*.
    """
    counts = [0] * 7
    for i in range(num_days):
        d = start + dt.timedelta(days=i)
        counts[weekday_of(d)] += 1
    return counts


def weekday_gini(start: dt.date, window: int = 30) -> float:
    """
    Compute the Gini coefficient of the weekday distribution over *window* days.
    """
    distribution = simulate_weekday_distribution(start, window)
    return gini_coefficient(distribution)


# ----------------------------------------------------------------------
# Hybrid core functions (the fused algorithm)
# ----------------------------------------------------------------------
def text_hypervector(
    text: str,
    dim: int = 4096,
) -> Vector:
    """
    Build a hypervector representing the stylometric profile of *text*.
    For each FUNCTION_CAT we:
      1. Generate a random hypervector seeded by the global text hash.
      2. Bind it to the symbol hypervector of the category name.
      3. Scale (repeat) the bound vector proportionally to the category's proportion.
    All bound vectors are bundled into a single hypervector.
    """
    seed = _text_hash_seed(text)
    proportions = stylometry_proportions(text)

    bound_vectors: List[Vector] = []
    for cat, prop in proportions.items():
        # Random vector seeded by (global seed XOR category hash)
        cat_seed = seed ^ int.from_bytes(hashlib.sha256(cat.encode()).digest()[:8], "big")
        rand_vec = random_vector(dim, cat_seed)
        sym_vec = symbol_vector(cat, dim)
        bound = bind(rand_vec, sym_vec)

        # Replicate proportionally – we approximate by scaling each component.
        scaled = [int(round(x * prop)) for x in bound]
        bound_vectors.append(scaled)

    return bundle(bound_vectors)


def gini_hypervector(gini: float, dim: int = 4096) -> Vector:
    """
    Encode a scalar Gini coefficient into a bipolar hypervector.
    The scalar is first mapped to a binary pattern via thresholding the bits of its
    IEEE‑754 binary representation, then converted to bipolar (+1 / -1).
    """
    # Clamp to [0,1] and map to an 8‑bit integer
    gini_clamped = max(0.0, min(1.0, gini))
    int_rep = int(round(gini_clamped * 255))
    bits = f"{int_rep:08b}"
    # Expand the 8‑bit pattern to the full dimension by repetition
    pattern = [1 if b == "1" else -1 for b in bits]
    repeats = dim // len(pattern) + 1
    vec = (pattern * repeats)[:dim]
    return vec


def hybrid_feature_vector(
    text: str,
    reference_date: dt.date,
    dim: int = 4096,
    weekday_window: int = 30,
) -> Vector:
    """
    Produce the final hybrid hypervector:
        HV = bind( text_hypervector(text), gini_hypervector(gini) )
    where *gini* is the inequality of weekday occurrence over *weekday_window* days
    starting from *reference_date*.
    """
    txt_vec = text_hypervector(text, dim)
    gini_val = weekday_gini(reference_date, weekday_window)
    gini_vec = gini_hypervector(gini_val, dim)
    return bind(txt_vec, gini_vec)


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _demo():
    sample_text = (
        "I cannot believe that you are not going to the party tonight, "
        "even though everyone else will be there. It is very common to feel "
        "left out, but you should not let that stop you."
    )
    today = dt.date.today()
    hv = hybrid_feature_vector(sample_text, today, dim=2048, weekday_window=60)
    # Simple sanity checks
    print(f"Hybrid vector length: {len(hv)}")
    print(f"First 10 components: {hv[:10]}")
    # Verify that the vector is bipolar
    unique_vals = set(hv)
    print(f"Unique component values (should be {{-1, 1}}): {unique_vals}")

    # Show the Gini value used
    gini_val = weekday_gini(today, 60)
    print(f"Gini coefficient of weekday distribution (60‑day window): {gini_val:.4f}")


if __name__ == "__main__":
    _demo()