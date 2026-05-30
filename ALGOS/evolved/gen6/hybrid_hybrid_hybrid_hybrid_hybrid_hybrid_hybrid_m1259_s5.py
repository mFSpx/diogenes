# DARWIN HAMMER — match 1259, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_doomsd_m857_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s5.py (gen4)
# born: 2026-05-29T23:34:56Z

"""
Hybrid Algorithm: Stylometry ↔ Geometric Algebra Bridge
=======================================================

Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_doomsd_m857_s1) computes
the proportion of words belonging to linguistic FUNCTION_CATS and uses a
deterministic hash of the input text to seed a pseudo‑random number generator.
Parent B (hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s5) defines a
geometric‑algebra `Multivector` class together with Gaussian‑beam and Fisher‑score
functions that operate on scalar coefficients.

**Mathematical Bridge**

Both parents rely on a *hash‑derived seed* to generate reproducible randomness.
We exploit this common seed to map the stylometric proportion vector onto the
grade‑1 (vector) part of a `Multivector`.  Each linguistic category is assigned a
unique basis blade `e_i` (represented by the frozenset `{i}`); the coefficient of
that blade is the stylometric proportion multiplied by a Bayesian‑style random
weight produced from the shared seed.  The resulting multivector can then be
processed with the Gaussian‑beam and Fisher‑score primitives of Parent B,
thereby fusing stylometry, Bayesian weighting, and hyperdimensional geometric
computations into a single unified system.
"""

import math
import random
import hashlib
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Parent A – Stylometry utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
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
    "quantifier": set("all many much most several some".split()),
}


def _hash_seed(text: str) -> int:
    """Deterministic integer seed derived from SHA‑256 of the input text."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16)  # use first 64 bits for speed


def stylometric_proportions(text: str) -> Dict[str, float]:
    """
    Compute the proportion of tokens belonging to each FUNCTION_CAT.
    Tokens are split on whitespace and lower‑cased.
    """
    tokens = [t.lower() for t in text.split()]
    total = len(tokens) or 1  # avoid division by zero
    counts = {cat: 0 for cat in FUNCTION_CATS}
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                counts[cat] += 1
                break  # a token belongs to at most one category
    return {cat: cnt / total for cat, cnt in counts.items()}


# ----------------------------------------------------------------------
# Parent B – Geometric Algebra core (trimmed for our use)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """
    Simple geometric‑algebra multivector limited to addition, scalar multiplication,
    and grade extraction.  Internally a dict maps frozenset‑blades to float coefficients.
    """

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k}, self.n
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({blade: coef * scalar for blade, coef in self.components.items()}, self.n)

    def __repr__(self) -> str:
        terms = [f"{coef:.3g}*e{sorted(blade)}" if blade else f"{coef:.3g}" for blade, coef in self.components.items()]
        return " + ".join(terms) or "0"


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity centered at `center` with standard deviation `width`."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.  Returns the (derivative)^2 / intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Hybrid Functions – Fusion of A and B
# ----------------------------------------------------------------------
def build_multivector_from_stylometry(
    proportions: Dict[str, float], seed: int, dim: int = 128
) -> Multivector:
    """
    Map stylometric proportions onto a grade‑1 multivector.

    * Each FUNCTION_CAT receives a unique basis index (0‑based).
    * A deterministic `random.Random(seed)` supplies a Bayesian‑style weight
      w_i ∈ (0,1) for every category.
    * Coefficient for blade e_i = proportion_i * w_i.
    * The multivector lives in an ambient space of dimension `dim` (unused
      beyond bookkeeping; blades are still singletons).
    """
    rng = random.Random(seed)
    # Assign deterministic indices based on sorted category names
    cat_to_idx = {cat: i for i, cat in enumerate(sorted(proportions))}
    components: Dict[frozenset, float] = {}
    for cat, prop in proportions.items():
        idx = cat_to_idx[cat]
        weight = rng.random()  # Bayesian random weight
        coef = prop * weight
        if abs(coef) > 1e-15:
            components[frozenset({idx})] = coef
    # Optionally add a scalar part equal to the average weight (captures global Bayesian bias)
    avg_weight = sum(rng.random() for _ in range(len(proportions))) / max(len(proportions), 1)
    components[frozenset()] = avg_weight
    return Multivector(components, dim)


def apply_fisher_to_multivector(
    mv: Multivector, center: float = 0.5, width: float = 0.2
) -> Multivector:
    """
    Produce a new multivector whose coefficients are the Fisher scores of the
    original coefficients interpreted as `theta` values.
    The scalar part (grade‑0) is transformed similarly.
    """
    new_components: Dict[frozenset, float] = {}
    for blade, theta in mv.components.items():
        score = fisher_score(theta, center, width)
        new_components[blade] = score
    return Multivector(new_components, mv.n)


def hybrid_similarity(mv1: Multivector, mv2: Multivector) -> float:
    """
    Compute a simple inner‑product‑like similarity between two hybrid multivectors.
    Only matching blades contribute (grade‑1 and scalar part).
    """
    sim = 0.0
    for blade, coef1 in mv1.components.items():
        coef2 = mv2.components.get(blade)
        if coef2 is not None:
            sim += coef1 * coef2
    return sim


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text_a = (
        "I think that the quick brown fox jumps over the lazy dog while "
        "you observe the scene from a distance."
    )
    sample_text_b = (
        "She cannot deny that the experiment was successful, but she "
        "doesn't want to reveal the results to anyone."
    )

    # Stylometry → proportions
    prop_a = stylometric_proportions(sample_text_a)
    prop_b = stylometric_proportions(sample_text_b)

    # Shared deterministic seed from each text
    seed_a = _hash_seed(sample_text_a)
    seed_b = _hash_seed(sample_text_b)

    # Build multivectors
    mv_a = build_multivector_from_stylometry(prop_a, seed_a)
    mv_b = build_multivector_from_stylometry(prop_b, seed_b)

    print("Multivector A (raw):", mv_a)
    print("Multivector B (raw):", mv_b)

    # Apply Fisher transformation
    mv_a_fisher = apply_fisher_to_multivector(mv_a)
    mv_b_fisher = apply_fisher_to_multivector(mv_b)

    print("Multivector A (Fisher):", mv_a_fisher)
    print("Multivector B (Fisher):", mv_b_fisher)

    # Hybrid similarity
    sim_raw = hybrid_similarity(mv_a, mv_b)
    sim_fisher = hybrid_similarity(mv_a_fisher, mv_b_fisher)

    print(f"Raw similarity: {sim_raw:.6g}")
    print(f"Fisher‑transformed similarity: {sim_fisher:.6g}")