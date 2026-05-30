# DARWIN HAMMER — match 1259, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_doomsd_m857_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s5.py (gen4)
# born: 2026-05-29T23:34:56Z

"""Hybrid Stylometry‑Geometric Fusion
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_doomsd_m857_s1.py (stylometry + Bayesian hypervectors)
- hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s5.py (geometric algebra + Fisher scoring)

Mathematical bridge:
Both parents rely on a deterministic hash to initialise pseudo‑random processes.
We use the SHA‑256 hash of the input text to seed Python’s ``random`` module.
The stylometric proportions (one scalar per FUNCTION_CAT) are then mapped to
basis blades of a geometric algebra (indices 0…k‑1).  These scalars become the
coefficients of a ``Multivector``.  Because the random generator is seeded by the
same hash, any auxiliary hyper‑vectors generated for the categories are
deterministically linked to the multivector representation.  The resulting
object can be processed with the geometric‑algebra primitives from Parent B and
the Fisher‑score kernel from Parent A, yielding a single unified hybrid system.
"""

import hashlib
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry utilities (excerpt & adaptation)
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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all many several few many much most some any each every".split()
    ),
}


def _hash_to_int(text: str) -> int:
    """Deterministic 64‑bit integer derived from SHA‑256 of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def stylometric_proportions(text: str) -> Dict[str, float]:
    """Return the proportion of tokens belonging to each FUNCTION_CAT."""
    tokens = [t.strip(".,!?;:()[]\"'").lower() for t in text.split()]
    total = len(tokens) or 1
    counts = {cat: 0 for cat in FUNCTION_CATS}
    for tok in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if tok in vocab:
                counts[cat] += 1
                break
    return {cat: cnt / total for cat, cnt in counts.items()}


def seeded_random_vector(dim: int, seed: int) -> np.ndarray:
    """Generate a bipolar (+1/‑1) random vector of length *dim* using *seed*."""
    rng = random.Random(seed)
    return np.array([1 if rng.random() < 0.5 else -1 for _ in range(dim)], dtype=int)


# ----------------------------------------------------------------------
# Parent B – geometric algebra utilities (as provided)
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
                sign *= 1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: frozenset[int], blade_b: frozenset[int]
) -> Tuple[frozenset[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Sparse representation of a multivector in an n‑dimensional GA."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
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
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                idx = "".join(str(i) for i in sorted(blade))
                terms.append(f"{coef:.3g}e{idx}")
            else:
                terms.append(f"{coef:.3g}")
        return " + ".join(terms) or "0"


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Hybrid functions – concrete mathematical fusion
# ----------------------------------------------------------------------
CATEGORY_ORDER = list(FUNCTION_CATS.keys())  # deterministic ordering


def build_multivector_from_text(text: str) -> Multivector:
    """
    1. Compute stylometric proportions (scalar per category).
    2. Map each category *c* to a basis blade indexed by its position in CATEGORY_ORDER.
    3. Assemble a Multivector where the coefficient of blade {i} is the proportion of *c*.
    """
    props = stylometric_proportions(text)
    components: Dict[frozenset, float] = {}
    for idx, cat in enumerate(CATEGORY_ORDER):
        val = props.get(cat, 0.0)
        if abs(val) > 1e-15:
            components[frozenset({idx})] = val
    return Multivector(components, n=len(CATEGORY_ORDER))


def apply_fisher_to_multivector(
    mv: Multivector, center: float = 0.5, width: float = 0.2
) -> Multivector:
    """
    Apply the Fisher‑score kernel to every scalar coefficient of *mv*.
    The result is a new Multivector with the same blade structure.
    """
    new_components: Dict[frozenset, float] = {}
    for blade, coef in mv.components.items():
        # Treat the coefficient as the “theta” parameter.
        new_coef = fisher_score(coef, center, width)
        new_components[blade] = new_coef
    return Multivector(new_components, mv.n)


def hyperdimensional_blend(text: str, dim: int = 1024) -> np.ndarray:
    """
    Create a deterministic hyper‑dimensional representation of *text*.
    For each FUNCTION_CAT we generate a bipolar random vector (seeded by the
    global hash combined with the category index) and scale it by the
    stylometric proportion.  The final representation is the sum of all scaled
    vectors.
    """
    seed_base = _hash_to_int(text)
    props = stylometric_proportions(text)
    accum = np.zeros(dim, dtype=int)
    for idx, cat in enumerate(CATEGORY_ORDER):
        cat_seed = seed_base ^ (idx + 1)  # simple mixing
        vec = seeded_random_vector(dim, cat_seed)
        accum += vec * props.get(cat, 0.0)
    return accum


def hybrid_score(text: str) -> float:
    """
    End‑to‑end hybrid metric:
    - Build a multivector from stylometry.
    - Transform it with the Fisher kernel.
    - Convert the transformed multivector to a scalar by summing absolute
      coefficients.
    - Blend the result with the L2 norm of the hyperdimensional vector.
    The two components are combined multiplicatively to emphasise agreement.
    """
    mv = build_multivector_from_text(text)
    mv_fisher = apply_fisher_to_multivector(mv)
    gauss_score = sum(abs(v) for v in mv_fisher.components.values())

    hd_vec = hyperdimensional_blend(text)
    hd_norm = np.linalg.norm(hd_vec)

    return gauss_score * hd_norm


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "I can't believe that you haven't finished the report, "
        "but nevertheless we will proceed with the data we have."
    )
    print("Stylometric proportions:")
    for cat, prop in stylometric_proportions(sample).items():
        print(f"  {cat:12s}: {prop:.3f}")

    mv = build_multivector_from_text(sample)
    print("\nMultivector representation:", mv)

    mv_f = apply_fisher_to_multivector(mv)
    print("\nAfter Fisher kernel:", mv_f)

    hd = hyperdimensional_blend(sample)
    print("\nHyper‑dimensional vector (first 16 components):", hd[:16])

    score = hybrid_score(sample)
    print("\nHybrid score:", score)