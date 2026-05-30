# DARWIN HAMMER — match 4491, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_krampus_brain_m1281_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_hybrid_m720_s0.py (gen4)
# born: 2026-05-29T23:56:06Z

"""Hybrid Algorithm combining Geometric Algebra (Parent A) and Epistemic‑Weighted Pheromone Signals (Parent B).

Mathematical bridge:
- Text is mapped to a multivector (geometric‑algebra representation) using deterministic random
  coefficients (Parent A).
- Each labeled text span carries an epistemic certainty scalar `c ∈ [0,1]` (Parent B).
- The geometric product of two multivectors `M_u ⊗ M_v` yields a new multivector whose blade
  coefficients are then weighted by the geometric mean `√(c_u·c_v)`.  
  This produces a confidence‑weighted pheromone signal that unifies the blade‑wise algebraic
  interaction with epistemic certainty.
- A global metric is obtained by the confidence‑weighted ℓ₂‑norm of the pheromone signal,
  merging the information‑gain perspective of Parent B with the algebraic structure of Parent A.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Core geometric‑algebra utilities (derived from Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return a sorted blade and its sign after cancelling duplicate indices.

    Duplicate indices annihilate (e_i ∧ e_i = 0).  The sign is the parity of the
    permutation needed to sort the remaining indices.
    """
    # Cancel pairs of identical indices
    i = 0
    while i < len(indices):
        j = i + 1
        while j < len(indices):
            if indices[i] == indices[j]:
                # remove both occurrences
                indices.pop(j)
                indices.pop(i)
                i = -1  # restart scanning
                break
            j += 1
        i += 1

    # Bubble‑sort while counting swaps for sign
    sign = 1
    n = len(indices)
    for i in range(n):
        for j in range(0, n - i - 1):
            if indices[j] > indices[j + 1]:
                indices[j], indices[j + 1] = indices[j + 1], indices[j]
                sign = -sign
    return indices, sign


def _multiply_blades(
    blade_a: FrozenSet[int], blade_b: FrozenSet[int]
) -> Tuple[FrozenSet[int], int]:
    """Geometric‑algebra blade multiplication.

    Returns (resulting_blade, sign) where sign ∈ {+1, -1}.
    """
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


def geometric_product(
    mv_a: Dict[FrozenSet[int], float],
    mv_b: Dict[FrozenSet[int], float],
) -> Dict[FrozenSet[int], float]:
    """Compute the geometric product of two multivectors."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            blade_c, sign = _multiply_blades(blade_a, blade_b)
            coeff_c = coeff_a * coeff_b * sign
            result[blade_c] = result.get(blade_c, 0.0) + coeff_c
    return result


# ----------------------------------------------------------------------
# Span representation with epistemic certainty (derived from Parent B)
# ----------------------------------------------------------------------
class Span:
    """A labeled text span carrying an epistemic certainty."""

    __slots__ = ("start", "end", "text", "label", "score")

    def __init__(
        self,
        start: int,
        end: int,
        text: str,
        label: str,
        score: float,
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.label = label
        self.score = score  # assumed to be in [0, 1]

    @property
    def certainty(self) -> float:
        """Epistemic certainty derived from the span's score."""
        return max(0.0, min(1.0, self.score))


# ----------------------------------------------------------------------
# Feature extraction → multivector conversion
# ----------------------------------------------------------------------
def extract_features(text: str, dim: int = 3) -> Dict[FrozenSet[int], float]:
    """Deterministically map a string to a multivector.

    The scalar part is set to 0.  For each basis index `i < dim` a random
    coefficient in [0,1) is generated using a hash‑seeded RNG.
    """
    rng = random.Random(hash(text))
    mv: Dict[FrozenSet[int], float] = {}
    # scalar part (empty blade)
    mv[frozenset()] = 0.0
    for i in range(dim):
        coeff = rng.random()
        mv[frozenset({i})] = coeff
    return mv


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def weighted_pheromone(
    span_u: Span,
    span_v: Span,
    mv_u: Dict[FrozenSet[int], float],
    mv_v: Dict[FrozenSet[int], float],
) -> Dict[FrozenSet[int], float]:
    """Compute a pheromone signal by weighting the geometric product with certainty.

    The weight is the geometric mean √(c_u·c_v).  The returned dict maps each blade
    to its confidence‑weighted coefficient.
    """
    weight = math.sqrt(span_u.certainty * span_v.certainty)
    gp = geometric_product(mv_u, mv_v)
    return {blade: coeff * weight for blade, coeff in gp.items()}


def confidence_weighted_l2_norm(pheromone: Dict[FrozenSet[int], float]) -> float:
    """ℓ₂‑norm of the pheromone signal, already confidence‑weighted."""
    return math.sqrt(sum(v * v for v in pheromone.values()))


def summarize_pheromone(pheromone: Dict[FrozenSet[int], float]) -> str:
    """Human‑readable summary of the pheromone multivector."""
    parts = []
    for blade, coeff in sorted(pheromone.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
        if not blade:
            name = "1"
        else:
            name = "e" + "^".join(str(i) for i in sorted(blade))
        parts.append(f"{coeff:.4f}{name}")
    return " + ".join(parts) if parts else "0"


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create two example spans
    span_a = Span(start=0, end=5, text="Hello", label="greeting", score=0.85)
    span_b = Span(start=6, end=11, text="World", label="noun", score=0.60)

    # Extract multivectors
    mv_a = extract_features(span_a.text)
    mv_b = extract_features(span_b.text)

    # Compute confidence‑weighted pheromone signal
    pheromone = weighted_pheromone(span_a, span_b, mv_a, mv_b)

    # Display results
    print("Span A:", span_a.text, "certainty:", span_a.certainty)
    print("Span B:", span_b.text, "certainty:", span_b.certainty)
    print("\nGeometric product (raw):")
    print(summarize_pheromone(geometric_product(mv_a, mv_b)))
    print("\nWeighted pheromone signal:")
    print(summarize_pheromone(pheromone))
    print("\nConfidence‑weighted ℓ₂ norm:", confidence_weighted_l2_norm(pheromone))
    sys.exit(0)