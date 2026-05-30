# DARWIN HAMMER — match 568, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_liquid_m39_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s5.py (gen4)
# born: 2026-05-29T23:29:50Z

import numpy as np
import re
import math
from typing import Dict, FrozenSet, Tuple, List

# ----------------------------------------------------------------------
# Regex feature sets (unchanged – used elsewhere in the system)
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
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|work)\b",
    re.I,
)

# ----------------------------------------------------------------------
# Geometric Algebra core
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Sort a list of basis indices and return the sorted list together with the sign
    of the permutation. Duplicate indices cancel (grade‑reduction)."""
    lst = list(indices)
    sign = 1
    # Bubble‑sort while tracking swaps
    for i in range(len(lst)):
        for j in range(len(lst) - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign = -sign
            elif lst[j] == lst[j + 1]:
                # Cancel duplicate basis vectors (e_i ^ e_i = 0)
                del lst[j : j + 2]
                sign = sign  # no sign change
                break
    return lst, sign


def _multiply_blades(
    blade_a: FrozenSet[int], blade_b: FrozenSet[int]
) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two blades represented as frozensets of basis indices."""
    combined = list(blade_a) + list(blade_b)
    sorted_indices, sign = _blade_sign(combined)
    return frozenset(sorted_indices), sign


class Multivector:
    """Sparse representation of a multivector in an n‑dimensional Euclidean space."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # prune near‑zero components for numerical stability
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }

    # ------------------------------------------------------------------
    # Basic constructors
    # ------------------------------------------------------------------
    @staticmethod
    def scalar(value: float, n: int) -> "Multivector":
        return Multivector({frozenset(): float(value)}, n)

    @staticmethod
    def vector(values: np.ndarray) -> "Multivector":
        n = values.shape[0]
        comps = {frozenset([i]): float(v) for i, v in enumerate(values)}
        return Multivector(comps, n)

    # ------------------------------------------------------------------
    # Grade extraction
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {
                blade: coef
                for blade, coef in self.components.items()
                if len(blade) == k
            },
            self.n,
        )

    # ------------------------------------------------------------------
    # Scalar part
    # ------------------------------------------------------------------
    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) - coef
        return Multivector(result, self.n)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()}, self.n)

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector(
            {blade: scalar * coef for blade, coef in self.components.items()},
            self.n,
        )

    def __mul__(self, other):
        """Geometric product if other is a Multivector, otherwise scalar multiplication."""
        if isinstance(other, Multivector):
            result: Dict[FrozenSet[int], float] = {}
            for blade_a, coef_a in self.components.items():
                for blade_b, coef_b in other.components.items():
                    blade_res, sign = _multiply_blades(blade_a, blade_b)
                    result[blade_res] = result.get(blade_res, 0.0) + sign * coef_a * coef_b
            return Multivector(result, self.n)
        else:  # scalar multiplication
            return self.__rmul__(float(other))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Multivector(n={self.n}, components={self.components})"


# ----------------------------------------------------------------------
# Fisher‑score utilities (vectorised)
# ----------------------------------------------------------------------
def gaussian_beam(theta: np.ndarray, center: float, width: float) -> np.ndarray:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return np.exp(-0.5 * z ** 2)


def fisher_score(
    theta: np.ndarray, center: float = 0.5, width: float = 1.0, eps: float = 1e-12
) -> np.ndarray:
    """Vectorised Fisher information for a Gaussian beam."""
    intensity = np.maximum(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return (derivative ** 2) / intensity


# ----------------------------------------------------------------------
# Weight modulation – now fully vectorised
# ----------------------------------------------------------------------
def modulate_weights(weights: np.ndarray, input_data: np.ndarray) -> np.ndarray:
    """Modulate each row of the weight matrix by the Fisher score of the
    corresponding input feature."""
    scores = fisher_score(input_data)[:, np.newaxis]  # (n,1)
    return weights * scores  # broadcasting over columns


# ----------------------------------------------------------------------
# Fusion: geometric‑algebra‑based decision
# ----------------------------------------------------------------------
def _weights_to_bivector(weights: np.ndarray) -> Multivector:
    """Convert a square weight matrix into a bivector (grade‑2) multivector.
    Only the upper‑triangular part is kept to avoid double‑counting."""
    n = weights.shape[0]
    comps: Dict[FrozenSet[int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            coeff = float(weights[i, j] - weights[j, i])  # antisymmetric part
            if abs(coeff) > 1e-15:
                comps[frozenset([i, j])] = coeff
    return Multivector(comps, n)


def hybrid_decision(input_data: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute a decision score by:
    1. Modulating the weight matrix with Fisher scores derived from the input.
    2. Embedding the input vector (after Fisher modulation) as a grade‑1 multivector.
    3. Embedding the antisymmetric part of the modulated weights as a bivector.
    4. Taking the geometric product of the two multivectors and returning its scalar part.
    """
    if input_data.ndim != 1:
        raise ValueError("input_data must be a 1‑D array")
    if weights.shape[0] != weights.shape[1] or weights.shape[0] != input_data.shape[0]:
        raise ValueError("weights must be square and match length of input_data")

    # 1. Fisher‑modulated weights
    mod_weights = modulate_weights(weights, input_data)

    # 2. Input multivector (grade‑1) with Fisher scores baked in
    input_scores = fisher_score(input_data)
    input_mv = Multivector(
        {frozenset([i]): float(v) for i, v in enumerate(input_scores)},
        n=input_data.shape[0],
    )

    # 3. Bivector from antisymmetric part of the modulated weight matrix
    weight_biv = _weights_to_bivector(mod_weights)

    # 4. Geometric product and scalar extraction
    decision_mv = input_mv * weight_biv
    return decision_mv.scalar_part()


# ----------------------------------------------------------------------
# Example usage (kept for manual testing; removed when imported as a module)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    input_vec = rng.random(8)
    weight_mat = rng.random((8, 8))
    score = hybrid_decision(input_vec, weight_mat)
    print("Decision score:", score)

    # Demonstrate that the multivector utilities work independently
    mv1 = Multivector.vector(input_vec)
    mv2 = Multivector.scalar(3.0, n=input_vec.shape[0])
    print("Vector + scalar =", mv1 + mv2)
    print("Geometric product (vector * bivector) scalar part:",
          (mv1 * _weights_to_bivector(weight_mat)).scalar_part())