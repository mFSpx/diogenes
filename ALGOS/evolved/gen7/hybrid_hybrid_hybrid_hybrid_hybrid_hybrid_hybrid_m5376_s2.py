# DARWIN HAMMER — match 5376, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s6.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1299_s1.py (gen6)
# born: 2026-05-30T00:01:31Z

"""Hybrid Geometry‑Morphology Engine

This module fuses the core of **Parent Algorithm A** (geometric algebra
operations on multivectors) with the core of **Parent Algorithm B**
(morphology statistics, min‑hash signatures and an endpoint circuit breaker).

Mathematical bridge
-------------------
A multivector 𝑀 = Σ c_B B (B – basis blade, c_B – real coefficient) lives in a
real vector space ℝ^N where N = 2ⁿ (n = algebra dimension).  By grouping the
coefficients by grade we obtain a 4‑dimensional norm vector

    v(M) = (‖grade‑0‖, ‖grade‑1‖, ‖grade‑2‖, ‖grade‑≥3‖)

which is exactly the shape required by the `Morphology` dataclass of the
second parent.  The bridge therefore consists of a **grade‑norm mapping**
`multivector_to_morphology`.  The resulting morphology can be analysed with
the Fisher‑score, while a compact MinHash of the multivector coefficients
provides a similarity measure that can be gated by the
`EndpointCircuitBreaker`.  All three mathematical layers are combined in the
high‑level function `hybrid_evaluate`.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# Geometric Algebra utilities (Parent A)
# ---------------------------------------------------------------------------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Bubble‑sort the index list, returning the sorted list and the sign
    of the permutation.  Duplicate indices cancel the blade."""
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
                # duplicate index → blade vanishes
                lst.pop(j)
                lst.pop(j)  # the element that shifted left
                return lst, sign
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sparse sum of basis blades."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        # keep only non‑zero coefficients
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    # -----------------------------------------------------------------------
    # Basic accessors
    # -----------------------------------------------------------------------
    def grade(self, k: int) -> 'Multivector':
        """Return a new Multivector containing only grade‑k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Coefficient of the empty blade (grade‑0)."""
        return self.components.get(frozenset(), 0.0)

    # -----------------------------------------------------------------------
    # Algebraic operations
    # -----------------------------------------------------------------------
    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __mul__(self, other: 'Multivector') -> 'Multivector':
        """Geometric product (full grade‑mixing)."""
        result: Dict[frozenset, float] = {}
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                blade, sign = _multiply_blades(b1, b2)
                result[blade] = result.get(blade, 0.0) + c1 * c2 * sign
        return Multivector(result, self.n)

    def __repr__(self) -> str:
        terms = [f"{coef:.3g}{''.join(str(i) for i in sorted(blade)) or '1'}"
                 for blade, coef in self.components.items()]
        return f"Multivector({', '.join(terms)})"


# ---------------------------------------------------------------------------
# Morphology utilities (Parent B)
# ---------------------------------------------------------------------------

class Morphology:
    """Four‑dimensional physical description."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        for name, value in (("length", length), ("width", width),
                            ("height", height), ("mass", mass)):
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.mass = float(mass)

    def as_vector(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height, self.mass], dtype=float)

    def __repr__(self) -> str:
        return (f"Morphology(length={self.length:.3g}, width={self.width:.3g}, "
                f"height={self.height:.3g}, mass={self.mass:.3g})")


class EndpointCircuitBreaker:
    """Simple failure‑count circuit breaker."""
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

    def status(self) -> Tuple[bool, int]:
        return self.open, self.failures


def fisher_score(morph: Morphology) -> float:
    """Variance‑to‑mean ratio plus a unit offset."""
    vec = morph.as_vector()
    variance = np.var(vec, ddof=1)
    mean = np.mean(vec)
    eps = 1e-12
    return variance / (mean + eps) + 1.0


# ---------------------------------------------------------------------------
# Hybrid layer – mathematical bridge
# ---------------------------------------------------------------------------

def multivector_to_morphology(mv: Multivector) -> Morphology:
    """Map a multivector to a Morphology by taking Euclidean norms of
    grade‑0, grade‑1, grade‑2 and all higher grades."""
    # grade‑0 (scalar)
    g0 = abs(mv.scalar_part())

    # grade‑1 (vectors)
    g1 = math.sqrt(sum(c * c for b, c in mv.components.items() if len(b) == 1))

    # grade‑2 (bivectors)
    g2 = math.sqrt(sum(c * c for b, c in mv.components.items() if len(b) == 2))

    # grade‑≥3 (remaining blades)
    g3 = math.sqrt(sum(c * c for b, c in mv.components.items() if len(b) >= 3))

    # Ensure strict positivity for the Morphology constructor
    eps = 1e-6
    return Morphology(length=g0 + eps, width=g1 + eps,
                      height=g2 + eps, mass=g3 + eps)


def minhash_signature(mv: Multivector, k: int = 64, seed: int = 42) -> List[int]:
    """Compact MinHash of the multivector's blade identifiers.
    Each hash function picks the minimum hashed value over all blades."""
    rng = np.random.RandomState(seed)
    signatures: List[int] = []
    blades = list(mv.components.keys())
    if not blades:
        return [0] * k
    for _ in range(k):
        min_hash = None
        salt = rng.randint(0, 2**31 - 1)
        for blade in blades:
            h = hash((blade, salt))
            if (min_hash is None) or (h < min_hash):
                min_hash = h
        signatures.append(min_hash if min_hash is not None else 0)
    return signatures


def signature_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Jaccard‑like similarity for equal‑length MinHash signatures."""
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def hybrid_evaluate(mv_a: Multivector,
                    mv_b: Multivector,
                    breaker: EndpointCircuitBreaker,
                    similarity_threshold: float = 0.5) -> Tuple[Multivector,
                                                               Morphology,
                                                               float,
                                                               float,
                                                               Tuple[bool, int]]:
    """
    1. Geometric product of the two multivectors.
    2. Convert the product to a Morphology (grade‑norm bridge).
    3. Compute Fisher score of the morphology.
    4. Compute MinHash similarity of the inputs.
    5. Use the EndpointCircuitBreaker to gate the operation.
    Returns (product, morphology, fisher, similarity, breaker_status).
    """
    product = mv_a * mv_b
    morph = multivector_to_morphology(product)
    fisher = fisher_score(morph)

    sig_a = minhash_signature(mv_a)
    sig_b = minhash_signature(mv_b)
    sim = signature_similarity(sig_a, sig_b)

    if sim >= similarity_threshold:
        breaker.record_success()
    else:
        breaker.record_failure()

    return product, morph, fisher, sim, breaker.status()


# ---------------------------------------------------------------------------
# Additional demonstration functions
# ---------------------------------------------------------------------------

def geometric_product_demo():
    """Simple demonstration of the geometric product."""
    # Define two 3‑dimensional GA elements (n=3)
    a = Multivector({frozenset(): 1.0, frozenset({0}): 2.0}, n=3)   # 1 + 2e0
    b = Multivector({frozenset({1}): 3.0, frozenset({0, 1}): 4.0}, n=3)  # 3e1 + 4e0e1
    return a * b


def morphology_fisher_demo():
    """Create a Morphology from a random multivector and evaluate Fisher."""
    rng = np.random.RandomState(0)
    comps = {frozenset({i}): rng.randn() for i in range(3)}
    mv = Multivector(comps, n=3)
    morph = multivector_to_morphology(mv)
    return fisher_score(morph)


def hybrid_pipeline_demo():
    """Run the full hybrid pipeline on two random multivectors."""
    rng = np.random.RandomState(1)
    comps1 = {frozenset({i}): rng.randn() for i in range(3)}
    comps2 = {frozenset({i, (i+1)%3}): rng.randn() for i in range(3)}
    mv1 = Multivector(comps1, n=3)
    mv2 = Multivector(comps2, n=3)
    breaker = EndpointCircuitBreaker(failure_threshold=2)
    return hybrid_evaluate(mv1, mv2, breaker)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Geometric product demo:", geometric_product_demo())
    print("Morphology Fisher demo:", morphology_fisher_demo())
    product, morph, fisher, sim, (open_state, fail_cnt) = hybrid_pipeline_demo()
    print("\nHybrid pipeline results")
    print("Product :", product)
    print("Morphology :", morph)
    print("Fisher score :", fisher)
    print("Signature similarity :", sim)
    print(f"Circuit breaker – open: {open_state}, failures: {fail_cnt}")