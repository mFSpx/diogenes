# DARWIN HAMMER — match 3752, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hdc_se_hybrid_hybrid_fracti_m265_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m917_s0.py (gen5)
# born: 2026-05-29T23:51:37Z

"""
Hybrid Morphology‑Geometric Hyperdimensional Computing (HM‑GHDC)

Parents:
* Parent A – hybrid_hybrid_hdc_se_hybrid_hybrid_fracti_m265_s3.py
  (morphology → dense vector → bipolar hypervector → fractional‑power weighting)
* Parent B – hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m917_s0.py
  (geometric algebra multivectors, Koopman operator, Shannon entropy,
   pheromone‑style probability weighting, Bayesian Count‑Min sketch)

Mathematical bridge:
1. A morphology is turned into a bipolar hypervector **hₘ** (Parent A).
2. **hₘ** is lifted to a geometric‑algebra multivector **M** by treating each
   hypervector component as the coefficient of a basis blade eᵢ.
3. The coefficient vector **c** of **M** is linearly evolved with a
   Koopman‑type operator **K** (Parent B) → **c′ = K·c**.
4. The Shannon entropy **H(c′)** of the normalized absolute coefficients modulates
   a fractional‑power exponent **α = 1 + H(c′)**, which is applied element‑wise:
   **c″ᵢ = sign(c′ᵢ)·|c′ᵢ|^{α}** (fractional‑power binding from Parent A).
5. A deterministic min‑hash of the accompanying text yields a bipolar hypervector
   **hₜ**.  A Bayesian Count‑Min sketch supplies a posterior weight **wₜ** for the
   text term distribution; **hₜ** is scaled by **wₜ**.
6. Final fused hypervector **h_f** is the element‑wise product
   **h_f = sign(c″) ⊙ hₜ** (geometric binding).

The three public functions below expose the hybrid pipeline, similarity
measurement, and a simple effect‑estimate proxy.
"""

import math
import random
import sys
import pathlib
import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Vector = List[float]
HV = np.ndarray          # bipolar hypervector (+1 / -1)
Blade = frozenset[int]   # basis blade identifier
Coeff = float

# ---------------------------------------------------------------------------
# Parent A – Morphology utilities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def _seed_from_morphology(m: Morphology) -> int:
    """Deterministic integer seed derived from morphology attributes."""
    s = f"{m.length:.6f}:{m.width:.6f}:{m.height:.6f}:{m.mass:.6f}"
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)


def random_vector(dim: int = 10000, seed: int | None = None) -> Vector:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def morphology_vector(m: Morphology, dim: int = 10000) -> Vector:
    """Dense real‑valued vector encoding morphology."""
    seed = _seed_from_morphology(m)
    return random_vector(dim, seed)


def bipolar_hypervector(vec: Vector) -> HV:
    """Binarise a real vector to bipolar (+1 / -1) hypervector."""
    arr = np.array(vec)
    median = np.median(arr)
    hv = np.where(arr >= median, 1, -1).astype(np.int8)
    return hv


# ---------------------------------------------------------------------------
# Parent B – Geometric algebra, entropy, Koopman, Bayesian sketch
# ---------------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return sorted indices and sign after bubble‑sorting, removing duplicates."""
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
                del lst[j:j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(a: Blade, b: Blade) -> Tuple[Blade, int]:
    """Geometric product of two basis blades."""
    combined = list(a) + list(b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


class Multivector:
    """Simple multivector in Cl(n,0) where each basis blade is a frozenset of ints."""

    def __init__(self, components: dict[Blade, Coeff], n: int):
        self.n = n
        # prune near‑zero coefficients
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}

    @classmethod
    def from_vector(cls, vec: HV) -> "Multivector":
        """Lift a bipolar hypervector to a multivector: each index i → blade {i}."""
        comps = {frozenset([i]): float(v) for i, v in enumerate(vec)}
        return cls(comps, n=len(vec))

    def coefficient_array(self) -> np.ndarray:
        """Return a dense array of coefficients ordered by blade index (scalar first)."""
        # For simplicity we only keep grade‑1 blades (vectors) in order.
        coeffs = np.zeros(self.n, dtype=np.float64)
        for blade, val in self.components.items():
            if len(blade) == 1:
                idx = next(iter(blade))
                coeffs[idx] = val
        return coeffs

    @classmethod
    def from_coeff_array(cls, coeffs: np.ndarray) -> "Multivector":
        comps = {frozenset([i]): float(v) for i, v in enumerate(coeffs)}
        return cls(comps, n=coeffs.size)

    def apply_koopman(self, K: np.ndarray) -> "Multivector":
        """Linear evolution via Koopman operator K (n×n matrix)."""
        c = self.coefficient_array()
        c_new = K @ c
        return Multivector.from_coeff_array(c_new)

    def fractional_power(self, alpha: float) -> "Multivector":
        """Raise each coefficient to exponent alpha, preserving sign."""
        new_comps = {}
        for blade, val in self.components.items():
            sign = 1.0 if val >= 0 else -1.0
            new_val = sign * (abs(val) ** alpha)
            new_comps[blade] = new_val
        return Multivector(new_comps, self.n)


def shannon_entropy(vals: np.ndarray) -> float:
    """Entropy of normalized absolute values."""
    probs = np.abs(vals)
    total = probs.sum()
    if total == 0:
        return 0.0
    probs = probs / total
    # avoid log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


# ---------------------------------------------------------------------------
# Bayesian Count‑Min Sketch (simplified)
# ---------------------------------------------------------------------------
class CountMinSketch:
    """Very small Count‑Min sketch with Bayesian posterior weighting."""

    def __init__(self, width: int = 256, depth: int = 4, seed: int | None = None):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        self.hash_seeds = [random.Random(seed + i).randint(0, 2**31 - 1) for i in range(depth)]

    def _hash(self, item: str, i: int) -> int:
        h = hashlib.blake2b(digest_size=4)
        h.update(item.encode())
        h.update(self.hash_seeds[i].to_bytes(4, "little"))
        return int.from_bytes(h.digest(), "little") % self.width

    def update(self, item: str, count: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.tables[i, idx] += count

    def estimate(self, item: str) -> int:
        mins = []
        for i in range(self.depth):
            idx = self._hash(item, i)
            mins.append(self.tables[i, idx])
        return min(mins)

    def posterior_weight(self, item: str, prior: float = 1.0) -> float:
        """Simple Bayesian update assuming Poisson likelihood and Gamma prior."""
        lam_hat = self.estimate(item) + 1e-9  # avoid zero
        # Gamma posterior mean with shape=prior+count, rate=1+1 (unit scale)
        shape = prior + lam_hat
        rate = 2.0
        return shape / rate


# ---------------------------------------------------------------------------
# Text → Hypervector utilities (min‑hash deterministic)
# ---------------------------------------------------------------------------
def _minhash_bits(text: str, dim: int = 10000) -> List[int]:
    """Deterministic min‑hash: for each of dim positions take the smallest hash of any word."""
    words = text.split()
    bits = []
    for i in range(dim):
        best = None
        for w in words:
            h = hashlib.sha256(f"{w}:{i}".encode()).hexdigest()
            val = int(h, 16)
            if best is None or val < best:
                best = val
        bits.append(1 if (best % 2 == 0) else -1)
    return bits


def text_hypervector(text: str, dim: int = 10000) -> HV:
    bits = _minhash_bits(text, dim)
    return np.array(bits, dtype=np.int8)


# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------
def hybrid_encode(morph: Morphology, text: str, dim: int = 10000) -> HV:
    """
    Encode a (morphology, text) pair into a fused hypervector.
    Steps:
        1. Morphology → bipolar hypervector h_m.
        2. Lift h_m to multivector M.
        3. Apply random Koopman matrix K (fixed per program run).
        4. Compute entropy of evolved coefficients → α = 1 + H.
        5. Fractional‑power Mᵅ.
        6. Text → hypervector h_t, weighted by Bayesian sketch posterior.
        7. Bind: sign(Mᵅ) ⊙ h_t  (element‑wise multiplication).
    Returns:
        Fused bipolar hypervector.
    """
    # 1. Morphology → hypervector
    vec = morphology_vector(morph, dim)
    h_m = bipolar_hypervector(vec)

    # 2. Lift to multivector
    M = Multivector.from_vector(h_m)

    # 3. Koopman operator (deterministic random matrix)
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    K = rng.standard_normal((dim, dim))
    # Optional orthogonalisation for stability
    Q, _ = np.linalg.qr(K)
    M_k = M.apply_koopman(Q)

    # 4. Entropy → exponent
    coeffs = M_k.coefficient_array()
    H = shannon_entropy(coeffs)
    alpha = 1.0 + H  # ensures alpha ≥ 1

    # 5. Fractional power
    M_fp = M_k.fractional_power(alpha)

    # 6. Text hypervector with Bayesian weight
    sketch = hybrid_encode.sketch  # static sketch attached to function
    for word in text.split():
        sketch.update(word)
    weight = sketch.posterior_weight(text)  # scalar weight
    h_t_raw = text_hypervector(text, dim)
    h_t = (h_t_raw.astype(np.float64) * weight).astype(np.int8)
    h_t[h_t >= 0] = 1
    h_t[h_t < 0] = -1

    # 7. Binding
    sign_vec = np.sign(np.array([v for v in M_fp.coefficient_array()]))
    sign_vec[sign_vec == 0] = 1  # treat zero as +1
    fused = (sign_vec * h_t).astype(np.int8)
    return fused


def hybrid_similarity(vec1: HV, vec2: HV) -> float:
    """Cosine similarity between two fused hypervectors."""
    a = vec1.astype(np.float64)
    b = vec2.astype(np.float64)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm != 0 else 0.0


def hybrid_effect_estimate(
    morph1: Morphology,
    text1: str,
    morph2: Morphology,
    text2: str,
    dim: int = 10000,
) -> float:
    """
    Proxy for causal effect: similarity of two (morph, text) encodings.
    Higher similarity → larger estimated effect.
    """
    hv1 = hybrid_encode(morph1, text1, dim)
    hv2 = hybrid_encode(morph2, text2, dim)
    return hybrid_similarity(hv1, hv2)


# Attach a persistent sketch to the encode function to avoid re‑creation.
hybrid_encode.sketch = CountMinSketch(width=512, depth=5, seed=12345)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple test objects
    m_a = Morphology(length=2.0, width=1.0, height=0.5, mass=3.2)
    m_b = Morphology(length=1.8, width=1.1, height=0.6, mass=2.9)

    txt_a = "the quick brown fox jumps over the lazy dog"
    txt_b = "a fast auburn rabbit leaps above a sleepy cat"

    hv_a = hybrid_encode(m_a, txt_a)
    hv_b = hybrid_encode(m_b, txt_b)

    sim = hybrid_similarity(hv_a, hv_b)
    effect = hybrid_effect_estimate(m_a, txt_a, m_b, txt_b)

    print(f"Similarity: {sim:.4f}")
    print(f"Effect estimate: {effect:.4f}")

    # Ensure dimensions match and values are bipolar
    assert hv_a.shape == hv_b.shape == (10000,)
    assert set(np.unique(hv_a)).issubset({-1, 1})
    assert set(np.unique(hv_b)).issubset({-1, 1})
    print("Smoke test passed.")