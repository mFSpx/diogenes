# DARWIN HAMMER — match 2826, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1252_s0.py (gen5)
# parent_b: hybrid_sketches_rlct_grokking_m5_s1.py (gen1)
# born: 2026-05-29T23:46:17Z

"""Hybrid Multivector‑Sketch‑RLCT Module
===================================

Parents
-------
* **Parent A** – *Hybrid Multivector MinHash* (``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1252_s0.py``)
  provides a ``Multivector`` class and a geometric product that yields a new
  multivector whose coefficients can be interpreted as a *frequency* distribution
  over basis blades.

* **Parent B** – *Hybrid Sketch‑RLCT* (``hybrid_sketches_rlct_grokking_m5_s1.py``)
  supplies Count‑Min sketch, a lightweight HyperLogLog estimator and MinHash
  utilities together with the RLCT (real‑log‑canonical‑threshold) asymptotic
  formulas.

Mathematical Bridge
-------------------
Both families manipulate **counts** that live in a logarithmic domain:

* The geometric product of two multivectors produces a multiset of blade
  coefficients.  The absolute values of these coefficients behave like
  frequencies of “compound symbols”.

* Sketch primitives approximate frequencies (Count‑Min), distinct symbol
  cardinalities (HyperLogLog) and set‑similarities (MinHash) without materialising
  the full distribution.

The bridge therefore consists of treating the coefficient map of a
``Multivector`` as a stream of items for the sketch structures.  The sketched
statistics are then fed into the RLCT formulas to obtain a free‑energy‑type
estimate that simultaneously reflects geometric algebra similarity and
statistical‑learning complexity.

The module below implements this fusion and exposes three representative
functions that showcase the hybrid behaviour.
"""

import math
import random
import sys
from collections import defaultdict
from hashlib import blake2b
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Multivector with geometric product
# ----------------------------------------------------------------------


class Multivector:
    """Element of a Euclidean Clifford algebra Cl(n,0).

    Internally stored as a mapping ``blade -> coefficient`` where a blade is
    represented by a tuple of sorted basis indices (e.g. (1,3) for e₁∧e₃).
    The empty tuple ``()`` denotes the scalar (grade‑0) part.
    """

    def __init__(self, components: Dict[Tuple[int, ...], float], n: int):
        # filter out zero coefficients and store a copy
        self.components = {
            tuple(blade): float(v) for blade, v in components.items() if v != 0.0
        }
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        """Return a new Multivector containing only blades of grade *k*."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) component."""
        return self.components.get((), 0.0)

    # ------------------------------------------------------------------
    # Geometric product (simplified: no sign handling, metric is Euclidean)
    # ------------------------------------------------------------------
    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Geometric product of two multivectors.

        The product of blades ``a`` and ``b`` is taken as the sorted concatenation
        ``tuple(sorted(a + b))``.  Coefficients multiply accordingly.  This
        simplified rule preserves the multivector structure while providing a
        deterministic frequency‑like map that can be sketched.
        """
        result: Dict[Tuple[int, ...], float] = defaultdict(float)
        for blade1, coef1 in self.components.items():
            for blade2, coef2 in other.components.items():
                new_blade = tuple(sorted(blade1 + blade2))
                result[new_blade] += coef1 * coef2
        return Multivector(dict(result), self.n)


# ----------------------------------------------------------------------
# Parent B – Sketch primitives (Count‑Min, HyperLogLog‑lite, MinHash)
# ----------------------------------------------------------------------


def _hash(item: str, seed: int) -> int:
    """Deterministic 64‑bit hash using blake2b with a numeric seed."""
    h = blake2b(digest_size=8)
    h.update(seed.to_bytes(4, "little") + item.encode("utf-8"))
    return int.from_bytes(h.digest(), "little")


def count_min_sketch(
    items: Iterable[str], width: int = 128, depth: int = 4
) -> List[List[int]]:
    """Return a Count‑Min sketch table for the supplied *items*."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = _hash(item, d) % width
            table[d][idx] += 1
    return table


def estimate_count_min(sketch: List[List[int]]) -> int:
    """Estimate total count from a Count‑Min sketch (minimum over rows)."""
    return min(sum(row) for row in sketch)


def hyperloglog_sketch(items: Iterable[str], p: int = 10) -> List[int]:
    """Light‑weight HyperLogLog sketch.

    *p* determines the number of registers (m = 2**p).  Each item contributes
    its leading‑zero count to the appropriate register.
    """
    m = 1 << p
    registers = [0] * m
    for item in items:
        h = _hash(item, 0)
        register = h & (m - 1)               # lower *p* bits
        w = h >> p                           # remaining bits
        leading_zeros = (w.bit_length() - w.bit_length())  # always 0, use fallback
        # Use Python's built‑in count of leading zeros in 64‑bit space
        lz = (64 - w.bit_length()) + 1 if w != 0 else 64
        registers[register] = max(registers[register], lz)
    return registers


def estimate_hyperloglog(registers: List[int]) -> float:
    """Cardinality estimate from HyperLogLog registers (raw harmonic mean)."""
    m = len(registers)
    alpha_m = 0.7213 / (1 + 1.079 / m)  # standard constant
    harmonic = sum(2.0 ** (-r) for r in registers)
    raw = alpha_m * m * m / harmonic
    # Small‑range correction (linear counting)
    if raw <= (5 / 2) * m:
        V = registers.count(0)
        if V != 0:
            return m * math.log(m / V)
    return raw


def minhash_signature(items: Iterable[str], num_perm: int = 64) -> List[int]:
    """Compute a MinHash signature (list of minima) for a set of *items*."""
    signature = [2 ** 64 - 1] * num_perm
    for item in items:
        for i in range(num_perm):
            hv = _hash(item, i)
            if hv < signature[i]:
                signature[i] = hv
    return signature


def minhash_jaccard(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must have equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def build_hybrid_structure(
    multivectors: List[Multivector],
    cm_width: int = 128,
    cm_depth: int = 4,
    hll_p: int = 10,
    mh_perm: int = 64,
) -> Dict[str, object]:
    """Construct a hybrid representation for a collection of multivectors.

    Returns a dictionary containing:
      * ``'cm'`` – a Count‑Min sketch of *all* blade identifiers (as strings)
      * ``'hll'`` – HyperLogLog registers estimating distinct blades
      * ``'mh'`` – a mapping ``index -> MinHash signature`` for each multivector
    """
    # Encode each blade as a string "i_j_k" to feed the sketches
    all_blades = []
    mh_map = {}
    for idx, mv in enumerate(multivectors):
        blade_strings = ["_".join(map(str, blade)) if blade else "scalar" for blade in mv.components]
        all_blades.extend(blade_strings)
        mh_map[idx] = minhash_signature(blade_strings, num_perm=mh_perm)

    cm = count_min_sketch(all_blades, width=cm_width, depth=cm_depth)
    hll = hyperloglog_sketch(all_blades, p=hll_p)

    return {"cm": cm, "hll": hll, "mh": mh_map}


def geometric_product_similarity(
    mv1: Multivector,
    mv2: Multivector,
    sketch_params: Dict[str, object] = None,
) -> Tuple[float, float]:
    """Compute a hybrid similarity between two multivectors.

    Returns a tuple ``(log_likelihood_est, jaccard_est)`` where:

    * ``log_likelihood_est`` – an approximation of ``log Σ |c|`` for the
      geometric product coefficients, obtained via a Count‑Min sketch.

    * ``jaccard_est`` – MinHash Jaccard similarity between the blade sets of
      ``mv1`` and ``mv2``.
    """
    # 1. Geometric product
    prod = mv1.geometric_product(mv2)

    # 2. Sketch the product coefficients as a multiset of blade identifiers
    blade_strings = [
        "_".join(map(str, blade)) if blade else "scalar"
        for blade, coeff in prod.components.items()
    ]

    # If external sketch tables are supplied, reuse them; otherwise build fresh.
    if sketch_params and "cm" in sketch_params:
        cm = sketch_params["cm"]
    else:
        cm = count_min_sketch(blade_strings)

    # Estimate total count (proxy for Σ|c|) and turn it into a log‑likelihood
    total_count = estimate_count_min(cm)
    log_likelihood_est = math.log(total_count + 1e-12)

    # 3. MinHash Jaccard of original blade sets
    set1 = {"_".join(map(str, b)) if b else "scalar" for b in mv1.components}
    set2 = {"_".join(map(str, b)) if b else "scalar" for b in mv2.components}
    sig1 = minhash_signature(set1)
    sig2 = minhash_signature(set2)
    jaccard_est = minhash_jaccard(sig1, sig2)

    return log_likelihood_est, jaccard_est


def hybrid_rlct_estimate(
    multivectors: List[Multivector],
    n_samples: int = 1000,
    cm_width: int = 128,
    cm_depth: int = 4,
    hll_p: int = 10,
) -> Dict[str, float]:
    """Estimate the Real Log Canonical Threshold (RLCT) for a set of multivectors.

    The procedure is:

    1. Build a Count‑Min sketch of all blade identifiers and obtain an
       estimate of the *empirical log‑likelihood* ``L`` (≈ log Σ counts).

    2. Use a HyperLogLog sketch to estimate the *effective model complexity*
       ``K`` (the number of distinct blades).

    3. Apply the asymptotic RLCT relation
          λ ≈ (L - K·log n) / log n
       where ``n`` is the supplied ``n_samples`` (dataset size).

    Returns a dictionary with keys ``'L'``, ``'K'``, ``'lambda'`` and the
    corresponding free‑energy estimate ``F = L - λ·log(n)``.
    """
    # Gather all blade identifiers
    all_blades = []
    for mv in multivectors:
        all_blades.extend(
            ["_".join(map(str, blade)) if blade else "scalar" for blade in mv.components]
        )

    # Sketches
    cm = count_min_sketch(all_blades, width=cm_width, depth=cm_depth)
    hll_regs = hyperloglog_sketch(all_blades, p=hll_p)

    # Empirical log‑likelihood from Count‑Min
    total_count = estimate_count_min(cm)
    L = math.log(total_count + 1e-12)

    # Model complexity from HyperLogLog
    K = estimate_hyperloglog(hll_regs)

    # RLCT λ estimate
    log_n = math.log(max(n_samples, 1))
    lam = (L - K * log_n) / log_n if log_n != 0 else 0.0

    # Free‑energy asymptotic
    F = L - lam * log_n

    return {"L": L, "K": K, "lambda": lam, "FreeEnergy": F}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create two simple multivectors in 3‑dimensional space
    mv_a = Multivector(
        {
            (): 1.0,                     # scalar
            (1,): 2.0,                   # e1
            (2,): -0.5,                  # e2
            (1, 2, 3): 0.3,              # e1∧e2∧e3
        },
        n=3,
    )
    mv_b = Multivector(
        {
            (): 0.7,
            (2,): 1.5,
            (3,): -0.2,
            (1, 3): 0.4,
        },
        n=3,
    )

    # 1. Build hybrid structure
    hybrid = build_hybrid_structure([mv_a, mv_b])
    print("Hybrid structure keys:", hybrid.keys())

    # 2. Hybrid geometric‑product similarity
    loglik, jacc = geometric_product_similarity(mv_a, mv_b, sketch_params=hybrid)
    print(f"Estimated log‑likelihood of product: {loglik:.4f}")
    print(f"Estimated Jaccard similarity of blade sets: {jacc:.4f}")

    # 3. RLCT estimate
    rlct_info = hybrid_rlct_estimate([mv_a, mv_b], n_samples=5000)
    print("RLCT estimate:", rlct_info)

    # Ensure no exceptions were raised
    sys.exit(0)