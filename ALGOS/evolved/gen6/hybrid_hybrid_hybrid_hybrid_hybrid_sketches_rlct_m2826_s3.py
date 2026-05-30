# DARWIN HAMMER — match 2826, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1252_s0.py (gen5)
# parent_b: hybrid_sketches_rlct_grokking_m5_s1.py (gen1)
# born: 2026-05-29T23:46:17Z

import math
import random
import sys
from collections import defaultdict
from hashlib import blake2b
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

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
        lz = 64 - w.bit_length() if w != 0 else 64
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

def build_hybrid_structure(
    multivectors: List[Multivector],
    cm_width: int = 128,
    cm_depth: int = 4,
    hll_p: int = 10,
    mh_perm: int = 64,
) -> Dict[str, object]:
    all_blades = []
    mh_map = {}
    for idx, mv in enumerate(multivectors):
        blade_strings = ["_".join(map(str, blade)) if blade else "scalar" for blade in mv.components]
        all_blades.extend(blade_strings)
        mh_map[idx] = minhash_signature(blade_strings, num_perm=mh_perm)

    cm = count_min_sketch(all_blades, width=cm_width, depth=cm_depth)
    hll = hyperloglog_sketch(all_blades, p=hll_p)

    return {
        'cm': cm,
        'hll': hll,
        'mh': mh_map
    }

def rlct_estimate(
    hybrid_structure: Dict[str, object],
    epsilon: float = 1e-6
) -> float:
    cm = hybrid_structure['cm']
    hll = hybrid_structure['hll']

    estimated_count = estimate_count_min(cm)
    estimated_cardinality = estimate_hyperloglog(hll)

    if estimated_count == 0 or estimated_cardinality == 0:
        return 0.0

    kl_divergence = estimated_count * np.log(estimated_count / estimated_cardinality)
    free_energy = -epsilon * kl_divergence
    return free_energy

def improved_build_hybrid_structure(
    multivectors: List[Multivector],
    cm_width: int = 128,
    cm_depth: int = 4,
    hll_p: int = 10,
    mh_perm: int = 64,
) -> Dict[str, object]:
    hybrid_structure = build_hybrid_structure(
        multivectors,
        cm_width=cm_width,
        cm_depth=cm_depth,
        hll_p=hll_p,
        mh_perm=mh_perm
    )

    # Use a more robust way to combine MinHash signatures
    mh_similarities = {}
    for i in range(len(multivectors)):
        for j in range(i+1, len(multivectors)):
            sig1 = hybrid_structure['mh'][i]
            sig2 = hybrid_structure['mh'][j]
            similarity = minhash_jaccard(sig1, sig2)
            mh_similarities[(i, j)] = similarity

    hybrid_structure['mh_similarities'] = mh_similarities
    return hybrid_structure

def improved_rlct_estimate(
    hybrid_structure: Dict[str, object],
    epsilon: float = 1e-6
) -> float:
    cm = hybrid_structure['cm']
    hll = hybrid_structure['hll']
    mh_similarities = hybrid_structure['mh_similarities']

    estimated_count = estimate_count_min(cm)
    estimated_cardinality = estimate_hyperloglog(hll)

    if estimated_count == 0 or estimated_cardinality == 0:
        return 0.0

    # Use a more sophisticated way to combine the estimates
    kl_divergence = estimated_count * np.log(estimated_count / estimated_cardinality)
    free_energy = -epsilon * kl_divergence

    # Incorporate MinHash similarities into the estimate
    similarity_contribution = sum(mh_similarities.values()) / len(mh_similarities)
    improved_estimate = free_energy * (1 + similarity_contribution)
    return improved_estimate