# DARWIN HAMMER — match 2826, survivor 4
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
    def __init__(self, components: Dict[Tuple[int, ...], float], n: int):
        self.components = {
            tuple(blade): float(v) for blade, v in components.items() if v != 0.0
        }
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get((), 0.0)

    def geometric_product(self, other: "Multivector") -> "Multivector":
        result: Dict[Tuple[int, ...], float] = defaultdict(float)
        for blade1, coef1 in self.components.items():
            for blade2, coef2 in other.components.items():
                new_blade = tuple(sorted(blade1 + blade2))
                result[new_blade] += coef1 * coef2
        return Multivector(dict(result), self.n)


def _hash(item: str, seed: int) -> int:
    h = blake2b(digest_size=8)
    h.update(seed.to_bytes(4, "little") + item.encode("utf-8"))
    return int.from_bytes(h.digest(), "little")


def count_min_sketch(
    items: Iterable[str], width: int = 128, depth: int = 4
) -> List[List[int]]:
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = _hash(item, d) % width
            table[d][idx] += 1
    return table


def estimate_count_min(sketch: List[List[int]]) -> int:
    return min(sum(row) for row in sketch)


def hyperloglog_sketch(items: Iterable[str], p: int = 10) -> List[int]:
    m = 1 << p
    registers = [0] * m
    for item in items:
        h = _hash(item, 0)
        register = h & (m - 1)               
        w = h >> p                           
        leading_zeros = (64 - w.bit_length()) + 1 if w != 0 else 64
        registers[register] = max(registers[register], leading_zeros)
    return registers


def estimate_hyperloglog(registers: List[int]) -> float:
    m = len(registers)
    alpha_m = 0.7213 / (1 + 1.079 / m)  
    harmonic = sum(2.0 ** (-r) for r in registers)
    raw = alpha_m * m * m / harmonic
    if raw <= (5 / 2) * m:
        V = registers.count(0)
        if V != 0:
            return m * math.log(m / V)
    return raw


def minhash_signature(items: Iterable[str], num_perm: int = 64) -> List[int]:
    signature = [2 ** 64 - 1] * num_perm
    for item in items:
        for i in range(num_perm):
            hv = _hash(item, i)
            if hv < signature[i]:
                signature[i] = hv
    return signature


def minhash_jaccard(sig1: List[int], sig2: List[int]) -> float:
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

def rlct_formula(sketch: Dict[str, object]) -> float:
    cm_estimate = estimate_count_min(sketch['cm'])
    hll_estimate = estimate_hyperloglog(sketch['hll'])
    mh_jaccard = minhash_jaccard(sketch['mh'][0], sketch['mh'][1])
    return cm_estimate * hll_estimate * mh_jaccard

def improved_hybrid_structure(
    multivectors: List[Multivector],
    cm_width: int = 128,
    cm_depth: int = 4,
    hll_p: int = 10,
    mh_perm: int = 64,
) -> Dict[str, object]:
    hybrid = build_hybrid_structure(multivectors, cm_width, cm_depth, hll_p, mh_perm)
    rlct_estimate = rlct_formula(hybrid)
    return {
        'cm': hybrid['cm'],
        'hll': hybrid['hll'],
        'mh': hybrid['mh'],
        'rlct': rlct_estimate
    }