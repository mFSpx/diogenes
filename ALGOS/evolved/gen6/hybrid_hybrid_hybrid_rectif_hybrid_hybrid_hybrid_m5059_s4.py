# DARWIN HAMMER — match 5059, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_rectified_flo_hybrid_hybrid_endpoi_m519_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_percep_m1761_s1.py (gen5)
# born: 2026-05-29T23:59:34Z

"""Hybrid Lexical‑Geometric‑Perceptual Allocation
Parents:
- hybrid_hybrid_rectified_flo_hybrid_hybrid_endpoi_m519_s3.py (lexical style matrix, stable hashing)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_percep_m1761_s1.py (geometric‑algebra multivector → hyper‑dimensional vector, RBF kernel with pheromone scaling)

Mathematical bridge:
The lexical style matrix (LSM) yields a probability distribution over a fixed
vocabulary of functional‑word categories.  We map each category to a basis
blade of a geometric‑algebra multivector; the LSM probabilities become the
blade coefficients.  By converting the multivector into a bipolar
hyper‑dimensional (HD) vector (using a deterministic hash‑seeded random
generator for each blade) we obtain a representation compatible with the
RBF machinery of Parent B.  The pheromone signal is injected as the kernel
width ε in the Gaussian RBF, thus unifying the algebraic, perceptual and
lexical components in a single adaptive allocation matrix.
"""

import math
import random
import sys
from pathlib import Path
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Sequence, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# 1. Lexical function categories (Parent A)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

def words(text: str) -> List[str]:
    """Tokenise a string into alphabetic lower‑case words."""
    return [w for w in (text or "").lower().split() if w.isalpha()]

def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a “lexical style matrix” (LSM) vector.
    Returns a probability for each FUNCTION_CATS entry; missing categories get 0.0.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    lsm: Dict[str, float] = {}
    for cat, vocab in FUNCTION_CATS.items():
        lsm[cat] = sum(cnt[w] for w in vocab) / total
    return lsm

def stable_hash(text: str) -> int:
    """Deterministic 64‑bit hash based on SHA‑256."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)

# ----------------------------------------------------------------------
# 2. Geometric‑Algebra utilities (Parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices: Sequence[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
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
                lst.pop(j)  # second copy now at same position
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def multiply_multivectors(mv_a: Dict[FrozenSet[int], float],
                         mv_b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """Geometric product of two multivectors represented as dicts."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            if not blade_res:
                continue  # scalar 0
            result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
    return result

# ----------------------------------------------------------------------
# 3. Hybrid constructions
# ----------------------------------------------------------------------
# Mapping from LSM categories to unique basis indices (chosen arbitrarily)
_CATEGORY_TO_INDEX: Dict[str, int] = {cat: i for i, cat in enumerate(FUNCTION_CATS.keys(), start=1)}

def lsm_to_multivector(lsm: Dict[str, float]) -> Dict[FrozenSet[int], float]:
    """
    Convert an LSM probability vector into a simple multivector.
    Each category becomes a 1‑vector blade with coefficient equal to its probability.
    """
    mv: Dict[FrozenSet[int], float] = {}
    for cat, prob in lsm.items():
        if prob <= 0.0:
            continue
        idx = _CATEGORY_TO_INDEX[cat]
        mv[frozenset({idx})] = prob
    # add a scalar part equal to the sum of probabilities (should be 1.0)
    mv[frozenset()] = sum(lsm.values())
    return mv

def multivector_to_hd(mv: Dict[FrozenSet[int], float],
                      dim: int = 10000) -> np.ndarray:
    """
    Embed a multivector into a bipolar hyper‑dimensional vector.
    Each blade gets a deterministic random bipolar seed (±1) based on its index set.
    The final HD vector is the weighted sum of these seeds, binarised to ±1.
    """
    rng = np.random.default_rng()
    hd = np.zeros(dim, dtype=np.int8)

    for blade, coeff in mv.items():
        # deterministic seed from blade indices
        seed_text = ','.join(str(i) for i in sorted(blade))
        seed = stable_hash(seed_text) & 0xffffffff
        rng_bit = np.random.default_rng(seed)
        bipolar = rng_bit.integers(0, 2, size=dim, dtype=np.int8) * 2 - 1  # ±1
        hd = hd + coeff * bipolar

    # Binarise: sign(0) -> 1
    hd = np.where(hd >= 0, 1, -1).astype(np.int8)
    return hd

def rbf_allocation(hd_vectors: np.ndarray,
                   pheromone: float,
                   epsilon_base: float = 1.0) -> np.ndarray:
    """
    Compute a Gaussian RBF allocation matrix from HD vectors.
    The pheromone signal scales the kernel width: ε = ε₀ / (1 + pheromone).
    Returns a row‑stochastic matrix where entry (i,j) is the influence of vector j on i.
    """
    if hd_vectors.ndim != 2:
        raise ValueError("hd_vectors must be a 2‑D array (samples × dimensions)")

    # Euclidean distances
    diff = hd_vectors[:, None, :] - hd_vectors[None, :, :]          # (N,N,D)
    dists = np.linalg.norm(diff, axis=2)                           # (N,N)

    epsilon = epsilon_base / (1.0 + max(0.0, pheromone))
    kernel = np.exp(- (dists ** 2) / (2.0 * epsilon ** 2))

    # Row‑normalise to obtain allocation probabilities
    row_sums = kernel.sum(axis=1, keepdims=True)
    allocation = kernel / np.where(row_sums == 0, 1, row_sums)
    return allocation

def hybrid_allocation(texts: List[str],
                      pheromone: float,
                      hd_dim: int = 10000) -> np.ndarray:
    """
    End‑to‑end hybrid pipeline:
    1. LSM → multivector for each text.
    2. Multivector → HD vector.
    3. RBF allocation with pheromone scaling.
    Returns the allocation matrix (len(texts) × len(texts)).
    """
    # Step 1 & 2
    hd_list = []
    for txt in texts:
        lsm = lsm_vector(txt)
        mv = lsm_to_multivector(lsm)
        hd = multivector_to_hd(mv, dim=hd_dim)
        hd_list.append(hd)

    hd_matrix = np.stack(hd_list, axis=0)  # shape (N, dim)
    # Step 3
    allocation = rbf_allocation(hd_matrix, pheromone)
    return allocation

# ----------------------------------------------------------------------
# 4. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I think therefore I am.",
        "The quick brown fox jumps over the lazy dog.",
        "Never gonna give you up, never gonna let you down."
    ]
    pheromone_signal = random.random() * 5.0  # arbitrary pheromone level

    alloc = hybrid_allocation(sample_texts, pheromone_signal, hd_dim=2048)
    np.set_printoptions(precision=3, suppress=True)
    print("Pheromone signal:", pheromone_signal)
    print("Allocation matrix:\n", alloc)