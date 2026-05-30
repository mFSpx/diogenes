# DARWIN HAMMER — match 4207, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_hard_t_m2229_s0.py (gen6)
# parent_b: hybrid_hybrid_korpus_text_h_hybrid_geometric_pro_m523_s5.py (gen3)
# born: 2026-05-29T23:54:15Z

import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, FrozenSet, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Geometric Algebra core (Cl(n, 0) with Euclidean metric)
# ----------------------------------------------------------------------
Blade = int  # bitmask representation of a basis blade, e.g. e1^e3 -> 0b1010
Scalar = float


def _blade_grade(b: Blade) -> int:
    """Number of set bits = grade of the blade."""
    return bin(b).count("1")


def _geometric_product_sign(a: Blade, b: Blade) -> int:
    """
    Sign factor for the geometric product of two blades a and b
    in an Euclidean metric.
    Overlapping basis vectors square to +1 and are removed (XOR).
    The sign is (-1)^{# swaps needed to bring the concatenated basis
    vectors into canonical order}. This equals
    (-1)^{popcount(a & b) * (popcount(a & b) - 1) // 2}.
    """
    common = a & b
    swaps = _blade_grade(common) * (_blade_grade(common) - 1) // 2
    return -1 if swaps % 2 else 1


class Multivector:
    """
    Simple multivector for Cl(n,0) using a dict:
        {blade_bitmask: coefficient}
    The scalar (grade‑0) blade is represented by the empty bitmask 0.
    """

    def __init__(self, components: Dict[Blade, Scalar], n: int):
        self.n = int(n)
        # prune near‑zero entries
        self.components: Dict[Blade, Scalar] = {
            b: float(c) for b, c in components.items() if abs(c) > 1e-15
        }

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = defaultdict(float, self.components)
        for b, c in other.components.items():
            result[b] += c
        return Multivector(dict(result), self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result = defaultdict(float)
        for a, ca in self.components.items():
            for b, cb in other.components.items():
                sign = _geometric_product_sign(a, b)
                blade = a ^ b  # overlapping indices cancel (e_i^2 = 1)
                result[blade] += ca * cb * sign
        return Multivector(dict(result), self.n)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def scalar_part(self) -> Scalar:
        """Return the grade‑0 component (0 if absent)."""
        return self.components.get(0, 0.0)

    def __repr__(self) -> str:
        terms = [f"{c:.3g}*e{b}" if b else f"{c:.3g}" for b, c in self.components.items()]
        return " + ".join(terms) if terms else "0"


# ----------------------------------------------------------------------
# Text processing utilities (minhash, shingle entropy, Fisher info proxy)
# ----------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Normalize whitespace, lower‑case and strip punctuation."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _shingles(text: str, k: int = 5) -> List[str]:
    """Return list of k‑character shingles."""
    return [text[i : i + k] for i in range(len(text) - k + 1)]


def minhash_signature(text: str, num_hashes: int = 64, shingle_size: int = 5) -> List[int]:
    """
    Compute a simple minhash signature.
    Each hash function is simulated by mixing the built‑in hash with a
    different seed.
    """
    clean = _clean_text(text)
    if len(clean) < shingle_size:
        return [0] * num_hashes

    shingles = _shingles(clean, shingle_size)
    # pre‑hash shingles once
    shingle_hashes = [hash(s) & 0xFFFFFFFF for s in shingles]

    signature = [0xFFFFFFFF] * num_hashes
    for i in range(num_hashes):
        seed = i * 0x9e3779b9  # large odd constant
        for h in shingle_hashes:
            mixed = (h ^ seed) & 0xFFFFFFFF
            if mixed < signature[i]:
                signature[i] = mixed
    return signature


def _hash_to_basis_index(h: int, n: int) -> int:
    """
    Map a 32‑bit hash value to a basis index in {0,…,n‑1}.
    """
    return h % n


def signature_to_point(sig: List[int]) -> Tuple[int, int]:
    """Project the first two hash values to a 2‑D integer point."""
    return sig[0], sig[1]


def fisher_information_proxy(text: str, shingle_size: int = 5) -> float:
    """
    Use the Shannon entropy of the shingle distribution as a proxy for
    Fisher information. The value is normalised to [0,1].
    """
    clean = _clean_text(text)
    if len(clean) < shingle_size:
        return 0.0
    shingles = _shingles(clean, shingle_size)
    counts = Counter(shingles)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()], dtype=float)
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    # Normalise by the maximum possible entropy (log2 of number of distinct shingles)
    max_entropy = math.log2(len(counts)) if counts else 1.0
    return entropy / max_entropy


def text_to_multivector(
    text: str, n: int = 64, shingle_size: int = 5, embed_fisher: bool = True
) -> Multivector:
    """
    Build a multivector from a text:
    - each minhash entry contributes a unit vector blade e_i,
    - the scalar part stores the (optional) Fisher‑information proxy.
    """
    sig = minhash_signature(text, num_hashes=n, shingle_size=shingle_size)
    components: Dict[Blade, Scalar] = defaultdict(float)

    for h in sig:
        idx = _hash_to_basis_index(h, n)
        blade = 1 << idx  # basis vector e_idx
        components[blade] += 1.0

    if embed_fisher:
        components[0] = fisher_information_proxy(text, shingle_size)

    return Multivector(dict(components), n)


# ----------------------------------------------------------------------
# Voronoi partitioning over text points
# ----------------------------------------------------------------------
def voronoi_partition(
    texts: List[str], seed_count: int = 5, n: int = 64
) -> List[List[str]]:
    """
    Partition texts into Voronoi cells using the 2‑D projection of their
    minhash signatures.
    Returns a list of text groups (one per seed).
    """
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if seed_count > len(texts):
        seed_count = len(texts)

    points = [signature_to_point(minhash_signature(t, num_hashes=n)) for t in texts]
    seeds = random.sample(points, seed_count)

    regions: List[List[str]] = [[] for _ in range(seed_count)]
    for txt, pt in zip(texts, points):
        distances = [np.linalg.norm(np.array(pt) - np.array(s)) for s in seeds]
        closest = int(np.argmin(distances))
        regions[closest].append(txt)

    # discard empty regions (possible when seed_count > unique points)
    return [r for r in regions if r]


def region_multivectors(
    regions: List[List[str]], n: int = 64, shingle_size: int = 5
) -> List[Multivector]:
    """
    Convert each Voronoi region into a single multivector by summing the
    multivectors of its member texts.
    """
    region_mv: List[Multivector] = []
    for texts in regions:
        agg = Multivector({}, n)
        for t in texts:
            agg = agg + text_to_multivector(t, n=n, shingle_size=shingle_size)
        region_mv.append(agg)
    return region_mv


def adjacent_region_products(region_mvs: List[Multivector]) -> List[Multivector]:
    """
    Compute geometric products between every unordered pair of region
    multivectors (treated as “adjacent” for the purpose of this hybrid).
    """
    products: List[Multivector] = []
    for i in range(len(region_mvs)):
        for j in range(i + 1, len(region_mvs)):
            products.append(region_mvs[i] * region_mvs[j])
    return products


# ----------------------------------------------------------------------
# Hybrid scoring function
# ----------------------------------------------------------------------
def hybrid_score(text_a: str, text_b: str, n: int = 64, shingle_size: int = 5) -> float:
    """
    Compute a scalar hybrid score by:
    1. Building multivectors for both texts (including Fisher information).
    2. Taking their geometric product.
    3. Returning the scalar (grade‑0) component.
    """
    mv_a = text_to_multivector(text_a, n=n, shingle_size=shingle_size)
    mv_b = text_to_multivector(text_b, n=n, shingle_size=shingle_size)
    prod = mv_a * mv_b
    return prod.scalar_part()


# ----------------------------------------------------------------------
# Demonstration / simple sanity test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)

    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy canine.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Testing the hybrid geometric‑linguistic module.",
        "Another example sentence for Voronoi partitioning.",
    ]

    # Partition corpus
    vor_regions = voronoi_partition(corpus, seed_count=3, n=64)
    print(f"Voronoi regions (texts per region): {list(map(len, vor_regions))}")

    # Build region multivectors and compute pairwise products
    region_mvs = region_multivectors(vor_regions, n=64)
    products = adjacent_region_products(region_mvs)
    print(f"Number of region products: {len(products)}")
    for idx, p in enumerate(products[:3]):
        print(f"Product {idx} scalar part: {p.scalar_part():.4f}")

    # Hybrid score between two specific texts
    s = hybrid_score(corpus[0], corpus[1])
    print(f"Hybrid score between first two texts: {s:.4f}")