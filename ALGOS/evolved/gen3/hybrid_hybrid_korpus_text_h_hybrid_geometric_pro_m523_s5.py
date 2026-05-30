# DARWIN HAMMER — match 523, survivor 5
# gen: 3
# parent_a: hybrid_korpus_text_hybrid_krampus_brain_m43_s0.py (gen2)
# parent_b: hybrid_geometric_product_voronoi_partition_m4_s0.py (gen1)
# born: 2026-05-29T23:29:20Z

"""Hybrid Text-Voronoi Geometric Product (HTVGP)

This module fuses two parent algorithms:

* **Parent A** – `hybrid_korpus_text_hybrid_krampus_brain_m43_s0.py`  
  Provides a *minhash* signature for arbitrary text and a rich
  feature extraction (`extract_master_vector`) that yields a dictionary of
  floating‑point metrics.

* **Parent B** – `hybrid_geometric_product_voronoi_partition_m4_s0.py`  
  Supplies a Clifford‑algebra **Multivector** implementation, a Voronoi
  partitioner for 2‑D points and the geometric product between multivectors.

**Mathematical bridge**

The bridge is built by interpreting the *minhash* signature of a text as a
compact high‑dimensional hash vector and projecting its first two components
to a 2‑D point. Those points seed a Voronoi diagram that partitions a corpus
of texts. Each Voronoi region aggregates the feature dictionaries of its
member texts into a **Multivector** (each feature becomes a basis blade).  The
geometric product is then applied between the multivectors of adjacent
Voronoi cells, yielding a hybrid operation that couples textual similarity
(minhash) with Clifford geometric algebra.

The resulting three public functions demonstrate the hybrid workflow:

* `text_signature_point(text, k=64)` – minhash → 2‑D point.
* `voronoi_partition_texts(texts, seed_count=5)` – texts → Voronoi regions.
* `region_geometric_products(regions)` – multivector construction + geometric
  products between neighboring regions.

All code relies only on the standard library and NumPy.
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import re

# ----------------------------------------------------------------------
# Parent A utilities (minhash, entropy, feature extraction)
# ----------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Normalize whitespace, lower‑case, and strip."""
    return re.sub(r"\s+", " ", text or "").strip().lower()

def minhash_for_text(text: str, k: int = 64) -> list[int]:
    """Return a minhash signature of length *k* for *text*."""
    text = _clean_text(text)
    if len(text) < 5:
        # not enough shingles – return a zero signature
        return [0] * k
    shingles = [text[i:i + 5] for i in range(len(text) - 4)]
    signature = np.full(k, 1_000_000, dtype=int)
    for s in shingles:
        h = hash(s)
        idx = h % k
        signature[idx] = min(signature[idx], h % 1_000_000)
    return signature.tolist()

def extract_master_vector(text: str) -> dict[str, float]:
    """Return a deterministic pseudo‑feature vector derived from *text*."""
    text = _clean_text(text)
    if not text:
        return {}
    # For reproducibility we base each metric on a deterministic hash.
    base = hash(text)
    random.seed(base)  # deterministic per‑text RNG
    keys = [
        "visceral_ratio", "tech_ratio", "legal_osint_ratio", "ledger_density",
        "recursion_score", "directive_ratio", "target_density",
        "forensic_shield_ratio", "poetic_entropy", "dissociative_index",
        "wrath_velocity", "bureaucratic_weaponization_index",
        "resource_exhaustion_metric", "swarm_orchestration_density",
        "logic_crucifixion_index", "conspiracy_grounding_ratio",
        "chaotic_good_tax", "corporate_grit_tension", "countdown_density",
        "asset_structuring_weight", "pitch_formatting_ratio",
        "agent_symmetry_ratio"
    ]
    return {k: random.random() for k in keys}

# ----------------------------------------------------------------------
# Parent B utilities (Clifford geometric product, Voronoi)
# ----------------------------------------------------------------------
Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    """Return index of the seed nearest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("seed list empty")
    return min(range(len(seeds)),
               key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    """Assign each point to the region of its nearest seed."""
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def _blade_sign(indices):
    """Canonicalize a blade (sorted tuple) and compute its sign."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # duplicate basis vectors cancel (e_i ^ e_i = 0)
            lst.pop(i)
            lst.pop(i)
            n -= 2
            i = max(i - 1, 0)
        else:
            i += 1
    return tuple(lst), sign

def _multiply_blades(blade_a: tuple[int, ...], blade_b: tuple[int, ...]):
    """Geometric product of two basis blades."""
    combined = blade_a + blade_b
    result, sign = _blade_sign(combined)
    return result, sign

class Multivector:
    """Sparse representation of a multivector in an n‑dimensional Clifford algebra."""
    def __init__(self, components: dict[tuple[int, ...], float], n: int):
        self.n = int(n)
        # remove zero coefficients
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}

    def __add__(self, other):
        if not isinstance(other, Multivector):
            return NotImplemented
        if self.n != other.n:
            raise ValueError("grade spaces must match")
        comp = defaultdict(float, self.components)
        for k, v in other.components.items():
            comp[k] += v
        return Multivector(dict(comp), self.n)

    def __mul__(self, other):
        """Geometric product."""
        if not isinstance(other, Multivector):
            return NotImplemented
        if self.n != other.n:
            raise ValueError("grade spaces must match")
        result = defaultdict(float)
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] += sign * coeff_a * coeff_b
        return Multivector(dict(result), self.n)

    def scalar_part(self) -> float:
        return self.components.get((), 0.0)

    def __repr__(self):
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                  key=lambda x: (len(x[0]), x[0])):
            if blade:
                basis = "^".join(f"e{idx}" for idx in blade)
                terms.append(f"{coef:.3g}*{basis}")
            else:
                terms.append(f"{coef:.3g}")
        return " + ".join(terms)

# ----------------------------------------------------------------------
# Hybrid layer – bridging the two worlds
# ----------------------------------------------------------------------
def text_signature_point(text: str, k: int = 64) -> Point:
    """
    Convert a text into a 2‑D point by:
    1. Computing its minhash signature (length *k*).
    2. Normalising the first two entries to the interval [0, 1].
    """
    sig = minhash_for_text(text, k)
    if not sig:
        return (0.0, 0.0)
    # Normalisation: map integer range [0, 1_000_000) → [0,1)
    x = (sig[0] % 1_000_000) / 1_000_000.0
    y = (sig[1] % 1_000_000) / 1_000_000.0
    return (x, y)

def _feature_to_blade(feature: str, n: int) -> tuple[int, ...]:
    """
    Deterministically map a textual feature name to a basis blade.
    The blade is represented by a sorted tuple of basis indices (0‑based).
    The mapping uses a simple hash modulo *n* and ensures uniqueness by
    allowing multi‑index blades when collisions occur.
    """
    base = hash(feature)
    idx = base % n
    # To increase variability we embed the hash's higher bits as a second index
    # when the first index already appears.
    secondary = (base // n) % n
    if secondary != idx:
        return tuple(sorted({idx, secondary}))
    return (idx,)

def region_multivector(texts: list[str], n: int = 5) -> Multivector:
    """
    Aggregate the master vectors of *texts* into a single Multivector.
    Each feature becomes a blade (via `_feature_to_blade`) and its coefficient
    is the average of that feature across the region.
    """
    if not texts:
        return Multivector({}, n)
    # Accumulate sums
    sums = defaultdict(float)
    counts = defaultdict(int)
    for txt in texts:
        vec = extract_master_vector(txt)
        for feat, val in vec.items():
            blade = _feature_to_blade(feat, n)
            sums[blade] += val
            counts[blade] += 1
    # Average
    components = {blade: sums[blade] / counts[blade] for blade in sums}
    return Multivector(components, n)

def voronoi_partition_texts(texts: list[str],
                            seed_count: int = 5,
                            k: int = 64) -> dict[int, list[str]]:
    """
    Partition *texts* into Voronoi regions.
    The first *seed_count* texts generate seed points; all texts are then
    assigned to the nearest seed based on their 2‑D minhash points.
    Returns a mapping ``region_index -> list_of_texts``.
    """
    if not texts:
        return {}
    # Ensure we have enough seeds
    seed_texts = texts[:seed_count]
    seed_points = [text_signature_point(t, k) for t in seed_texts]

    # Map every text to a point and then to a region
    region_map = defaultdict(list)
    for txt in texts:
        pt = text_signature_point(txt, k)
        region = nearest(pt, seed_points)
        region_map[region].append(txt)
    return dict(region_map)

def region_geometric_products(regions: dict[int, list[str]],
                              n: int = 5) -> dict[tuple[int, int], Multivector]:
    """
    For each pair of neighboring Voronoi regions (i.e., regions whose seeds are
    nearest neighbours), compute the geometric product of their region
    multivectors. Returns a mapping ``(region_i, region_j) -> product_mv``.
    """
    if not regions:
        return {}
    # Derive seed points again (required for neighbour detection)
    # The ordering of seeds is the sorted region keys.
    sorted_keys = sorted(regions.keys())
    seed_points = [text_signature_point(regions[k][0]) for k in sorted_keys]

    # Pre‑compute multivectors per region
    mv_per_region = {k: region_multivector(txts, n) for k, txts in regions.items()}

    # Determine adjacency via nearest‑seed graph (undirected)
    adjacency = set()
    for i, p_i in enumerate(seed_points):
        # find the nearest other seed
        nearest_idx = min((j for j in range(len(seed_points)) if j != i),
                          key=lambda j: distance(p_i, seed_points[j]))
        a, b = sorted((sorted_keys[i], sorted_keys[nearest_idx]))
        adjacency.add((a, b))

    # Compute geometric products for each adjacent pair
    products = {}
    for a, b in adjacency:
        products[(a, b)] = mv_per_region[a] * mv_per_region[b]
    return products

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "In a galaxy far, far away, an empire rises.",
        "Quantum entanglement defies classical intuition.",
        "Artificial intelligence reshapes modern society.",
        "The rain in Spain stays mainly in the plain.",
        "To be, or not to be, that is the question.",
        "All models are wrong, but some are useful.",
        "E = mc^2 relates mass and energy.",
        "Gödel's incompleteness theorems limit formal systems."
    ]

    # 1. Voronoi partition the corpus
    regions = voronoi_partition_texts(sample_texts, seed_count=3, k=64)
    print("Voronoi regions (index -> count):")
    for idx, txts in regions.items():
        print(f"  Region {idx}: {len(txts)} texts")

    # 2. Build multivectors per region and show a sample
    mv_examples = {idx: region_multivector(txts, n=5) for idx, txts in regions.items()}
    print("\nSample multivectors per region:")
    for idx, mv in mv_examples.items():
        print(f"  Region {idx}: {mv}")

    # 3. Compute geometric products between neighboring regions
    products = region_geometric_products(regions, n=5)
    print("\nGeometric products between adjacent regions:")
    for (a, b), prod_mv in products.items():
        print(f"  Regions {a}-{b}: {prod_mv}")

    sys.exit(0)