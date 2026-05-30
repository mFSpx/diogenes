# DARWIN HAMMER — match 1761, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m84_s2.py (gen4)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hdc_serpentin_m327_s3.py (gen4)
# born: 2026-05-29T23:38:42Z

"""Hybrid Multivector‑Perceptual RBF Allocation
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m84_s2.py (Multivector with pheromone‑modulated time constants)
- hybrid_hybrid_perceptual_de_hybrid_hdc_serpentin_m327_s3.py (Perceptual hashing + hyper‑dimensional RBF surrogate)

Mathematical Bridge:
The bridge is a *distance‑based similarity* that exists in both parents.
Parent A yields a geometric‑algebra element (Multivector); after extracting its
grade‑k components and modulating them with store‑state and pheromone signals we
obtain a real‑valued vector.  Parent B builds a Gaussian RBF kernel from the
Euclidean distances of hyper‑dimensional vectors.  By converting each
Multivector into a (bipolar) hyper‑dimensional vector we can reuse the RBF
machinery, while the pheromone signal now appears as a scaling factor inside the
kernel (ε‑parameter).  The resulting linear system provides an adaptive
work‑share allocation that respects both algebraic structure and perceptual
clustering.
"""

import math
import random
import sys
from pathlib import Path
import hashlib
import numpy as np
from typing import Sequence, List, Dict, Tuple, FrozenSet

# ----------------------------------------------------------------------
# Parent A – Multivector utilities (geometric algebra)
# ----------------------------------------------------------------------
def _blade_sign(indices: Sequence[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate index → blade vanishes
                lst.pop(j)
                lst.pop(j)  # second copy now at same position
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # discard zero components for compactness
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}
        self.n = int(n)

    def grade(self, k: int, store_state: Dict[str, float], pheromone_signal: float) -> "Multivector":
        """
        Return a new Multivector keeping only grade‑k blades.
        Each kept component is modulated by the current store state and pheromone signal.
        """
        scale = 1.0 + pheromone_signal
        store_factor = store_state.get("scale", 1.0)
        new_comp = {}
        for blade, val in self.components.items():
            if len(blade) == k:
                new_comp[blade] = val * scale * store_factor
        return Multivector(new_comp, self.n)

    def to_vector(self, order: List[FrozenSet[int]]) -> np.ndarray:
        """
        Convert the multivector into a dense real vector following the supplied blade order.
        Missing blades are filled with zeros.
        """
        return np.array([self.components.get(blade, 0.0) for blade in order], dtype=float)

    def __add__(self, other: "Multivector") -> "Multivector":
        assert self.n == other.n
        comp = self.components.copy()
        for blade, val in other.components.items():
            comp[blade] = comp.get(blade, 0.0) + val
        return Multivector(comp, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (simplified, without metric handling)."""
        assert self.n == other.n
        result: Dict[FrozenSet[int], float] = {}
        for b1, v1 in self.components.items():
            for b2, v2 in other.components.items():
                blade, sign = _multiply_blades(b1, b2)
                result[blade] = result.get(blade, 0.0) + sign * v1 * v2
        return Multivector(result, self.n)

# ----------------------------------------------------------------------
# Parent B – Perceptual hashing & hyper‑dimensional RBF utilities
# ----------------------------------------------------------------------
Vector = Sequence[float]


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(vec: Vector, bits: int = 64) -> str:
    """
    Compute a deterministic perceptual hash of a real‑valued vector.
    The vector is first quantised to 8‑bit integers, then hashed with SHA‑256.
    The first `bits` bits of the hex digest are returned.
    """
    # Simple quantisation
    quant = bytes(int(255 * (v - min(vec)) / (max(vec) - min(vec) + 1e-12)) for v in vec)
    digest = hashlib.sha256(quant).hexdigest()
    # Convert hex to binary and truncate
    bin_digest = bin(int(digest, 16))[2:].zfill(256)
    return bin_digest[:bits]


def cluster_by_phash(vectors: List[Vector], bits: int = 16) -> Dict[str, List[int]]:
    """
    Group vector indices by the prefix of their perceptual hash.
    Returns a dict mapping hash‑prefix → list of indices.
    """
    clusters: Dict[str, List[int]] = {}
    for idx, vec in enumerate(vectors):
        h = compute_phash(vec, bits=bits)
        clusters.setdefault(h, []).append(idx)
    return clusters


def morphology_influenced_vector(seed: int, dim: int = 1024) -> np.ndarray:
    """
    Generate a bipolar (+1/‑1) hyper‑dimensional vector whose pattern depends on a seed.
    The seed could be derived from a morphology metric (e.g., sphericity index).
    """
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=dim)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Component‑wise multiplication (binding) of two hyper‑dimensional vectors."""
    return a * b


# ----------------------------------------------------------------------
# Hybrid Functions – integrating both parents
# ----------------------------------------------------------------------
def multivector_to_hdv(
    mv: Multivector,
    store_state: Dict[str, float],
    pheromone_signal: float,
    blade_order: List[FrozenSet[int]],
    dim: int = 1024,
) -> np.ndarray:
    """
    Convert a Multivector into a bipolar hyper‑dimensional vector.
    Steps:
    1. Extract grade‑2 part (as an example) modulated by store_state & pheromone.
    2. Convert the resulting components into a dense real vector.
    3. Quantise and map to a bipolar vector of dimension `dim`.
    """
    # 1. Grade‑2 extraction (can be changed)
    mv_g2 = mv.grade(k=2, store_state=store_state, pheromone_signal=pheromone_signal)

    # 2. Dense representation
    real_vec = mv_g2.to_vector(blade_order)

    # 3. Derive a deterministic seed from the real vector (hash → int)
    seed_hash = int(hashlib.sha256(real_vec.tobytes()).hexdigest(), 16) % (2 ** 32)

    # 4. Produce bipolar vector and bind with a morphology‑influenced seed
    morph_vec = morphology_influenced_vector(seed_hash, dim=dim)
    hdv = bind(morph_vec, np.sign(real_vec[:dim]) if len(real_vec) >= dim else np.sign(np.pad(real_vec, (0, dim - len(real_vec)))))
    return hdv.astype(float)


def build_rbf_kernel(vectors: List[np.ndarray], epsilon: float) -> np.ndarray:
    """
    Build a symmetric Gaussian RBF kernel matrix K_ij = exp(- (ε * ||v_i - v_j||)^2 ).
    """
    n = len(vectors)
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        K[i, i] = 1.0  # distance zero → exp(0) = 1
        for j in range(i + 1, n):
            r = euclidean(vectors[i], vectors[j])
            val = gaussian(r, epsilon)
            K[i, j] = K[j, i] = val
    return K


def allocate_workshare(
    multivectors: List[Multivector],
    store_state: Dict[str, float],
    pheromone_signal: float,
    epsilon: float = 1.0,
) -> Dict[str, float]:
    """
    Hybrid allocation pipeline:
    1. Convert each Multivector to a hyper‑dimensional vector (HDV).
    2. Cluster HDVs by perceptual hash (prefix length 16 bits).
    3. For each cluster compute a representative (mean) HDV.
    4. Build an RBF kernel from the representatives.
    5. Solve K * w = b where b encodes store‑state priority (here a simple scaled sum).
    6. Distribute the solved weights back to original groups.
    Returns a mapping group_name → allocation proportion (sums to 1).
    """
    # Determine a deterministic blade ordering for all possible grade‑2 blades
    # For n dimensions we generate all 2‑combinations.
    n = multivectors[0].n
    grade2_blades = [frozenset({i, j}) for i in range(n) for j in range(i + 1, n)]

    # 1. Convert to HDVs
    hdvs = [
        multivector_to_hdv(mv, store_state, pheromone_signal, grade2_blades, dim=1024)
        for mv in multivectors
    ]

    # 2. Cluster by perceptual hash
    clusters = cluster_by_phash([v.tolist() for v in hdvs], bits=16)

    # 3. Representative per cluster (mean vector)
    rep_vectors: List[np.ndarray] = []
    cluster_keys: List[str] = []
    for h, idxs in clusters.items():
        cluster_keys.append(h)
        rep = np.mean([hdvs[i] for i in idxs], axis=0)
        rep_vectors.append(rep)

    # 4. RBF kernel on representatives
    K = build_rbf_kernel(rep_vectors, epsilon)

    # 5. Target vector b – derived from store_state (e.g., priority per cluster)
    # Here we map each cluster to a synthetic priority using its hash integer.
    priorities = np.array(
        [int(h, 2) % 100 for h in cluster_keys], dtype=float
    )
    b = priorities / priorities.sum() if priorities.sum() != 0 else np.ones_like(priorities) / len(priorities)

    # Solve for weights (regularised solve to avoid singularity)
    try:
        w = np.linalg.solve(K, b)
    except np.linalg.LinAlgError:
        # fallback to least‑squares
        w = np.linalg.lstsq(K, b, rcond=None)[0]

    # 6. Map weights back to original multivectors
    allocation: Dict[str, float] = {}
    for h, idxs in clusters.items():
        weight = max(w[cluster_keys.index(h)], 0.0)  # enforce non‑negative
        share_per_item = weight / len(idxs) if idxs else 0.0
        for i in idxs:
            group_name = f"group_{i % len(GROUPS)}"
            allocation[group_name] = allocation.get(group_name, 0.0) + share_per_item

    # Normalise to sum to 1
    total = sum(allocation.values())
    if total > 0:
        for k in allocation:
            allocation[k] /= total
    return allocation


# ----------------------------------------------------------------------
# Constants used by the smoke test
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small population of random multivectors (dimension n=6)
    n_dim = 6
    random.seed(42)
    mv_list: List[Multivector] = []
    for _ in range(8):
        # Randomly pick a few grade‑2 blades and assign random coefficients
        comp: Dict[FrozenSet[int], float] = {}
        for _ in range(random.randint(1, 4)):
            i, j = random.sample(range(n_dim), 2)
            blade = frozenset({i, j})
            comp[blade] = random.uniform(-1.0, 1.0)
        mv_list.append(Multivector(comp, n=n_dim))

    # Mock store state and pheromone signal
    store = {"scale": 0.85}
    pheromone = random.uniform(0.0, 0.3)

    # Run the hybrid allocation
    alloc = allocate_workshare(mv_list, store, pheromone, epsilon=0.7)

    print("Allocation per group:")
    for grp, val in sorted(alloc.items()):
        print(f"  {grp}: {val:.4f}")
    # Verify that allocations sum to ~1
    print("Total allocation:", sum(alloc.values()))