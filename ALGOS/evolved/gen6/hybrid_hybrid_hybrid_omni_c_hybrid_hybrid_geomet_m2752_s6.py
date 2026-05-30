# DARWIN HAMMER — match 2752, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_liquid_m1302_s2.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s0.py (gen5)
# born: 2026-05-29T23:45:38Z

"""Hybrid JEPA‑Geometric‑Algebra Engine
====================================

This module fuses the two parent algorithms:

* **Parent A – hybrid_hybrid_omni_chaotic__hybrid_hybrid_liquid_m1302_s2.py**  
  Provides the JEPA energy formulation `E(x,y,z)=||sθ(x)-pφ(sθ(y),z)||²` where
  `sθ` is an encoder and `pφ` a predictor that mixes the encoded parent with a
  latent variable `z`.

* **Parent B – hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s0.py**  
  Implements geometric‑algebra (GA) blade arithmetic and a `Multivector`
  container with addition and scalar multiplication.

**Mathematical Bridge**

The bridge is built by interpreting every encoded node representation as a
multivector in the Clifford algebra `Cl(d,0)`.  
The predictor `pφ` is realised as a *geometric product* between the parent
multivector and a blade derived from the latent variable `z`.  The JEPA
energy then becomes the squared Euclidean distance between the coefficient
vectors of the child multivector and the predicted multivector.

Additionally, a lightweight MinHash signature (originating from the HTR‑LTCMH
side of Parent A) is concatenated to the raw node features before encoding,
injecting a similarity‑preserving fingerprint into the GA embedding.

The resulting hybrid can be used on graph‑structured data, on point clouds,
or on any collection where a latent transition between entities is to be
modelled.

"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# GA blade utilities (from Parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Each transposition flips the sign (anti‑commutativity).  Duplicate indices
    cancel because e_i*e_i = 1.
    """
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
                # e_i * e_i = 1  → remove both and keep sign
                lst.pop(j)
                lst.pop(j)  # second element now occupies position j
                n -= 2
                i = -1  # restart outer loop because length changed
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Geometric product of two basis blades.

    Returns (result_blade, sign).  The result may be the scalar blade ``frozenset()``.
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# ----------------------------------------------------------------------
# Multivector class (from Parent B)
# ----------------------------------------------------------------------
class Multivector:
    """Element of the Clifford algebra Cl(d,0).

    Internally stored as ``components`` mapping ``frozenset`` of basis indices
    to a float coefficient.  The empty frozenset represents the scalar part.
    """

    def __init__(self, components: Dict[frozenset, float] = None):
        self.components: Dict[frozenset, float] = components or {}

    def __add__(self, other: "Multivector") -> "Multivector":
        new = self.components.copy()
        for blade, coeff in other.components.items():
            new[blade] = new.get(blade, 0.0) + coeff
        return Multivector(new)

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({blade: scalar * coeff for blade, coeff in self.components.items()})

    def __repr__(self) -> str:
        return f"Multivector({self.components})"


def geometric_product(mv1: Multivector, mv2: Multivector) -> Multivector:
    """Full geometric product between two multivectors."""
    result: Dict[frozenset, float] = {}
    for blade_a, coeff_a in mv1.components.items():
        for blade_b, coeff_b in mv2.components.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            coeff = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff
    return Multivector(result)


# ----------------------------------------------------------------------
# JEPA‑GA specific utilities (from Parent A)
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def minhash_signature(data: bytes, num_perm: int = 16) -> np.ndarray:
    """Very lightweight MinHash: for each permutation seed, hash the data and keep the minimum."""
    mins = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for i in range(num_perm):
        h = hash((i, data))
        # Python's hash may be negative; convert to unsigned 64‑bit
        h_uint = np.uint64(h & ((1 << 64) - 1))
        if h_uint < mins[i]:
            mins[i] = h_uint
    return mins.astype(np.float64) / np.iinfo(np.uint64).max  # normalise to [0,1]


def encode_node(features: np.ndarray, embed_dim: int = 8) -> Multivector:
    """Encode a feature vector as a multivector.

    - The first ``embed_dim`` entries become coefficients of the 1‑vectors
      ``e_0, e_1, …``.
    - Remaining entries (if any) are ignored for simplicity.
    """
    components: Dict[frozenset, float] = {}
    for idx in range(min(embed_dim, features.shape[0])):
        coeff = float(features[idx])
        if coeff != 0.0:
            components[frozenset({idx})] = coeff
    return Multivector(components)


def latent_to_blade(z: float, max_dim: int = 8) -> frozenset:
    """Map a scalar latent variable to a basis blade.

    The integer part of ``z`` modulo ``max_dim`` determines the index; the blade
    is a single‑vector ``e_i``.  For negative ``z`` we flip the sign by using
    the scalar blade (empty frozenset) and later multiply by -1.
    """
    idx = int(abs(z)) % max_dim
    blade = frozenset({idx})
    return blade if z >= 0 else frozenset()  # empty set will be used to flip sign later


def predictor(parent_mv: Multivector, z: float, embed_dim: int = 8) -> Multivector:
    """JEPA predictor realised as a geometric product with a latent blade.

    The latent blade is built from ``z``; its sign is applied after the product.
    """
    blade = latent_to_blade(z, embed_dim)
    # Turn the blade into a multivector (single component with coeff 1)
    latent_mv = Multivector({blade: 1.0}) if blade else Multivector({frozenset(): 1.0})
    pred = geometric_product(parent_mv, latent_mv)
    # If z was negative we flip the whole multivector
    if z < 0:
        pred = -1.0 * pred
    return pred


def jepa_energy(child_mv: Multivector, pred_mv: Multivector) -> float:
    """Squared L2 distance between coefficient vectors of two multivectors."""
    # Union of blades present in either multivector
    all_blades = set(child_mv.components) | set(pred_mv.components)
    diff_sq = 0.0
    for blade in all_blades:
        c1 = child_mv.components.get(blade, 0.0)
        c2 = pred_mv.components.get(blade, 0.0)
        diff_sq += (c1 - c2) ** 2
    return diff_sq


def vicreg_regularizer(mvs: List[Multivector], eps: float = 1e-4) -> float:
    """Simple VICReg‑style regularizer: variance + covariance + invariance.

    For this hybrid we only implement a variance term that encourages
    non‑degenerate coefficient magnitudes.
    """
    # Stack coefficient vectors (same ordering of blades)
    all_blades = set()
    for mv in mvs:
        all_blades.update(mv.components.keys())
    blades = list(all_blades)
    if not blades:
        return 0.0
    coeff_matrix = np.stack(
        [
            np.array([mv.components.get(b, 0.0) for b in blades], dtype=np.float64)
            for mv in mvs
        ],
        axis=0,
    )
    variances = np.var(coeff_matrix, axis=0)
    # Encourage variance to be close to 1 (scaled)
    return np.mean((variances - 1.0) ** 2) + eps


# ----------------------------------------------------------------------
# Hybrid loss over a graph
# ----------------------------------------------------------------------
def hybrid_graph_loss(
    node_features: np.ndarray,
    edges: List[Tuple[int, int, float]],
    embed_dim: int = 8,
    minhash_perm: int = 16,
) -> float:
    """
    Compute the hybrid loss for a directed graph.

    Parameters
    ----------
    node_features : (N, D) array
        Raw feature vectors for each node.
    edges : list of (parent_idx, child_idx, weight)
        Directed edges with a scalar latent weight.
    embed_dim : int
        Dimensionality of the GA embedding (must be ≤ D).
    minhash_perm : int
        Number of permutations for the MinHash fingerprint.

    Returns
    -------
    total_loss : float
    """
    N = node_features.shape[0]
    # Pre‑compute encoded multivectors (including MinHash fingerprint)
    encoded: List[Multivector] = []
    for i in range(N):
        feats = node_features[i]
        # MinHash of the raw bytes of the feature vector
        mh = minhash_signature(feats.tobytes(), num_perm=minhash_perm)
        # Concatenate fingerprint to original features (simple concatenation)
        extended = np.concatenate([feats[: embed_dim - minhash_perm], mh])
        encoded.append(encode_node(extended, embed_dim=embed_dim))

    # JEPA energy term
    energy = 0.0
    for parent_idx, child_idx, weight in edges:
        pred_mv = predictor(encoded[parent_idx], weight, embed_dim=embed_dim)
        energy += jepa_energy(encoded[child_idx], pred_mv)

    # VICReg regularizer on all node embeddings
    reg = vicreg_regularizer(encoded)

    return energy + reg


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Synthetic graph: 20 nodes, feature dim 16
    NUM_NODES = 20
    FEATURE_DIM = 16
    EMBED_DIM = 8  # must be >= minhash_perm
    MINHASH_PERM = 4

    # Random features
    node_feats = np.random.randn(NUM_NODES, FEATURE_DIM).astype(np.float64)

    # Random edges (directed) with latent weight drawn from a normal distribution
    edges: List[Tuple[int, int, float]] = []
    for _ in range(30):
        src = random.randint(0, NUM_NODES - 1)
        dst = random.randint(0, NUM_NODES - 1)
        weight = np.random.randn()
        edges.append((src, dst, weight))

    loss = hybrid_graph_loss(
        node_features=node_feats,
        edges=edges,
        embed_dim=EMBED_DIM,
        minhash_perm=MINHASH_PERM,
    )
    print(f"Hybrid graph loss: {loss:.6f}")