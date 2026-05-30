# DARWIN HAMMER — match 2752, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_liquid_m1302_s2.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s0.py (gen5)
# born: 2026-05-29T23:45:38Z

"""Hybrid JEPA‑Geometric Algebra Engine
====================================

This module fuses the two parent algorithms:

* **Parent A** – *hybrid_hybrid_omni_chaotic__hybrid_hybrid_liquid_m1302_s2.py*  
  Provides the JEPA energy formulation `E(x,y,z)=||sθ(x)‑pφ(sθ(y),z)||₂²` together
  with a sigmoid encoder and a simple affine predictor.

* **Parent B** – *hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s0.py*  
  Supplies a full geometric‑algebra (Clifford) implementation: blade sign handling,
  blade multiplication, and a `Multivector` container.

**Mathematical bridge**

Node embeddings `sθ(·)` are now represented as *multivectors* rather than plain
vectors.  The predictor `pφ` is defined via the *geometric product* of the parent
multivector with a latent multivector derived from the edge weight `z`.  The
resulting multivector is projected back to a real‑valued vector for the L2‑energy
computation.  This unifies the JEPA energy with the algebraic structure of the
geometric product, allowing similarity (via blade overlap) to influence the
prediction while preserving the original JEPA loss.

The hybrid loss combines


L = Σ_edges E(x,y,z)                # JEPA energy on multivector embeddings
    + λ·VICReg(sθ)                  # variance‑invariance‑covariance regulariser
    + μ·‖sθ(x)‑decode(sθ(x))‖₂²    # simple reconstruction term


Only the core operations are implemented; the regulariser is a lightweight
proxy that penalises deviation from unit variance per embedding dimension.

"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
EMBED_DIM = 64               # dimension of the *vector* part of a multivector
SEED = 42                    # deterministic randomness
MAX_NODES = 1000
MAX_EDGES = 2000
LAMBDA_REG = 0.1
MU_RECON = 0.05

random.seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------
# Utility functions (Parent A)
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ----------------------------------------------------------------------
# Geometric Algebra helpers (Parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices: Iterable[int]) -> Tuple[Tuple[int, ...], int]:
    """Return a sorted tuple of indices and the sign introduced by anti‑commutation.

    Duplicate indices cancel (e_i * e_i = 1) and are removed.
    """
    lst = list(indices)
    sign = 1
    i = 0
    while i < len(lst):
        # Remove duplicates on the fly
        if lst[i] in lst[i + 1 :]:
            # find the next occurrence and cancel both
            j = lst[i + 1 :].index(lst[i]) + i + 1
            lst.pop(j)
            lst.pop(i)
            sign *= 1  # e_i*e_i = +1, sign unchanged
            i = 0  # restart because list changed
            continue
        i += 1

    # Bubble sort to count swaps
    n = len(lst)
    for a in range(n):
        for b in range(n - 1 - a):
            if lst[b] > lst[b + 1]:
                lst[b], lst[b + 1] = lst[b + 1], lst[b]
                sign *= -1
    return tuple(lst), sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """Geometric product of two basis blades.

    Returns a new blade (as frozenset of indices) and the sign factor.
    """
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


# ----------------------------------------------------------------------
# Multivector class (Parent B)
# ----------------------------------------------------------------------
class Multivector:
    """Clifford algebra element represented as a sparse map blade → coefficient."""

    def __init__(self, components: Dict[frozenset[int], float] | None = None):
        self.components: Dict[frozenset[int], float] = components if components else {}

    def __add__(self, other: "Multivector") -> "Multivector":
        result = self.components.copy()
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) + coeff
        return Multivector(result)

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({blade: scalar * coeff for blade, coeff in self.components.items()})

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Geometric product (Clifford multiplication) of two multivectors."""
        out: Dict[frozenset[int], float] = {}
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                blade, sign = _multiply_blades(b1, b2)
                out[blade] = out.get(blade, 0.0) + sign * c1 * c2
        return Multivector(out)

    def to_vector(self, dim: int = EMBED_DIM) -> np.ndarray:
        """Flatten to a real vector of length `dim` using a deterministic hash."""
        vec = np.zeros(dim, dtype=np.float64)
        for blade, coeff in self.components.items():
            # deterministic index from blade hash
            idx = (hash(blade) % dim)
            vec[idx] += coeff
        return vec

    @staticmethod
    def from_vector(vec: np.ndarray) -> "Multivector":
        """Create a simple multivector where each vector entry becomes a 1‑blade."""
        comps: Dict[frozenset[int], float] = {}
        for i, v in enumerate(vec):
            if abs(v) > 1e-12:
                comps[frozenset({i})] = float(v)
        return Multivector(comps)


# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
def encode_node(features: np.ndarray) -> Multivector:
    """Encode raw node features into a multivector.

    The raw features are first passed through a sigmoid (as in Parent A) and
    then mapped to a multivector via `Multivector.from_vector`.
    """
    if features.shape[0] != EMBED_DIM:
        raise ValueError(f"features must be of length {EMBED_DIM}")
    activated = sigmoid(features)
    return Multivector.from_vector(activated)


def geometric_predictor(parent_mv: Multivector, latent_z: np.ndarray) -> Multivector:
    """Predict child embedding from parent multivector and edge latent.

    The latent vector `latent_z` is turned into a multivector and multiplied
    with the parent using the geometric product, mimicking the affine predictor
    of JEPA but enriched by blade interactions.
    """
    latent_mv = Multivector.from_vector(latent_z)
    # Simple geometric product as predictor
    pred_mv = parent_mv.geometric_product(latent_mv)
    return pred_mv


def jepa_energy(child_mv: Multivector, parent_mv: Multivector, latent_z: np.ndarray) -> float:
    """JEPA energy for a single edge using multivector embeddings."""
    pred_mv = geometric_predictor(parent_mv, latent_z)
    child_vec = child_mv.to_vector()
    pred_vec = pred_mv.to_vector()
    return float(np.mean((child_vec - pred_vec) ** 2))


def vicreg_regularizer(embeddings: Iterable[Multivector]) -> float:
    """Light‑weight VICReg‑style regularizer.

    Enforces unit variance per dimension across a batch of embeddings.
    """
    vecs = np.stack([e.to_vector() for e in embeddings], axis=0)  # (B, D)
    var = np.var(vecs, axis=0, ddof=1) + 1e-6
    return float(np.mean((1.0 - np.sqrt(var)) ** 2))


def reconstruction_loss(mv: Multivector) -> float:
    """Simple auto‑encoding loss: compare multivector with its decoded version."""
    vec = mv.to_vector()
    decoded_mv = Multivector.from_vector(vec)
    decoded_vec = decoded_mv.to_vector()
    return float(np.mean((vec - decoded_vec) ** 2))


def hybrid_loss(
    edges: Iterable[Tuple[int, int, np.ndarray]],
    node_embeddings: Dict[int, Multivector],
) -> float:
    """Compute the total hybrid loss over a set of edges.

    Parameters
    ----------
    edges : iterable of (parent_id, child_id, latent_z)
        `latent_z` must be a vector of length `EMBED_DIM`.
    node_embeddings : dict node_id → Multivector
        Pre‑computed multivector embeddings for each node.
    """
    energy_terms = []
    for parent_id, child_id, z in edges:
        energy = jepa_energy(
            child_mv=node_embeddings[child_id],
            parent_mv=node_embeddings[parent_id],
            latent_z=z,
        )
        energy_terms.append(energy)

    avg_energy = float(np.mean(energy_terms)) if energy_terms else 0.0
    reg = vicreg_regularizer(node_embeddings.values())
    recon = float(
        np.mean([reconstruction_loss(mv) for mv in node_embeddings.values()])
    )
    total = avg_energy + LAMBDA_REG * reg + MU_RECON * recon
    return total


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic graph
    N = 5
    node_features = {
        i: np.random.randn(EMBED_DIM) for i in range(N)
    }

    # Encode nodes into multivectors
    embeddings = {i: encode_node(feat) for i, feat in node_features.items()}

    # Random edges with latent vectors
    edges = []
    for _ in range(7):
        parent = random.randint(0, N - 1)
        child = random.randint(0, N - 1)
        # avoid self‑loops for variety
        while child == parent:
            child = random.randint(0, N - 1)
        latent = np.random.randn(EMBED_DIM)
        edges.append((parent, child, latent))

    loss = hybrid_loss(edges, embeddings)
    print(f"Hybrid loss on synthetic graph: {loss:.6f}")