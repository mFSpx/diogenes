# DARWIN HAMMER — match 5574, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_jepa_energy_h_hybrid_hybrid_hybrid_m1517_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1378_s4.py (gen5)
# born: 2026-05-30T00:03:09Z

"""Hybrid Fusion of Model Pool Management and Text‑Feature Graph Weighting

Parents:
- hybrid_hybrid_jepa_energy_h_hybrid_hybrid_hybrid_m1517_s1.py (ModelPool with energy, truth allocation, tier‑based penalties)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1378_s4.py (Text feature extraction → Euclidean cost matrix C,
  epistemic multiplier φ(flag) → fused weight matrix W = C·½(φ_i+φ_j))

Mathematical Bridge:
The *truth allocation* value (or tier‑derived certainty) of each model in the ModelPool is interpreted as the
epistemic certainty flag ϕᵢ used in the text‑graph algorithm.  For a set of texts we compute a symmetric Euclidean
cost matrix **C** from their min‑hash signatures.  The fused edge‑weight matrix **W** is

    Wᵢⱼ = Cᵢⱼ · ½·(ϕᵢ + ϕⱼ)

where ϕᵢ = 1 + α·truth_allocationᵢ (α is a scaling constant).  The resulting **W** guides model‑loading decisions:
nodes (texts) with larger aggregate fused weight are deemed more “expensive” to keep in memory, prompting eviction,
while low‑weight nodes are preferentially loaded.  This unifies the two topologies into a single
energy‑aware, privacy‑sensitive allocation system.
"""

from __future__ import annotations
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# 1. Model Pool (Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Immutable descriptor of a model."""
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


class ModelPool:
    """Manages a pool of loaded models with an energy budget."""

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0
        self._truth_allocation: Dict[str, float] = {}  # per‑model epistemic certainty

    # ------------------------------------------------------------------
    # bookkeeping helpers
    # ------------------------------------------------------------------
    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _update_truth(self, name: str, delta: float) -> None:
        self._truth_allocation[name] = self._truth_allocation.get(name, 0.0) + delta

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def add_model(self, model: ModelTier) -> None:
        """Insert a model, applying tier‑conflict and memory penalties."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10  # tier conflict penalty
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy += 1e6   # memory overflow penalty
        self.loaded[model.name] = model
        # initialise truth allocation if absent
        self._truth_allocation.setdefault(model.name, 0.0)

    def load(self, model: ModelTier) -> None:
        """Load a model, rewarding the operation."""
        self._energy -= 1e4
        self.add_model(model)

    def evict(self, name: str) -> None:
        """Evict a model, rewarding the freeing of memory."""
        if name in self.loaded:
            self._energy -= 1e3
            del self.loaded[name]
            self._truth_allocation.pop(name, None)

    def load_with_eviction(self, model: ModelTier) -> None:
        """Load a model, evicting the least‑important ones until space is available."""
        while model.ram_mb + self._used() > self.ram_ceiling_mb and self.loaded:
            # evict the model with the smallest truth allocation (least certain)
            victim = min(self.loaded.keys(),
                         key=lambda n: self._truth_allocation.get(n, 0.0))
            self.evict(victim)
        self.load(model)

    # ------------------------------------------------------------------
    # energy / truth inspection
    # ------------------------------------------------------------------
    @property
    def energy(self) -> float:
        return self._energy

    def truth_vector(self, model_names: List[str]) -> np.ndarray:
        """Return a vector ϕ of epistemic multipliers for the supplied model names."""
        # scaling constant α determines how strongly truth allocation influences φ
        α = 0.5
        vec = np.array([1.0 + α * self._truth_allocation.get(name, 0.0) for name in model_names],
                       dtype=float)
        return vec

    def boost_truth(self, name: str, amount: float = 1.0) -> None:
        """Increase the truth allocation of a model (e.g., after a successful inference)."""
        self._update_truth(name, amount)


# ----------------------------------------------------------------------
# 2. Text Feature Extraction & Cost Matrix (Parent B)
# ----------------------------------------------------------------------


def _shingles(text: str, width: int = 5) -> List[str]:
    """Generate overlapping substrings (shingles) of given width."""
    return [text[i: i + width] for i in range(len(text) - width + 1)]


def minhash_signature(text: str, k: int = 64, width: int = 5) -> np.ndarray:
    """
    Deterministic min‑hash: return the k smallest 64‑bit hashes of the shingles,
    normalised to the interval [0, 1].
    """
    if not text:
        return np.zeros(k, dtype=float)
    shingles = _shingles(text.lower(), width)
    hashes = [hash(s) & 0xFFFFFFFFFFFFFFFF for s in shingles]
    hashes.sort()
    # pad / truncate to k elements
    padded = (hashes[:k] + [0] * k)[:k]
    return np.array(padded, dtype=float) / float(0xFFFFFFFFFFFFFFFF)


def euclidean_cost_matrix(features: np.ndarray) -> np.ndarray:
    """
    Given an (N, D) matrix of feature vectors, return the symmetric Euclidean
    squared distance matrix C where Cᵢⱼ = ‖vᵢ−vⱼ‖².
    """
    diff = features[:, None, :] - features[None, :, :]
    C = np.einsum('ijk,ijk->ij', diff, diff)
    return C


def epistemic_multiplier_vector(truth_vec: np.ndarray) -> np.ndarray:
    """
    Identity wrapper – in the fusion the truth vector already plays the role of φ.
    """
    return truth_vec


def fused_weight_matrix(cost_matrix: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute the fused edge‑weight matrix W using the bridge formula

        Wᵢⱼ = Cᵢⱼ · ½·(φᵢ + φⱼ)

    Parameters
    ----------
    cost_matrix : (N, N) symmetric ndarray
        Euclidean squared distances C.
    phi : (N,) ndarray
        Epistemic multipliers for each node (derived from ModelPool truth allocation).

    Returns
    -------
    W : (N, N) ndarray
        Symmetric fused weight matrix.
    """
    # outer average of phi
    phi_avg = 0.5 * (phi[:, None] + phi[None, :])
    W = cost_matrix * phi_avg
    return W


# ----------------------------------------------------------------------
# 3. Hybrid Allocation Logic
# ----------------------------------------------------------------------


def allocate_models_based_on_weights(
    pool: ModelPool,
    model_catalog: Dict[str, ModelTier],
    weight_matrix: np.ndarray,
    node_names: List[str],
) -> None:
    """
    Load models into the pool guided by the fused weight matrix.

    Nodes with **lower** aggregate fused weight are considered cheaper to keep,
    so we attempt to load the corresponding models first.  If memory is insufficient,
    models attached to high‑weight nodes are evicted.

    Parameters
    ----------
    pool : ModelPool
        The pool to operate on.
    model_catalog : dict
        Mapping from node name → ModelTier descriptor.
    weight_matrix : (N, N) ndarray
        Fused edge‑weight matrix W.
    node_names : list of str
        Order of nodes that matches rows/columns of `weight_matrix`.
    """
    # Compute a scalar importance per node: sum of its incident fused weights
    importance = weight_matrix.sum(axis=1)  # shape (N,)

    # Sort nodes by ascending importance (cheapest first)
    sorted_indices = np.argsort(importance)

    for idx in sorted_indices:
        name = node_names[idx]
        model = model_catalog.get(name)
        if model is None:
            continue  # no model associated with this node
        if not pool.is_loaded(name):
            pool.load_with_eviction(model)
        else:
            # reward staying loaded by boosting truth allocation
            pool.boost_truth(name, amount=0.2)


def compute_fused_system(
    texts: List[str],
    pool: ModelPool,
    model_catalog: Dict[str, ModelTier],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End‑to‑end pipeline:
    1. Extract min‑hash signatures from `texts`.
    2. Build Euclidean cost matrix C.
    3. Retrieve epistemic multipliers φ from the pool's truth allocation.
    4. Fuse to obtain W.
    5. Allocate models based on W.

    Returns
    -------
    C : ndarray
        Euclidean squared distance matrix.
    W : ndarray
        Fused weight matrix.
    """
    # 1. Feature extraction
    signatures = np.stack([minhash_signature(t) for t in texts])  # (N, k)

    # 2. Cost matrix
    C = euclidean_cost_matrix(signatures)

    # 3. Epistemic multipliers
    phi = pool.truth_vector([t for t in texts])  # align order with texts

    # 4. Fusion
    W = fused_weight_matrix(C, phi)

    # 5. Allocation
    allocate_models_based_on_weights(pool, model_catalog, W, [t for t in texts])

    return C, W


# ----------------------------------------------------------------------
# 6. Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Sample texts (nodes)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Artificial intelligence and machine learning are transformative.",
        "Python code can be both concise and expressive."
    ]

    # Create a ModelPool with a modest memory ceiling
    pool = ModelPool(ram_ceiling_mb=2000)

    # Build a catalog of dummy models, one per text node
    model_catalog = {}
    for i, txt in enumerate(sample_texts):
        name = txt  # using the raw text as identifier (for demo purposes)
        ram = random.randint(200, 600)  # random RAM consumption per model
        tier = random.choice(["T1", "T2", "T3"])
        model_catalog[name] = ModelTier(name=name, ram_mb=ram, tier=tier)

    # Run the fused system
    C, W = compute_fused_system(sample_texts, pool, model_catalog)

    # Simple sanity prints
    print("Cost matrix C:")
    print(np.round(C, 3))
    print("\nFused weight matrix W:")
    print(np.round(W, 3))
    print("\nLoaded models after allocation:")
    for mdl in pool.loaded.values():
        print(f"- {mdl.name[:30]:30s} | RAM: {mdl.ram_mb:4d} MB | Tier: {mdl.tier}")

    print(f"\nFinal energy state: {pool.energy:.2e}")