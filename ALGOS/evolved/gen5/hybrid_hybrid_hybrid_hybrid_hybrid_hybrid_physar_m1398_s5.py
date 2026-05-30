# DARWIN HAMMER — match 1398, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s4.py (gen3)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_hybrid_cockpi_m1134_s0.py (gen4)
# born: 2026-05-29T23:35:59Z

"""Hybrid Model Loading and Physarum Conductance Network

This module fuses the core topologies of two parent algorithms:

* **Parent A** – a model‑pool optimizer that uses stylometry features together with
  morphology‑based indices (sphericity, flatness) to decide which models to load
  given a RAM ceiling.
* **Parent B** – a Physarum‑inspired conductance network where edge conductances are
  updated by a flux equation and the update is modulated by a stylometry feature
  vector.

**Mathematical bridge**

1. Each model is represented as a node *i* with a pressure *p_i* that encodes its
   morphology indices:

   \[
   p_i = \alpha\,\text{sphericity}_i - \beta\,\text{flatness}_i
   \]

   where \(\alpha,\beta\) are scaling constants.

2. The edge between nodes *i* and *j* has a conductance \(g_{ij}\) and a physical
   length \(l_{ij}>0\). The flux follows the Physarum primitive of Parent B:

   \[
   q_{ij}= \frac{g_{ij}}{l_{ij}}\,(p_i-p_j)
   \]

3. Conductance updates are modulated by a stylometry feature vector
   \(\mathbf{f}\) extracted from the model’s textual descriptor (e.g. its name).
   The hybrid update (Parent B) is applied element‑wise:

   \[
   g_{ij}\leftarrow\max\bigl(0,\;g_{ij}+ \Delta t\,
        (\gamma\,|f_{ij}|-\delta\,g_{ij})\bigr)
   \]

   where \(\gamma\) and \(\delta\) are gain/decay parameters.

4. After a network step the net inflow \(\sum_j q_{ji}\) of a node is interpreted
   as a loading priority. Nodes whose priority exceeds a threshold are loaded
   into the `ModelPool` while respecting the global RAM ceiling.

The code below implements this hybrid system with three public functions that
demonstrate the combined operation."""

import numpy as np
import math
import random
import sys
import pathlib
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Parent‑A‑like structures (model pool and morphology utilities)
# ----------------------------------------------------------------------
class ModelTier:
    """Simple container for a model description."""
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    """Manages loaded models under a global RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def current_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        return self.current_ram() + model.ram_mb <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> bool:
        if self.is_loaded(model.name):
            return True
        if self.can_load(model):
            self.loaded[model.name] = model
            return True
        return False

    def unload(self, name: str) -> None:
        self.loaded.pop(name, None)

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def __repr__(self) -> str:
        return f"<ModelPool ram={self.current_ram()}/{self.ram_ceiling_mb} MB models={list(self.loaded)}>"


def _hash_to_float(s: str, seed: int = 0) -> float:
    """Deterministic hash → [0,1] float."""
    h = hash((s, seed)) & 0xffffffff
    return h / 0xffffffff

def compute_morphology_indices(name: str) -> Tuple[float, float]:
    """
    Produce pseudo‑sphericity and flatness indices from a model name.
    The implementation is deliberately simple (hash‑based) but provides
    reproducible values in [0,1].
    """
    sphericity = _hash_to_float(name, seed=1)
    flatness   = _hash_to_float(name, seed=2)
    return sphericity, flatness

# ----------------------------------------------------------------------
# Parent‑B‑like utilities (stylometry and Physarum primitives)
# ----------------------------------------------------------------------
def stylometry_feature_vector(text_data: str) -> np.ndarray:
    """
    Very lightweight stylometry: count occurrences of three pronoun groups.
    Returns a column vector (3,) that can be broadcast onto conductance matrices.
    """
    tokens = text_data.lower().split()
    vec = np.zeros(3)
    pronoun_groups = [
        {"i", "me", "my", "mine", "myself"},
        {"you", "your", "yours", "yourself"},
        {"he", "him", "his", "himself"}
    ]
    for i, group in enumerate(pronoun_groups):
        vec[i] = sum(1 for w in tokens if w in group)
    # Normalise to [0,1] to keep updates stable
    if vec.max() > 0:
        vec = vec / vec.max()
    return vec  # shape (3,)


def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def hybrid_conductance_update(conductance: np.ndarray,
                              feature_vector: np.ndarray,
                              dt: float = 1.0,
                              gain: float = 1.0,
                              decay: float = 0.05) -> np.ndarray:
    """
    Element‑wise conductance update where `feature_vector` (shape (3,))
    is broadcast across the conductance matrix.
    """
    # Broadcast to matrix shape
    f_mat = np.abs(feature_vector).reshape(1, -1)  # (1,3)
    # If conductance is square, repeat the vector across rows
    if conductance.shape[0] != f_mat.shape[1]:
        # Pad or truncate to match dimensions
        repeat = math.ceil(conductance.shape[0] / f_mat.shape[1])
        f_mat = np.tile(f_mat, (repeat, 1))[:conductance.shape[0], :conductance.shape[1]]
    return np.maximum(0.0, conductance + dt * (gain * f_mat - decay * conductance))

# ----------------------------------------------------------------------
# Hybrid engine: combine morphology‑driven pressures with stylometry‑modulated conductance
# ----------------------------------------------------------------------
def build_initial_network(models: List[ModelTier]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a symmetric conductance matrix `G` and an edge‑length matrix `L`
    for a fully‑connected undirected graph of `n` models.
    Conductances start from a small positive constant; lengths are random
    positive values to avoid division by zero.
    """
    n = len(models)
    G = np.full((n, n), 0.1, dtype=float)
    np.fill_diagonal(G, 0.0)  # No self‑loops
    L = np.random.uniform(0.5, 2.0, size=(n, n))
    L = (L + L.T) / 2.0
    np.fill_diagonal(L, 0.0)
    return G, L

def compute_node_pressures(models: List[ModelTier],
                           alpha: float = 1.0,
                           beta: float = 1.0) -> np.ndarray:
    """
    Convert morphology indices of each model into a scalar pressure.
    """
    pressures = []
    for m in models:
        sph, flat = compute_morphology_indices(m.name)
        p = alpha * sph - beta * flat
        pressures.append(p)
    return np.array(pressures, dtype=float)  # shape (n,)

def hybrid_network_step(models: List[ModelTier],
                        G: np.ndarray,
                        L: np.ndarray,
                        dt: float = 1.0,
                        gain: float = 1.0,
                        decay: float = 0.05,
                        alpha: float = 1.0,
                        beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one iteration of the hybrid Physarum‑Morphology system.

    Returns
    -------
    G_new : np.ndarray
        Updated conductance matrix.
    flux_matrix : np.ndarray
        Antisymmetric matrix of fluxes q_{ij}.
    priority : np.ndarray
        Net inflow per node (positive values indicate attraction).
    """
    n = len(models)
    # 1. Node pressures from morphology
    p = compute_node_pressures(models, alpha, beta)  # (n,)

    # 2. Compute fluxes for each edge
    flux_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            q = flux(G[i, j], L[i, j], p[i], p[j])
            flux_matrix[i, j] = q
            flux_matrix[j, i] = -q  # antisymmetry

    # 3. Aggregate absolute flux per edge to feed conductance update
    #    Here we reuse the stylometry vector derived from the concatenated model names.
    combined_text = " ".join(m.name for m in models)
    f_vec = stylometry_feature_vector(combined_text)  # (3,)

    # 4. Conductance update (stylometry‑modulated)
    G_new = hybrid_conductance_update(G, f_vec, dt=dt, gain=gain, decay=decay)

    # 5. Optional: enforce symmetry and zero diagonal
    G_new = (G_new + G_new.T) / 2.0
    np.fill_diagonal(G_new, 0.0)

    # 6. Compute loading priority as net inflow (sum of incoming positive flux)
    priority = np.sum(np.maximum(0.0, -flux_matrix), axis=0)  # inflow to node i

    return G_new, flux_matrix, priority

def run_hybrid_simulation(models: List[ModelTier],
                          steps: int = 10,
                          ram_ceiling_mb: int = 6000,
                          priority_threshold: float = 0.1) -> ModelPool:
    """
    Execute a short simulation, updating the network and loading/unloading models
    according to the computed priorities.
    """
    pool = ModelPool(ram_ceiling_mb=ram_ceiling_mb)
    G, L = build_initial_network(models)

    for step in range(steps):
        G, flux_mat, priority = hybrid_network_step(models, G, L)

        # Decide loading based on priority
        for idx, model in enumerate(models):
            if priority[idx] > priority_threshold:
                pool.load(model)
            else:
                pool.unload(model.name)

    return pool

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a few dummy models with varying RAM footprints
    dummy_models = [
        ModelTier(name="AlphaEngine", ram_mb=1200, tier="A"),
        ModelTier(name="BetaProcessor", ram_mb=800, tier="B"),
        ModelTier(name="GammaCircuit", ram_mb=1500, tier="A"),
        ModelTier(name="DeltaModule", ram_mb=600, tier="C"),
        ModelTier(name="EpsilonUnit", ram_mb=900, tier="B")
    ]

    final_pool = run_hybrid_simulation(dummy_models, steps=15, ram_ceiling_mb=4000, priority_threshold=0.05)

    print("Final loaded models:", final_pool)
    print("Total RAM used:", final_pool.current_ram(), "MB")