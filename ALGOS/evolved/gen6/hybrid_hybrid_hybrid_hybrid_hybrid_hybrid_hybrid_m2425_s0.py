# DARWIN HAMMER — match 2425, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m555_s1.py (gen5)
# born: 2026-05-29T23:42:16Z

"""
Hybrid Fusion Module
====================

Parents:
    - hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s1.py (Algorithm A)
    - hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m555_s1.py (Algorithm B)

Mathematical Bridge
-------------------
Algorithm A defines a 3‑dimensional *resource vector*  
    eᵢ = [ dᵢ , pᵢ , sᵢ ]  
where *dᵢ* is a haversine distance, *pᵢ = β·σᵢ* encodes a binary signature‑collision flag and *sᵢ* is a decision‑hygiene score.

Algorithm B treats *sections* of a sheaf as arbitrary vectors, applies a random linear map
    Tₙ ∈ ℝ^{d_out×d_in}
to obtain a transformed tensor *tₙ = Tₙ·sectionₙ* and then feeds *tₙ* into a Dense Associative Memory (DAM) whose energy is  

    E(t) = -β·∑_k ‖t - pattern_k‖² .

The fusion identifies the resource vector *eᵢ* with a sheaf section. The bandit‑propensity term *pᵢ* is used as a **regret‑weight** that biases a simulated‑annealing style acceptance probability for the DAM‑energy‑driven update. Thus the pipeline is:


entity → eᵢ (resource vector) → sheaf section
section ──► T·section = tₙ (ttt transformation)
tₙ ──► DAM → energy E(tₙ)
E(tₙ) ──► hybrid update with regret‑weighted acceptance (pᵢ)


The code below implements this unified system, providing three public functions that
demonstrate the hybrid operation:
    1. `calculate_resource_vector`
    2. `regret_weighted_distribution`
    3. `hybrid_step` (full end‑to‑end processing of a batch of entities).

"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Helper utilities from Algorithm A
# ----------------------------------------------------------------------
def haversine_distance(loc1: tuple[float, float], loc2: tuple[float, float]) -> float:
    """Return the haversine distance (metres) between two (lat, lon) points."""
    R = 6371000.0  # Earth radius in metres
    lat1, lon1 = math.radians(loc1[0]), math.radians(loc1[1])
    lat2, lon2 = math.radians(loc2[0]), math.radians(loc2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_resource_vector(entity: dict,
                              reference_location: tuple[float, float],
                              beta: float = 1.0) -> np.ndarray:
    """
    Compute the 3‑dimensional resource vector eᵢ for a single entity.

    Parameters
    ----------
    entity : dict
        Must contain keys ``'location'`` (lat,lon), ``'signature'`` (hashable)
        and ``'score'`` (float).
    reference_location : tuple[float, float]
        Latitude and longitude used as the distance anchor.
    beta : float, optional
        Scaling factor for the collision flag.

    Returns
    -------
    np.ndarray
        Vector ``[dᵢ, pᵢ, sᵢ]``.
    """
    distance = haversine_distance(entity["location"], reference_location)

    # Collision flag σᵢ: 1 if another entity shares the same signature, else 0.
    # The caller is responsible for providing the global signature map; we
    # assume the entity already carries its σᵢ value under key ``'collision'``.
    sigma = float(entity.get("collision", 0))
    pi = beta * sigma

    score = float(entity.get("score", 0.0))
    return np.array([distance, pi, score], dtype=np.float64)


# ----------------------------------------------------------------------
# Sheaf and Dense Associative Memory (Algorithm B)
# ----------------------------------------------------------------------
class Sheaf:
    """
    Minimal sheaf implementation.
    Nodes are identified by hashable keys; each node stores a section vector.
    Edges are stored only for bookkeeping – restrictions are optional.
    """
    def __init__(self, node_dims: dict, edges: list[tuple]):
        self.node_dims = node_dims            # {node: dimension}
        self.edges = edges                    # list of (src, dst) tuples
        self.sections: dict = {}              # {node: np.ndarray}
        self.restrictions: dict = {}          # {(src,dst): (src_map, dst_map)}

    def set_section(self, node, value: np.ndarray) -> None:
        expected_dim = self.node_dims.get(node)
        if expected_dim is None:
            raise KeyError(f"Node {node} not declared in node_dims.")
        if value.shape[0] != expected_dim:
            raise ValueError(f"Section dimension mismatch for node {node}: "
                             f"expected {expected_dim}, got {value.shape[0]}")
        self.sections[node] = value

    def get_section(self, node):
        return self.sections.get(node)

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        self.restrictions[edge] = (src_map, dst_map)


class DenseAssociativeMemory:
    """
    Simplified DAM: energy is negative β times the minimum squared distance
    to any stored pattern.
    """
    def __init__(self, patterns: np.ndarray, beta: float = 1.0):
        """
        patterns : shape (n_patterns, dim)
        """
        self.patterns = np.asarray(patterns, dtype=np.float64)
        self.beta = beta

    def compute_energy(self, vector: np.ndarray) -> float:
        """Return -β * min_k ||vector - pattern_k||²."""
        diffs = self.patterns - vector
        dists_sq = np.einsum('ij,ij->i', diffs, diffs)
        return -self.beta * np.min(dists_sq)


# ----------------------------------------------------------------------
# Regret‑weighted distribution (Algorithm B component)
# ----------------------------------------------------------------------
def regret_weighted_distribution(regrets: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert a vector of non‑negative regrets into a probability distribution.
    Uses a softmax over -regret / T, so higher regret ⇒ lower probability.

    Parameters
    ----------
    regrets : np.ndarray
        1‑D array of regret values (≥0).
    temperature : float
        Controls sharpness; larger T → flatter distribution.

    Returns
    -------
    np.ndarray
        Probabilities summing to 1.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = -regrets / temperature
    max_shift = np.max(scaled)  # for numerical stability
    exp_vals = np.exp(scaled - max_shift)
    probs = exp_vals / np.sum(exp_vals)
    return probs


# ----------------------------------------------------------------------
# Hybrid Model integrating both parents
# ----------------------------------------------------------------------
class HybridModel:
    """
    Core object that holds a Sheaf, a Dense Associative Memory, and
    implements the hybrid update rule that couples the resource vector
    (Algorithm A) with the sheaf‑DAM pipeline (Algorithm B).
    """
    def __init__(self,
                 node_dims: dict,
                 edges: list[tuple],
                 patterns: np.ndarray,
                 beta_dam: float = 1.0,
                 beta_pi: float = 1.0):
        self.sheaf = Sheaf(node_dims, edges)
        self.dam = DenseAssociativeMemory(patterns, beta=beta_dam)
        self.beta_pi = beta_pi          # scaling for the bandit propensity term pᵢ

    def embed_resource_vectors(self,
                               resource_vectors: dict[Any, np.ndarray]) -> None:
        """
        Store each entity's resource vector as a sheaf section.
        ``resource_vectors`` maps node identifiers to 3‑D vectors.
        """
        for node, vec in resource_vectors.items():
            self.sheaf.set_section(node, vec)

    def ttt_transform(self, node) -> np.ndarray | None:
        """
        Apply a random linear map (the “ttt” matrix) to the node's section.
        The matrix is regenerated on each call to keep the system stochastic,
        mirroring the original algorithm's behaviour.
        """
        section = self.sheaf.get_section(node)
        if section is None:
            return None
        rng = np.random.default_rng(hash(node) % (2**32))
        ttt = rng.standard_normal((section.shape[0], section.shape[0])) * 0.01
        return ttt @ section

    def compute_node_energy(self, node) -> float | None:
        """
        Transform the node's section and evaluate DAM energy.
        """
        transformed = self.ttt_transform(node)
        if transformed is None:
            return None
        return self.dam.compute_energy(transformed)

    def hybrid_update(self,
                      node,
                      target: np.ndarray,
                      regret: float,
                      temperature: float = 1.0) -> bool:
        """
        Simulated‑annealing style update for a node's section.

        Acceptance probability:
            a = min(1, exp( -(E_new - E_old - λ·pᵢ) / T ))
        where λ = β_pi and pᵢ is the propensity term (second component of the
        resource vector). The regret term shifts the effective temperature
        via ``regret_weighted_distribution``.

        Returns True if the update was accepted.
        """
        old_section = self.sheaf.get_section(node)
        if old_section is None:
            return False

        # Compute energies before and after a tentative replacement
        old_energy = self.compute_node_energy(node)
        # Temporarily set the new section to evaluate its energy
        self.sheaf.set_section(node, target)
        new_energy = self.compute_node_energy(node)

        # Restore old section in case of rejection
        self.sheaf.set_section(node, old_section)

        if old_energy is None or new_energy is None:
            return False

        # Extract propensity pᵢ (second component) from the old resource vector
        p_i = old_section[1] * self.beta_pi

        # Regret‑weighted temperature scaling
        regret_factor = regret_weighted_distribution(np.array([regret]), temperature)[0]
        effective_T = temperature * (1.0 + regret_factor)

        delta = new_energy - old_energy - p_i
        acceptance = math.exp(-delta / effective_T) if delta > 0 else 1.0
        if random.random() < acceptance:
            self.sheaf.set_section(node, target)
            return True
        return False


# ----------------------------------------------------------------------
# Public API functions demonstrating the hybrid operation
# ----------------------------------------------------------------------
def compute_resource_vectors(entities: list[dict],
                             reference_location: tuple[float, float],
                             beta: float = 1.0) -> dict[int, np.ndarray]:
    """
    Compute resource vectors for a list of entities and return a mapping
    from integer node ids (0 … N‑1) to vectors.
    """
    vectors = {}
    for idx, ent in enumerate(entities):
        vec = calculate_resource_vector(ent, reference_location, beta)
        vectors[idx] = vec
    return vectors


def hybrid_step(model: HybridModel,
                entities: list[dict],
                reference_location: tuple[float, float],
                beta: float = 1.0,
                temperature: float = 1.0) -> None:
    """
    One full hybrid iteration:
        1. Compute resource vectors.
        2. Embed them as sheaf sections.
        3. For each node, propose a random perturbation and apply the hybrid
           update rule using a regret value derived from the node's propensity.
    """
    # 1. Resource vectors
    res_vectors = compute_resource_vectors(entities, reference_location, beta)

    # 2. Embed into sheaf
    model.embed_resource_vectors(res_vectors)

    # 3. Node‑wise updates
    for node, vec in res_vectors.items():
        # Simple perturbation: add Gaussian noise
        rng = np.random.default_rng(hash(node) % (2**32))
        proposal = vec + rng.normal(scale=0.05, size=vec.shape)

        # Regret derived from the propensity term (higher pᵢ ⇒ higher regret)
        regret = vec[1]  # pᵢ itself serves as a regret proxy

        model.hybrid_update(node, proposal, regret, temperature)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy entities
    dummy_entities = [
        {"location": (37.7749, -122.4194), "signature": "A", "score": 0.8, "collision": 0},
        {"location": (34.0522, -118.2437), "signature": "B", "score": 0.6, "collision": 1},
        {"location": (40.7128, -74.0060),  "signature": "C", "score": 0.9, "collision": 0},
    ]

    ref_loc = (36.0, -120.0)

    # Simple pattern set for DAM (3 patterns, 3‑dimensional)
    patterns = np.array([
        [1000.0, 0.0, 0.5],
        [2000.0, 1.0, 0.7],
        [3000.0, 0.0, 0.9],
    ])

    # Node dimensions: each node stores a 3‑D vector
    node_dims = {i: 3 for i in range(len(dummy_entities))}
    edges = []  # No explicit edges needed for this demo

    hybrid_model = HybridModel(node_dims=node_dims,
                               edges=edges,
                               patterns=patterns,
                               beta_dam=0.5,
                               beta_pi=0.3)

    # Run a few hybrid steps to ensure no runtime errors
    for step in range(3):
        hybrid_step(hybrid_model,
                    dummy_entities,
                    reference_location=ref_loc,
                    beta=1.2,
                    temperature=0.8)
        # Print energies for inspection
        energies = [hybrid_model.compute_node_energy(n) for n in node_dims]
        print(f"Step {step+1} energies: {energies}")
    sys.exit(0)