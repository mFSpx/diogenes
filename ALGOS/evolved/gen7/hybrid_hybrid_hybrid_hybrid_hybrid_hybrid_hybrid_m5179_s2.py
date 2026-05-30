# DARWIN HAMMER — match 5179, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m555_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m2045_s0.py (gen6)
# born: 2026-05-30T00:00:19Z

"""Hybrid Fusion of Hybrid_Hybrid_Hybrid_Hybrid_Hybrid_Hybrid_Model (Sheaf + Dense Associative Memory) 
and Hybrid_Hybrid_Hybrid_Bandit (Gaussian Beam, Variational Free Energy, Pheromone Modulation).

Mathematical Bridge
------------------
- The *section vectors* of the Sheaf are interpreted as *policy vectors*.
- A LeaderElection process extracts a *leader signature* L from these policies.
- The *pheromone vector* P stored in a simple honey‑bee store is compared to L using a
  MinHash‑like similarity σ = 1 – mean|L – P| (higher σ ⇒ more similar).
- σ modulates the *effective temperature* of the simulated‑annealing acceptance rule:
      T_eff = T / (1 + λ·σ)
- The *energy* of a node combines two contributions:
      ΔE = ε – G
  where ε is a Hoeffding‑type bound (approximated by the variance of the policy)
  and G is a tropical gain derived from the variational free energy between the
  leader signature and the pheromone vector.
- Acceptance probability follows the Boltzmann form:
      p_accept = exp( –ΔE / T_eff )
- A regret‑weighted probability distribution (based on the regret of the current
  energy w.r.t. the best observed energy) further scales the acceptance, linking
  the Regret‑Weighted Tree of the original parent A with the bandit‑style updates
  of parent B.

The code below implements this fused system, providing three core functions that
demonstrate the hybrid operation:
    1. `compute_combined_energy`
    2. `acceptance_probability`
    3. `hybrid_anneal_step`
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Minimal building blocks extracted from the parents
# ----------------------------------------------------------------------
class Sheaf:
    """Simple sheaf structure holding node sections and edge restrictions."""
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims            # node -> dimension
        self.edges = edges                    # list of (src, dst)
        self.sections = {}                    # node -> np.ndarray
        self.restrictions = {}                # edge -> (src_map, dst_map)

    def set_section(self, node, value: np.ndarray):
        if node not in self.node_dims:
            raise KeyError(f"Unknown node {node}")
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("Dimension mismatch for node section")
        self.sections[node] = value

    def get_section(self, node):
        return self.sections.get(node, None)

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray):
        if edge not in self.edges:
            raise KeyError(f"Edge {edge} not defined")
        self.restrictions[edge] = (src_map, dst_map)


class DenseAssociativeMemory:
    """Energy based on distance to nearest stored pattern."""
    def __init__(self, patterns: np.ndarray, beta: float = 1.0):
        self.patterns = patterns      # shape (n_patterns, dim)
        self.beta = beta

    def energy(self, vector: np.ndarray) -> float:
        # negative log‑likelihood like energy
        dists = np.linalg.norm(self.patterns - vector, axis=1)
        min_dist = np.min(dists)
        return self.beta * min_dist


class LeaderElection:
    """Select the leader as the section with maximal L2 norm."""
    def elect(self, sections: dict) -> np.ndarray:
        if not sections:
            raise ValueError("No sections to elect from")
        # Return the vector with largest norm
        leader = max(sections.values(), key=lambda v: np.linalg.norm(v))
        return leader.copy()


class PheromoneStore:
    """Stores a pheromone vector that can be updated by bandit‑style rules."""
    def __init__(self, dim: int):
        self.vector = np.zeros(dim)

    def update(self, delta: np.ndarray, learning_rate: float = 0.1):
        self.vector = (1 - learning_rate) * self.vector + learning_rate * delta


# ----------------------------------------------------------------------
# Functions from Parent B (Bandit side)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam and alpha must be non-negative")
    return math.exp(-lam * t ** alpha)


def variational_free_energy(policy: np.ndarray, pheromone: np.ndarray) -> float:
    """
    Approximate variational free energy using an L1‑based similarity.
    The similarity metric is the mean absolute difference (a simple MinHash proxy).
    """
    if policy.shape != pheromone.shape:
        raise ValueError("policy and pheromone must have the same shape")
    similarity = 1.0 - np.mean(np.abs(policy - pheromone))   # in [0,1]
    # Avoid log(0)
    eps = 1e-12
    return -math.log(max(similarity, eps))


# ----------------------------------------------------------------------
# Hybrid core functions (the required three)
# ----------------------------------------------------------------------
def compute_combined_energy(sheaf: Sheaf,
                            dam: DenseAssociativeMemory,
                            leader_election: LeaderElection,
                            pheromone_store: PheromoneStore,
                            node: any,
                            lambda_gain: float = 0.5) -> float:
    """
    Compute ΔE = ε - G for a given node.

    ε (epsilon)  : Hoeffding‑type bound approximated by the variance of the node's section.
    G (gain)    : Tropical gain derived from variational free energy between the elected leader
                  and the pheromone vector, scaled by λ_gain.
    """
    section = sheaf.get_section(node)
    if section is None:
        raise ValueError(f"No section defined for node {node}")

    # ε – use variance as a proxy for Hoeffding bound
    epsilon = np.var(section)

    # Leader signature L
    leader = leader_election.elect(sheaf.sections)

    # G – tropical gain (negative free energy)
    free_energy = variational_free_energy(leader, pheromone_store.vector)
    gain = lambda_gain * (-free_energy)   # tropical gain is negative free energy

    delta_e = epsilon - gain
    return delta_e


def acceptance_probability(delta_e: float,
                           temperature: float,
                           similarity_sigma: float,
                           lambda_temp: float = 0.3) -> float:
    """
    Compute the Boltzmann acceptance probability with temperature modulation.

    T_eff = T / (1 + λ_temp * σ)
    p = exp( -ΔE / T_eff )
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    t_eff = temperature / (1.0 + lambda_temp * similarity_sigma)
    # Guard against overflow
    exponent = -delta_e / t_eff
    # Clip exponent to avoid underflow/overflow
    exponent = max(min(exponent, 700), -700)
    return math.exp(exponent)


def hybrid_anneal_step(model,
                       node: any,
                       temperature: float,
                       lambda_gain: float = 0.5,
                       lambda_temp: float = 0.3,
                       learning_rate: float = 0.1) -> bool:
    """
    Perform a single simulated‑annealing step on `node`.

    Returns True if the new section was accepted, False otherwise.
    """
    # Current section and its energy
    current_section = model.sheaf.get_section(node).copy()
    current_energy = compute_combined_energy(
        model.sheaf,
        model.dense_associative_memory,
        model.leader_election,
        model.pheromone_store,
        node,
        lambda_gain=lambda_gain,
    )

    # Propose a new section via a Gaussian perturbation (bandit‑style exploration)
    perturbation = np.random.normal(scale=0.05, size=current_section.shape)
    proposed_section = current_section + perturbation
    model.sheaf.set_section(node, proposed_section)

    # Energy of the proposal
    proposed_energy = compute_combined_energy(
        model.sheaf,
        model.dense_associative_memory,
        model.leader_election,
        model.pheromone_store,
        node,
        lambda_gain=lambda_gain,
    )

    delta_e = proposed_energy - current_energy

    # Similarity σ between leader and pheromone (used for temperature scaling)
    leader = model.leader_election.elect(model.sheaf.sections)
    sigma = 1.0 - np.mean(np.abs(leader - model.pheromone_store.vector))

    p_accept = acceptance_probability(delta_e, temperature, sigma, lambda_temp=lambda_temp)

    # Regret‑weighted scaling (regret = max(0, proposed_energy - best_energy_seen))
    if not hasattr(model, "best_energy"):
        model.best_energy = current_energy
    regret = max(0.0, proposed_energy - model.best_energy)
    regret_weight = math.exp(-regret)  # higher regret ⇒ lower weight
    p_accept *= regret_weight

    if random.random() < p_accept:
        # Accept: update best energy and pheromone store
        if proposed_energy < model.best_energy:
            model.best_energy = proposed_energy
        # Update pheromone toward the accepted leader
        model.pheromone_store.update(leader, learning_rate=learning_rate)
        return True
    else:
        # Reject: revert section
        model.sheaf.set_section(node, current_section)
        return False


# ----------------------------------------------------------------------
# Wrapper class that assembles all components
# ----------------------------------------------------------------------
class HybridFusionModel:
    def __init__(self,
                 node_dims: dict,
                 edges: list,
                 patterns: np.ndarray,
                 beta: float = 1.0,
                 pheromone_dim: int = None):
        self.sheaf = Sheaf(node_dims, edges)
        self.dense_associative_memory = DenseAssociativeMemory(patterns, beta)
        self.leader_election = LeaderElection()
        dim = pheromone_dim if pheromone_dim is not None else max(node_dims.values())
        self.pheromone_store = PheromoneStore(dim)

    # Helper to initialise a node with a random section
    def init_random_section(self, node, seed=None):
        rng = np.random.default_rng(seed)
        dim = self.sheaf.node_dims[node]
        vec = rng.standard_normal(dim)
        self.sheaf.set_section(node, vec)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    node_dims = {"A": 4, "B": 4}
    edges = [("A", "B")]

    # Random patterns for the associative memory
    patterns = np.array([[0.5, -0.2, 0.1, 0.0],
                         [-0.3, 0.8, -0.5, 0.2]])

    model = HybridFusionModel(node_dims, edges, patterns, beta=1.0)

    # Initialise sections
    model.init_random_section("A", seed=42)
    model.init_random_section("B", seed=43)

    # Run a few annealing steps
    temperature = 1.0
    for step in range(10):
        accepted_A = hybrid_anneal_step(model, "A", temperature)
        accepted_B = hybrid_anneal_step(model, "B", temperature)
        temperature *= 0.95  # cooling schedule
        print(f"Step {step+1:2d}: accept A={accepted_A}, accept B={accepted_B}, "
              f"best_energy={model.best_energy:.4f}")

    print("Final pheromone vector:", model.pheromone_store.vector)