# DARWIN HAMMER — match 2132, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m536_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_fisher_m22_s3.py (gen4)
# born: 2026-05-29T23:40:56Z

"""Hybrid Algorithm combining Sheaf‑Cohomology weighted tree scoring (Parent A) with
Pheromone‑Entropy and Fisher‑Information analysis (Parent B).

Mathematical bridge:
- Each node carries a sheaf section vector  s_i . Its Euclidean norm ‖s_i‖ is used
  as the *center* parameter of a Gaussian beam.
- For an edge (i, j) we obtain a pheromone probability p_{ij}. This probability
  serves as the *θ* argument of the Gaussian beam.
- The Gaussian intensity I_{ij}=gaussian_beam(p_{ij},‖s_i‖,w) (with a global
  width w) is interpreted as a likelihood for the edge.
- Fisher information F_{ij}=fisher_score(p_{ij},‖s_i‖,w) quantifies the
  uncertainty of that likelihood.
- A Bayesian update combines a prior edge weight w_{ij}^{prior} (derived from
  the sheaf restriction maps) with the likelihood I_{ij} to produce a posterior
  weight w_{ij}^{post}=bayesian_update(w_{ij}^{prior},I_{ij}).
- Finally a tree score aggregates the posterior weights, penalised by the
  entropy of the pheromone distribution over all edges.

The three core functions below demonstrate this fused pipeline."""
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Sheaf infrastructure (from Parent A)
# ----------------------------------------------------------------------
class Sheaf:
    """Simple sheaf storing node sections and edge restriction maps."""

    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        """
        node_dims: mapping node -> dimension of its section vector
        edge_list: list of (u, v) edges
        """
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}

    def set_section(self, node: int, value: List[float]) -> None:
        """Assign a section vector to a node."""
        dim = self.node_dims.get(node, len(value))
        arr = np.array(value, dtype=float)
        if arr.shape[0] != dim:
            raise ValueError(f"Section dimension mismatch for node {node}")
        self._sections[node] = arr

    def set_restriction(
        self, edge: Tuple[int, int], src_map: List[float], dst_map: List[float]
    ) -> None:
        """Define linear restriction maps for an edge."""
        u, v = edge
        self._restrictions[(u, v)] = (np.array(src_map, dtype=float), np.array(dst_map, dtype=float))

    def get_section(self, node: int) -> np.ndarray:
        return self._sections[node]

    def edge_prior_weight(self, u: int, v: int) -> float:
        """
        Produce a scalar prior weight for edge (u,v) using the restriction maps.
        We simply take the L2 norm of the mapped source section.
        """
        if (u, v) not in self._restrictions:
            raise KeyError(f"No restriction defined for edge {(u, v)}")
        src_map, _ = self._restrictions[(u, v)]
        src_sec = self.get_section(u)
        mapped = src_map @ src_sec
        return float(np.linalg.norm(mapped))


# ----------------------------------------------------------------------
# Pheromone & Entropy (from Parent B)
# ----------------------------------------------------------------------
def calculate_pheromone_probabilities(limit: int) -> List[float]:
    """Generate a normalized random pheromone distribution of given size."""
    pheromones = [random.random() for _ in range(limit)]
    total = sum(pheromones)
    if total == 0:
        raise ValueError("Pheromone total mass must be positive")
    return [p / total for p in pheromones]


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = [p / total for p in probabilities if p > 0]
    return -sum(p * math.log(max(p, eps)) for p in probs)


# ----------------------------------------------------------------------
# Gaussian beam, Fisher information (from Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile I(θ)."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam evaluated at θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / max(intensity, eps)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_hybrid_edge_weight(
    sheaf: Sheaf,
    u: int,
    v: int,
    pheromone_prob: float,
    beam_width: float = 0.1,
) -> Tuple[float, float]:
    """
    Return (posterior_weight, fisher_information) for edge (u,v).

    Steps:
    1. Prior weight w_prior from sheaf restriction maps.
    2. Center = ||section_u|| (norm of node u's sheaf section).
    3. Likelihood = Gaussian intensity I(θ=pheromone_prob, center, width).
    4. Posterior weight = Bayesian update of w_prior with likelihood.
    5. Fisher information computed at the same parameters.
    """
    # 1. prior
    w_prior = sheaf.edge_prior_weight(u, v)

    # 2. center from node u's section norm
    sec_u = sheaf.get_section(u)
    center = float(np.linalg.norm(sec_u))

    # 3. likelihood from Gaussian beam
    likelihood = gaussian_beam(pheromone_prob, center, beam_width)

    # 4. Bayesian update (simple proportional update)
    posterior = bayesian_update(w_prior, likelihood)

    # 5. Fisher information
    fisher = fisher_score(pheromone_prob, center, beam_width)

    return posterior, fisher


def bayesian_update(prior: float, likelihood: float, epsilon: float = 1e-12) -> float:
    """
    Perform a minimal Bayesian update assuming a uniform evidence term.
    posterior ∝ prior * likelihood.
    """
    unnorm = prior * likelihood
    # Normalise to keep scale comparable; we use (prior + likelihood) as a proxy for evidence.
    evidence = prior + likelihood + epsilon
    return unnorm / evidence


def evaluate_hybrid_tree(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    sheaf: Sheaf,
    beam_width: float = 0.1,
) -> float:
    """
    Compute a global score for a tree defined by `edges`.
    The score aggregates posterior edge weights, penalised by the entropy of the
    pheromone distribution over those edges and the total Fisher information.
    """
    if not edges:
        raise ValueError("Tree must contain at least one edge")

    # Generate a pheromone probability for each edge
    pher_probs = calculate_pheromone_probabilities(len(edges))

    total_posterior = 0.0
    total_fisher = 0.0
    for (u, v), p in zip(edges, pher_probs):
        post, fisher = compute_hybrid_edge_weight(sheaf, u, v, p, beam_width)
        total_posterior += post
        total_fisher += fisher

    # Entropy term (higher entropy -> larger penalty)
    ent = entropy(pher_probs)

    # Final hybrid score: higher posterior & lower entropy & lower Fisher (more certain)
    score = total_posterior - ent - total_fisher
    return score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph (3 nodes, 2 edges forming a tree)
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2)]

    # Create a sheaf with 2‑dimensional sections for each node
    node_dims = {0: 2, 1: 2, 2: 2}
    sheaf = Sheaf(node_dims, edges)

    # Assign random sections
    random.seed(42)
    for n in nodes:
        vec = [random.uniform(-1, 1) for _ in range(node_dims[n])]
        sheaf.set_section(n, vec)

    # Define simple restriction maps (identity scaled)
    for (u, v) in edges:
        dim = node_dims[u]
        src_map = np.identity(dim).flatten().tolist()  # identity
        dst_map = np.identity(dim).flatten().tolist()
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Evaluate the hybrid tree score
    score = evaluate_hybrid_tree(nodes, edges, sheaf, beam_width=0.2)
    print(f"Hybrid tree score: {score:.6f}")