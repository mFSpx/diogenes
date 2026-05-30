# DARWIN HAMMER — match 3737, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1512_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1753_s5.py (gen5)
# born: 2026-05-29T23:51:24Z

"""
Hybrid Algorithm integrating:
- Parent A: pheromone-based maximal independent set, MinHash similarity, simulated annealing acceptance.
- Parent B: Gaussian RBF surrogate model and Euclidean geometry.

Mathematical Bridge:
Both parents rely on a similarity measure that can be expressed as a kernel matrix.
Parent B provides a Gaussian RBF kernel K_{ij}=exp(-ε²‖x_i−x_j‖²) for any pair of
feature vectors.  We reuse this kernel as the “perceptual similarity” surface that
weights pheromone diffusion in Parent A and as the energy landscape for the
acceptance probability in the leader‑election step.  Thus the RBF kernel becomes
the common matrix linking pheromone update, MinHash‑style hashing, and the
simulated‑annealing decision rule.
"""

import sys
import random
import math
import pathlib
import hashlib
from dataclasses import dataclass
from typing import List, Set, Tuple, Mapping, Hashable, Sequence, Dict

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Perceptual hash: 1 bit per value indicating >= average (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return bin(a ^ b).count("1")

def minhash_signature(tokens: Set[str], num_hashes: int = 7) -> List[int]:
    """Very small MinHash implementation using MD5."""
    signatures: List[int] = []
    for seed in range(num_hashes):
        min_hash = None
        for token in tokens:
            h = hashlib.md5((token + str(seed)).encode()).hexdigest()
            hv = int(h, 16)
            if min_hash is None or hv < min_hash:
                min_hash = hv
        signatures.append(min_hash if min_hash is not None else 0)
    return signatures

def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Simulated‑annealing acceptance rule."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("Invalid cooling parameters")
    return t0 * (alpha ** k)

# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("Dimension mismatch")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def rbf_kernel_matrix(features: List[Vector], epsilon: float = 1.0) -> np.ndarray:
    """Compute the symmetric RBF kernel matrix K_{ij}=exp(-ε²‖x_i−x_j‖²)."""
    n = len(features)
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            r = euclidean(features[i], features[j])
            val = gaussian(r, epsilon)
            K[i, j] = K[j, i] = val
    return K

# ----------------------------------------------------------------------
# Core data structure from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """RBF surrogate prediction."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

# ----------------------------------------------------------------------
# Hybrid Functions (the required three+ functions)
# ----------------------------------------------------------------------
def compute_similarity_and_phash(
    node_features: Dict[Node, Vector]
) -> Tuple[np.ndarray, Dict[Node, int]]:
    """
    Build the RBF similarity matrix for all nodes (bridge to Parent B) and
    compute a perceptual hash for each node's feature vector (bridge to Parent A).

    Returns
    -------
    K : np.ndarray
        Symmetric similarity matrix K_{ij}=exp(-ε²‖f_i−f_j‖²).
    phash_map : dict
        Mapping node → 64‑bit perceptual hash of its feature values.
    """
    nodes = list(node_features.keys())
    feature_list = [list(node_features[n]) for n in nodes]

    # RBF similarity (Parent B)
    K = rbf_kernel_matrix(feature_list, epsilon=1.0)

    # Perceptual hash (Parent A)
    phash_map = {
        n: compute_phash(list(node_features[n])) for n in nodes
    }

    return K, phash_map


def hybrid_pheromone_update(
    graph: Graph,
    pheromones: Dict[Node, float],
    similarity: np.ndarray,
    decay: float = 0.9,
    alpha: float = 0.5,
) -> Dict[Node, float]:
    """
    Diffuse pheromones over the graph weighted by the RBF similarity matrix.
    The update rule blends:
        - decay of existing pheromone,
        - influx proportional to the similarity‑weighted sum of neighbours'
          pheromones (Parent A diffusion),
        - a global reinforcement term alpha * average_pheromone.

    Parameters
    ----------
    graph : adjacency mapping
    pheromones : current pheromone levels per node
    similarity : pre‑computed RBF kernel matrix aligned with graph node order
    decay, alpha : scalar hyper‑parameters

    Returns
    -------
    new_pheromones : dict with updated values
    """
    nodes = list(graph.keys())
    idx = {n: i for i, n in enumerate(nodes)}
    new_pheromones: Dict[Node, float] = {}

    avg_ph = np.mean(list(pheromones.values())) if pheromones else 0.0

    for n in nodes:
        i = idx[n]
        neigh = graph[n]
        # similarity‑weighted neighbour sum
        sim_sum = 0.0
        for nb in neigh:
            j = idx[nb]
            sim_sum += similarity[i, j] * pheromones.get(nb, 0.0)
        updated = decay * pheromones.get(n, 0.0) + alpha * (sim_sum / (len(neigh) or 1)) + alpha * avg_ph
        new_pheromones[n] = max(updated, 0.0)  # pheromones stay non‑negative
    return new_pheromones


def hybrid_leader_election(
    candidates: Set[Node],
    pheromones: Dict[Node, float],
    surrogate: RBFSurrogate,
    node_features: Dict[Node, Vector],
    temperature: float,
) -> Node:
    """
    Perform a probabilistic leader election.
    Energy of a candidate = - (pheromone level + surrogate prediction).
    The candidate with the lowest energy is the current best.
    A new candidate is accepted with the simulated‑annealing probability.

    Returns the elected leader node.
    """
    # Compute energies
    energies: Dict[Node, float] = {}
    for n in candidates:
        phi = pheromones.get(n, 0.0)
        pred = surrogate.predict(list(node_features[n]))
        energies[n] = - (phi + pred)  # lower energy = more attractive

    # Current best
    current = min(energies, key=energies.get)
    current_energy = energies[current]

    # Propose a random alternative
    alternative = random.choice(list(candidates - {current}))
    alt_energy = energies[alternative]

    delta_e = alt_energy - current_energy
    if acceptance_probability(delta_e, temperature) > random.random():
        return alternative
    else:
        return current


def hybrid_bandit_reward(
    node: Node,
    pheromones: Dict[Node, float],
    surrogate: RBFSurrogate,
    node_features: Dict[Node, Vector],
    exploration_coef: float = 0.1,
) -> float:
    """
    Estimate a bandit‑style reward for a node.
    Reward = surrogate prediction + exploration term based on pheromone variance.
    """
    pred = surrogate.predict(list(node_features[node]))
    # Exploration term: high variance in neighbour pheromones encourages sampling
    neigh_pher = [pheromones.get(nb, 0.0) for nb in node_features.get(node, [])]
    variance = np.var(neigh_pher) if neigh_pher else 0.0
    return pred + exploration_coef * math.sqrt(variance)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic graph
    G: Graph = {
        0: {1, 2},
        1: {0, 2, 3},
        2: {0, 1, 3},
        3: {1, 2},
    }

    # Random 2‑D features for each node
    random.seed(42)
    node_feats: Dict[int, Vector] = {
        n: (random.random(), random.random()) for n in G
    }

    # Initialise pheromones uniformly
    pher: Dict[int, float] = {n: 1.0 for n in G}

    # Build a tiny RBF surrogate with random centres/weights
    centers = [tuple(random.random() for _ in range(2)) for _ in range(3)]
    weights = [random.uniform(-1, 1) for _ in range(3)]
    surrogate = RBFSurrogate(centers=centers, weights=weights, epsilon=1.5)

    # 1. Compute similarity matrix and perceptual hashes
    K, phash = compute_similarity_and_phash(node_feats)

    # 2. Update pheromones using the hybrid diffusion rule
    pher = hybrid_pheromone_update(G, pher, K, decay=0.85, alpha=0.3)

    # 3. Run a few leader election steps with cooling temperature
    temperature = cooling_temperature(k=0, t0=1.0, alpha=0.9)
    leader = hybrid_leader_election(set(G.keys()), pher, surrogate, node_feats, temperature)

    # 4. Compute a bandit reward for the elected leader
    reward = hybrid_bandit_reward(leader, pher, surrogate, node_feats)

    # Print results (no external libraries used)
    print("Perceptual hashes:", phash)
    print("Updated pheromones:", pher)
    print("Selected leader:", leader)
    print("Leader reward estimate:", reward)