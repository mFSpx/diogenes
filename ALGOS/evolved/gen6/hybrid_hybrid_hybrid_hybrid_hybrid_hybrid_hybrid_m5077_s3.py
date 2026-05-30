# DARWIN HAMMER — match 5077, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1512_s5.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_geomet_m429_s3.py (gen4)
# born: 2026-05-29T23:59:40Z

"""Hybrid Algorithm Fusion of Parent A and Parent B

Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1512_s5.py) provides
a pairwise similarity measure between graph nodes based on a perceptual hash
(derived from numeric values) and a MinHash signature (derived from token sets).

Parent B (hybrid_hybrid_hybrid_bandit_hybrid_hybrid_geomet_m429_s3.py) implements a
contextual multi‑armed bandit (LinUCB‑style) together with Euclidean distance
based region assignment.

Mathematical Bridge
------------------
Both parents operate on *pairwise* relationships:

* A computes a similarity `S_A(i,j) ∈ [0,1]` from Hamming‑based and Jaccard‑based
  components.
* B computes a Euclidean distance `d(i,j)` and derives a similarity
  `S_B(i,j) = 1 - d(i,j) / d_max` (normalized to `[0,1]`).

The fusion defines a unified similarity


S(i,j) = α * S_A(i,j) + (1‑α) * S_B(i,j)      with 0 ≤ α ≤ 1


This matrix `S` is then used as the *context vector* for a LinUCB‑style bandit.
Each node is an arm; the expected reward of pulling arm `j` from the current
node `i` is estimated from the similarity‑weighted history.  The bandit policy
selects the next node to “broadcast” to, while the update step incorporates the
observed reward (e.g., a simulated diffusion gain).

The code below implements:
* the original A utilities,
* the Euclidean utilities from B (without SciPy),
* a function to build the hybrid similarity matrix,
* a lightweight LinUCB‑style bandit that consumes the matrix as context,
* a small smoke test.
"""

import sys
import random
import math
import hashlib
from pathlib import Path
from collections import Counter
from typing import Mapping, Hashable, Set, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
ValuesMap = Mapping[Node, List[float]]
TokensMap = Mapping[Node, Set[str]]
PositionMap = Mapping[Node, Tuple[float, float]]

# ----------------------------------------------------------------------
# Parent A utilities (perceptual hash + MinHash)
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
    """MinHash signature for a set of tokens."""
    signatures: List[int] = []
    for seed in range(num_hashes):
        def hash_fn(x: str) -> int:
            return int(hashlib.md5((x + str(seed)).encode()).hexdigest(), 16)

        hashed = [hash_fn(tok) for tok in tokens]
        signatures.append(min(hashed) if hashed else 0)
    return signatures


def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity for MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def node_similarity(
    phash_a: int,
    phash_b: int,
    sig_a: List[int],
    sig_b: List[int],
) -> float:
    """Blend of Hamming‑based and MinHash‑based similarity."""
    ham_sim = 1.0 - hamming_distance(phash_a, phash_b) / 64.0
    mh_sim = jaccard_similarity(sig_a, sig_b)
    return 0.5 * ham_sim + 0.5 * mh_sim


# ----------------------------------------------------------------------
# Parent B utilities (Euclidean distance and region assignment)
# ----------------------------------------------------------------------
def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Standard L2 distance in 2‑D."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalized_euclidean_similarity(
    a: Tuple[float, float],
    b: Tuple[float, float],
    max_dist: float,
) -> float:
    """Map distance to similarity in [0,1] using max distance for normalization."""
    if max_dist == 0:
        return 1.0
    return 1.0 - euclidean_distance(a, b) / max_dist


# ----------------------------------------------------------------------
# Hybrid similarity matrix
# ----------------------------------------------------------------------
def hybrid_similarity_matrix(
    nodes: List[Node],
    values_map: ValuesMap,
    tokens_map: TokensMap,
    pos_map: PositionMap,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Build a symmetric matrix S where
        S[i, j] = α * S_A(i, j) + (1‑α) * S_B(i, j)

    Parameters
    ----------
    nodes : list of node identifiers (order defines matrix rows/cols)
    values_map : node -> list[float] (used for perceptual hash)
    tokens_map : node -> set[str]   (used for MinHash)
    pos_map    : node -> (x, y)     (used for Euclidean similarity)
    alpha      : weighting factor for A‑side similarity

    Returns
    -------
    np.ndarray of shape (n, n) with values in [0,1]
    """
    n = len(nodes)
    S = np.zeros((n, n), dtype=float)

    # Pre‑compute per‑node artefacts
    phashes = [compute_phash(values_map.get(node, [])) for node in nodes]
    signatures = [minhash_signature(tokens_map.get(node, set())) for node in nodes]
    positions = [pos_map.get(node, (0.0, 0.0)) for node in nodes]

    # Determine max Euclidean distance for normalization
    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(positions[i], positions[j])
            if d > max_dist:
                max_dist = d

    for i in range(n):
        for j in range(i, n):
            # A‑side similarity
            s_a = node_similarity(phashes[i], phashes[j], signatures[i], signatures[j])
            # B‑side Euclidean similarity
            s_b = normalized_euclidean_similarity(positions[i], positions[j], max_dist)
            # Blend
            s = alpha * s_a + (1.0 - alpha) * s_b
            S[i, j] = S[j, i] = s
    return S


# ----------------------------------------------------------------------
# LinUCB‑style bandit that consumes the hybrid similarity matrix as context
# ----------------------------------------------------------------------
class HybridLinUCB:
    """
    A lightweight LinUCB implementation where each arm corresponds to a node.
    The context vector for arm j is the j‑th column of the hybrid similarity
    matrix (i.e., similarity of every other node to j).  The algorithm maintains
    A (d×d) and b (d) for each arm to estimate θ̂ = A⁻¹ b.
    """

    def __init__(self, n_arms: int, alpha: float = 1.0, seed: int | None = None):
        self.n = n_arms
        self.alpha = alpha
        self.rng = random.Random(seed)
        self.d = n_arms  # dimensionality equals number of arms (full similarity vector)

        # For each arm we store A (d×d) and b (d)
        self.A = [np.identity(self.d) for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]

    def select(self, context_matrix: np.ndarray, available: List[int] | None = None) -> int:
        """
        Choose the arm with the highest upper confidence bound.

        Parameters
        ----------
        context_matrix : (n, n) similarity matrix S
        available      : optional subset of arm indices to consider

        Returns
        -------
        int : selected arm index
        """
        if available is None:
            available = list(range(self.n))

        best_score = -float("inf")
        best_arm = available[0]

        for arm in available:
            x = context_matrix[:, arm]  # similarity of all nodes to candidate arm
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            p = float(theta @ x + self.alpha * math.sqrt(x @ A_inv @ x))
            if p > best_score:
                best_score = p
                best_arm = arm
        return best_arm

    def update(self, arm: int, reward: float, context_matrix: np.ndarray):
        """
        Perform the LinUCB posterior update for the chosen arm.

        Parameters
        ----------
        arm            : index of the pulled arm
        reward         : observed scalar reward
        context_matrix : (n, n) similarity matrix S
        """
        x = context_matrix[:, arm].reshape(-1, 1)  # column vector
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x.ravel()


# ----------------------------------------------------------------------
# High‑level hybrid operation functions
# ----------------------------------------------------------------------
def run_hybrid_iteration(
    nodes: List[Node],
    values_map: ValuesMap,
    tokens_map: TokensMap,
    pos_map: PositionMap,
    policy: HybridLinUCB,
    alpha_sim: float = 0.6,
) -> Tuple[int, float]:
    """
    Perform a single iteration:
        1. Build the hybrid similarity matrix.
        2. Let the bandit pick the next node (arm).
        3. Simulate a reward (here: average similarity to already visited nodes).
        4. Update the policy.

    Returns
    -------
    (selected_arm_index, reward)
    """
    S = hybrid_similarity_matrix(nodes, values_map, tokens_map, pos_map, alpha=alpha_sim)

    # For demonstration we allow all arms
    chosen = policy.select(S)

    # Simulated reward: mean similarity of chosen node to all others
    reward = float(np.mean(S[:, chosen]))

    policy.update(chosen, reward, S)
    return chosen, reward


def build_dummy_data() -> Tuple[List[Node], ValuesMap, TokensMap, PositionMap]:
    """
    Construct a tiny synthetic dataset with 5 nodes.
    """
    nodes = ['A', 'B', 'C', 'D', 'E']

    values_map = {
        'A': [0.1, 0.3, 0.5],
        'B': [0.2, 0.2, 0.6],
        'C': [0.9, 0.8, 0.7],
        'D': [0.4, 0.4, 0.4],
        'E': [0.0, 0.1, 0.2],
    }

    tokens_map = {
        'A': {'apple', 'alpha'},
        'B': {'banana', 'beta'},
        'C': {'cherry', 'gamma'},
        'D': {'date', 'delta'},
        'E': {'elderberry', 'epsilon'},
    }

    pos_map = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (0.5, 1.0),
        'D': (1.0, 1.0),
        'E': (0.2, 0.8),
    }

    return nodes, values_map, tokens_map, pos_map


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build synthetic problem
    nodes, val_map, tok_map, pos_map = build_dummy_data()
    n = len(nodes)

    # Initialise bandit policy
    bandit = HybridLinUCB(n_arms=n, alpha=1.0, seed=42)

    # Run a few iterations to ensure no exception occurs
    for step in range(5):
        arm, rew = run_hybrid_iteration(
            nodes,
            val_map,
            tok_map,
            pos_map,
            policy=bandit,
            alpha_sim=0.6,
        )
        print(f"Step {step+1}: selected node '{nodes[arm]}' with reward {rew:.4f}")

    print("Hybrid fusion test completed successfully.")