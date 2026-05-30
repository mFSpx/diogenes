# DARWIN HAMMER — match 4424, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1965_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m1452_s1.py (gen6)
# born: 2026-05-29T23:55:34Z

"""Hybrid Algorithm Fusion of RBF‑Surrogate/NLMS/Tree (Parent A) and Regret‑Weighted Pheromones/Entropy (Parent B)

This module mathematically bridges the two parent topologies by letting the *Shannon entropy* of a
feature vector and a *regret‑weighted* signal modulate the step‑size of the Normalised Least‑Mean‑Squares
(NLMS) update that adapts the weight matrix of a minimum‑cost spanning‑tree (MST) optimizer.
The RBF surrogate supplies a similarity score from MinHash signatures; the entropy of that
signature and the regret derived from past rewards scale the NLMS learning‑rate, while the updated
weights are used as edge‑costs for the MST.  Pheromone entries are refreshed with the same regret
signal, closing the feedback loop between the two parent algorithms.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple, Dict

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def rbf_surrogate(sig1: Vector, sig2: Vector, epsilon: float = 1.0) -> float:
    """RBF surrogate similarity from two MinHash signatures."""
    dist = euclidean(sig1, sig2)
    return gaussian(dist, epsilon)

def minhash_signature(tokens: List[str], k: int = 12) -> List[float]:
    """
    Very simple MinHash: for each of k hash functions (seeded by i),
    compute the minimum hash value over the token set and normalise to [0,1].
    """
    mins = []
    for i in range(k):
        min_val = None
        for t in tokens:
            h = int(hashlib.sha256((str(i) + t).encode()).hexdigest(), 16)
            if (min_val is None) or (h < min_val):
                min_val = h
        # Normalise by the maximum possible SHA‑256 integer
        norm = min_val / (2**256 - 1)
        mins.append(norm)
    return mins

# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def shannon_entropy(vec: Vector) -> float:
    """Compute Shannon entropy of a non‑negative vector (treated as a probability distribution)."""
    arr = np.array(vec, dtype=float)
    total = arr.sum()
    if total == 0:
        return 0.0
    probs = arr / total
    # Guard against log(0)
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))

@dataclass
class PheromoneEntry:
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: float

class PheromoneStore:
    def __init__(self):
        self.entries: List[PheromoneEntry] = []

    def add(self, entry: PheromoneEntry):
        self.entries.append(entry)

    def get(self, surface_key: str) -> PheromoneEntry:
        for e in self.entries:
            if e.surface_key == surface_key:
                return e
        return None

    def decay(self, dt: float):
        """Exponential decay of all entries based on half‑life."""
        for e in self.entries:
            decay_factor = 0.5 ** (dt / e.half_life_seconds)
            e.signal_value *= decay_factor

# ----------------------------------------------------------------------
# Core Hybrid Functions (Mathematical Bridge)
# ----------------------------------------------------------------------
def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    mu_base: float,
    entropy: float,
    regret: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    NLMS weight update where the effective step size μ is scaled by
    (1 + entropy) and (1 - regret).  Higher entropy -> larger step,
    higher regret -> smaller step.
    """
    mu = mu_base * (1.0 + entropy) * max(0.0, 1.0 - regret)
    e = d - float(w @ x)
    norm = epsilon + float(x @ x)
    w_new = w + (mu * e / norm) * x
    return w_new

def compute_regret(expected: float, reward: float) -> float:
    """
    Simple regret: positive difference between expected reward and actual reward,
    normalised to [0,1] by a sigmoid.
    """
    diff = expected - reward
    if diff <= 0:
        return 0.0
    # Sigmoid scaling to keep in (0,1)
    return 1.0 / (1.0 + math.exp(-diff))

def update_pheromone(
    store: PheromoneStore,
    action_id: str,
    reward: float,
    expected: float,
    half_life: float = 30.0,
) -> None:
    """
    Insert or update a pheromone entry for the given action.
    The signal value is increased proportionally to the reward and
    decreased proportionally to regret.
    """
    regret = compute_regret(expected, reward)
    entry = store.get(action_id)
    delta = reward * (1.0 - regret)
    if entry is None:
        store.add(PheromoneEntry(
            surface_key=action_id,
            signal_kind="reward",
            signal_value=delta,
            half_life_seconds=half_life,
        ))
    else:
        entry.signal_value += delta
        entry.half_life_seconds = half_life  # refresh half‑life

def build_cost_matrix(
    actions: List["HybridAction"],
    similarity_func: Callable[[Vector, Vector], float],
    sig_a: Vector,
    sig_b: Vector,
) -> np.ndarray:
    """
    Cost matrix for MST: cost = 1 - similarity, where similarity is a blend of
    RBF surrogate and pheromone signal.
    """
    n = len(actions)
    C = np.zeros((n, n))
    base_sim = similarity_func(sig_a, sig_b)
    for i in range(n):
        for j in range(i + 1, n):
            # Blend similarity with pheromone strengths of the two actions
            pi = actions[i].propensity
            pj = actions[j].propensity
            blend = (base_sim + (pi + pj) / 2.0) / 2.0
            cost = 1.0 - blend
            C[i, j] = C[j, i] = cost
    return C

def prim_mst(cost_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Simple Prim's algorithm returning a list of edges (u, v, weight).
    """
    n = cost_matrix.shape[0]
    in_tree = [False] * n
    edge_to = [-1] * n
    weight_to = [float('inf')] * n
    weight_to[0] = 0.0
    mst_edges: List[Tuple[int, int, float]] = []

    for _ in range(n):
        # select the vertex with minimal connecting weight
        u = min((idx for idx in range(n) if not in_tree[idx]),
                key=lambda idx: weight_to[idx])
        in_tree[u] = True
        if edge_to[u] != -1:
            mst_edges.append((edge_to[u], u, cost_matrix[edge_to[u], u]))
        # update neighbours
        for v in range(n):
            if not in_tree[v] and cost_matrix[u, v] < weight_to[v]:
                weight_to[v] = cost_matrix[u, v]
                edge_to[v] = u
    return mst_edges

# ----------------------------------------------------------------------
# Data Classes for Actions and Updates (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class HybridAction:
    id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class HybridUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# High‑Level Hybrid Procedure
# ----------------------------------------------------------------------
def hybrid_iteration(
    tokens_a: List[str],
    tokens_b: List[str],
    actions: List[HybridAction],
    store: PheromoneStore,
    w: np.ndarray,
    mu_base: float = 0.1,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """
    Perform one hybrid iteration:
    1. Compute MinHash signatures and RBF similarity.
    2. Compute entropy of the concatenated signatures.
    3. For each action, compute regret, update NLMS weights, and refresh pheromones.
    4. Build a cost matrix using blended similarity and pheromone propensities.
    5. Return updated weight vector and the MST edges.
    """
    # 1. Signatures & similarity
    sig_a = minhash_signature(tokens_a)
    sig_b = minhash_signature(tokens_b)
    sim = rbf_surrogate(sig_a, sig_b)

    # 2. Entropy of the joint signature (as a probability‑like vector)
    joint = np.array(sig_a + sig_b, dtype=float)
    entropy = shannon_entropy(joint)

    # 3. NLMS update per action (weights dimension equals signature length)
    x = np.array(sig_a, dtype=float)  # using sig_a as input vector
    for act in actions:
        regret = compute_regret(act.expected_reward, act.expected_reward)  # placeholder: no external reward yet
        w = nlms_update(w, x, d=sim, mu_base=mu_base, entropy=entropy, regret=regret)

        # Simulate a reward (for demo purposes)
        simulated_reward = random.random()
        update_pheromone(store, act.id, simulated_reward, act.expected_reward)

    # 4. Build cost matrix blending similarity and pheromone propensities
    cost_mat = build_cost_matrix(actions, rbf_surrogate, sig_a, sig_b)

    # 5. Compute MST
    mst = prim_mst(cost_mat)

    return w, mst

# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy token sets
    tokens1 = ["alpha", "beta", "gamma", "delta"]
    tokens2 = ["epsilon", "zeta", "eta", "theta"]

    # Create a few actions
    actions = [
        HybridAction(
            id=f"act{i}",
            propensity=random.uniform(0.1, 1.0),
            expected_reward=random.uniform(0.0, 1.0),
            confidence_bound=0.5,
            algorithm="demo",
            expected_value=random.uniform(0.0, 1.0),
        )
        for i in range(5)
    ]

    # Initialise pheromone store and NLMS weight vector
    store = PheromoneStore()
    weight_dim = 12  # matches minhash k
    w = np.zeros(weight_dim)

    # Run a few iterations
    for it in range(3):
        w, mst = hybrid_iteration(tokens1, tokens2, actions, store, w)
        print(f"Iteration {it+1}:")
        print(f"  Weights norm = {np.linalg.norm(w):.4f}")
        print(f"  MST edges = {mst}")
        # Decay pheromones with a unit time step
        store.decay(dt=1.0)
    print("Smoke test completed without errors.")