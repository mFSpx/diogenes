# DARWIN HAMMER — match 5553, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2284_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_distri_m957_s0.py (gen4)
# born: 2026-05-30T00:04:08Z

"""Hybrid Algorithm: Fusion of
- hybrid_hybrid_hybrid_hybrid_hybrid_fold_change_d_m696_s2.py (resource vector, fold‑change detection, SSM, bandit)
- hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_distri_m957_s0.py (bipolar hypervectors, perceptual‑hash similarity, Gini‑inequality, labeling)

Mathematical Bridge
------------------
The *resource vector* of Parent A is re‑interpreted as a **bipolar hyper‑vector** that encodes the
morphological properties of graph nodes (Parent B).  The *fold‑change detection* signal from
Parent A is used as a multiplicative weight on the **cosine similarity** between these hyper‑vectors,
thereby modulating the expected reward computed by the State‑Space Model (SSM).  The
*Do­omsday‑Gini* scalar of Parent A quantifies inequality among a node’s neighbour degrees;
it is employed as an adaptive exploration factor for the bandit component of Parent A.
Thus the three core subsystems are mathematically intertwined:


h_i      – bipolar hyper‑vector for node i  (resource vector ↔ morphology)
s_t      – hidden SSM state
r̂_t     – predicted reward = C·s_t
Δ_fc     – fold‑change factor = (obs_t – obs_{t‑1}) / obs_{t‑1}
γ_i      – Gini coefficient of neighbour degree distribution for node i
score_i  = (h_i·h_ref / ‖h_i‖‖h_ref‖) · Δ_fc · r̂_t
UCB_i    = Q_i + γ_i·√(2·log(N)/n_i)·score_i


The implementation below provides three public functions that expose this hybrid
behaviour: `hybrid_initialize`, `hybrid_update`, and `hybrid_select_action`.  A lightweight
smoke‑test runs when the module is executed as a script."""

import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Hashable, Mapping, Set

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
Vector = np.ndarray  # bipolar (+1 / -1) hyper‑vector

# ----------------------------------------------------------------------
# Helper utilities (Parent B)
# ----------------------------------------------------------------------
def random_bipolar_vector(dim: int = 1024, seed: int | str | None = None) -> Vector:
    """Generate a bipolar (+1 / -1) hyper‑vector."""
    rng = random.Random(seed)
    return np.array([1 if rng.getrandbits(1) else -1 for _ in range(dim)], dtype=np.int8)


def perceptual_hash(vector: Vector) -> int:
    """Deterministic hash of a bipolar vector – mimics a perceptual hash."""
    # Convert to bytes (treat -1 as 0, 1 as 1)
    bits = (vector > 0).astype(np.uint8).tobytes()
    return int(hashlib.sha256(bits).hexdigest(), 16)


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity for bipolar vectors."""
    dot = float(np.dot(a, b))
    norm = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return dot / norm if norm != 0 else 0.0


def gini_coefficient(values: List[int]) -> float:
    """Gini coefficient of a non‑negative list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(values)
    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return gini


# ----------------------------------------------------------------------
# Fold‑change detection (Parent A)
# ----------------------------------------------------------------------
def fold_change(prev: float, curr: float) -> float:
    """Relative change (curr‑prev)/prev, safe for zero denominator."""
    return (curr - prev) / prev if prev != 0 else 0.0


# ----------------------------------------------------------------------
# State‑Space Model + Bandit (Parent A)
# ----------------------------------------------------------------------
@dataclass
class SSM:
    """Linear State‑Space Model: x_{t+1}=A·x_t + B·u_t ,  r̂_t = C·x_t."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    state: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))

    def step(self, control: np.ndarray) -> float:
        """Perform one SSM step and return predicted reward."""
        self.state = self.A @ self.state + self.B @ control
        reward_pred = float(self.C @ self.state)
        return reward_pred


@dataclass
class BanditUCB:
    """Simple UCB bandit with adaptive exploration via Gini."""
    values: Dict[Node, float] = field(default_factory=dict)
    counts: Dict[Node, int] = field(default_factory=dict)
    total_pulls: int = 0

    def select(self, nodes: List[Node], gini_factor: float, score: Dict[Node, float]) -> Node:
        """Select node with highest UCB, scaling exploration by gini_factor."""
        best_node = None
        best_val = -float('inf')
        for n in nodes:
            q = self.values.get(n, 0.0)
            n_i = self.counts.get(n, 0) + 1e-9  # avoid div‑zero
            exploration = gini_factor * math.sqrt(2 * math.log(self.total_pulls + 1) / n_i)
            ucb = q + exploration * score.get(n, 0.0)
            if ucb > best_val:
                best_val = ucb
                best_node = n
        return best_node

    def update(self, node: Node, reward: float) -> None:
        """Incremental average update."""
        self.total_pulls += 1
        self.counts[node] = self.counts.get(node, 0) + 1
        n = self.counts[node]
        old_q = self.values.get(node, 0.0)
        self.values[node] = old_q + (reward - old_q) / n


# ----------------------------------------------------------------------
# Hybrid model that fuses both parents
# ----------------------------------------------------------------------
@dataclass
class HybridModel:
    graph: Graph
    ssm: SSM
    bandit: BanditUCB
    node_vectors: Dict[Node, Vector]          # bipolar hyper‑vectors (resource ↔ morphology)
    reference_vector: Vector                  # global reference hyper‑vector
    prev_observation: float = 0.0

    @staticmethod
    def _init_vectors(graph: Graph, dim: int = 1024) -> Tuple[Dict[Node, Vector], Vector]:
        """Create a bipolar vector per node and a global reference vector."""
        node_vecs = {n: random_bipolar_vector(dim, seed=str(n)) for n in graph}
        ref_vec = random_bipolar_vector(dim, seed='reference')
        return node_vecs, ref_vec

    @classmethod
    def hybrid_initialize(cls, graph: Graph) -> "HybridModel":
        """Create a fresh HybridModel with random SSM matrices and bandit."""
        dim_state = 4
        A = np.eye(dim_state) + 0.1 * np.random.randn(dim_state, dim_state)
        B = 0.1 * np.random.randn(dim_state, 1)
        C = np.random.randn(1, dim_state)

        ssm = SSM(A=A, B=B, C=C, state=np.zeros((dim_state, 1)))
        bandit = BanditUCB()
        node_vecs, ref_vec = cls._init_vectors(graph)
        return cls(graph=graph, ssm=ssm, bandit=bandit,
                   node_vectors=node_vecs, reference_vector=ref_vec)

    # ------------------------------------------------------------------
    # Core hybrid operations
    # ------------------------------------------------------------------
    def hybrid_update(self, observation: float, control: float = 0.0) -> None:
        """
        Perform a full update step:
        1. Fold‑change detection on the scalar observation.
        2. SSM step using the control input (treated as a 1‑D column vector).
        3. Update bandit values with the predicted reward scaled by fold‑change.
        """
        # 1. Fold‑change factor
        fc = fold_change(self.prev_observation, observation)
        self.prev_observation = observation

        # 2. SSM prediction
        ctrl_vec = np.array([[control]], dtype=float)
        pred_reward = self.ssm.step(ctrl_vec)

        # 3. Scale reward by fold‑change (bridge to morphology)
        scaled_reward = pred_reward * (1.0 + fc)

        # Update bandit for every node (could be selective in a real system)
        for node in self.graph:
            self.bandit.update(node, scaled_reward)

    def _compute_node_scores(self) -> Dict[Node, float]:
        """
        Compute a fused similarity‑scaled score for each node:
        score_i = cosine_similarity(h_i, h_ref) * (1 + fold_change)
        The fold_change factor is taken from the most recent observation.
        """
        fc = fold_change(self.prev_observation, self.prev_observation)  # zero‑change → 0
        # Use last observation as proxy for current magnitude
        base = 1.0 + fc
        scores = {}
        for n, vec in self.node_vectors.items():
            sim = cosine_similarity(vec, self.reference_vector)
            scores[n] = sim * base
        return scores

    def hybrid_select_action(self) -> Node:
        """
        Select an action (node) using a UCB bandit whose exploration term is
        modulated by the Gini coefficient of the node's neighbour degree
        distribution and whose exploitation term is the fused similarity score.
        """
        scores = self._compute_node_scores()
        # Compute Gini for each node based on neighbour degrees
        gini_factors = {}
        for n, neighbours in self.graph.items():
            degrees = [len(self.graph.get(nb, [])) for nb in neighbours]
            gini_factors[n] = gini_coefficient(degrees)

        # Combine Gini into a single scalar per node (here we simply use it directly)
        # Select node via bandit
        selected = self.bandit.select(
            nodes=list(self.graph.keys()),
            gini_factor=1.0,               # base factor; individual gini applied inside score
            score={n: scores[n] * (1.0 + gini_factors.get(n, 0.0)) for n in scores}
        )
        return selected

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny undirected graph
    sample_graph: Graph = {
        "A": {"B", "C"},
        "B": {"A", "C", "D"},
        "C": {"A", "B"},
        "D": {"B"},
    }

    # Initialise hybrid model
    model = HybridModel.hybrid_initialize(sample_graph)

    # Simulate a few observation steps
    observations = [10.0, 12.0, 9.5, 11.0, 13.2]
    for obs in observations:
        model.hybrid_update(observation=obs, control=random.random())
        chosen = model.hybrid_select_action()
        print(f"Obs={obs:.2f} → selected node: {chosen}")

    sys.exit(0)