# DARWIN HAMMER — match 5547, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m2543_s1.py (gen6)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_hybrid_m1247_s3.py (gen6)
# born: 2026-05-30T00:02:43Z

"""Hybrid module combining DARWIN HAMMER match 43 (curvature & pheromones) and 
match 1247 (RBF kernel & expected‑value policy). 

Mathematical bridge
-------------------
- Parent A provides a scalar Ollivier‑Ricci curvature `c_i` for each node *i* of a
  graph built from the actions.
- Parent B builds an RBF kernel `K_ij = exp(-ε·‖x_i−x_j‖²)` from feature vectors `x_i`.
- The hybrid treats the curvature `c_i` as an extra dimension appended to the
  original feature vector, i.e. `x'_i = [x_i , c_i]`.  
  The kernel is then recomputed on the augmented vectors, letting geometry
  (curvature) directly influence similarity.
- Pheromone entries are generated from the curvature values; their decayed
  signals weight the expected‑value computation, fusing the pheromone dynamics
  of Parent A with the kernel‑based expectation of Parent B.

The three core functions below implement this pipeline:
1. `build_action_graph` – creates node dimensions and a full edge list.
2. `curvature_augmented_kernel` – computes curvature, augments vectors, and
   returns the RBF kernel.
3. `hybrid_expected_values` – combines kernel‑based expectations with pheromone
   signals to produce a final policy estimate.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared between the two parent designs)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Immutable description of a mathematical action."""
    id: str
    tokens: Tuple[str, ...]          # symbolic tokens describing the action
    expected_value: float            # a priori expected reward
    cost: float = 0.0
    risk: float = 0.0


@dataclass
class PheromoneEntry:
    """Pheromone signal attached to a node."""
    feature: str
    value: float                     # raw value (here curvature)
    half_life: float                 # decay half‑life in arbitrary steps
    signal: float = field(init=False)  # decayed signal, computed on init

    def __post_init__(self):
        # initial signal equals the raw value
        self.signal = self.value


# ----------------------------------------------------------------------
# Helper utilities (adapted from parent A)
# ----------------------------------------------------------------------
def compute_ollivier_ricci_curvature(node_dims: Dict[str, np.ndarray],
                                    edge_list: List[Tuple[str, str]]) -> Dict[str, float]:
    """Simple proxy for Ollivier‑Ricci curvature: degree / total nodes."""
    curvature = {}
    total_nodes = len(node_dims)
    for node in node_dims:
        incident = sum(1 for e in edge_list if node in e)
        curvature[node] = incident / total_nodes if total_nodes else 0.0
    return curvature


def decay_signal(entry: PheromoneEntry, steps: int = 1) -> None:
    """Exponential decay of a pheromone signal using its half‑life."""
    decay_factor = 0.5 ** (steps / entry.half_life)
    entry.signal *= decay_factor


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def build_action_graph(actions: List[MathAction]) -> Tuple[Dict[str, np.ndarray],
                                                          List[Tuple[str, str]]]:
    """
    Constructs a graph from a list of actions.

    *Node dimensions* are numeric vectors derived from the token strings.
    Each token is hashed to a float; the vector is the list of token hashes.
    *Edge list* is the complete undirected graph (every pair of actions is linked),
    which is sufficient for the curvature proxy.
    """
    node_dims: Dict[str, np.ndarray] = {}
    for act in actions:
        # deterministic but pseudo‑random mapping from token to float
        vec = np.array([float(abs(hash(tok)) % 10_000) / 10_000 for tok in act.tokens],
                       dtype=np.float64)
        node_dims[act.id] = vec

    # Complete graph edges (unordered pairs)
    edge_list = [(actions[i].id, actions[j].id)
                 for i in range(len(actions))
                 for j in range(i + 1, len(actions))]
    return node_dims, edge_list


def curvature_augmented_kernel(actions: List[MathAction],
                               epsilon: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    """
    1. Builds the action graph.
    2. Computes Ollivier‑Ricci curvature for each node.
    3. Appends curvature as an extra dimension to each node's feature vector.
    4. Returns the RBF kernel matrix on the augmented vectors and the ordering of nodes.
    """
    node_dims, edge_list = build_action_graph(actions)
    curvature = compute_ollivier_ricci_curvature(node_dims, edge_list)

    # Augment each feature vector with its curvature value
    augmented: Dict[str, np.ndarray] = {}
    for act in actions:
        base = node_dims[act.id]
        curv = np.array([curvature[act.id]], dtype=np.float64)
        augmented[act.id] = np.concatenate([base, curv])

    nodes = [act.id for act in actions]
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        vi = augmented[nodes[i]]
        for j in range(i, n):
            vj = augmented[nodes[j]]
            dist = np.linalg.norm(vi - vj)
            val = math.exp(- (epsilon * dist) ** 2)
            K[i, j] = val
            K[j, i] = val
    return K, nodes


def hybrid_expected_values(actions: List[MathAction],
                           kernel: np.ndarray,
                           nodes: List[str],
                           half_life: float = 10.0,
                           decay_steps: int = 1) -> Dict[str, float]:
    """
    Computes a hybrid expected value for each action.

    *Kernel‑based expectation*: same as Parent B – weighted average of the
    `expected_value` of other actions using the kernel similarity.
    *Pheromone weighting*: a pheromone entry is created from the curvature
    (extracted from the diagonal of the kernel‑augmented vectors).  Its decayed
    signal multiplies the kernel‑based expectation, thereby fusing the
    pheromone infotaxis dynamics.
    """
    # 1️⃣ Build curvature‑derived pheromone entries (curvature = 1 - K_ii)
    #    (Since K_ii = 1 after augmentation, we use a small perturbation)
    pheromones: Dict[str, PheromoneEntry] = {}
    for idx, node in enumerate(nodes):
        curvature_est = 1.0 - kernel[idx, idx]  # placeholder, will be near 0
        # Use a minimal positive curvature if the estimate is 0
        curvature_est = max(curvature_est, 1e-6)
        entry = PheromoneEntry(feature=node,
                               value=curvature_est,
                               half_life=half_life)
        decay_signal(entry, steps=decay_steps)
        pheromones[node] = entry

    # 2️⃣ Kernel‑based expected values (excluding self‑contribution)
    expected: Dict[str, float] = {}
    m = len(actions)
    for i, act_i in enumerate(actions):
        weighted_sum = 0.0
        weight_total = 0.0
        for j, act_j in enumerate(actions):
            if i == j:
                continue
            sim = kernel[i, j]
            weighted_sum += sim * act_j.expected_value
            weight_total += sim
        base_exp = weighted_sum / weight_total if weight_total > 0 else 0.0
        # 3️⃣ Apply pheromone signal as a multiplicative bias
        pher_signal = pheromones[act_i.id].signal
        hybrid_val = base_exp * (1.0 + pher_signal)  # boost proportional to signal
        expected[act_i.id] = hybrid_val
    return expected


# ----------------------------------------------------------------------
# Optional convenience: policy update using hybrid expectations
# ----------------------------------------------------------------------
def update_policy_hybrid(updates: List[Tuple[str, float]],
                         actions: List[MathAction],
                         hybrid_expectations: Dict[str, float]) -> Dict[str, float]:
    """
    Mirrors Parent B's `update_policy` but weights rewards by the hybrid
    expected values instead of the raw `expected_value`.
    """
    policy: Dict[str, float] = {}
    action_map = {a.id: a for a in actions}
    for action_id, reward in updates:
        if action_id in action_map:
            weight = hybrid_expectations.get(action_id, 0.0)
            policy[action_id] = policy.get(action_id, 0.0) + reward * weight
    return policy


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny set of actions
    actions = [
        MathAction(id="A", tokens=("sin", "x"), expected_value=1.2),
        MathAction(id="B", tokens=("cos", "x"), expected_value=0.9),
        MathAction(id="C", tokens=("exp", "y"), expected_value=1.5),
    ]

    # 1. Build curvature‑augmented kernel
    K, order = curvature_augmented_kernel(actions, epsilon=0.8)

    # 2. Compute hybrid expectations
    hybrid_exp = hybrid_expected_values(actions, K, order,
                                        half_life=12.0, decay_steps=2)

    # 3. Simulate a few policy updates
    updates = [("A", 0.3), ("B", -0.1), ("C", 0.5)]
    policy = update_policy_hybrid(updates, actions, hybrid_exp)

    # Simple output to verify execution
    print("Kernel matrix (augmented):")
    print(K)
    print("\nHybrid expected values:")
    for aid, val in hybrid_exp.items():
        print(f"  {aid}: {val:.4f}")
    print("\nPolicy after updates:")
    for aid, val in policy.items():
        print(f"  {aid}: {val:.4f}")