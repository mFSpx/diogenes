# DARWIN HAMMER — match 9, survivor 4
# gen: 3
# parent_a: hybrid_bandit_router_honeybee_store_m9_s0.py (gen1)
# parent_b: hybrid_hybrid_krampus_brain_ttt_linear_m4_s1.py (gen2)
# born: 2026-05-29T23:25:06Z

"""
Hybrid Algorithm: Bandit-Router + Honeybee-Store fused with Graph Curvature (Krampus) & Linear Test-Time Training

Parents:
- hybrid_bandit_router_honeybee_store_m9_s0.py (Bandit action selection + store dynamics)
- hybrid_hybrid_krampus_brain_ttt_linear_m4_s1.py (Feature extraction, graph‑based Ollivier‑Ricci curvature proxy, linear matrix updates)

Mathematical Bridge:
The bridge is a *node‑wise curvature proxy* computed from a graph adjacency matrix that
encodes feature similarity. The curvature value of a node serves as the *expected reward*
for the bandit selector. After an action (node) is chosen, the honeybee store dynamics
update a scalar “resource store”. The resulting store delta is fed back to the graph:
edge weights incident to the selected node are scaled proportionally to the delta,
thereby performing a linear test‑time training step on the adjacency matrix.
Thus the bandit’s reward estimation, the store differential equation, and the graph
matrix update are mathematically intertwined.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Simple in‑memory policy store (Parent A)
# ----------------------------------------------------------------------
_POLICY: dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

# ----------------------------------------------------------------------
# Feature extraction (Parent B)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric", "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index", "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax", "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density", "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio", "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline", "telemetry_manic_velocity",
    ]
    return {k: rnd.random() * 10 for k in keys}

def extract_master_vector(text: str) -> Dict[str, float]:
    if not text.strip():
        return {}
    f = extract_full_features(text)
    # Collapse to a smaller, deterministic vector
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0),
    }

# ----------------------------------------------------------------------
# Graph construction and curvature proxy (Parent B)
# ----------------------------------------------------------------------
def build_adjacency(nodes: List[str], master_vectors: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Create a symmetric adjacency matrix where edge weight = exp(-||v_i - v_j||²)."""
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    for i, ni in enumerate(nodes):
        vi = np.array(list(master_vectors[ni].values()))
        for j, nj in enumerate(nodes):
            if i == j:
                continue
            vj = np.array(list(master_vectors[nj].values()))
            dist_sq = np.sum((vi - vj) ** 2)
            A[i, j] = math.exp(-dist_sq)
    # Symmetrize
    A = (A + A.T) / 2.0
    return A

def node_curvature_proxy(adj: np.ndarray, node_index: int) -> float:
    """
    Simple curvature proxy: C_i = sum_j (A_ij - A_ji) / degree_i.
    For a symmetric matrix this reduces to zero; we therefore use a
    Laplacian‑based proxy: C_i = degree_i - average_neighbor_weight.
    """
    degree = np.sum(adj[node_index, :])
    if degree == 0:
        return 0.0
    avg_neighbor = degree / (adj.shape[0] - 1)
    return degree - avg_neighbor

def curvature_vector(adj: np.ndarray) -> np.ndarray:
    """Return curvature proxy for every node."""
    return np.array([node_curvature_proxy(adj, i) for i in range(adj.shape[0])])

# ----------------------------------------------------------------------
# Store dynamics (Parent A)
# ----------------------------------------------------------------------
def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

# ----------------------------------------------------------------------
# Hybrid action selection (Bandit) using curvature as reward (Bridge)
# ----------------------------------------------------------------------
def select_action(
    context: Dict[str, float],
    node_names: List[str],
    curvature: np.ndarray,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    if not node_names:
        raise ValueError("node_names required")
    rng = random.Random(seed)

    # Populate temporary policy with curvature as pseudo‑reward
    for name, cur in zip(node_names, curvature):
        if name not in _POLICY:
            _POLICY[name] = [cur, 1.0]  # init with curvature as average reward

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(node_names)
    elif algorithm == "thompson":
        chosen = max(
            node_names,
            key=lambda a: rng.betavariate(
                1 + max(0, _reward(a)), 1 + max(0, 1 - _reward(a))
            ),
        )
    else:  # LinUCB‑style proxy
        scale = math.sqrt(
            sum(float(v) * float(v) for v in context.values())
        ) if context else 1.0
        chosen = max(
            node_names,
            key=lambda a: _reward(a)
            + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
        )

    return BanditAction(
        action_id=chosen,
        propensity=1.0 / len(node_names),
        expected_reward=_reward(chosen),
        confidence_bound=1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1]),
        algorithm=algorithm,
    )

# ----------------------------------------------------------------------
# Hybrid update: combine store dynamics, graph weight adaptation, and policy update
# ----------------------------------------------------------------------
def hybrid_step(
    store: float,
    adj: np.ndarray,
    node_names: List[str],
    master_vectors: Dict[str, Dict[str, float]],
    context: Dict[str, float],
    algorithm: str = "linucb",
) -> Tuple[float, np.ndarray, BanditAction]:
    """
    1. Compute curvature proxy for each node.
    2. Select a node via the bandit selector (reward = curvature).
    3. Treat the selected node as a resource sink:
       - inflow = weights of incoming edges,
       - outflow = weights of outgoing edges.
    4. Update the scalar store using the honeybee dynamics.
    5. Propagate the store delta back to the graph:
       increase outgoing edges from the selected node proportionally to delta.
    6. Record the observed reward (curvature) into the bandit policy.
    """
    # 1. curvature
    curv_vec = curvature_vector(adj)

    # 2. action selection
    action = select_action(context, node_names, curv_vec, algorithm=algorithm)

    idx = node_names.index(action.action_id)

    # 3. inflow / outflow (undirected graph => treat symmetric)
    inflow = adj[:, idx].tolist()
    outflow = adj[idx, :].tolist()

    # 4. store update
    new_store, delta = update_store(store, inflow, outflow)

    # 5. graph weight adaptation (linear test‑time training step)
    if delta != 0.0:
        # scale outgoing edges; keep symmetry by mirroring the change
        scale_factor = 1.0 + 0.05 * delta
        adj[idx, :] = np.clip(adj[idx, :] * scale_factor, 0.0, None)
        adj[:, idx] = adj[idx, :]  # enforce symmetry

    # 6. bandit policy update (reward = curvature of chosen node)
    observed_reward = curv_vec[idx]
    update_policy(
        [
            BanditUpdate(
                context_id="hybrid_step",
                action_id=action.action_id,
                reward=observed_reward,
                propensity=action.propensity,
            )
        ]
    )

    return new_store, adj, action

# ----------------------------------------------------------------------
# Utility to initialise a demo graph from random text snippets
# ----------------------------------------------------------------------
def initialise_demo(num_nodes: int = 5) -> Tuple[float, np.ndarray, List[str], Dict[str, Dict[str, float]]]:
    dummy_texts = [f"sample text {i}" for i in range(num_nodes)]
    node_names = [f"node_{i}" for i in range(num_nodes)]
    master_vectors = {name: extract_master_vector(txt) for name, txt in zip(node_names, dummy_texts)}
    adj = build_adjacency(node_names, master_vectors)
    initial_store = 10.0
    return initial_store, adj, node_names, master_vectors

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    store, adj, names, vectors = initialise_demo()
    ctx = {"temperature": 0.5, "pressure": 1.2}
    for step in range(3):
        store, adj, act = hybrid_step(store, adj, names, vectors, ctx, algorithm="linucb")
        print(f"Step {step+1}: selected={act.action_id}, store={store:.3f}, reward≈{act.expected_reward:.3f}")
    print("Final adjacency matrix (symmetrized, trimmed):")
    print(np.round(adj, 3))