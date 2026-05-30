# DARWIN HAMMER — match 9, survivor 5
# gen: 3
# parent_a: hybrid_bandit_router_honeybee_store_m9_s0.py (gen1)
# parent_b: hybrid_hybrid_krampus_brain_ttt_linear_m4_s1.py (gen2)
# born: 2026-05-29T23:25:06Z

"""Hybrid Bandit‑Honeybee‑Graph Curvature Algorithm
==================================================
Parents:
- **hybrid_bandit_router_honeybee_store_m9_s0.py** (Bandit action selection + Honeybee store dynamics)
- **hybrid_hybrid_krampus_brain_ttt_linear_m4_s1.py** (Feature extraction + graph‑matrix operations for Ollivier‑Ricci curvature)

Mathematical Bridge
-------------------
Both parents manipulate a *state representation*:
* Bandit A uses a **context vector** (features) to select an action via an Upper‑Confidence‑Bound style rule.
* Krampus B builds a **feature vector** from raw text and then applies **matrix operations** on a graph adjacency matrix to obtain curvature estimates.

The fusion treats the **selected bandit action** as a *graph node* whose incident edge weights are modulated by the **Honeybee store delta** (the net resource flow).  
After the edge update, a simple curvature matrix is recomputed from the modified adjacency.  
Thus the algorithm closes the loop:


features → bandit.select_action → node i
store + inflow/outflow → Δ
adjacency[i,:] , adjacency[:,i] ← adjacency + f(Δ)
curvature ← g(adjacency)


The three core functions below implement this hybrid pipeline.

"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared Bandit / Store components (Parent A)
# ----------------------------------------------------------------------

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Clear the global reward statistics."""
    _POLICY.clear()

def update_policy(updates: List["BanditUpdate"]) -> None:
    """Accumulate rewards for each action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def _reward(action: str) -> float:
    """Mean reward observed for *action*."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

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

def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """Choose an action according to a simple LinUCB / epsilon‑greedy rule."""
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Beta posterior approximation
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0, _reward(a)), 1 + max(0, 1 - _reward(a))
            ),
        )
    else:  # LinUCB‑like
        scale = math.sqrt(
            sum(float(v) * float(v) for v in context.values())
        ) if context else 1.0
        chosen = max(
            actions,
            key=lambda a: _reward(a)
            + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
        )
    return BanditAction(
        action_id=chosen,
        propensity=1.0 / len(actions),
        expected_reward=_reward(chosen),
        confidence_bound=1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1]),
        algorithm=algorithm,
    )

def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    """Honeybee store dynamics."""
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

# ----------------------------------------------------------------------
# Graph / Feature components (Parent B)
# ----------------------------------------------------------------------

def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature generation from a string."""
    rng = random.Random(hash(text))
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
    return {k: rng.random() * 10 for k in keys}

def extract_master_vector(text: str) -> Dict[str, float]:
    """Compress the full feature set to a smaller, semantically grouped vector."""
    if not text.strip():
        return {}
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0),
        "directive_ratio": f.get("operator_directive_ratio", 0.0),
        "target_density": f.get("operator_target_density", 0.0),
        "forensic_shield": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
    }

def curvature_from_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Very lightweight Ollivier‑Ricci‑inspired curvature estimate.
    For an undirected weighted graph we treat the normalized adjacency
    matrix as a transport plan; curvature_ij = 1 - d_ij where d_ij is
    the (row‑wise) L1 distance between the probability vectors of nodes i
    and j.  The result lies in [-1, 1] and is symmetric.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    # Ensure non‑negative weights
    adj = np.maximum(adj, 0.0)
    # Row‑wise normalization to obtain a stochastic matrix P
    row_sums = adj.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    P = adj / row_sums
    # Pairwise L1 distance between rows
    diff = np.abs(P[:, None, :] - P[None, :, :]).sum(axis=2)
    # Curvature as 1 - distance (scaled to [-1,1])
    curvature = 1.0 - diff
    # Symmetrize explicitly (numerical noise may break exact symmetry)
    curvature = (curvature + curvature.T) / 2.0
    return curvature

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------

def adjust_adjacency_for_delta(
    adj: np.ndarray,
    node_idx: int,
    delta: float,
    scale: float = 0.5,
) -> np.ndarray:
    """
    Increase (or decrease) all edges incident to *node_idx* proportionally
    to the store delta.  The update rule is:

        A_ij ← A_ij + scale * Δ   for i == node_idx or j == node_idx

    The adjacency matrix remains symmetric and non‑negative.
    """
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must be square")
    if not (0 <= node_idx < adj.shape[0]):
        raise IndexError("node_idx out of bounds")
    inc = scale * delta
    # Broadcast update to row and column
    adj[node_idx, :] = np.maximum(0.0, adj[node_idx, :] + inc)
    adj[:, node_idx] = np.maximum(0.0, adj[:, node_idx] + inc)
    # Preserve symmetry explicitly
    adj = (adj + adj.T) / 2.0
    return adj

def hybrid_step(
    text: str,
    actions: List[str],
    store: float,
    inflow: List[float],
    outflow: List[float],
    adjacency: np.ndarray,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> Tuple[float, np.ndarray, np.ndarray, BanditAction]:
    """
    Execute one hybrid iteration:

    1. Convert *text* into a context vector (master features).
    2. Select a bandit action → interpreted as a graph node.
    3. Update the honeybee store and obtain Δ.
    4. Modulate the adjacency matrix using Δ.
    5. Re‑compute the curvature matrix.

    Returns the new store, updated adjacency, curvature matrix, and the
    BanditAction object.
    """
    # 1️⃣ Feature extraction → context
    context = extract_master_vector(text)

    # 2️⃣ Bandit selection
    bandit_action = select_action(context, actions, algorithm, epsilon, seed)

    # Map action identifier to a node index (simple hash modulo)
    node_idx = hash(bandit_action.action_id) % adjacency.shape[0]

    # 3️⃣ Store dynamics
    new_store, delta = update_store(store, inflow, outflow)

    # 4️⃣ Graph weight adjustment
    updated_adj = adjust_adjacency_for_delta(adjacency.copy(), node_idx, delta)

    # 5️⃣ Curvature computation
    curvature = curvature_from_adjacency(updated_adj)

    # Record the observed reward (here we use delta as a proxy reward)
    update_policy(
        [BanditUpdate(context_id=text, action_id=bandit_action.action_id, reward=delta, propensity=bandit_action.propensity)]
    )

    return new_store, updated_adj, curvature, bandit_action

def simulate_hybrid(
    steps: int = 5,
    node_count: int = 6,
    seed: int = 42,
) -> None:
    """Run a short deterministic simulation printing key statistics."""
    random.seed(seed)
    np.random.seed(seed)

    # Initialise a random symmetric adjacency matrix
    raw = np.random.rand(node_count, node_count)
    adj = (raw + raw.T) / 2.0  # make symmetric
    np.fill_diagonal(adj, 0.0)  # no self‑loops

    store = 10.0
    actions = [f"node_{i}" for i in range(node_count)]

    for step in range(steps):
        inflow = [random.random() for _ in range(2)]
        outflow = [random.random() for _ in range(2)]
        text = f"step-{step}-payload"

        store, adj, curvature, ba = hybrid_step(
            text=text,
            actions=actions,
            store=store,
            inflow=inflow,
            outflow=outflow,
            adjacency=adj,
            algorithm="linucb",
            epsilon=0.05,
            seed=seed + step,
        )

        print(f"Step {step+1}")
        print(f"  Selected action : {ba.action_id}")
        print(f"  Store delta     : {store:.3f}")
        print(f"  Curvature mean  : {curvature.mean():.4f}")
        print("-" * 30)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    simulate_hybrid()