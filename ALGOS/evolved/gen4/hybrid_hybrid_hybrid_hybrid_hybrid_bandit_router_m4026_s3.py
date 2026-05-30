# DARWIN HAMMER — match 4026, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_ternary_lens__m1111_s1.py (gen3)
# parent_b: hybrid_bandit_router_honeybee_store_m9_s0.py (gen1)
# born: 2026-05-29T23:53:14Z

"""Hybrid Algorithm integrating adjacency matrix representation (Parent A) with
bandit action selection and store dynamics (Parent B).

Mathematical bridge:
- Parent A can encode a set of extracted features as a weighted adjacency matrix
  A where A_{ij}=cosine_similarity(f_i,f_j).  The node degrees d_i = Σ_j A_{ij}
  form a natural context vector for a contextual bandit.
- Parent B selects actions based on a contextual linear‑UCB (or other) rule and
  maintains a reward policy.  The reward obtained for the selected action is fed
  back to modify the adjacency matrix (e.g., by reinforcing the edges incident
  to the chosen node) and to update the store dynamics.

Thus the hybrid system learns a graph representation, uses it to drive bandit
decisions, and then closes the loop by adapting the graph and a scalar store
according to the observed reward."""
import numpy as np
import random
import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------- Parent A core (feature extraction) ----------
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

# ---------- Parent B core (bandit + store) ----------
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

_POLICY: dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def select_action(
    context: dict[str, float],
    actions: List[str],
    algorithm: str = 'linucb',
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0, _reward(a)), 1 + max(0, 1 - _reward(a))
            ),
        )
    else:  # default linear UCB‑like rule
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
    delta = alpha * sum(inflow) - beta * sum(outflow)
    return max(0.0, store + dt * delta), delta

# ---------- Hybrid Functions ----------
def build_adjacency(features: Dict[str, float]) -> np.ndarray:
    """Create a symmetric adjacency matrix from a feature dict using cosine similarity."""
    vec = np.array(list(features.values()), dtype=float)
    if vec.size == 0:
        raise ValueError("Feature vector is empty")
    norm = np.linalg.norm(vec)
    if norm == 0:
        norm = 1.0
    normalized = vec / norm
    # outer product gives cosine similarity for identical vectors; we add a small epsilon
    adj = np.outer(normalized, normalized) + 1e-6
    # enforce symmetry and zero diagonal (no self‑loops)
    np.fill_diagonal(adj, 0.0)
    return adj

def adjacency_to_context(adj: np.ndarray) -> Dict[str, float]:
    """Map node degrees of the adjacency matrix to a context dictionary."""
    degrees = adj.sum(axis=1)
    total = degrees.sum()
    if total == 0:
        total = 1.0
    normalized = degrees / total
    return {f"node_{i}": float(val) for i, val in enumerate(normalized)}

def hybrid_step(
    text: str,
    actions: List[str],
    store: float,
    algorithm: str = 'linucb',
    seed: int | str | None = 7,
) -> Tuple[BanditAction, float, np.ndarray]:
    """
    One hybrid iteration:
    1. Extract features from text (Parent A).
    2. Build adjacency matrix and derive a bandit context.
    3. Select an action using the contextual bandit (Parent B).
    4. Simulate a reward, update the bandit policy, and adjust the store.
    5. Reinforce the adjacency matrix edges incident to the chosen node with the
       obtained reward (closing the loop).
    Returns the selected action, the new store value, and the updated adjacency.
    """
    # 1. Feature extraction
    feats = extract_full_features(text)

    # 2. Graph construction & context
    adj = build_adjacency(feats)
    context = adjacency_to_context(adj)

    # 3. Action selection
    chosen = select_action(context, actions, algorithm=algorithm, seed=seed)

    # 4. Simulated reward (deterministic from hash for reproducibility)
    rng = random.Random(hash((chosen.action_id, seed)))
    reward = rng.random()  # reward ∈ [0,1)

    # Update bandit policy
    update_policy(
        [BanditUpdate(context_id='hybrid', action_id=chosen.action_id, reward=reward, propensity=chosen.propensity)]
    )

    # Store dynamics: treat reward as inflow, a small constant as outflow
    new_store, delta = update_store(store, inflow=[reward], outflow=[0.01])

    # 5. Edge reinforcement: add reward to edges adjacent to the selected node
    node_idx = int(chosen.action_id.split('_')[-1]) if '_' in chosen.action_id else 0
    if 0 <= node_idx < adj.shape[0]:
        adj[node_idx, :] += reward * 0.1
        adj[:, node_idx] += reward * 0.1
        # re‑symmetrize and keep diagonal zero
        adj = (adj + adj.T) / 2.0
        np.fill_diagonal(adj, 0.0)

    return chosen, new_store, adj

def demo_hybrid():
    """Run a short demo of the hybrid algorithm."""
    sample_text = "The quick brown fox jumps over the lazy dog."
    actions = [f"node_{i}" for i in range(5)]  # five possible graph nodes as actions
    store = 10.0
    print("Initial store:", store)
    for step in range(3):
        action, store, adj = hybrid_step(sample_text, actions, store, seed=step)
        print(f"Step {step+1}:")
        print("  Chosen action :", action.action_id)
        print("  Expected reward:", f"{action.expected_reward:.3f}")
        print("  Updated store  :", f"{store:.3f}")
        print("  Adjacency slice:", adj[:2, :2].round(3))
    print("Demo completed.")

if __name__ == "__main__":
    demo_hybrid()