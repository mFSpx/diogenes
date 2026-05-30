# DARWIN HAMMER — match 3272, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2508_s2.py (gen6)
# born: 2026-05-29T23:48:54Z

"""Hybrid Decision‑Bandit Algorithm
===================================

This module fuses the two parent algorithms:

* **Parent A** – *hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s2.py*  
  Provides a 3‑dimensional resource vector `e_i = [d_i, p_i, s_i]` for each
  entity (spatial distance, privacy‑collision penalty, decision‑hygiene score)
  and a linear‑constraint selection model.

* **Parent B** – *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2508_s2.py*  
  Defines immutable `MathAction` objects with a regret‑weighted scalar and
  a MinHash‑based signature utility for similarity computation.

**Mathematical Bridge**

The bridge is the *signature similarity* used in Parent B.  
We reuse it in Parent A to compute the privacy‑collision term `p_i` and to
modulate the bandit propensity of each action. Concretely:

* `σ_i = 1` if the entity’s signature collides (similarity ≥ θ) with any
  other entity, otherwise `0`.  
  `p_i = β·σ_i` (Parent A).

* For each action we also compute a MinHash signature. The similarity
  `γ_{i,j} = similarity(sig_entity_i, sig_action_j)` is used to weight the
  regret weight `w_j = action.regret_weight`:

  
  adjusted_weight_{i,j} = w_j * (1 + γ_{i,j})
  

Thus the entity‑action interaction matrix combines the resource vectors of
Parent A with the regret‑weighted bandit scores of Parent B, enabling a
single greedy selection that respects spatial, privacy, and decision
budgets while favouring actions whose signatures resemble the entity’s
signature.

The public API consists of three core functions demonstrating the hybrid
operation:

1. `calculate_resource_vector` – builds `e_i` for an entity.
2. `build_hybrid_matrix` – stacks entity and action vectors into a single
   resource matrix and computes the adjusted bandit weights.
3. `greedy_hybrid_selector` – selects a feasible subset under the three
   linear budgets using a greedy heuristic.

A lightweight smoke test is provided under ``if __name__ == "__main__"``
to verify that the module runs without error.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict

import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Parent B – immutable action definitions and MinHash utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Immutable description of a decision option."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0
    ram: float = 0.0          # RAM consumption (MB) – required for hybrid matrix
    tier: int = 1             # Tier factor τ∈{1,2,3}
    tokens: Tuple[str, ...] = ()  # Token set for signature generation

    @property
    def regret_weight(self) -> float:
        """Regret‑weighted scalar used throughout the hybrid model."""
        return self.expected_value - self.cost - self.risk


@dataclass(frozen=True)
class MathCounterfactual:
    """Result of a counter‑factual evaluation for a single action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on a seed and a token string."""
    h = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def signature(tokens: Iterable[str], k: int = 128) -> np.ndarray:
    """Min‑hash signature of a token set."""
    toks = [t for t in tokens if t]
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not toks:
        return np.full(k, (1 << 64) - 1, dtype=np.uint64)

    sig = np.empty(k, dtype=np.uint64)
    for i in range(k):
        hashes = [_hash(i, t) for t in toks]
        sig[i] = min(hashes)
    return sig


def similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Normalized Jaccard‑like similarity for two MinHash signatures."""
    if sig_a.shape != sig_b.shape:
        raise ValueError("Signatures must have the same length")
    return np.mean(sig_a == sig_b)


# ----------------------------------------------------------------------
# Parent A – resource vector computation
# ----------------------------------------------------------------------


def haversine_distance(loc1: Tuple[float, float],
                       loc2: Tuple[float, float]) -> float:
    """Return distance in metres between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, loc1)
    lat2, lon2 = map(math.radians, loc2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    R = 6371000.0  # Earth radius in metres
    return R * c


def calculate_resource_vector(entity: Dict,
                              reference_location: Tuple[float, float],
                              beta: float,
                              alpha: float,
                              all_entity_sigs: List[np.ndarray],
                              collision_threshold: float = 0.8) -> np.ndarray:
    """
    Compute the 3‑dimensional resource vector e_i = [d_i, p_i, s_i].

    * d_i – haversine distance from reference_location.
    * p_i – β·σ_i, where σ_i = 1 if the entity's signature collides with any
      other entity (similarity ≥ threshold), else 0.
    * s_i – decision‑hygiene score (here simulated as a random float in [0,1]).
    """
    # Spatial component
    d_i = haversine_distance(entity["location"], reference_location)

    # Signature component
    sig_i = entity["signature"]
    sigma_i = 0
    for other_sig in all_entity_sigs:
        if other_sig is sig_i:
            continue
        if similarity(sig_i, other_sig) >= collision_threshold:
            sigma_i = 1
            break
    p_i = beta * sigma_i

    # Decision‑hygiene score (placeholder – in real code this would be computed)
    s_i = random.random()

    return np.array([d_i, p_i, s_i], dtype=float)


# ----------------------------------------------------------------------
# Hybrid matrix construction and greedy selector
# ----------------------------------------------------------------------


def build_hybrid_matrix(entities: List[Dict],
                        actions: List[MathAction],
                        reference_location: Tuple[float, float],
                        beta: float,
                        alpha: float,
                        privacy_risk_mean: float,
                        collision_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the combined resource matrix A and the adjusted bandit weight
    matrix W.

    Returns
    -------
    A : np.ndarray, shape (N_entities + N_actions, 3)
        Each row is a 3‑dimensional vector:
        * For entities → [d_i, p_i, s_i] (see ``calculate_resource_vector``).
        * For actions   → [RAM_j, α·τ_j·μ, expected_value_j].

    W : np.ndarray, shape (N_entities, N_actions)
        Adjusted regret weights:
        ``W[i, j] = action_j.regret_weight * (1 + similarity(sig_i, sig_j))``.
    """
    # Pre‑compute signatures for entities and actions
    entity_sigs = [entity["signature"] for entity in entities]
    action_sigs = [signature(action.tokens) for action in actions]

    # Build entity resource rows
    entity_rows = []
    for ent, sig in zip(entities, entity_sigs):
        vec = calculate_resource_vector(
            ent,
            reference_location,
            beta,
            alpha,
            entity_sigs,
            collision_threshold,
        )
        entity_rows.append(vec)

    # Build action resource rows
    action_rows = []
    for act in actions:
        ram_component = act.ram
        privacy_component = alpha * act.tier * privacy_risk_mean
        score_component = act.expected_value  # using expected value as the “score”
        action_rows.append([ram_component, privacy_component, score_component])

    A = np.vstack([np.array(entity_rows), np.array(action_rows)], dtype=float)

    # Adjusted bandit weight matrix
    N_e = len(entities)
    N_a = len(actions)
    W = np.empty((N_e, N_a), dtype=float)
    for i, sig_e in enumerate(entity_sigs):
        for j, sig_a in enumerate(action_sigs):
            sim = similarity(sig_e, sig_a)
            W[i, j] = actions[j].regret_weight * (1.0 + sim)

    return A, W


def greedy_hybrid_selector(A: np.ndarray,
                           W: np.ndarray,
                           budgets: Tuple[float, float, float]) -> np.ndarray:
    """
    Greedy selection under linear budgets.

    Parameters
    ----------
    A : np.ndarray, shape (N_total, 3)
        Combined resource matrix (entities + actions).
    W : np.ndarray, shape (N_entities, N_actions)
        Adjusted regret weights.
    budgets : (spatial_budget, privacy_budget, decision_budget)

    Returns
    -------
    x : np.ndarray, shape (N_total,)
        Binary indicator vector (1 = selected, 0 = not selected).
    """
    N_entities = W.shape[0]
    N_actions = W.shape[1]
    N_total = A.shape[0]

    # Initialise selection vector
    x = np.zeros(N_total, dtype=int)

    # Track remaining budgets
    remaining = np.array(budgets, dtype=float)

    # Helper: compute marginal gain of picking action j for entity i
    def marginal_gain(i: int, j: int) -> float:
        return W[i, j]

    # Greedy loop – we iterate over actions, picking the best entity‑action pair
    # that fits within the remaining budgets.
    while True:
        best_gain = -np.inf
        best_pair = None

        for i in range(N_entities):
            if x[i] == 1:   # entity already selected (we allow at most one action per entity)
                continue
            for j in range(N_actions):
                action_idx = N_entities + j
                if x[action_idx] == 1:
                    continue
                # Resource consumption if we add this pair (entity i + action j)
                consumption = A[i] + A[action_idx]
                if np.all(consumption <= remaining):
                    gain = marginal_gain(i, j)
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, action_idx, consumption)

        if best_pair is None:
            break  # no feasible addition

        i_sel, a_sel, cons = best_pair
        x[i_sel] = 1
        x[a_sel] = 1
        remaining -= cons

    return x


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------


def _generate_dummy_entities(num: int,
                             reference_location: Tuple[float, float]) -> List[Dict]:
    """Create dummy entities with random locations and token signatures."""
    entities = []
    for idx in range(num):
        # Random location within ~10 km radius
        lat_offset = random.uniform(-0.09, 0.09)
        lon_offset = random.uniform(-0.09, 0.09)
        loc = (reference_location[0] + lat_offset,
               reference_location[1] + lon_offset)

        # Random token set
        tokens = [f"tok{random.randint(0, 1000)}" for _ in range(10)]
        sig = signature(tokens)

        entities.append({
            "id": f"E{idx}",
            "location": loc,
            "tokens": tokens,
            "signature": sig,
        })
    return entities


def _generate_dummy_actions(num: int) -> List[MathAction]:
    """Create dummy actions with random attributes and token signatures."""
    actions = []
    for idx in range(num):
        tokens = [f"act{random.randint(0, 1000)}" for _ in range(8)]
        actions.append(
            MathAction(
                id=f"A{idx}",
                expected_value=random.uniform(0, 100),
                cost=random.uniform(0, 20),
                risk=random.uniform(0, 10),
                ram=random.uniform(50, 500),   # MB
                tier=random.choice([1, 2, 3]),
                tokens=tuple(tokens),
            )
        )
    return actions


if __name__ == "__main__":
    # Settings
    REF_LOC = (37.7749, -122.4194)   # San Francisco (lat, lon)
    BETA = 5.0
    ALPHA = 1.2
    PRIVACY_RISK_MEAN = 0.35
    BUDGETS = (1e6, 500.0, 300.0)   # spatial (m), privacy, decision

    # Generate synthetic data
    entities = _generate_dummy_entities(5, REF_LOC)
    actions = _generate_dummy_actions(4)

    # Build hybrid structures
    A_matrix, W_matrix = build_hybrid_matrix(
        entities,
        actions,
        REF_LOC,
        BETA,
        ALPHA,
        PRIVACY_RISK_MEAN,
    )

    # Perform greedy selection
    selection = greedy_hybrid_selector(A_matrix, W_matrix, BUDGETS)

    # Simple verification output
    selected_indices = np.where(selection == 1)[0]
    print("Selected rows (indices):", selected_indices.tolist())
    print("Remaining budgets after selection:",
          (np.array(BUDGETS) - A_matrix[selected_indices].sum(axis=0)).tolist())
    sys.exit(0)