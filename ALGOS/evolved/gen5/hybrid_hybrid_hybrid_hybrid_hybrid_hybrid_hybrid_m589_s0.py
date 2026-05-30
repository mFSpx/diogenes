# DARWIN HAMMER — match 589, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s5.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s0.py (gen4)
# born: 2026-05-29T23:29:52Z

"""Hybrid Fisher‑Bandit Resource Scheduler
========================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s5.py``  
  Provides a linear‑constraint resource model **A·x ≤ budgets** and a scalar
  “VRAM store” that evolves as  

  ``d store / dt = α·reward – β·store``  

  The store value is used as a *budget multiplier* for bandit propensities.

* **Parent B** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s0.py``  
  Supplies a Fisher‑information based angle estimator and a contextual
  multi‑armed bandit whose ``propensity`` and ``confidence_bound`` drive action
  selection.

Mathematical Bridge
-------------------
The bridge is the **scalar confidence/fisher factor** that can simultaneously
scale:

1. the **resource feasibility** of an entity (by multiplying the right‑hand
   side of the linear constraints with the current ``store``), and
2. the **bandit propensity** (by multiplying a raw propensity with the
   Fisher information vector derived from the current sensory features).

Thus a single scalar – the product ``store * fisher_factor`` – modulates both
the feasibility test and the stochastic bandit choice, yielding a unified
hybrid scheduler.

The module below implements this fusion with three core functions:

* ``build_resource_matrix`` – constructs the **A** matrix from entity resource
  vectors.
* ``fisher_information`` – computes Fisher‑information weights from features,
  angles and per‑feature importance.
* ``hybrid_select_action`` – performs feasibility filtering, scales bandit
  propensities with the combined ``store * fisher_factor`` and returns a
  ``BanditAction``.

A small smoke test demonstrates end‑to‑end execution."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (derived from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float            # scaled inflow rate
    expected_reward: float
    confidence_bound: float      # outflow / uncertainty
    algorithm: str


@dataclass(frozen=True)
class Entity:
    """An entity that consumes resources."""
    entity_id: str
    resources: Tuple[float, float, float]   # (distance, privacy_load, decision_score)


# ----------------------------------------------------------------------
# 1. Build the linear‑constraint matrix (Parent A)
# ----------------------------------------------------------------------
def build_resource_matrix(entities: List[Entity]) -> np.ndarray:
    """
    Assemble the resource matrix **A** where each row corresponds to an entity
    and columns are the three resource dimensions (d, p, s).

    Parameters
    ----------
    entities: list of Entity
        The entities whose resource consumption vectors are to be stacked.

    Returns
    -------
    np.ndarray
        A 2‑D array of shape (len(entities), 3).
    """
    if not entities:
        raise ValueError("Entity list must not be empty")
    A = np.array([list(e.resources) for e in entities], dtype=float)
    return A


# ----------------------------------------------------------------------
# 2. Fisher‑information computation (Parent B)
# ----------------------------------------------------------------------
def fisher_information(
    feature_vector: np.ndarray,
    angles: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute a Fisher‑information‑like scalar for each angle.

    The implementation follows the original parent: each feature contributes
    a Gaussian bump centred at a proportional angle; the gradient of the
    resulting intensity profile yields the Fisher information.

    Parameters
    ----------
    feature_vector : np.ndarray, shape (F,)
        Sensor or feature magnitudes.
    angles : np.ndarray, shape (A,)
        Discrete candidate angles (radians).
    weights : np.ndarray, shape (F,)
        Per‑feature importance; larger weight ⇒ narrower Gaussian.

    Returns
    -------
    np.ndarray, shape (A,)
        Fisher information values for each angle.
    """
    if feature_vector.shape != weights.shape:
        raise ValueError("feature_vector and weights must have the same length")
    intensities = np.zeros_like(angles, dtype=float)

    # Build intensity profile
    for i, (feature, weight) in enumerate(zip(feature_vector, weights)):
        sigma = 1.0 / math.sqrt(weight) if weight > 0 else 1.0
        centre = (i / len(feature_vector)) * 2 * math.pi
        diff = angles - centre
        intensities += feature * np.exp(-0.5 * (diff / sigma) ** 2)

    # Avoid division by zero
    eps = np.finfo(float).eps
    fisher = (np.gradient(intensities) ** 2) / (intensities + eps)
    return fisher


# ----------------------------------------------------------------------
# 3. Hybrid action selection (fusion of both parents)
# ----------------------------------------------------------------------
def hybrid_select_action(
    entities: List[Entity],
    actions: List[str],
    feature_vector: np.ndarray,
    angles: np.ndarray,
    weights: np.ndarray,
    store: float,
    budgets: np.ndarray,
    alpha: float = 0.1,
    beta: float = 0.01,
    epsilon: float = 0.1,
    rng: Optional[random.Random] = None,
) -> Tuple[BanditAction, float]:
    """
    Select an action using a contextual bandit whose propensities are
    modulated by Fisher information and a VRAM‑like store.  The selection is
    constrained by the linear resource feasibility test.

    Steps
    -----
    1. Compute Fisher information → ``fisher`` (vector over angles).
    2. Derive a *scalar confidence factor* ``f_factor`` as the mean of the
       top‑k Fisher values (k = min(3, len(fisher))).
    3. Scale the current ``store`` by ``f_factor`` to obtain the *effective
       budget multiplier* ``budget_mult``.
    4. Build matrix **A** from ``entities`` and keep rows whose resource
       consumption satisfies ``A_i ≤ budgets * budget_mult``.
    5. For each feasible action, draw a raw propensity from a simple
       epsilon‑greedy policy and then scale it by ``budget_mult``.
    6. Return the action with the highest scaled propensity together with the
       updated store value.

    Parameters
    ----------
    entities : List[Entity]
        All candidate entities (resource consumers).
    actions : List[str]
        Action identifiers; one action per entity is assumed (order‑preserving).
    feature_vector, angles, weights : np.ndarray
        Inputs to the Fisher information routine.
    store : float
        Current VRAM store value.
    budgets : np.ndarray, shape (3,)
        Base resource budgets for (d, p, s).
    alpha, beta : float
        ODE parameters for store update (see Parent A).
    epsilon : float
        Exploration probability for the epsilon‑greedy bandit.
    rng : random.Random | None
        Random generator; if ``None`` a new ``Random`` with system time seed is used.

    Returns
    -------
    Tuple[BanditAction, float]
        The selected action and the *new* store value after incorporating the
        observed reward (simulated as the scaled propensity).
    """
    if rng is None:
        rng = random.Random()

    # 1. Fisher information
    fisher = fisher_information(feature_vector, angles, weights)

    # 2. Confidence factor (mean of top‑k values)
    k = min(3, len(fisher))
    top_k = np.partition(fisher, -k)[-k:]
    f_factor = float(np.mean(top_k))

    # 3. Effective budget multiplier
    budget_mult = store * f_factor if store > 0 else f_factor

    # 4. Feasibility filter
    A = build_resource_matrix(entities)                     # shape (N,3)
    scaled_budgets = budgets * budget_mult                  # shape (3,)
    feasible_mask = np.all(A <= scaled_budgets, axis=1)     # shape (N,)
    feasible_indices = np.where(feasible_mask)[0]

    if feasible_indices.size == 0:
        # No feasible entity – fall back to a random action with zero propensity
        chosen_id = rng.choice(actions)
        action = BanditAction(
            action_id=chosen_id,
            propensity=0.0,
            expected_reward=0.0,
            confidence_bound=0.0,
            algorithm="hybrid_fisher_bandit",
        )
        new_store = max(0.0, store - beta * store)  # decay only
        return action, new_store

    # 5. Bandit propensities for feasible actions
    propensities: List[float] = []
    expected_rewards: List[float] = []
    confidence_bounds: List[float] = []
    for idx in feasible_indices:
        # epsilon‑greedy: with prob epsilon explore uniformly,
        # otherwise use a deterministic estimate proportional to Fisher.
        if rng.random() < epsilon:
            raw_propensity = rng.random()
        else:
            raw_propensity = fisher[idx % len(fisher)]  # wrap if needed
        scaled_propensity = raw_propensity * budget_mult
        propensities.append(scaled_propensity)

        # Simulated expected reward – linear in raw propensity
        expected_rewards.append(raw_propensity * 10.0)

        # Confidence bound inversely related to Fisher (more info → less uncertainty)
        confidence_bounds.append(1.0 / (fisher[idx % len(fisher)] + 1e-9))

    # Choose the feasible action with maximal scaled propensity
    best_idx_rel = int(np.argmax(propensities))
    best_idx = feasible_indices[best_idx_rel]
    chosen_action_id = actions[best_idx]

    # Simulated observed reward (for store update)
    observed_reward = expected_rewards[best_idx_rel]

    # 6. Store update (ODE discretised with Euler step Δt = 1)
    new_store = max(0.0, store + alpha * observed_reward - beta * store)

    selected = BanditAction(
        action_id=chosen_action_id,
        propensity=propensities[best_idx_rel],
        expected_reward=expected_rewards[best_idx_rel],
        confidence_bound=confidence_bounds[best_idx_rel],
        algorithm="hybrid_fisher_bandit",
    )
    return selected, new_store


# ----------------------------------------------------------------------
# Helper to initialise a simple scenario (used in the smoke test)
# ----------------------------------------------------------------------
def _demo_scenario() -> Tuple[List[Entity], List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Create a tiny deterministic scenario for quick testing."""
    entities = [
        Entity("E0", (2.0, 1.5, 0.5)),
        Entity("E1", (1.0, 2.0, 1.0)),
        Entity("E2", (0.5, 0.5, 2.0)),
    ]
    actions = [e.entity_id for e in entities]  # one‑to‑one mapping
    feature_vector = np.array([0.8, 0.3, 0.5])
    angles = np.linspace(0, 2 * math.pi, num=9, endpoint=False)
    weights = np.array([1.0, 2.0, 1.5])
    return entities, actions, feature_vector, angles, weights


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise demo data
    ents, acts, feats, angs, wts = _demo_scenario()
    base_budgets = np.array([3.0, 3.0, 3.0])   # generous budgets
    store_val = 1.0                           # initial VRAM store

    # Run a few iterations to illustrate learning / decay
    for step in range(5):
        action, store_val = hybrid_select_action(
            entities=ents,
            actions=acts,
            feature_vector=feats,
            angles=angs,
            weights=wts,
            store=store_val,
            budgets=base_budgets,
            alpha=0.2,
            beta=0.05,
            epsilon=0.2,
            rng=random.Random(step),  # deterministic per‑step RNG
        )
        print(f"Step {step}: selected {action.action_id}, "
              f"propensity={action.propensity:.4f}, "
              f"store={store_val:.4f}")
    sys.exit(0)