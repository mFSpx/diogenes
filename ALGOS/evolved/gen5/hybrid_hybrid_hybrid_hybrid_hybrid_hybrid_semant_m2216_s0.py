# DARWIN HAMMER — match 2216, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s5.py (gen4)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_bayes_update__m1099_s1.py (gen3)
# born: 2026-05-29T23:41:20Z

"""Hybrid Decision‑Bandit + Semantic Bayesian Scheduler
=====================================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s5.py``  
  Provides a 3‑dimensional resource vector **eᵢ = [dᵢ, pᵢ, sᵢ]** for each
  entity and a linear‑constraint feasibility test ``A @ x ≤ budgets``.
* **Parent B** – ``hybrid_hybrid_semantic_neig_hybrid_bayes_update__m1099_s1.py``  
  Supplies a semantic‑similarity neighbourhood function and a Bayesian
  Beta‑posterior update for binary rewards.

Mathematical Bridge
-------------------
Both parents expose a *scalar* that governs the dynamics of the system:

* In Parent A the scalar is the **budget multiplier** that scales the
  feasibility region.
* In Parent B the scalar is the **VRAM store** that is updated by an ODE
  ``d store / dt = α·reward – β·store`` and also drives the bandit’s
  propensity scaling.

The hybrid algorithm therefore:

1. Builds the resource matrix **A** from the entities’ 3‑D resource vectors.
2. For a candidate action computes a **semantic context** vector using the
   neighbour similarity routine of Parent B.
3. Forms a **propensity** by weighting the Bayesian expected reward with the
   semantic context and the current ``store`` value.
4. Filters actions whose resource row violates ``store * budgets``.
5. Samples the feasible action, observes a binary reward, updates the
   Beta‑posterior (expected reward) and the VRAM ``store`` in one step.

The three core functions below implement this fused workflow.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class Entity:
    """Physical entity with a 3‑D resource vector and an embedding."""
    id: str
    resources: Tuple[float, float, float]          # (distance, privacy, decision)
    embedding: np.ndarray                          # 5‑D semantic vector


@dataclass
class BanditStats:
    """Beta‑posterior parameters for a binary reward."""
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def expected(self) -> float:
        """Mean of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)


@dataclass
class SchedulerState:
    """Global mutable state."""
    store: float = 1.0                              # VRAM scalar
    alpha: float = 0.1                              # store increase coefficient
    beta: float = 0.05                              # store decay coefficient
    budgets: np.ndarray = np.array([10.0, 10.0, 10.0])  # per‑resource caps


# ----------------------------------------------------------------------
# Helper functions from Parent B (semantic neighbours)
# ----------------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if den == 0 else float(np.dot(a, b) / den)


def semantic_neighbors(
    target_id: str,
    target_vec: np.ndarray,
    corpus: List[Entity],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Return *k* most similar entity ids with cosine similarity."""
    sims = [
        (e.id, _cosine(target_vec, e.embedding))
        for e in corpus
        if e.id != target_id
    ]
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[:k]


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def build_resource_matrix(entities: List[Entity]) -> Tuple[np.ndarray, List[str]]:
    """
    Assemble the resource matrix **A** (shape N×3) from the entities.
    Returns the matrix and the ordered list of entity ids.
    """
    A = np.array([list(e.resources) for e in entities], dtype=float)
    ids = [e.id for e in entities]
    return A, ids


def compute_contextual_feature(
    entity: Entity,
    corpus: List[Entity],
    k: int = 5,
) -> float:
    """
    Produce a single scalar context for the bandit by averaging the cosine
    similarities of the *k* nearest semantic neighbours.
    """
    neighbours = semantic_neighbors(entity.id, entity.embedding, corpus, k)
    if not neighbours:
        return 0.0
    avg_sim = sum(sim for _, sim in neighbours) / len(neighbours)
    return avg_sim


def hybrid_bandit_step(
    entity: Entity,
    entities: List[Entity],
    A: np.ndarray,
    ids: List[str],
    stats: Dict[str, BanditStats],
    state: SchedulerState,
    rng: random.Random,
) -> Tuple[Optional[str], float]:
    """
    Perform one decision‑bandit iteration for *entity*.

    1. Compute semantic context.
    2. Form a propensity proportional to
       ``store * context * expected_reward``.
    3. Discard actions whose resource row violates
       ``store * budgets``.
    4. Sample a feasible action proportionally to its propensity.
    5. Simulate a binary reward (here a stochastic function of similarity).
    6. Update the Beta posterior and the VRAM store.

    Returns the selected action id (or ``None`` if no feasible action) and
    the observed reward.
    """
    # 1 – semantic context
    context = compute_contextual_feature(entity, entities, k=5)

    # 2 – raw propensities
    raw_props = {}
    for idx, aid in enumerate(ids):
        if aid == entity.id:
            continue  # skip self‑selection
        exp_r = stats[aid].expected
        raw = state.store * context * exp_r
        raw_props[aid] = max(raw, 0.0)

    if not raw_props:
        return None, 0.0

    # 3 – feasibility under scaled budgets
    scaled_budget = state.store * state.budgets
    feasible = {}
    for aid, prop in raw_props.items():
        row = A[ids.index(aid)]
        if np.all(row <= scaled_budget):
            feasible[aid] = prop

    if not feasible:
        return None, 0.0

    # 4 – sample action
    total = sum(feasible.values())
    pick = rng.random() * total
    cum = 0.0
    chosen_id = None
    for aid, prop in feasible.items():
        cum += prop
        if pick <= cum:
            chosen_id = aid
            break

    if chosen_id is None:
        return None, 0.0

    # 5 – simulate reward (binary). Higher similarity → higher chance.
    chosen_entity = next(e for e in entities if e.id == chosen_id)
    sim = _cosine(entity.embedding, chosen_entity.embedding)
    reward = 1.0 if rng.random() < sim else 0.0

    # 6 – Bayesian update
    stat = stats[chosen_id]
    stat.alpha += reward
    stat.beta += 1.0 - reward

    #    store ODE discretisation
    state.store += state.alpha * reward - state.beta * state.store
    state.store = max(state.store, 0.0)  # keep non‑negative

    return chosen_id, reward


# ----------------------------------------------------------------------
# Utility to initialise the system
# ----------------------------------------------------------------------
def initialise_system(num_entities: int = 10, seed: int = 42) -> Tuple[
    List[Entity], np.ndarray, List[str], Dict[str, BanditStats], SchedulerState
]:
    rng = random.Random(seed)
    np.random.seed(seed)

    entities = []
    for i in range(num_entities):
        eid = f"entity{i}"
        resources = (
            rng.uniform(0.5, 3.0),   # distance
            rng.uniform(0.5, 3.0),   # privacy
            rng.uniform(0.5, 3.0),   # decision
        )
        embedding = np.random.rand(5)
        entities.append(Entity(eid, resources, embedding))

    A, ids = build_resource_matrix(entities)
    stats = {e.id: BanditStats() for e in entities}
    state = SchedulerState()
    return entities, A, ids, stats, state


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = random.Random(123)

    entities, A, ids, stats, state = initialise_system(num_entities=8)

    # Run a few hybrid steps
    for step in range(5):
        # pick a random "requesting" entity
        requester = rng.choice(entities)
        chosen, rew = hybrid_bandit_step(
            requester,
            entities,
            A,
            ids,
            stats,
            state,
            rng,
        )
        print(
            f"Step {step+1}: requester={requester.id} -> "
            f"chosen={chosen} reward={rew:.1f} store={state.store:.3f}"
        )
    sys.exit(0)