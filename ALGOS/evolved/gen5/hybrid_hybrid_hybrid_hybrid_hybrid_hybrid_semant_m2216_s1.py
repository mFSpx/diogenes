# DARWIN HAMMER — match 2216, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s5.py (gen4)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_bayes_update__m1099_s1.py (gen3)
# born: 2026-05-29T23:41:20Z

"""Hybrid Decision‑Bandit & Semantic Bayesian Scheduler
=====================================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s5.py``  
  Provides a 3‑dimensional resource vector **eᵢ = [dᵢ, pᵢ, sᵢ]** for each entity,
  a linear feasibility test ``eᵢ ≤ budget`` and a contextual bandit whose
  propensity is scaled by a scalar *VRAM store* that follows  

  ``store_{t+1} = (1‑β)·store_t + α·reward_t`` .

* **Parent B** – ``hybrid_hybrid_semantic_neig_hybrid_bayes_update__m1099_s1.py``  
  Supplies semantic similarity via cosine distance and a Bayesian update of
  reward expectations (Beta‑Bernoulli conjugacy).

Mathematical Bridge
-------------------
Both parents treat a **single scalar** as a global regulator:

* In Parent A the scalar is the *VRAM store* that multiplies each bandit
  propensity.
* In Parent B the scalar appears as the *posterior mean* of a Beta
  distribution (α/(α+β)) that quantifies confidence in an action’s reward.

The hybrid algorithm therefore **fuses** the two scalars by letting the
VRAM store act as the *α* parameter of a Beta prior, while the observed
reward updates the corresponding *β* parameter.  The resulting posterior
mean directly scales the bandit propensity.  Feasibility is still
checked against the resource budget.

The three core functions below implement:

1. ``build_resource_matrix`` – builds **A** from entity resource vectors.
2. ``bayesian_update`` – updates Beta parameters and returns the posterior
   mean (expected reward) for every action.
3. ``hybrid_select_action`` – computes a propensity that combines
   (posterior mean) × (semantic similarity) × (store scaling), filters by
   feasibility, and returns the chosen ``BanditAction``.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """An entity that can be selected by the scheduler."""
    entity_id: str
    resources: Tuple[float, float, float]          # (distance, privacy, decision)
    features: np.ndarray                           # semantic embedding vector


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float            # scaled inflow rate
    expected_reward: float       # posterior mean from Bayesian update
    confidence_bound: float      # posterior variance proxy
    algorithm: str = "HybridDecisionBanditSemantic"


@dataclass
class VRAMStore:
    """Scalar store that evolves with reward feedback."""
    value: float = 0.0
    alpha: float = 0.1          # learning rate for reward influx
    beta: float = 0.01          # decay rate

    def update(self, reward: float) -> None:
        """Discrete ODE update: store ← (1‑β)·store + α·reward."""
        self.value = (1.0 - self.beta) * self.value + self.alpha * reward

    def scaling(self) -> float:
        """Return a positive scaling factor derived from the store."""
        # Prevent division by zero; shift by 1 to keep scaling ≈1 when store≈0
        return 1.0 + self.value


# ----------------------------------------------------------------------
# Helper mathematics
# ----------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1‑D arrays."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_resource_matrix(entities: List[Entity]) -> np.ndarray:
    """
    Assemble the resource matrix **A** (shape N×3) where each row is
    the resource vector eᵢ = [dᵢ, pᵢ, sᵢ] of an entity.
    """
    return np.array([list(ent.resources) for ent in entities], dtype=float)


def bayesian_update(
    successes: Dict[str, int],
    failures: Dict[str, int],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Perform a Beta‑Bernoulli Bayesian update for each action.

    Returns:
        posterior_means – α/(α+β) for every action
        posterior_vars  – α·β/[(α+β)²·(α+β+1)] (used as a confidence proxy)
    """
    posterior_means: Dict[str, float] = {}
    posterior_vars: Dict[str, float] = {}
    for aid in successes.keys() | failures.keys():
        a = prior_alpha + successes.get(aid, 0)
        b = prior_beta + failures.get(aid, 0)
        mean = a / (a + b)
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        posterior_means[aid] = mean
        posterior_vars[aid] = var
    return posterior_means, posterior_vars


def hybrid_select_action(
    entities: List[Entity],
    budgets: Tuple[float, float, float],
    store: VRAMStore,
    successes: Dict[str, int],
    failures: Dict[str, int],
) -> BanditAction:
    """
    Choose an entity respecting resource budgets and using a
    propensity that fuses:

        posterior_mean  ×  semantic_similarity  ×  store_scaling

    The selected action updates the VRAM store and returns the
    corresponding ``BanditAction``.
    """
    # 1. Feasibility mask based on budgets
    resource_mat = build_resource_matrix(entities)                     # N×3
    budget_arr = np.array(budgets, dtype=float)
    feasible_mask = np.all(resource_mat <= budget_arr, axis=1)         # N‑bool

    if not np.any(feasible_mask):
        raise RuntimeError("No feasible entity under current budgets.")

    # 2. Compute posterior means and confidence bounds
    posterior_means, posterior_vars = bayesian_update(successes, failures)

    # 3. Compute pairwise semantic similarity to a *reference* embedding.
    #    For simplicity we use the first feasible entity as reference.
    ref_idx = int(np.where(feasible_mask)[0][0])
    ref_feat = entities[ref_idx].features
    similarities = np.array([
        cosine_similarity(ref_feat, ent.features) if feasible else 0.0
        for ent, feasible in zip(entities, feasible_mask)
    ])

    # 4. Assemble propensity
    scaling = store.scaling()
    propensities = np.array([
        posterior_means.get(ent.entity_id, 0.5) * similarities[i] * scaling
        for i, ent in enumerate(entities)
    ])

    # 5. Mask infeasible propensities
    propensities[~feasible_mask] = -np.inf

    # 6. Select the action with maximal propensity
    chosen_idx = int(np.argmax(propensities))
    chosen_entity = entities[chosen_idx]

    # 7. Create BanditAction result
    action = BanditAction(
        action_id=chosen_entity.entity_id,
        propensity=propensities[chosen_idx],
        expected_reward=posterior_means.get(chosen_entity.entity_id, 0.5),
        confidence_bound=posterior_vars.get(chosen_entity.entity_id, 0.0),
    )
    return action


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create a small pool of dummy entities
    num_entities = 6
    entities: List[Entity] = []
    for i in range(num_entities):
        eid = f"ent{i}"
        # Random resources in [0, 5)
        resources = tuple(random.uniform(0, 5) for _ in range(3))
        # Random 5‑dimensional semantic feature vector
        features = np.random.rand(5)
        entities.append(Entity(entity_id=eid, resources=resources, features=features))

    # Define a modest budget that makes only a subset feasible
    budgets = (3.0, 3.0, 3.0)

    # Initialise VRAM store and Bayesian counters
    store = VRAMStore(value=0.0, alpha=0.2, beta=0.05)
    successes: Dict[str, int] = {}
    failures: Dict[str, int] = {}

    # Run a few iterations of the hybrid scheduler
    for step in range(5):
        try:
            action = hybrid_select_action(
                entities=entities,
                budgets=budgets,
                store=store,
                successes=successes,
                failures=failures,
            )
        except RuntimeError as exc:
            print(f"[step {step}] {exc}")
            break

        # Simulate a stochastic reward (Bernoulli with p=0.6)
        reward = 1.0 if random.random() < 0.6 else 0.0

        # Update VRAM store
        store.update(reward)

        # Update Bayesian counters
        if reward > 0:
            successes[action.action_id] = successes.get(action.action_id, 0) + 1
        else:
            failures[action.action_id] = failures.get(action.action_id, 0) + 1

        # Print iteration diagnostics
        print(
            f"[step {step}] Selected {action.action_id} | "
            f"propensity={action.propensity:.4f} | reward={reward} | "
            f"store={store.value:.4f}"
        )
    sys.exit(0)