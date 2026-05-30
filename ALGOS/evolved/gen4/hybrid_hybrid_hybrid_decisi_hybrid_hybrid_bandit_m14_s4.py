# DARWIN HAMMER — match 14, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s3.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (gen2)
# born: 2026-05-29T23:26:19Z

"""
Hybrid Decision-Bandit Scheduler
================================

This module fuses the **resource‑knapsack topology** of *hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s3.py*
(Parent A) with the **contextual bandit + virtual‑VRAM store** topology of
*hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py* (Parent B).

Mathematical Bridge
-------------------
* Parent A defines a resource matrix **A** whose rows are *entities* (or models) and
  columns are three loads: spatial distance **d**, privacy load **p**, and decision
  score **s**. A binary selection vector **x** must satisfy  

  ``A.T @ x <= [spatial_budget, privacy_budget, decision_budget]``

* Parent B introduces a contextual bandit whose actions have a *propensity* (inflow)
  and a *confidence bound* (outflow).  The bandit’s learning‑rate ``η`` is modulated
  by a *virtual VRAM store* **S**, which evolves as  

  ``dS/dt = α·(inflow − outflow) − β·S``  

  and is discretised with a time step ``dt``.

The fusion treats each **bandit action** as an *entity* in the resource matrix:
``d_i``  ← propensity,  
``p_i``  ← confidence_bound,  
``s_i``  ← expected_reward.  

Model rows keep the original RAM‑load, privacy‑risk, and score from Parent A.
The VRAM store **S** is interpreted as the *spatial budget* for the knapsack,
so the bandit’s learning dynamics directly affect the feasibility region of the
resource selection.  The combined algorithm proceeds as:

1. Build the joint resource matrix **A** (actions ∪ models).
2. At each step, the bandit proposes an action using its current propensities.
3. If adding the action keeps ``A.T @ x`` within the dynamic budgets
   (spatial_budget = S, privacy_budget, decision_budget), the action is accepted.
4. The VRAM store is updated with the action’s inflow/outflow, which in turn
   rescales the learning‑rate for future bandit updates.

The three core functions below demonstrate this hybrid operation.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# ----------------------------------------------------------------------
# Data structures (union of both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """Represents an entity/action that participates in the resource knapsack."""
    id: str
    propensity: float          # inflow rate (spatial load)
    confidence_bound: float    # outflow rate (privacy load)
    expected_reward: float     # decision score


@dataclass(frozen=True)
class Model:
    """Represents a model with static resource consumption."""
    id: str
    ram_mb: float
    privacy_risk: float
    decision_score: float


# ----------------------------------------------------------------------
# Hybrid resource matrix builder
# ----------------------------------------------------------------------
def build_resource_matrix(
    entities: List[Entity],
    models: List[Model],
    privacy_risk_mean: float,
    alpha_priv: float = 1.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Construct the combined resource matrix A (rows = entities ∪ models,
    columns = [spatial, privacy, score]).

    For entities:
        spatial  = propensity
        privacy  = confidence_bound
        score    = expected_reward

    For models:
        spatial  = ram_mb                           (treated as spatial load)
        privacy  = alpha_priv * privacy_risk_mean   (scaled privacy load)
        score    = decision_score
    """
    rows = []
    labels = []

    # Entity rows
    for e in entities:
        rows.append([e.propensity, e.confidence_bound, e.expected_reward])
        labels.append(f"E:{e.id}")

    # Model rows
    for m in models:
        rows.append([m.ram_mb,
                     alpha_priv * privacy_risk_mean,
                     m.decision_score])
        labels.append(f"M:{m.id}")

    A = np.array(rows, dtype=float)
    return A, labels


# ----------------------------------------------------------------------
# Virtual VRAM store (bandit side)
# ----------------------------------------------------------------------
class VRAMStore:
    """
    Simulates a virtual VRAM store whose level S(t) influences the spatial budget.
    The dynamics follow the discretised differential equation:

        S_{t+1} = (1 - decay) * S_t + dt * (α * (inflow - outflow) - β * S_t)

    The store also rescales the bandit learning rate:
        η = base_eta * (1 + S / budget)
    """

    def __init__(
        self,
        budget_mb: float = 8192.0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        decay: float = 0.99,
    ) -> None:
        self.budget = budget_mb
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.decay = decay
        self.level = 0.0  # initial store occupancy

    @property
    def spatial_budget(self) -> float:
        """Current spatial budget equals the remaining VRAM."""
        return max(self.budget - self.level, 0.0)

    @property
    def learning_rate(self) -> float:
        """Learning rate modulated by the current store level."""
        return self.base_eta * (1.0 + self.level / self.budget)

    def step(self, inflow: float, outflow: float) -> None:
        """Update store level with one discretised step."""
        delta = self.alpha * (inflow - outflow) - self.beta * self.level
        self.level = self.decay * self.level + self.dt * delta
        # Clamp to [0, budget]
        self.level = max(min(self.level, self.budget), 0.0)


# ----------------------------------------------------------------------
# Simple contextual bandit (linear Thompson Sampling style)
# ----------------------------------------------------------------------
class LinearBandit:
    """
    Maintains a weight vector w ∈ ℝ^d for d‑dimensional context.
    The expected reward for an action a with context vector φ(a) is φ(a)·w.
    Propensity (inflow) is used as a scalar feature; confidence bound as another.
    """

    def __init__(self, d: int, rng: Optional[np.random.Generator] = None):
        self.d = d
        self.rng = rng or np.random.default_rng()
        self.w = np.zeros(d)  # start with zero weights

    def predict(self, phi: np.ndarray) -> float:
        """Linear prediction φ·w."""
        return float(phi @ self.w)

    def update(self, phi: np.ndarray, reward: float, lr: float) -> None:
        """Stochastic gradient ascent on the squared‑error loss."""
        error = reward - self.predict(phi)
        self.w += lr * error * phi


# ----------------------------------------------------------------------
# Hybrid selection loop
# ----------------------------------------------------------------------
def hybrid_select(
    entities: List[Entity],
    models: List[Model],
    privacy_budget: float,
    decision_budget: float,
    store: VRAMStore,
    bandit: LinearBandit,
) -> Tuple[List[str], np.ndarray]:
    """
    Perform a greedy hybrid selection.

    Returns
    -------
    selected_labels : list of identifiers (entity/model) that were accepted.
    final_x        : binary selection vector aligned with the order of
                     ``build_resource_matrix`` rows.
    """
    # Pre‑compute static privacy load for models (uses mean privacy risk)
    mean_privacy = np.mean([m.privacy_risk for m in models]) if models else 0.0
    A, labels = build_resource_matrix(entities, models, mean_privacy)

    n = A.shape[0]
    x = np.zeros(n, dtype=int)   # selection vector
    used = np.zeros(3, dtype=float)  # cumulative loads [spatial, privacy, decision]

    # Helper to test feasibility
    def feasible(candidate_idx: int) -> bool:
        new_used = used + A[candidate_idx]
        return (new_used[0] <= store.spatial_budget and
                new_used[1] <= privacy_budget and
                new_used[2] <= decision_budget)

    # Greedy loop over entities (bandit‑driven ordering)
    remaining = set(range(len(entities)))  # indices of entity rows
    while remaining:
        # Build feature vectors φ_i = [propensity, confidence, 1] for each candidate
        phi_matrix = np.array([
            [entities[i].propensity, entities[i].confidence_bound, 1.0]
            for i in remaining
        ])
        # Predict rewards
        preds = np.array([bandit.predict(phi) for phi in phi_matrix])
        # Choose highest predicted reward that is feasible
        sorted_idx = np.argsort(-preds)  # descending
        chosen = None
        for idx in sorted_idx:
            ent_idx = list(remaining)[idx]
            row_idx = ent_idx  # entity rows are first in A
            if feasible(row_idx):
                chosen = ent_idx
                break
        if chosen is None:
            # No feasible entity left
            break

        # Accept the chosen entity
        row_idx = chosen
        x[row_idx] = 1
        used += A[row_idx]

        # Update bandit with observed reward (use true expected_reward)
        phi = np.array([entities[chosen].propensity,
                        entities[chosen].confidence_bound, 1.0])
        bandit.update(phi, entities[chosen].expected_reward, store.learning_rate)

        # Update VRAM store (inflow = propensity, outflow = confidence_bound)
        store.step(inflow=entities[chosen].propensity,
                   outflow=entities[chosen].confidence_bound)

        remaining.remove(chosen)

    # After entity selection, attempt to pack models greedily by score
    model_indices = range(len(entities), n)
    for idx in model_indices:
        if feasible(idx):
            x[idx] = 1
            used += A[idx]

    selected_labels = [lbl for flag, lbl in zip(x, labels) if flag]
    return selected_labels, x


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def random_entities(k: int, rng: Optional[np.random.Generator] = None) -> List[Entity]:
    """Generate *k* synthetic entities with random attributes."""
    rng = rng or np.random.default_rng()
    ents = []
    for i in range(k):
        prop = rng.uniform(10, 200)          # propensity (spatial)
        conf = rng.uniform(5, 100)           # confidence (privacy)
        rew = rng.uniform(0, 1)              # expected reward (score)
        ents.append(Entity(id=f"e{i}", propensity=prop,
                           confidence_bound=conf,
                           expected_reward=rew))
    return ents


def random_models(k: int, rng: Optional[np.random.Generator] = None) -> List[Model]:
    """Generate *k* synthetic models."""
    rng = rng or np.random.default_rng()
    mods = []
    for i in range(k):
        ram = rng.uniform(100, 2000)          # MB
        priv = rng.uniform(0.1, 1.0)          # privacy risk
        score = rng.uniform(0, 1)            # decision score
        mods.append(Model(id=f"m{i}", ram_mb=ram,
                          privacy_risk=priv,
                          decision_score=score))
    return mods


def smoke_test() -> None:
    """Run a minimal end‑to‑end execution to verify that the hybrid works."""
    rng = np.random.default_rng(42)

    ents = random_entities(8, rng)
    mods = random_models(4, rng)

    store = VRAMStore(budget_mb=8192.0,
                      base_eta=0.01,
                      alpha=0.8,
                      beta=0.2,
                      dt=1.0,
                      decay=0.98)

    bandit = LinearBandit(d=3, rng=rng)

    privacy_budget = 1500.0
    decision_budget = 3.0

    selected, x = hybrid_select(
        entities=ents,
        models=mods,
        privacy_budget=privacy_budget,
        decision_budget=decision_budget,
        store=store,
        bandit=bandit,
    )

    print("Selected items:", selected)
    print("Final VRAM store level:", store.level)
    print("Final learning rate:", store.learning_rate)
    print("Selection vector:", x)


if __name__ == "__main__":
    smoke_test()