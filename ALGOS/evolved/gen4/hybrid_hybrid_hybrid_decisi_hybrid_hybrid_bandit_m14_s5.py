# DARWIN HAMMER — match 14, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s3.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (gen2)
# born: 2026-05-29T23:26:19Z

"""Hybrid Decision‑Bandit Resource Scheduler
================================================

This module fuses the two parent algorithms:

* **Parent A** – *hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s3.py*  
  Provides a 3‑dimensional resource vector **eᵢ = [dᵢ, pᵢ, sᵢ]** for every
  entity (distance, privacy‑load, decision‑score) and a linear‑constraint
  selection problem ``Aᵀ·x ≤ budgets``.

* **Parent B** – *hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py*  
  Implements a contextual bandit whose propensity and learning‑rate are
  modulated by a virtual VRAM store and a linear “TTT” weight matrix.

**Mathematical bridge**

The bridge is the *resource budget* that both algorithms treat as a
scalar quantity:

* In Parent A the budget appears as the right‑hand side of the linear
  constraints.
* In Parent B the virtual VRAM store ``store`` is a scalar that is updated
  by an ODE ``d store / dt = α·reward – β·store`` and is used to scale the
  bandit’s propensity.

The hybrid algorithm therefore:

1. Builds the resource matrix **A** (entities ∪ models) exactly as in
   Parent A.
2. Uses the current ``store`` value as a *budget multiplier* that scales
   each action’s propensity.
3. Selects an action with the bandit, but only if the corresponding row
   of **A** is feasible under the (scaled) budgets.
4. Updates the VRAM store with the observed reward and decays it,
   feeding the next iteration.

The three core functions below demonstrate this integration.

"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional

import numpy as np

# ----------------------------------------------------------------------
# Data structures (derived from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float            # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float      # interpreted as outflow rate
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Observed reward for a given action."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Hybrid bandit‑TTT scheduler (Parent B core, corrected)
# ----------------------------------------------------------------------
class HybridBanditTTT:
    """
    Contextual bandit whose learning‑rate and propensity are modulated by a
    virtual VRAM store.  The TTT weight matrix ``W`` is a linear model that
    maps a context vector to expected rewards.
    """

    DEFAULT_BUDGET_MB = 8192  # total VRAM budget for reporting (unused in core)

    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        store_decay: float = 0.99,
    ) -> None:
        """
        Parameters
        ----------
        d_in, d_out : dimensions of the TTT weight matrix.
        seed        : RNG seed for reproducibility.
        base_eta    : Baseline learning rate before modulation.
        alpha, beta : Coefficients for the store differential equation.
        dt          : Time step for store integration.
        store_decay : Exponential decay applied to the store each step.
        """
        self.rng = np.random.default_rng(seed)
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.store_decay = store_decay

        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        # Initialise a simple linear weight matrix (TTT model)
        self.W = self.rng.normal(scale=0.1, size=(self.d_out, self.d_in))

        # Virtual VRAM store (scalar)
        self.store: float = self.DEFAULT_BUDGET_MB * 0.5  # start at 50 % capacity

    # ------------------------------------------------------------------
    # Core bandit operations
    # ------------------------------------------------------------------
    def _predict(self, context: np.ndarray) -> np.ndarray:
        """Linear prediction: r̂ = W·context."""
        return self.W @ context

    def select_action(
        self,
        context: np.ndarray,
        candidate_actions: List[BanditAction],
    ) -> BanditAction:
        """
        Choose the action with the highest *scaled* propensity.
        Scaling = store / (store + 1) to keep the factor in (0,1).
        """
        scale = self.store / (self.store + 1.0)
        best = max(
            candidate_actions,
            key=lambda a: a.propensity * scale,
        )
        return best

    def update(
        self,
        context: np.ndarray,
        chosen_action: BanditAction,
        observed_reward: float,
    ) -> None:
        """
        Perform a single stochastic gradient step on the TTT model and update
        the VRAM store according to the ODE from Parent B.
        """
        # ----- TTT weight update (simple SGD) -----
        pred = self._predict(context)
        error = observed_reward - pred[0]  # assume scalar reward
        grad = -2.0 * error * context  # dL/dW[0,:]
        self.W[0, :] -= self.base_eta * grad

        # ----- Store ODE integration (Euler) -----
        dstore = self.alpha * observed_reward - self.beta * self.store
        self.store += self.dt * dstore
        # Apply exponential decay to simulate eviction
        self.store *= self.store_decay

        # Keep store non‑negative
        self.store = max(self.store, 0.0)


# ----------------------------------------------------------------------
# Resource matrix construction (Parent A core)
# ----------------------------------------------------------------------
def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Return distance in metres between two lat/lon points."""
    R = 6371000.0  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def build_resource_matrix(
    entities: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    reference_location: Tuple[float, float],
    beta_privacy: float = 1.0,
    alpha_score: float = 1.0,
) -> np.ndarray:
    """
    Construct the combined resource matrix A (rows = entities ∪ models,
    columns = [spatial, privacy, decision]).

    Entity row:
        dᵢ = haversine distance to reference location
        pᵢ = β·σᵢ   (σᵢ = 1 if signature collides, else 0)
        sᵢ = α·score

    Model row:
        dⱼ = RAM consumption (treated as spatial load)
        pⱼ = α·τⱼ·μ   (privacy‑load analogue)
        sⱼ = 0        (models have no decision score)
    """
    # ---- Entity part -------------------------------------------------
    # Detect signature collisions
    signature_counts: Dict[Any, int] = {}
    for e in entities:
        sig = e.get("signature")
        signature_counts[sig] = signature_counts.get(sig, 0) + 1

    entity_rows = []
    for e in entities:
        lat, lon = e["lat"], e["lon"]
        d = haversine_distance(lat, lon, *reference_location)
        sigma = 1 if signature_counts[e["signature"]] > 1 else 0
        p = beta_privacy * sigma
        s = alpha_score * e.get("score", 0.0)
        entity_rows.append([d, p, s])

    # ---- Model part --------------------------------------------------
    # Compute mean privacy risk μ across provided records (use entities)
    privacy_risks = [e.get("privacy_risk", 0.0) for e in entities]
    mu = float(np.mean(privacy_risks)) if privacy_risks else 0.0

    tier_factor = {1: 1, 2: 2, 3: 3}
    model_rows = []
    for m in models:
        ram = m.get("RAM", 0.0)
        tier = tier_factor.get(m.get("tier", 1), 1)
        p = alpha_score * tier * mu
        s = 0.0
        model_rows.append([ram, p, s])

    # Combine
    A = np.array(entity_rows + model_rows, dtype=float)
    return A


# ----------------------------------------------------------------------
# Greedy feasibility selector (Parent A core)
# ----------------------------------------------------------------------
def greedy_feasible_selection(
    A: np.ndarray,
    budgets: Tuple[float, float, float],
) -> np.ndarray:
    """
    Return a binary indicator vector x (shape = rows of A) that selects a
    subset of rows while respecting the linear constraints
    Aᵀ·x ≤ budgets.

    The greedy heuristic sorts rows by descending decision score (column 2)
    and picks them as long as the remaining budget permits.
    """
    spatial_budget, privacy_budget, decision_budget = budgets
    n = A.shape[0]
    x = np.zeros(n, dtype=int)

    # Sort indices by decision score (higher is better)
    sorted_idx = np.argsort(-A[:, 2])

    remaining_spatial = spatial_budget
    remaining_privacy = privacy_budget
    remaining_decision = decision_budget

    for idx in sorted_idx:
        d, p, s = A[idx]
        if (
            d <= remaining_spatial
            and p <= remaining_privacy
            and s <= remaining_decision
        ):
            x[idx] = 1
            remaining_spatial -= d
            remaining_privacy -= p
            remaining_decision -= s

    return x


# ----------------------------------------------------------------------
# Hybrid iteration that ties everything together
# ----------------------------------------------------------------------
def run_hybrid_iteration(
    entities: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    bandit: HybridBanditTTT,
    reference_location: Tuple[float, float],
    budgets: Tuple[float, float, float],
    context: np.ndarray,
) -> Tuple[BanditAction, np.ndarray, np.ndarray]:
    """
    One iteration of the hybrid algorithm:

    1. Build resource matrix A.
    2. Compute a feasibility mask via the greedy selector.
    3. Generate candidate actions (one per row) with propensities derived
       from the row's decision score.
    4. Let the bandit pick an action respecting the feasibility mask.
    5. Observe a synthetic reward (here we reuse the row's decision score)
       and update the bandit/store.
    6. Return the chosen action, the feasibility vector, and the resource
       matrix for inspection.
    """
    A = build_resource_matrix(entities, models, reference_location)

    # Feasibility mask (binary)
    feasible_mask = greedy_feasible_selection(A, budgets)

    # Build candidate actions only for feasible rows
    candidate_actions: List[BanditAction] = []
    for idx, feasible in enumerate(feasible_mask):
        if not feasible:
            continue
        # Use decision score as a proxy for propensity (scaled)
        propensity = max(A[idx, 2], 0.0) + 1e-3
        action = BanditAction(
            action_id=str(idx),
            propensity=propensity,
            expected_reward=A[idx, 2],
            confidence_bound=0.1,  # placeholder
            algorithm="HybridDecisionBandit",
        )
        candidate_actions.append(action)

    if not candidate_actions:
        # No feasible actions – return a dummy no‑op
        dummy = BanditAction(
            action_id="none",
            propensity=0.0,
            expected_reward=0.0,
            confidence_bound=0.0,
            algorithm="HybridDecisionBandit",
        )
        return dummy, feasible_mask, A

    # Bandit selects
    chosen = bandit.select_action(context, candidate_actions)

    # Synthetic reward: use the expected_reward field (could be replaced by
    # an external evaluation)
    reward = chosen.expected_reward

    # Update bandit and store
    bandit.update(context, chosen, reward)

    return chosen, feasible_mask, A


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic data
    entities = [
        {"lat": 40.0, "lon": -74.0, "signature": "abc", "score": 5.0, "privacy_risk": 0.2},
        {"lat": 40.1, "lon": -74.1, "signature": "def", "score": 3.0, "privacy_risk": 0.1},
        {"lat": 40.2, "lon": -74.2, "signature": "abc", "score": 4.0, "privacy_risk": 0.3},
    ]

    models = [
        {"RAM": 1024.0, "tier": 1},
        {"RAM": 2048.0, "tier": 2},
    ]

    reference_location = (40.05, -74.05)

    # Budgets: generous enough to allow some selections
    budgets = (1e7, 10.0, 10.0)  # spatial (m), privacy, decision

    # Context vector for the bandit (dimension matches d_in)
    context = np.array([0.5, 1.2, -0.3])

    # Initialise hybrid bandit‑TTT
    bandit = HybridBanditTTT(d_in=context.shape[0], seed=42)

    # Run a few iterations to demonstrate learning / store dynamics
    for step in range(5):
        action, mask, A = run_hybrid_iteration(
            entities,
            models,
            bandit,
            reference_location,
            budgets,
            context,
        )
        print(f"Step {step+1}:")
        print(f"  Chosen action id: {action.action_id}")
        print(f"  Propensity (scaled): {action.propensity:.4f}")
        print(f"  Expected reward: {action.expected_reward:.4f}")
        print(f"  Store after update: {bandit.store:.2f} MB")
        print(f"  Feasible rows: {np.where(mask)[0].tolist()}")
        print("-" * 40)

    sys.exit(0)