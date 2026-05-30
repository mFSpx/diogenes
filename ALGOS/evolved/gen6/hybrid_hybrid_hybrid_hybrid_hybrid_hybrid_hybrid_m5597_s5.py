# DARWIN HAMMER — match 5597, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_jepa_e_hybrid_hybrid_hybrid_m2401_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m449_s1.py (gen4)
# born: 2026-05-30T00:03:23Z

"""Hybrid Energy‑Risk Bandit Model Pool
====================================

This module fuses the two parent algorithms:

* **Parent A** – a memory‑constrained ``ModelPool`` that tracks an
  ``energy`` score while loading/evicting models and that can compute a
  similarity matrix via a dot‑product of model feature vectors.
* **Parent B** – a lightweight Thompson‑sampling bandit that maintains a
  Beta posterior for each possible action (here: loading a particular model).

**Mathematical bridge**

The bridge is a *risk‑aware selection* rule:

1. The similarity matrix **S** ∈ ℝⁿˣⁿ (dot‑product of feature vectors) gives a
   context‑dependent *expected reward* for loading model *i*:
   \[
   r_i = \frac{1}{n}\sum_j S_{ij}
   \]
   (average similarity to the current pool – higher similarity means lower
   risk of incompatibility).

2. The Thompson‑sampling bandit draws a sample
   \(\theta_i \sim \mathrm{Beta}(\alpha_i,\beta_i)\) for each model *i*.
   The **selection score** for model *i* is the product
   \[
   \sigma_i = \theta_i \cdot r_i .
   \]
   The model with maximal \(\sigma_i\) is loaded (or evicted) by the
   ``ModelPool``.  

3. After the load operation the pool’s energy change ΔE is observed.
   Normalising ΔE to a probability‑like reward
   \(\rho = \mathrm{sigmoid}(-\Delta E)\) (energy reduction is good) yields a
   scalar in \([0,1]\) that updates the bandit’s Beta posterior:
   \[
   \alpha_i \leftarrow \alpha_i + \rho,\qquad
   \beta_i  \leftarrow \beta_i + (1-\rho).
   \]

Thus the bandit’s posterior guides model selection, while the pool’s
energy feedback continuously refines the posterior – a true hybrid of the
two parent topologies.
"""

import sys
import math
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable

# ----------------------------------------------------------------------
# Parent‑A inspired structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str
    features: Tuple[float, ...]  # fixed‑length feature vector for similarity


class ModelPool:
    """Memory‑constrained pool that tracks an abstract ``energy`` score."""

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0

    # ------------------------------------------------------------------
    # Energy helpers
    # ------------------------------------------------------------------
    @property
    def energy(self) -> float:
        """Current energy (lower is better)."""
        return self._energy

    def _penalise(self, amount: float) -> None:
        self._energy += amount

    def _reward(self, amount: float) -> None:
        self._energy -= amount

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_fit(self, model: ModelTier) -> bool:
        return self._used() + model.ram_mb <= self.ram_ceiling_mb

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def add_model(self, model: ModelTier) -> None:
        """Add model to the pool respecting tier‑conflict penalties."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._penalise(1e10)                     # tier conflict
        if not self.can_fit(model):
            self._penalise(1e6)                      # out‑of‑memory
        self.loaded[model.name] = model

    def load(self, model: ModelTier) -> None:
        """Load a model, rewarding the operation."""
        self._reward(1e4)
        self.add_model(model)

    def evict(self, model_name: str) -> None:
        """Evict a model, rewarding the eviction."""
        if model_name in self.loaded:
            self._reward(1e3)
            del self.loaded[model_name]

    # ------------------------------------------------------------------
    # Similarity matrix (dot‑product of feature vectors)
    # ------------------------------------------------------------------
    def similarity_matrix(self) -> np.ndarray:
        """Return the pairwise dot‑product matrix of loaded model features."""
        if not self.loaded:
            return np.zeros((0, 0))
        feats = np.stack([np.array(m.features) for m in self.loaded.values()])
        return feats @ feats.T  # shape (n_loaded, n_loaded)


# ----------------------------------------------------------------------
# Parent‑B inspired Thompson Bandit
# ----------------------------------------------------------------------
class ThompsonBandit:
    """Lightweight Thompson‑sampling bandit for continuous (0‑1) rewards."""

    def __init__(self, actions: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self._alpha: Dict[str, float] = {a: prior_alpha for a in actions}
        self._beta:  Dict[str, float] = {a: prior_beta  for a in actions}
        self._actions = actions

    def sample(self) -> str:
        """Draw a beta sample for each action and return the best."""
        draws = {a: np.random.beta(self._alpha[a], self._beta[a]) for a in self._actions}
        return max(draws, key=draws.get)

    def update(self, action: str, reward: float) -> None:
        """
        Update the Beta posterior for *action* using a reward in [0,1].
        The reward is interpreted as a pseudo‑success count.
        """
        if action not in self._actions:
            raise ValueError(f"Unknown action {action}")
        # Clip reward to avoid pathological values
        rho = max(0.0, min(1.0, reward))
        self._alpha[action] += rho
        self._beta[action]  += (1.0 - rho)

    def posterior_means(self) -> Dict[str, float]:
        """Return the mean of each Beta posterior."""
        return {a: self._alpha[a] / (self._alpha[a] + self._beta[a]) for a in self._actions}


# ----------------------------------------------------------------------
# Hybrid functions (mathematical fusion)
# ----------------------------------------------------------------------
def average_similarity(pool: ModelPool) -> Dict[str, float]:
    """
    Compute, for each loaded model, the average similarity to all other
    loaded models.  The result is used as the *contextual reward* r_i.
    """
    S = pool.similarity_matrix()
    if S.size == 0:
        return {}
    n = S.shape[0]
    # Zero out diagonal to avoid self‑similarity bias
    np.fill_diagonal(S, 0.0)
    avg = S.sum(axis=1) / max(1, n - 1)
    return dict(zip(pool.loaded.keys(), avg.tolist()))


def select_and_load(pool: ModelPool, bandit: ThompsonBandit,
                    candidate_models: List[ModelTier]) -> Tuple[bool, str]:
    """
    Hybrid selection step:

    1. Compute average similarity scores r_i for the *currently* loaded models.
    2. For every candidate model compute a combined score σ_i = θ_i * r_i',
       where θ_i is a Thompson sample and r_i' is the similarity of the
       candidate to the current pool (average dot‑product with loaded features).
    3. Load the candidate with maximal σ_i (if it fits); otherwise evict the
       lowest‑energy model and retry.

    Returns a tuple (loaded_successfully, chosen_model_name).
    """
    # Pre‑compute similarity of candidates to the current pool
    loaded_feats = np.stack([np.array(m.features) for m in pool.loaded.values()]) \
        if pool.loaded else np.empty((0, len(candidate_models[0].features)))
    cand_scores: Dict[str, float] = {}

    for cand in candidate_models:
        if loaded_feats.size == 0:
            # No loaded models – treat similarity as 1.0 (neutral)
            sim = 1.0
        else:
            cand_vec = np.array(cand.features)
            sims = loaded_feats @ cand_vec
            sim = float(sims.mean())
        # Thompson sample for this candidate's action name
        theta = np.random.beta(
            bandit._alpha.get(cand.name, 1.0),
            bandit._beta.get(cand.name, 1.0)
        )
        cand_scores[cand.name] = theta * sim

    # Choose best candidate
    chosen_name = max(cand_scores, key=cand_scores.get)
    chosen_model = next(m for m in candidate_models if m.name == chosen_name)

    # Try to load; if not enough memory, evict the *largest* model and retry once
    if pool.can_fit(chosen_model):
        pool.load(chosen_model)
        return True, chosen_name
    else:
        # Evict the model with the highest RAM usage (simple heuristic)
        if pool.loaded:
            evict_name = max(pool.loaded, key=lambda n: pool.loaded[n].ram_mb)
            pool.evict(evict_name)
            if pool.can_fit(chosen_model):
                pool.load(chosen_model)
                return True, chosen_name
        # Still cannot fit – abort
        return False, chosen_name


def observe_and_update(pool: ModelPool, bandit: ThompsonBandit,
                       action_name: str, prev_energy: float) -> None:
    """
    Observe the change in energy after an action and update the bandit.

    The reward is defined as ρ = sigmoid(−ΔE) where ΔE = new_energy − prev_energy.
    A reduction in energy yields a reward close to 1.
    """
    delta_e = pool.energy - prev_energy
    # Sigmoid maps large negative ΔE (energy drop) → ≈1, positive ΔE → ≈0
    reward = 1.0 / (1.0 + math.exp(delta_e))
    bandit.update(action_name, reward)


def minimum_spanning_tree_cost(pool: ModelPool) -> float:
    """
    Compute a simple MST cost over the current similarity matrix using Prim's
    algorithm.  Edge weight is defined as 1 − similarity (so higher similarity
    → lower cost).  The total cost is a proxy for the “risk” of the pool
    configuration.
    """
    S = pool.similarity_matrix()
    if S.shape[0] <= 1:
        return 0.0
    n = S.shape[0]
    visited = np.zeros(n, dtype=bool)
    # Start from node 0
    visited[0] = True
    total_cost = 0.0
    while not visited.all():
        # Mask edges that connect visited → unvisited
        mask = np.outer(visited, ~visited)
        # Edge cost = 1 - similarity
        costs = (1.0 - S) * mask
        # Find minimum cost edge
        i, j = np.unravel_index(np.argmin(costs + np.eye(n) * 1e9), costs.shape)
        total_cost += costs[i, j]
        visited[j] = True
    return total_cost


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create a small catalogue of candidate models
    candidates = [
        ModelTier(name="A", ram_mb=1500, tier="T1", features=tuple(np.random.rand(5))),
        ModelTier(name="B", ram_mb=2000, tier="T2", features=tuple(np.random.rand(5))),
        ModelTier(name="C", ram_mb=1200, tier="T3", features=tuple(np.random.rand(5))),
        ModelTier(name="D", ram_mb=1800, tier="T1", features=tuple(np.random.rand(5))),
    ]

    # Initialise pool with a modest ceiling
    pool = ModelPool(ram_ceiling_mb=5000)

    # Initialise bandit with the candidate names as actions
    bandit = ThompsonBandit(actions=[m.name for m in candidates])

    # Run a few hybrid cycles
    for step in range(5):
        prev_e = pool.energy
        loaded, chosen = select_and_load(pool, bandit, candidates)
        if loaded:
            observe_and_update(pool, bandit, chosen, prev_e)
        else:
            # If load failed, still give a small penalty reward
            bandit.update(chosen, 0.1)

        # Optional diagnostics
        print(f"Step {step+1}: loaded={loaded}, model={chosen}, energy={pool.energy:.2f}")
        print("  Posterior means:", {k: f'{v:.3f}' for k, v in bandit.posterior_means().items()})
        print("  MST cost:", minimum_spanning_tree_cost(pool))
        print("-" * 40)

    sys.exit(0)