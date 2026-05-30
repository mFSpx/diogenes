# DARWIN HAMMER — match 3727, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1458_s1.py (gen6)
# parent_b: hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s0.py (gen3)
# born: 2026-05-29T23:51:28Z

"""Hybrid Thompson‑Voronoi‑Caputo Bandit

This module fuses the two parent algorithms:

* **Parent A** – *HybridThompsonVoronoiCaputo* which combined a Thompson‑sampling
  bandit, Voronoi partitioning of a feature space and a Caputo fractional
  derivative acting on the reward estimates.
* **Parent B** – a lightweight Thompson / LinUCB bandit together with a feature
  extractor (`extract_full_features`).

**Mathematical bridge**

The bridge is the *reward‑value field* ϱ(t) defined on the Voronoi cells.
Parent B updates ϱ by simple additive rewards per action.  
Parent A evolves ϱ in continuous (fractional) time using the Caputo derivative
\(D^{\alpha}_{0+} ϱ(t)\).  

In the hybrid we (i) extract a dense feature vector from the raw context,
(ii) assign the feature to a Voronoi region, (iii) update the per‑action
reward estimate of that region, and (iv) evolve the whole vector of estimates
with a discrete Caputo operator before feeding the evolved estimate into a
Thompson‑sampling selector.  The result is a single unified system where the
fractional dynamics and the bandit policy are mathematically coupled.

"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

# ----------------------------------------------------------------------
# Utilities from Parent B
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
    propensity: float = 1.0

def extract_full_features(text: str) -> dict[str, float]:
    """Deterministic pseudo‑feature extractor used in Parent B."""
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_signal_to_noise", "operator_entropy"
    ]
    return {k: rnd.random() for k in keys}

# ----------------------------------------------------------------------
# Core of the hybrid algorithm
# ----------------------------------------------------------------------
def caputo_fractional_derivative(values: np.ndarray, alpha: float, dt: float) -> np.ndarray:
    """
    Discrete Caputo derivative (Grünwald‑Letnikov approximation).

    D^α f(t_n) ≈ (1/dt^α) * Σ_{k=0}^{n} w_k * (f_{n‑k} - f_0)
    with weights w_k = (-1)^k * binom(α, k).

    Parameters
    ----------
    values : np.ndarray
        1‑D array of function samples f(t_k), k = 0…n.
    alpha : float
        Fractional order 0 < α ≤ 1.
    dt : float
        Sampling interval.

    Returns
    -------
    np.ndarray
        Approximation of the Caputo derivative at the last time point.
    """
    n = len(values) - 1
    if n < 0:
        raise ValueError("values must contain at least one element")
    # compute binomial coefficients recursively
    w = np.empty(n + 1, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, n + 1):
        w[k] = w[k - 1] * (alpha - (k - 1)) / k
    w = (-1) ** np.arange(n + 1) * w
    diff = values - values[0]  # (f_{n‑k} - f_0) for each k
    derivative = (1.0 / dt ** alpha) * np.dot(w, diff[::-1])
    return np.full_like(values, derivative)  # broadcast to same shape

def voronoi_partition(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Assign each feature vector to the nearest Voronoi centre.

    Returns an integer array of region indices, same length as ``features``.
    """
    # Euclidean distance matrix (samples × centres)
    dists = np.linalg.norm(
        features[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2
    )
    return np.argmin(dists, axis=1)

class HybridThompsonVoronoiCaputo:
    """
    Unified hybrid algorithm.

    * Maintains a Thompson‑sampling Beta posterior per (action, region).
    * Updates posteriors with observed rewards (Parent B logic).
    * After each batch, evolves the posterior means with a Caputo fractional
      derivative (Parent A logic).
    * Uses Voronoi partitioning of the extracted feature space to map a
      context to a region.
    """
    def __init__(
        self,
        actions: List[str],
        region_centers: np.ndarray,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        frac_alpha: float = 0.8,
        dt: float = 1.0,
    ):
        self.actions = actions
        self.centers = region_centers  # shape (R, d)
        self.R = self.centers.shape[0]

        # Beta parameters per (region, action)
        self._alpha: np.ndarray = np.full((self.R, len(actions)), prior_alpha, dtype=np.float64)
        self._beta: np.ndarray = np.full((self.R, len(actions)), prior_beta, dtype=np.float64)

        # Fractional‑derivative bookkeeping
        self.frac_alpha = frac_alpha
        self.dt = dt
        self._reward_history: List[np.ndarray] = []  # store mean reward vectors per step

    # ------------------------------------------------------------------
    # Policy update utilities (Parent B)
    # ------------------------------------------------------------------
    def _reward_mean(self) -> np.ndarray:
        """Mean reward per (region, action) from current Beta posteriors."""
        return self._alpha / (self._alpha + self._beta)

    def update_policy(self, updates: List[BanditUpdate], region_ids: List[int]) -> None:
        """
        Apply a batch of BanditUpdate objects.

        ``region_ids`` must be aligned with ``updates`` – the i‑th update is
        assumed to have originated from ``region_ids[i]``.
        """
        for upd, rid in zip(updates, region_ids):
            if upd.action_id not in self.actions:
                continue
            a_idx = self.actions.index(upd.action_id)
            # Simple Bernoulli‑like update (reward in [0,1])
            self._alpha[rid, a_idx] += upd.reward
            self._beta[rid, a_idx] += 1.0 - upd.reward

    # ------------------------------------------------------------------
    # Fractional evolution (Parent A)
    # ------------------------------------------------------------------
    def evolve_rewards(self) -> None:
        """
        Evolve the mean reward field with a Caputo derivative.
        The derivative is computed on the flattened reward vector across
        all regions and actions and then added back (Euler step).
        """
        current = self._reward_mean().ravel()
        self._reward_history.append(current.copy())
        if len(self._reward_history) < 2:
            # Need at least two points for a meaningful derivative
            return
        hist = np.vstack(self._reward_history)  # shape (t, R*|A|)
        deriv = caputo_fractional_derivative(hist[-1], self.frac_alpha, self.dt)
        # Simple explicit Euler integration
        updated = current + self.dt * deriv
        # Project back to [0,1] and re‑parameterise the Beta distribution
        updated = np.clip(updated, 0.0, 1.0)
        # Reset Beta parameters to reflect the new mean while keeping total count
        total_counts = self._alpha + self._beta
        self._alpha = updated * total_counts
        self._beta = (1.0 - updated) * total_counts

    # ------------------------------------------------------------------
    # Action selection (bridge of both parents)
    # ------------------------------------------------------------------
    def select_action(
        self,
        context_text: str,
        algorithm: str = "thompson",
        seed: Optional[int | str] = 7,
    ) -> BanditAction:
        """
        1. Extract dense features from the raw text.
        2. Assign the feature vector to a Voronoi region.
        3. Sample from the Thompson posterior of that region.
        4. Return the chosen BanditAction.
        """
        rng = random.Random(seed)
        feats = np.array(list(extract_full_features(context_text).values()), dtype=np.float64)
        # Pad / truncate to match centre dimensionality
        d = self.centers.shape[1]
        if feats.shape[0] < d:
            feats = np.pad(feats, (0, d - feats.shape[0]), constant_values=0.0)
        elif feats.shape[0] > d:
            feats = feats[:d]

        region = voronoi_partition(feats[np.newaxis, :], self.centers)[0]

        if algorithm == "thompson":
            samples = [
                rng.betavariate(self._alpha[region, i] + 1e-8, self._beta[region, i] + 1e-8)
                for i in range(len(self.actions))
            ]
            chosen_idx = int(np.argmax(samples))
        else:
            # fallback to greedy on posterior mean
            means = self._reward_mean()[region]
            chosen_idx = int(np.argmax(means))

        chosen_action = self.actions[chosen_idx]
        propensity = 1.0 / len(self.actions)
        expected = self._reward_mean()[region, chosen_idx]
        confidence = 1.0 / math.sqrt(1.0 + self._alpha[region, chosen_idx] + self._beta[region, chosen_idx])
        return BanditAction(chosen_action, propensity, expected, confidence, algorithm)

# ----------------------------------------------------------------------
# Demonstration functions (require at least three)
# ----------------------------------------------------------------------
def demo_initialisation() -> HybridThompsonVoronoiCaputo:
    """Create a hybrid instance with three actions and random centres."""
    actions = ["click", "scroll", "ignore"]
    # 5 random 4‑dimensional centres
    centers = np.random.RandomState(42).uniform(0, 1, size=(5, 4))
    return HybridThompsonVoronoiCaputo(actions, centers)

def demo_batch_update(agent: HybridThompsonVoronoiCaputo) -> None:
    """Generate a synthetic batch of updates and feed them to the agent."""
    updates = []
    region_ids = []
    rng = random.Random(123)
    for _ in range(20):
        ctx = f"sample_{rng.randint(0,1000)}"
        feats = np.array(list(extract_full_features(ctx).values()))
        region = voronoi_partition(feats[np.newaxis, :], agent.centers)[0]
        act = rng.choice(agent.actions)
        reward = rng.random()  # continuous reward in [0,1]
        updates.append(BanditUpdate(context_id=ctx, action_id=act, reward=reward))
        region_ids.append(region)
    agent.update_policy(updates, region_ids)
    agent.evolve_rewards()

def demo_action_selection(agent: HybridThompsonVoronoiCaputo) -> BanditAction:
    """Select an action for a new context after a few update cycles."""
    return agent.select_action("User visited page with content X.", algorithm="thompson")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    hybrid_agent = demo_initialisation()
    # Run a few update‑evolve cycles
    for _ in range(3):
        demo_batch_update(hybrid_agent)
    action = demo_action_selection(hybrid_agent)
    print("Selected action:", action)
    # Show internal Beta parameters for inspection
    print("Alpha matrix:\n", hybrid_agent._alpha)
    print("Beta matrix:\n", hybrid_agent._beta)