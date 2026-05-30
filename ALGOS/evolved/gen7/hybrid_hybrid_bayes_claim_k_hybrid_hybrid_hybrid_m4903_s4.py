# DARWIN HAMMER — match 4903, survivor 4
# gen: 7
# parent_a: hybrid_bayes_claim_kernel_hybrid_hybrid_hybrid_m1718_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_percep_hybrid_hybrid_path_s_m2078_s0.py (gen6)
# born: 2026-05-29T23:58:37Z

"""
Hybrid Bayesian‑RBF Regret Engine
=================================

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** – ``hybrid_bayes_claim_kernel_hybrid_hybrid_hybrid_m1718_s0.py``  
  Provides a Bayesian update rule for hypotheses and uses Kullback‑Leibler (KL)
  divergence together with Shannon entropy to modulate probability‑based
  quantities.

* **Parent B** – ``hybrid_hybrid_hybrid_percep_hybrid_hybrid_path_s_m2078_s0.py``  
  Supplies a pheromone‑based Radial‑Basis‑Function (RBF) system that evaluates
  ``MathAction`` expected values and incorporates regret‑weighted decision making.

**Mathematical bridge**

Both parents operate on *probability distributions* – Parent A over
hypotheses, Parent B over actions (through pheromone strengths).  The hybrid
engine therefore:

1. **Updates** a hypothesis posterior with Bayes’ rule (Parent A).
2. **Computes** the KL divergence between the updated posterior and a
   reference prior; the resulting divergence is used as a *regret weight*.
3. **Feeds** the posterior vector into the RBF system; the entropy of the
   posterior modulates the RBF widths, making the kernel sharper when the
   distribution is certain and flatter when it is uncertain.
4. **Produces** action scores that combine the RBF‑estimated expected value,
   the regret weight (KL), and the original action cost/risk.

The functions below demonstrate this integrated workflow.  All code runs
with the standard library and NumPy only.
"""

import sys
import math
import random
import pathlib
from dataclasses import dataclass, replace
import numpy as np
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Data structures (shared between both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathClaim:
    id: str

@dataclass(frozen=True)
class MathEvidence:
    id: str

@dataclass(frozen=True)
class MathHypothesis:
    id: str
    prior: float          # prior probability before the latest evidence
    posterior: float      # posterior probability after the latest evidence
    evidence_ids: Tuple[str, ...] = ()

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ----------------------------------------------------------------------
# Parent‑A functionality (Bayesian update + KL / entropy)
# ----------------------------------------------------------------------
def update_hypothesis(
    hypothesis: MathHypothesis,
    evidence: MathEvidence,
    likelihood_ratio: float,
) -> MathHypothesis:
    """Bayesian odds update (same as the original Parent A)."""
    if likelihood_ratio < 0:
        raise ValueError("likelihood_ratio must be non‑negative")
    p = max(0.0, min(1.0, hypothesis.posterior))
    if p <= 0.0 or likelihood_ratio == 0.0:
        posterior = 0.0
    elif p >= 1.0:
        posterior = 1.0
    else:
        odds = p / (1.0 - p)
        new_odds = odds * likelihood_ratio
        posterior = new_odds / (1.0 + new_odds)
    posterior = max(0.0, min(1.0, posterior))
    ids = tuple(list(hypothesis.evidence_ids) + [evidence.id])
    return MathHypothesis(
        id=hypothesis.id,
        prior=hypothesis.posterior,
        posterior=posterior,
        evidence_ids=ids,
    )

def shannon_entropy(probs: np.ndarray) -> float:
    """Compute H(p) = -∑ p log p (base e). Zero probabilities are ignored."""
    probs = probs[np.where(probs > 0)]
    return -float(np.sum(probs * np.log(probs)))

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p‖q) = ∑ p log(p/q). Both vectors must sum to 1."""
    if not np.isclose(p.sum(), 1.0) or not np.isclose(q.sum(), 1.0):
        raise ValueError("Both distributions must be normalized")
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

# ----------------------------------------------------------------------
# Parent‑B functionality (RBF system)
# ----------------------------------------------------------------------
class HybridRBFSystem:
    """
    Pheromone‑driven RBF network.
    * ``centers`` – location of each RBF in the action‑space.
    * ``widths``  – isotropic width of each RBF (modulated by entropy).
    * ``rbf_weights`` – learned linear combination of RBF activations.
    * ``pheromones`` – per‑action strength that decays over simulated time.
    """
    def __init__(
        self,
        n_actions: int = 5,
        n_rbf: int = 12,
        alpha: float = 0.05,
        beta: float = 0.02,
    ):
        self.n_actions = n_actions
        self.n_rbf = n_rbf
        self.alpha = alpha          # learning rate for pheromone reinforcement
        self.beta = beta            # decay coefficient
        self.centers = np.random.rand(n_rbf, n_actions)
        self.widths = np.ones(n_rbf) * 0.5
        self.rbf_weights = np.random.rand(n_rbf)
        # pheromone record: action_id → (strength, creation_timestamp)
        self.pheromones: Dict[str, Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Time utilities (simple deterministic pseudo‑clock)
    # ------------------------------------------------------------------
    def _current_time(self) -> float:
        """Deterministic monotonic time based on a hidden counter."""
        if not hasattr(self, "_time_counter"):
            self._time_counter = 0.0
        self._time_counter += 0.1
        return self._time_counter

    def _decayed_pheromone(self, strength: float, created: float, half_life: float = 5.0) -> float:
        """Exponential decay based on half‑life."""
        elapsed = self._current_time() - created
        decay_factor = 0.5 ** (elapsed / half_life)
        return strength * decay_factor

    # ------------------------------------------------------------------
    # Core RBF evaluation
    # ------------------------------------------------------------------
    def rbf_activation(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian RBF activations for input vector ``x``.
        Returns a vector of length ``n_rbf``.
        """
        diff = self.centers - x  # shape (n_rbf, n_actions)
        sq_norm = np.sum(diff ** 2, axis=1)
        return np.exp(-sq_norm / (2.0 * self.widths ** 2))

    def estimate_expected_values(self, actions: List[MathAction]) -> np.ndarray:
        """
        Produce an estimated expected value for each action using the RBF net.
        The input ``x`` is the vector of raw ``expected_value`` fields.
        """
        raw = np.array([a.expected_value for a in actions])  # shape (n_actions,)
        phi = self.rbf_activation(raw)                     # shape (n_rbf,)
        estimate = phi @ self.rbf_weights                 # scalar
        # Spread the scalar estimate uniformly; later we add action‑specific terms.
        return np.full(self.n_actions, estimate)

    # ------------------------------------------------------------------
    # Pheromone handling
    # ------------------------------------------------------------------
    def reinforce(self, action_id: str, reward: float) -> None:
        """Increase pheromone strength for ``action_id`` proportional to ``reward``."""
        now = self._current_time()
        strength, _ = self.pheromones.get(action_id, (0.0, now))
        new_strength = strength + self.alpha * reward
        self.pheromones[action_id] = (new_strength, now)

    def get_pheromone_strength(self, action_id: str) -> float:
        """Return decayed pheromone strength (0 if absent)."""
        if action_id not in self.pheromones:
            return 0.0
        strength, created = self.pheromones[action_id]
        return self._decayed_pheromone(strength, created, half_life=10.0)

    # ------------------------------------------------------------------
    # Integration with Bayesian posterior (the hybrid bridge)
    # ------------------------------------------------------------------
    def adapt_widths_by_entropy(self, posterior: np.ndarray) -> None:
        """
        Shrink widths when entropy is low (distribution is certain) and
        expand them when entropy is high.  Widths stay within [0.1, 2.0].
        """
        H = shannon_entropy(posterior)
        # Map entropy ∈ [0, log(k)] → scale ∈ [0.5, 2.0]
        k = posterior.size
        max_H = math.log(k) if k > 1 else 0.0
        scale = 0.5 + 1.5 * (H / max_H) if max_H > 0 else 1.0
        self.widths = np.clip(self.widths * scale, 0.1, 2.0)

# ----------------------------------------------------------------------
# Hybrid functions (demonstrate the fused mathematics)
# ----------------------------------------------------------------------
def bayes_to_distribution(hypotheses: List[MathHypothesis]) -> np.ndarray:
    """Collect posterior probabilities into a normalized vector."""
    probs = np.array([h.posterior for h in hypotheses], dtype=float)
    total = probs.sum()
    if total == 0:
        # Avoid division by zero – fall back to uniform distribution
        return np.full_like(probs, 1.0 / len(probs))
    return probs / total

def hybrid_action_scores(
    actions: List[MathAction],
    hypotheses: List[MathHypothesis],
    rbf_system: HybridRBFSystem,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a hybrid score for each action.

    Steps
    -----
    1. Build the posterior distribution over hypotheses.
    2. Compute KL divergence between posterior and the uniform prior (regret weight).
    3. Adapt RBF widths using the posterior entropy (bridge from Parent A to B).
    4. Obtain RBF‑based value estimates.
    5. Combine:  score = rbf_estimate + pheromone - KL_weight * (cost + risk)

    Returns
    -------
    scores : np.ndarray
        Final hybrid scores (higher is better).
    kl_vals : np.ndarray
        KL divergence used for each action (identical across actions but
        returned for inspection).
    """
    posterior = bayes_to_distribution(hypotheses)

    # Uniform prior over the same number of hypotheses
    uniform = np.full_like(posterior, 1.0 / posterior.size)
    kl = kl_divergence(posterior, uniform)

    # Adapt RBF kernel widths according to entropy (entropy modulates KL handling)
    rbf_system.adapt_widths_by_entropy(posterior)

    # RBF gives a base estimate for every action
    base_estimates = rbf_system.estimate_expected_values(actions)  # shape (n_actions,)

    scores = np.empty(len(actions))
    for idx, act in enumerate(actions):
        pher = rbf_system.get_pheromone_strength(act.id)
        penalty = kl * (act.cost + act.risk)
        scores[idx] = base_estimates[idx] + pher - penalty
    return scores, np.full(len(actions), kl)

def hybrid_update_cycle(
    hypotheses: List[MathHypothesis],
    evidences: List[MathEvidence],
    likelihoods: List[float],
    actions: List[MathAction],
    rbf_system: HybridRBFSystem,
) -> Tuple[List[MathHypothesis], np.ndarray]:
    """
    One full hybrid iteration:

    * Bayesian update of each hypothesis with the supplied evidences.
    * Compute hybrid action scores.
    * Reinforce the best action's pheromone using its score as reward.

    Returns the updated hypothesis list and the array of action scores.
    """
    # 1. Bayesian update (Parent A)
    updated = []
    for hyp, ev, lr in zip(hypotheses, evidences, likelihoods):
        updated.append(update_hypothesis(hyp, ev, lr))

    # 2. Hybrid scoring (bridge)
    scores, _ = hybrid_action_scores(actions, updated, rbf_system)

    # 3. Reinforce the selected action
    best_idx = int(np.argmax(scores))
    best_action = actions[best_idx]
    rbf_system.reinforce(best_action.id, reward=scores[best_idx])

    return updated, scores

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny hypothesis pool
    hyps = [
        MathHypothesis(id=f"H{i}", prior=0.2, posterior=0.2, evidence_ids=())
        for i in range(4)
    ]

    # Synthetic evidences and likelihood ratios
    evs = [MathEvidence(id=f"E{i}") for i in range(4)]
    lrs = [random.uniform(0.5, 2.0) for _ in range(4)]

    # Define a few actions
    acts = [
        MathAction(id=f"A{i}", expected_value=random.uniform(-1, 1), cost=0.1*i, risk=0.05*i)
        for i in range(4)
    ]

    # Initialise the hybrid RBF system
    rbf = HybridRBFSystem(n_actions=len(acts), n_rbf=8)

    # Run a single hybrid cycle
    new_hyps, action_scores = hybrid_update_cycle(hyps, evs, lrs, acts, rbf)

    # Print results (purely for demonstration; not part of the required output)
    print("Updated hypotheses posteriors:")
    for h in new_hyps:
        print(f"  {h.id}: {h.posterior:.4f}")

    print("\nHybrid action scores:")
    for act, sc in zip(acts, action_scores):
        print(f"  {act.id}: {sc:.4f}")

    sys.exit(0)