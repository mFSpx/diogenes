# DARWIN HAMMER — match 3192, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1213_s6.py (gen6)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_ternar_m1253_s0.py (gen3)
# born: 2026-05-29T23:48:31Z

"""HybridFusionAlgorithm
Combines:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1213_s6.py
  Provides morphology similarity `S`, token entropy `H`, and BIC‑derived weight `w_B`.
- Parent B: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_ternar_m1253_s0.py
  Supplies a contextual bandit (HybridBanditTTT) whose learning‑rate and
  propensity are modulated by information‑theoretic quantities from Parent A.

Mathematical Bridge
------------------
Both parents expose logarithmic complexity penalties:
    * Entropy `H` (Parent A) and Bayesian Information Criterion `B` (Parent B).
      We map `B` to a normalized weight `w_B∈[0,1]` via a sigmoid and combine
      them into a joint complexity factor  
          C = (1 − β·H)·w_B .
    * The Real Log‑Canonical Threshold `λ` (Parent B) scales the NLMS‑style
      learning‑rate `η = 1/(1+λ)`.

The unified Hybrid Recovery Score is  

    Ψ = (α·S + (1‑α)·R̄)·C  

where `R̄` is the average regret of the bandit actions.  `Ψ` is used to
adjust the bandit’s step size and to weight the expected reward when
selecting an action.

The module below implements the fused system, exposing three core
functions and a smoke‑test."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology, similarity, entropy, BIC utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Return a simple sphericity proxy (closer to 1 → more sphere‑like)."""
    a = length * width
    b = width * height
    c = height * length
    surface = 2 * (a + b + c)
    volume = length * width * height
    if surface == 0:
        return 0.0
    return (math.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface


def morphology_vector(m: Morphology) -> np.ndarray:
    """Pack morphology fields into a 4‑D vector."""
    return np.array([m.length, m.width, m.height, m.mass], dtype=float)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two non‑zero vectors."""
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def compute_similarity(m1: Morphology, m2: Morphology) -> float:
    """Similarity `S` used in the hybrid score."""
    v1 = morphology_vector(m1)
    v2 = morphology_vector(m2)
    return cosine_similarity(v1, v2)


def token_entropy(token_counts: Dict[str, int]) -> float:
    """Shannon entropy `H` of a discrete token distribution."""
    total = sum(token_counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(token_counts.values()), dtype=float) / total
    # protect against log(0)
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


def bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """Bayesian Information Criterion."""
    if n_samples <= 0:
        return float('inf')
    return -2.0 * log_likelihood + n_params * math.log(n_samples)


def bic_weight(B: float) -> float:
    """Map BIC to a [0,1] weight via a sigmoid centered at 0."""
    # Larger BIC → smaller weight
    return 1.0 / (1.0 + math.exp(B))


def joint_complexity_factor(H: float, B: float, beta: float = 0.5) -> float:
    """Combined complexity factor `C`."""
    w_B = bic_weight(B)
    C = (1.0 - beta * H) * w_B
    # Clamp to [0,1] for stability
    return max(0.0, min(1.0, C))


# ----------------------------------------------------------------------
# Parent B – Contextual bandit (HybridBanditTTT)
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


class HybridBanditTTT:
    """
    Minimal contextual bandit whose learning‑rate can be externally scaled.
    The internal state is a simple linear model `θ` mapping contexts to
    expected rewards.
    """

    DEFAULT_BUDGET_MB = 8192

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
        self.rng = np.random.default_rng(seed)
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.store_decay = store_decay
        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        # Linear parameters initialised near zero
        self.theta = np.zeros((self.d_out, self.d_in), dtype=float)
        # Simple VRAM‑like store for momentum
        self.vram_store = np.zeros_like(self.theta)

    def _context_vector(self, context_id: str) -> np.ndarray:
        """Deterministic pseudo‑random vector for a given context."""
        random_state = np.random.RandomState(abs(hash(context_id)) % (2 ** 32))
        return random_state.rand(self.d_in).astype(float)

    def predict(self, context_id: str) -> List[BanditAction]:
        """Return a list of candidate actions with propagated scores."""
        x = self._context_vector(context_id)  # shape (d_in,)
        scores = self.theta @ x                # shape (d_out,)
        actions: List[BanditAction] = []
        for i, sc in enumerate(scores):
            action_id = f"act_{i}"
            propensity = max(0.0, math.tanh(sc))  # inflow‑like
            expected = float(sc)
            # Confidence bound as a simple function of variance estimate
            conf = self.beta / (1.0 + abs(sc))
            actions.append(
                BanditAction(
                    action_id=action_id,
                    propensity=propensity,
                    expected_reward=expected,
                    confidence_bound=conf,
                    algorithm="HybridBanditTTT",
                )
            )
        return actions

    def update(
        self,
        context_id: str,
        action_id: str,
        reward: float,
        propensity: float,
        eta_scale: float = 1.0,
    ) -> None:
        """NLMS‑style update of the linear model."""
        x = self._context_vector(context_id)
        # locate action index
        try:
            idx = int(action_id.split("_")[1])
        except Exception:
            return
        # prediction error
        pred = float(self.theta[idx] @ x)
        error = reward - pred
        # NLMS step size
        eta = self.base_eta * eta_scale / (1.0 + np.dot(x, x))
        # Parameter update with store decay (VRAM effect)
        self.theta[idx] += eta * error * x
        self.vram_store[idx] = self.store_decay * self.vram_store[idx] + (1 - self.store_decay) * np.abs(eta * error * x)


# ----------------------------------------------------------------------
# Hybrid Fusion Core
# ----------------------------------------------------------------------


def hybrid_recovery_score(
    morph_a: Morphology,
    morph_b: Morphology,
    token_counts: Dict[str, int],
    log_likelihood: float,
    n_params: int,
    n_samples: int,
    lambda_rlct: float,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    Compute the unified Hybrid Recovery Score Ψ.

    Parameters
    ----------
    morph_a, morph_b : Morphology
        Objects describing two entities to be compared.
    token_counts : dict
        Frequency map of observed tokens (used for entropy H).
    log_likelihood, n_params, n_samples : float/int
        Statistics for BIC computation.
    lambda_rlct : float
        Real Log‑Canonical Threshold (scales learning rate).
    alpha, beta : float
        Weighting coefficients (α for similarity, β for entropy in C).

    Returns
    -------
    Ψ : float
        Hybrid recovery score in [0, 1] (approximately).
    """
    S = compute_similarity(morph_a, morph_b)                     # similarity term
    H = token_entropy(token_counts)                             # entropy term
    B = bic(log_likelihood, n_params, n_samples)                # BIC
    C = joint_complexity_factor(H, B, beta=beta)                # joint complexity
    # λ influences the learning‑rate factor; we embed it into Ψ via a simple
    # monotonic mapping to keep Ψ bounded.
    rlct_factor = 1.0 / (1.0 + lambda_rlct)
    # The regret part R̄ will be injected later; for the pure score we set it to 0.
    Ψ = (alpha * S) * C * rlct_factor
    # Clamp for numerical safety
    return max(0.0, min(1.0, Ψ))


def average_regret(actions: List[BanditAction]) -> float:
    """Compute the mean regret of a set of actions."""
    if not actions:
        return 0.0
    max_reward = max(a.expected_reward for a in actions)
    regrets = [max_reward - a.expected_reward for a in actions]
    return float(np.mean(regrets))


def hybrid_bandit_update(
    bandit: HybridBanditTTT,
    context_id: str,
    reward: float,
    token_counts: Dict[str, int],
    log_likelihood: float,
    n_params: int,
    n_samples: int,
    lambda_rlct: float,
    morph_ref: Morphology,
    morph_cur: Morphology,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> None:
    """
    Perform a bandit update where the NLMS step size is modulated by the
    hybrid recovery score Ψ.
    """
    # Predict to obtain the action with highest propensity (used as proxy)
    actions = bandit.predict(context_id)
    if not actions:
        return
    # Choose the action with maximal expected reward (standard bandit choice)
    chosen = max(actions, key=lambda a: a.expected_reward)
    # Compute Ψ using current morphology pair and statistical information
    Ψ = hybrid_recovery_score(
        morph_ref,
        morph_cur,
        token_counts,
        log_likelihood,
        n_params,
        n_samples,
        lambda_rlct,
        alpha=alpha,
        beta=beta,
    )
    # Scale learning rate by Ψ (acts as C·η factor)
    eta_scale = Ψ
    bandit.update(
        context_id=context_id,
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
        eta_scale=eta_scale,
    )


def select_action_hybrid(
    bandit: HybridBanditTTT,
    context_id: str,
    token_counts: Dict[str, int],
    log_likelihood: float,
    n_params: int,
    n_samples: int,
    lambda_rlct: float,
    morph_ref: Morphology,
    morph_cur: Morphology,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Optional[BanditAction]:
    """
    Choose the best action after weighting expected rewards with Ψ and
    incorporating average regret.
    """
    actions = bandit.predict(context_id)
    if not actions:
        return None

    # Compute average regret for the current action set
    R_bar = average_regret(actions)

    # Base similarity and entropy terms
    S = compute_similarity(morph_ref, morph_cur)
    H = token_entropy(token_counts)
    B = bic(log_likelihood, n_params, n_samples)
    C = joint_complexity_factor(H, B, beta=beta)
    rlct_factor = 1.0 / (1.0 + lambda_rlct)

    # Full hybrid score Ψ (includes regret term)
    Ψ = (alpha * S + (1.0 - alpha) * R_bar) * C * rlct_factor
    Ψ = max(0.0, min(1.0, Ψ))

    # Adjust each action's expected reward by Ψ and pick the max
    adjusted = [
        (
            a,
            a.expected_reward * Ψ
            - a.confidence_bound * (1.0 - Ψ)  # penalise uncertainty when Ψ is low
        )
        for a in actions
    ]
    best_action = max(adjusted, key=lambda tup: tup[1])[0]
    return best_action


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Dummy morphologies
    morph_a = Morphology(length=1.2, width=0.8, height=0.5, mass=2.3)
    morph_b = Morphology(length=1.0, width=0.9, height=0.6, mass=2.1)

    # Token frequency example
    tokens = {"alpha": 10, "beta": 5, "gamma": 15}

    # Statistics for BIC
    loglik = -123.4
    n_params = 8
    n_samples = 250

    # RLCT (chosen arbitrarily)
    lambda_rlct = 0.7

    # Initialise bandit
    bandit = HybridBanditTTT(d_in=4, seed=42)

    # Simulate