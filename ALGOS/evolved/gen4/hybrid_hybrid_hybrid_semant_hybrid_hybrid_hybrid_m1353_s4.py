# DARWIN HAMMER — match 1353, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_semantic_neig_hybrid_sketches_hybr_m57_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m202_s3.py (gen3)
# born: 2026-05-29T23:35:43Z

import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Geometry / Morphology (Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Physical shape descriptors of a document."""
    length: float
    width: float
    height: float
    mass: float


def _check_positive(*values: float) -> None:
    """Utility to ensure all supplied values are strictly positive."""
    for v in values:
        if v <= 0:
            raise ValueError("All geometric parameters must be > 0")


def sphericity_index(length: float, width: float, height: float) -> float:
    """Compactness ratio: geometric mean of dimensions over the longest side."""
    _check_positive(length, width, height)
    gm = (length * width * height) ** (1.0 / 3.0)
    return gm / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness ratio: larger when height is small relative to the base."""
    _check_positive(length, width, height)
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """
    Approximate time for a body to right itself after being tipped.
    A simple physics‑inspired surrogate.
    """
    _check_positive(m.mass, neck_lever)
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass * (m.length ** b)) / (k * neck_lever * fi)


def compute_recovery_priority(m: Morphology) -> float:
    """
    Smooth priority in [0, 1] that grows with “good” morphology.
    Combines sphericity, flatness and righting time via a weighted geometric mean.
    """
    sph = sphericity_index(m.length, m.width, m.height)          # ∈ (0,1]
    flt = flatness_index(m.length, m.width, m.height)           # >0
    rti = righting_time_index(m)                                # >0

    # Normalise flatness to (0,1) using a soft‑clipping exponential.
    flt_norm = 1.0 - exp(-flt / 10.0)

    # Righting time is better when small; map via decreasing exponential.
    rti_norm = exp(-rti / 5.0)

    # Weighted geometric mean – λ controls influence of each term.
    λ_sph, λ_flt, λ_rti = 0.4, 0.3, 0.3
    priority = (sph ** λ_sph) * (flt_norm ** λ_flt) * (rti_norm ** λ_rti)

    # Clamp for safety.
    return max(0.0, min(1.0, priority))


# ----------------------------------------------------------------------
# Thompson Sampling Bandit (Parent B)
# ----------------------------------------------------------------------


@dataclass
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float          # Estimated probability that this action is optimal
    expected_reward: float     # Posterior mean of the chosen arm
    confidence_bound: float    # Posterior standard deviation
    algorithm: str = "thompson_sampling"


@dataclass
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0    # Not used by the vanilla bandit but kept for API compatibility


class ThompsonBandit:
    """Lightweight Thompson‑sampling bandit for Bernoulli‑like rewards."""

    def __init__(self, actions: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta prior parameters must be > 0")
        self._alpha: Dict[str, float] = {a: prior_alpha for a in actions}
        self._beta: Dict[str, float] = {a: prior_beta for a in actions}
        self._actions = actions

    # ------------------------------------------------------------------
    # Internal helpers – keep the public API clean
    # ------------------------------------------------------------------
    def _sample_beta(self, a: str, alpha: float, beta: float) -> float:
        """Draw a single sample from a Beta(alpha, beta) distribution."""
        return np.random.beta(alpha, beta)

    def _posterior_params(self, a: str) -> Tuple[float, float]:
        """Return current (alpha, beta) for arm ``a``."""
        return self._alpha[a], self._beta[a]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def sample(self) -> str:
        """Return the action with the highest Thompson sample."""
        samples = {
            a: self._sample_beta(a, *self._posterior_params(a))
            for a in self._actions
        }
        return max(samples, key=samples.get)

    def expected(self) -> Dict[str, float]:
        """Posterior mean for each arm."""
        return {
            a: self._alpha[a] / (self._alpha[a] + self._beta[a])
            for a in self._actions
        }

    def variance(self) -> Dict[str, float]:
        """Posterior variance for each arm."""
        return {
            a: (self._alpha[a] * self._beta[a])
            / ((self._alpha[a] + self._beta[a]) ** 2 * (self._alpha[a] + self._beta[a] + 1))
            for a in self._actions
        }

    def update(self, upd: BanditUpdate) -> None:
        """Update the posterior for ``upd.action_id`` with a clipped reward."""
        if upd.action_id not in self._actions:
            raise KeyError(f"Unknown action {upd.action_id}")

        r = max(0.0, min(1.0, upd.reward))          # Bernoulli clipping
        self._alpha[upd.action_id] += r
        self._beta[upd.action_id] += 1.0 - r


# ----------------------------------------------------------------------
# Hybrid Wrapper – deeper integration of morphology
# ----------------------------------------------------------------------


class HybridBandit:
    """
    A Thompson‑sampling bandit whose belief is *hierarchically* conditioned
    on a morphology‑derived recovery priority ``ρ ∈ [0,1]``.

    Instead of naively scaling both α and β (which only inflates pseudo‑counts
    without moving the posterior mean), we treat ``ρ`` as a *soft prior* that
    nudges the posterior mean toward the morphology‑informed guess
    ``μ_ρ = ρ``.  Concretely, for each arm we augment the Beta parameters by

        α' = α + λ·ρ·α₀
        β' = β + λ·(1‑ρ)·β₀

    where ``α₀, β₀`` are the original prior hyper‑parameters and ``λ ≥ 0`` controls
    the strength of the morphological influence.  This formulation preserves
    variance scaling while allowing the morphology to shift the mean in a
    principled Bayesian way.

    The same ``ρ`` also scales the observed reward during the update step,
    reflecting that a more plausible document should have a larger impact on
    learning.
    """

    def __init__(
        self,
        actions: List[str],
        lambda_: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        if lambda_ < 0:
            raise ValueError("lambda_ must be non‑negative")
        self._bandit = ThompsonBandit(actions, prior_alpha, prior_beta)
        self.lambda_ = lambda_
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self.actions = actions

    # ------------------------------------------------------------------
    # Morphology‑aware posterior tilting
    # ------------------------------------------------------------------
    def _tilted_params(self, rho: float) -> Dict[str, Tuple[float, float]]:
        """
        Return per‑arm (α', β') after incorporating the morphology‑derived prior.
        The tilt respects the original posterior while adding a soft prior
        centred at ``ρ`` with strength proportional to ``λ``.
        """
        factor = self.lambda_
        tilted: Dict[str, Tuple[float, float]] = {}
        for a in self.actions:
            alpha, beta = self._bandit._posterior_params(a)
            # Soft prior contribution
            alpha_t = alpha + factor * rho * self._prior_alpha
            beta_t = beta + factor * (1.0 - rho) * self._prior_beta
            tilted[a] = (alpha_t, beta_t)
        return tilted

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def sample(self, morphology: Morphology, context_id: str = "") -> BanditAction:
        """
        Select an action using morphology‑aware Thompson sampling.
        Returns a ``BanditAction`` enriched with diagnostics.
        """
        rho = compute_recovery_priority(morphology)
        tilted = self._tilted_params(rho)

        # Draw a single Thompson sample from each tilted posterior.
        samples = {
            a: np.random.beta(alpha, beta) for a, (alpha, beta) in tilted.items()
        }
        chosen = max(samples, key=samples.get)

        # Estimate propensity as the fraction of samples (Monte‑Carlo) where the
        # chosen arm is optimal.  Using 200 draws provides a low‑variance estimate
        # without excessive cost.
        mc_draws = 200
        wins = 0
        for _ in range(mc_draws):
            draw = {
                a: np.random.beta(alpha, beta) for a, (alpha, beta) in tilted.items()
            }
            if max(draw, key=draw.get) == chosen:
                wins += 1
        propensity = wins / mc_draws

        expected = self._bandit.expected()[chosen]
        variance = self._bandit.variance()[chosen]
        confidence = sqrt(variance)

        return BanditAction(
            action_id=chosen,
            propensity=propensity,
            expected_reward=expected,
            confidence_bound=confidence,
            algorithm="hybrid_thompson_sampling",
        )

    # ------------------------------------------------------------------
    # Update with morphology‑scaled reward
    # ------------------------------------------------------------------
    def update(self, upd: BanditUpdate, morphology: Morphology) -> None:
        """
        Update the underlying bandit.  The observed reward is scaled by the
        recovery priority, ensuring that high‑quality morphology yields a
        stronger learning signal.
        """
        rho = compute_recovery_priority(morphology)
        scaled_reward = max(0.0, min(1.0, upd.reward * (1.0 + self.lambda_ * rho)))
        scaled_upd = BanditUpdate(
            context_id=upd.context_id,
            action_id=upd.action_id,
            reward=scaled_reward,
            propensity=upd.propensity,
        )
        self._bandit.update(scaled_upd)

    # ------------------------------------------------------------------
    # Convenience wrappers mirroring the original API
    # ------------------------------------------------------------------
    def expected(self) -> Dict[str, float]:
        """Expose the underlying bandit's posterior means."""
        return self._bandit.expected()

    def variance(self) -> Dict[str, float]:
        """Expose the underlying bandit's posterior variances."""
        return self._bandit.variance()


# ----------------------------------------------------------------------
# Example utilities (unchanged semantics, but type‑annotated)
# ----------------------------------------------------------------------


def hybrid_select_action(bandit: HybridBandit, morphology: Morphology) -> BanditAction:
    """Convenient one‑liner for action selection."""
    return bandit.sample(morphology)


def hybrid_update(
    bandit: HybridBandit,
    upd: BanditUpdate,
    morphology: Morphology,
) -> None:
    """Convenient one‑liner for updating the hybrid bandit."""
    bandit.update(upd, morphology)


def similarity_matrix(actions: List[str]) -> np.ndarray:
    """
    Dummy similarity matrix: identity with a small off‑diagonal jitter.
    Demonstrates that matrix operations can be shared across both parent
    algorithms (e.g., for contextual extensions).
    """
    n = len(actions)
    base = np.eye(n)
    jitter = np.random.uniform(-0.01, 0.01, size=(n, n))
    return base + jitter

# End of module.