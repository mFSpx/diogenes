# DARWIN HAMMER — match 1353, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_semantic_neig_hybrid_sketches_hybr_m57_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m202_s3.py (gen3)
# born: 2026-05-29T23:35:43Z

"""Hybrid Semantic‑Morphology Thompson Bandit
================================================

This module fuses the two parent algorithms:

* **Parent A** – provides geometric descriptors (sphericity, flatness,
  righting‑time) and a *recovery_priority* concept derived from a document's
  morphology.
* **Parent B** – implements a minimal Thompson‑sampling Bernoulli bandit.

The mathematical bridge is the **recovery priority** `ρ ∈ [0,1]`.  In the
parent‑A world `ρ` is computed from morphology indices; in the parent‑B world
`ρ` can be used to *tilt* the Beta posterior of the Thompson sampler.  We
multiply the posterior parameters (α, β) by a factor `1 + λ·ρ` (λ≥0) before
sampling, and we also scale the observed reward during the update step.
Thus the bandit’s belief is continuously informed by the physical plausibility
of the underlying document, yielding a unified hybrid system.

The code below implements:
* geometric helpers (sphericity, flatness, righting‑time)
* `recovery_priority` – a smooth function of those helpers
* `ThompsonBandit` – the original sampler with a priority‑aware `sample`
* `HybridBandit` – a thin wrapper that injects the priority into sampling
  and updating
* three demonstration functions:
    - `compute_recovery_priority`
    - `hybrid_select_action`
    - `hybrid_update`
* a tiny similarity‑matrix helper to showcase matrix operations from both
  parents.
"""

import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of the three dimensions to the longest side."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """How flat an object is: larger when height is small compared to the base."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """
    Approximate time for a body to right itself after being tipped.
    Formula taken from the parent A implementation (truncated for brevity).
    """
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    # Simple physics‑inspired surrogate:
    return (m.mass * (m.length ** b)) / (k * neck_lever * fi)


def compute_recovery_priority(m: Morphology) -> float:
    """
    A smooth priority in [0,1] that grows with “good” morphology.
    The function combines sphericity (favoring compactness) and flatness
    (favoring stability) via a logistic transform.
    """
    sph = sphericity_index(m.length, m.width, m.height)          # ≈0–1
    flt = flatness_index(m.length, m.width, m.height)           # >0
    rti = righting_time_index(m)                                # >0

    # Normalise flatness to [0,1] by a soft‑clipping exponential.
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
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "thompson_sampling"


@dataclass
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0


class ThompsonBandit:
    """Lightweight Thompson‑sampling bandit for Bernoulli‑like rewards."""

    def __init__(self, actions: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self._alpha: Dict[str, float] = {a: prior_alpha for a in actions}
        self._beta: Dict[str, float] = {a: prior_beta for a in actions}
        self._actions = actions

    def _sample_posteriors(self) -> Dict[str, float]:
        """Draw a sample from each Beta posterior."""
        return {
            a: np.random.beta(self._alpha[a], self._beta[a])
            for a in self._actions
        }

    def sample(self) -> str:
        """Return the action with the highest sampled value."""
        samples = self._sample_posteriors()
        return max(samples, key=samples.get)

    def expected(self) -> Dict[str, float]:
        """Return the mean of each Beta posterior."""
        return {
            a: self._alpha[a] / (self._alpha[a] + self._beta[a])
            for a in self._actions
        }

    def update(self, upd: BanditUpdate) -> None:
        """Update the posterior for ``upd.action_id`` with a clipped reward."""
        r = max(0.0, min(1.0, upd.reward))
        self._alpha[upd.action_id] += r
        self._beta[upd.action_id] += 1.0 - r


# ----------------------------------------------------------------------
# Hybrid Wrapper – merges geometry with the bandit
# ----------------------------------------------------------------------


class HybridBandit:
    """
    Wraps a :class:`ThompsonBandit` and injects the morphology‑derived
    ``recovery_priority`` into both sampling and updating.

    The priority scales the Beta parameters by a factor ``1 + λ·ρ`` where
    ``λ`` (lambda_) controls how aggressively geometry influences the belief.
    """

    def __init__(
        self,
        actions: List[str],
        lambda_: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        self._bandit = ThompsonBandit(actions, prior_alpha, prior_beta)
        self.lambda_ = max(0.0, lambda_)
        self.actions = actions

    def _tilted_alpha_beta(self, rho: float) -> Dict[str, tuple[float, float]]:
        """Return α,β after scaling by the priority factor."""
        factor = 1.0 + self.lambda_ * rho
        tilted = {}
        for a in self.actions:
            tilted[a] = (self._bandit._alpha[a] * factor,
                         self._bandit._beta[a] * factor)
        return tilted

    def sample(self, morphology: Morphology, context_id: str = "") -> BanditAction:
        """Select an action using priority‑aware Thompson sampling."""
        rho = compute_recovery_priority(morphology)
        tilted = self._tilted_alpha_beta(rho)

        # Sample from the tilted posteriors.
        samples = {
            a: np.random.beta(alpha, beta)
            for a, (alpha, beta) in tilted.items()
        }
        chosen = max(samples, key=samples.get)

        # Build a BanditAction with additional diagnostics.
        expected = self._bandit.expected()[chosen]
        confidence = np.sqrt(
            (self._bandit._alpha[chosen] * self._bandit._beta[chosen])
            / ((self._bandit._alpha[chosen] + self._bandit._beta[chosen]) ** 2
               * (self._bandit._alpha[chosen] + self._bandit._beta[chosen] + 1))
        )
        return BanditAction(
            action_id=chosen,
            propensity=rho,
            expected_reward=expected,
            confidence_bound=confidence,
            algorithm="priority_thompson",
        )

    def update(self, upd: BanditUpdate, morphology: Morphology) -> None:
        """Update the underlying bandit, scaling reward by the recovery priority."""
        rho = compute_recovery_priority(morphology)
        # Scale reward toward the optimistic side when priority is high.
        scaled_reward = upd.reward * (0.5 + 0.5 * rho)  # stays in [0,1]
        scaled_upd = BanditUpdate(
            context_id=upd.context_id,
            action_id=upd.action_id,
            reward=scaled_reward,
            propensity=upd.propensity,
        )
        self._bandit.update(scaled_upd)


# ----------------------------------------------------------------------
# Matrix‑level demonstration (mix of both parents)
# ----------------------------------------------------------------------


def similarity_matrix(doc_vectors: List[List[float]]) -> np.ndarray:
    """
    Compute a cosine‑similarity matrix for a list of document vectors.
    This mirrors the linear‑algebra flavour of the parents and provides a
    shared data structure that could be fed into the hybrid bandit as context.
    """
    if not doc_vectors:
        raise ValueError("doc_vectors must be non‑empty")
    arr = np.array(doc_vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid division by zero.
    norms[norms == 0] = 1.0
    normalized = arr / norms
    return normalized @ normalized.T


def hybrid_select_action(
    hybrid_bandit: HybridBandit,
    morphology: Morphology,
    doc_vectors: List[List[float]],
) -> BanditAction:
    """
    High‑level helper that computes a similarity matrix (matrix op from parent B),
    extracts a simple context identifier, and delegates to ``HybridBandit.sample``.
    """
    sim = similarity_matrix(doc_vectors)
    # Use the mean similarity as a dummy context identifier.
    context_id = f"ctx_{sim.mean():.3f}"
    return hybrid_bandit.sample(morphology, context_id=context_id)


def hybrid_update(
    hybrid_bandit: HybridBandit,
    action: BanditAction,
    reward: float,
    morphology: Morphology,
) -> None:
    """
    High‑level helper that builds a :class:`BanditUpdate` from the chosen action
    and forwards it to the hybrid bandit.
    """
    upd = BanditUpdate(
        context_id=f"upd_{datetime.now(timezone.utc).isoformat()}",
        action_id=action.action_id,
        reward=reward,
    )
    hybrid_bandit.update(upd, morphology)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a tiny set of actions.
    actions = ["search", "recommend", "summarize"]
    hb = HybridBandit(actions, lambda_=0.8)

    # Create a mock morphology.
    morph = Morphology(length=12.0, width=8.0, height=3.0, mass=5.0)

    # Fake document vectors (e.g., embeddings).
    docs = [
        [0.1, 0.3, 0.5],
        [0.2, 0.1, 0.4],
        [0.4, 0.4, 0.2],
    ]

    # Select an action.
    chosen_action = hybrid_select_action(hb, morph, docs)
    print("Chosen action:", chosen_action)

    # Simulate a reward (e.g., user click = 1, no click = 0).
    simulated_reward = random.choice([0.0, 1.0])
    print("Simulated reward:", simulated_reward)

    # Update the bandit with the simulated outcome.
    hybrid_update(hb, chosen_action, simulated_reward, morph)

    # Show updated expectations.
    expectations = hb._bandit.expected()
    print("Posterior expectations after update:", expectations)