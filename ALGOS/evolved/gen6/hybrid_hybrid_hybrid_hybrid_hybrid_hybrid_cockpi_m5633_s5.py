# DARWIN HAMMER — match 5633, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_epistemic_certainty_m1153_s0.py (gen5)
# parent_b: hybrid_hybrid_cockpit_metri_hybrid_hybrid_liquid_m488_s1.py (gen4)
# born: 2026-05-30T00:03:39Z

"""Hybrid Bandit‑RBF with Cockpit‑Metrics‑Guided Diffusion

Parents:
- hybrid_hybrid_hybrid_hybrid_epistemic_certainty_m1153_s0 (Bandit + RBF surrogate + epistemic certainty)
- hybrid_hybrid_cockpit_metri_hybrid_hybrid_liquid_m488_s1 (Cockpit metrics + liquid‑time diffusion)

Mathematical bridge:
Both parents expose scalar quality measures that can be used as weighting factors.
We fuse them by letting the *honesty* and *anti‑slop* metrics modulate the
exploration term of the bandit (the propensity noise) while the epistemic
certainty (confidence bound) scales the RBF surrogate’s prediction.  The
resulting hybrid propensity 𝜋̂ is

    𝜋̂ = softmax( (μ̂ + λ·CB) / (1 + α·H + β·S) ),

where
    μ̂   – RBF surrogate prediction for the context,
    CB   – epistemic confidence bound,
    H    – cockpit_honesty,
    S    – anti_slop_ratio,
    λ,α,β – tunable scalars.

The diffusion noise injected into the particle‑swarm‑style social interaction is
scaled by a similarity‑driven factor γ·sim·(1‑H), ensuring that low‑trust
situations damp the swarm’s movement.

The module provides three core functions demonstrating this hybrid dynamics:
    1. hybrid_predict – RBF prediction with certainty scaling.
    2. hybrid_propensity – computes the modified propensity for each action.
    3. diffusion_noise – returns a noise vector for social interaction based on
       similarity and cockpit honesty.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Callable, Sequence

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Core data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBanditRBF"


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            object.__setattr__(self, "generated_at", "2026-05-29T23:26:28Z")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """RBF weighted sum prediction."""
        return sum(
            w * math.exp(-((self.epsilon * self.euclidean(x, c)) ** 2))
            for w, c in zip(self.weights, self.centers)
        )

    @staticmethod
    def euclidean(a: Vector, b: Vector) -> float:
        """Euclidean distance between two vectors."""
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ----------------------------------------------------------------------
# Cockpit‑metrics utilities from Parent B
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Fraction of claims that are backed by evidence, clipped to [0,1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Honesty = proportion of displayed claims that are known to be OK."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity in [0,1] for non‑negative vectors."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def social_interaction(
    x: Vector,
    g_best: Vector,
    k: int = 1,
    r1: float | None = None,
    seed: int | str | None = None,
) -> List[float]:
    """Particle‑swarm‑style attraction toward the global best."""
    if len(x) != len(g_best):
        raise ValueError("x and g_best must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    # Simple PSO update: x_new = x + k * r * (g_best - x)
    return [(xi + k * r * (gi - xi)) for xi, gi in zip(x, g_best)]


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_predict(
    surrogate: RBFSurrogate,
    context: Vector,
    certainty: CertaintyFlag,
    lambda_cb: float = 0.5,
) -> float:
    """
    Predict expected reward for a context while scaling by epistemic certainty.

    μ̂ = surrogate.predict(context)
    μ̂_hybrid = μ̂ + λ·CB

    where CB = confidence bound derived from the certainty flag (bps → probability).
    """
    base = surrogate.predict(context)
    # Convert basis points to a probability‑like scale (0‑1)
    cb = certainty.confidence_bps / 10_000.0
    return base + lambda_cb * cb


def hybrid_propensity(
    actions: List[BanditAction],
    surrogate: RBFSurrogate,
    context: Vector,
    certainty: CertaintyFlag,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
    claims_with_evidence: int,
    total_claims_emitted: int,
    lambda_cb: float = 0.5,
    alpha_hon: float = 2.0,
    beta_slop: float = 2.0,
    temperature: float = 0.1,
) -> Dict[str, float]:
    """
    Compute a soft‑max propensity for each action after fusing:
      • RBF prediction + certainty (hybrid_predict)
      • Cockpit honesty (H) and anti‑slop (S) as denominator scaling
    Returns a dict mapping action_id → new propensity.
    """
    H = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    S = anti_slop_ratio(claims_with_evidence, total_claims_emitted)

    # Hybrid score for each action
    scores = {}
    for act in actions:
        # Expected reward is replaced by hybrid prediction for the given context
        hybrid_reward = hybrid_predict(surrogate, context, certainty, lambda_cb)
        # Blend with the action's own expected reward (optional)
        blended = 0.7 * hybrid_reward + 0.3 * act.expected_reward
        # Scale by certainty bound and cockpit metrics
        denom = 1.0 + alpha_hon * H + beta_slop * S
        scores[act.action_id] = blended / denom

    # Soft‑max conversion
    max_score = max(scores.values())
    exp_vals = {aid: math.exp((s - max_score) / temperature) for aid, s in scores.items()}
    total = sum(exp_vals.values())
    propensities = {aid: ev / total for aid, ev in exp_vals.items()}
    return propensities


def diffusion_noise(
    token_set_a: Vector,
    token_set_b: Vector,
    honesty: float,
    base_noise: float = 0.05,
    gamma: float = 1.5,
) -> np.ndarray:
    """
    Generate a noise vector that will be added to a particle’s position.
    The magnitude is proportional to (1‑H) and to the cosine similarity of the
    two token sets, implementing the “similarity‑driven diffusion” bridge.
    """
    sim = cosine_similarity(token_set_a, token_set_b)  # in [0,1]
    scale = base_noise * gamma * sim * (1.0 - honesty)
    rng = np.random.default_rng()
    return rng.normal(loc=0.0, scale=scale, size=len(token_set_a))


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy RBF surrogate with two centers in 3‑D space
    surrogate = RBFSurrogate(
        centers=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        weights=[0.6, 0.4],
        epsilon=1.2,
    )

    # Certainty flag (e.g., 9500 bps = 0.95 confidence)
    certainty = CertaintyFlag(
        label="high_confidence",
        confidence_bps=9500,
        authority_class="expert",
        rationale="validated by external audit",
    )

    # Define three possible actions
    actions = [
        BanditAction(action_id="A", propensity=0.33, expected_reward=1.2, confidence_bound=0.1),
        BanditAction(action_id="B", propensity=0.33, expected_reward=0.8, confidence_bound=0.2),
        BanditAction(action_id="C", propensity=0.34, expected_reward=1.0, confidence_bound=0.15),
    ]

    # Context vector for prediction
    context = [0.2, 0.5, 0.3]

    # Cockpit metrics (arbitrary numbers)
    displayed_ok = 80
    unknown_displayed_as_ok = 20
    claims_with_evidence = 70
    total_claims_emitted = 100

    # Compute hybrid propensities
    new_props = hybrid_propensity(
        actions=actions,
        surrogate=surrogate,
        context=context,
        certainty=certainty,
        displayed_ok=displayed_ok,
        unknown_displayed_as_ok=unknown_displayed_as_ok,
        claims_with_evidence=claims_with_evidence,
        total_claims_emitted=total_claims_emitted,
    )
    print("Hybrid propensities:", new_props)

    # Demonstrate diffusion noise
    token_a = [0.1, 0.4, 0.5]
    token_b = [0.2, 0.3, 0.6]
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    noise_vec = diffusion_noise(token_a, token_b, honesty)
    print("Diffusion noise vector:", noise_vec)

    # Apply a single step of social interaction with noise
    g_best = [0.5, 0.5, 0.5]
    updated = social_interaction(context, g_best, k=1, seed=42)
    updated_noisy = np.add(updated, noise_vec).tolist()
    print("Updated position with noise:", updated_noisy)