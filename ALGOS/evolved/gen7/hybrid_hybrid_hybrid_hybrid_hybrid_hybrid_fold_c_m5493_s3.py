# DARWIN HAMMER — match 5493, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hdc_se_m172_s2.py (gen4)
# parent_b: hybrid_hybrid_fold_change_d_hybrid_hybrid_hybrid_m1947_s1.py (gen6)
# born: 2026-05-30T00:02:19Z

"""Hybrid Fisher‑JEPA & Bandit‑Hyperdimensional Fusion

Parents:
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hdc_se_m172_s2.py (Fisher‑based information density + JEPA)
- hybrid_hybrid_fold_change_d_hybrid_hybrid_hybrid_m1947_s1.py (Bandit policy + hyperdimensional multivector operations)

Mathematical Bridge
------------------
The Fisher information *F(θ)* computed by the Gaussian‑beam model is treated as a
scalar “confidence” that modulates a bandit’s propensity to select an action.
Simultaneously, *F(θ)* is embedded into a high‑dimensional bipolar vector
*𝑣_F* ∈ {‑1,+1}^D via a random projection.  Action identifiers are also mapped to
bipolar vectors *𝑣_a*.  The similarity 𝑠 = 𝑣_F·𝑣_a (dot product) provides a
hyperdimensional cue that, together with the scalar confidence, drives a
UCB‑style policy update:

    propensity_a ← σ(F(θ))                (σ = sigmoid)
    confidence_bound_a ← √(2·ln N / n_a)   (UCB)
    score_a ← propensity_a + confidence_bound_a + α·s

where *N* is the total number of pulls, *n_a* the pull count of action *a*, and
α a weighting factor for the hyperdimensional similarity.  The resulting
score is used to select the next action.  This code implements the core
functions of that hybrid system."""


import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Iterable
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Parent A – Fisher‑based Gaussian beam utilities
# ---------------------------------------------------------------------------

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float,
                 center: float = 0.0,
                 width: float = 1.0,
                 eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ---------------------------------------------------------------------------
# Parent B – Bandit structures and hyperdimensional primitives
# ---------------------------------------------------------------------------

EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE"
)


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
    propensity: float


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.utcnow().isoformat() + "Z")

    def as_dict(self) -> Dict[str, any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Hyperdimensional primitives (bipolar vectors)
# ---------------------------------------------------------------------------

_DIMENSION = 10_000                     # dimensionality of the HD space
_RNG = np.random.default_rng(42)        # reproducible randomness
_RANDOM_MATRIX = _RNG.standard_normal((_DIMENSION, 1))  # projection matrix


def _bipolar(vec: np.ndarray) -> np.ndarray:
    """Convert a real‑valued vector to bipolar {‑1,+1}."""
    return np.where(vec >= 0, 1, -1).astype(np.int8)


def encode_scalar_to_hd(value: float) -> np.ndarray:
    """
    Encode a scalar (e.g., Fisher information) into a bipolar HD vector.
    The encoding uses a fixed random projection followed by a sign threshold.
    """
    proj = _RANDOM_MATRIX.squeeze() * value
    return _bipolar(proj)


def bind(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Component‑wise XOR (binding) for bipolar vectors."""
    return np.where(v1 == v2, 1, -1).astype(np.int8)


def similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine‑like similarity (dot product normalized by dimension)."""
    return float(np.dot(v1, v2) / _DIMENSION)


# ---------------------------------------------------------------------------
# Hybrid Bandit policy that consumes Fisher information
# ---------------------------------------------------------------------------

_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, pulls]
_ACTION_VECTORS: Dict[str, np.ndarray] = {}  # action_id → bipolar HD vector
_TOTAL_PULLS: int = 0                        # global pull counter
_ALPHA: float = 0.5                          # weight of HD similarity in score


def reset_policy() -> None:
    """Clear all policy state."""
    global _POLICY, _ACTION_VECTORS, _TOTAL_PULLS
    _POLICY.clear()
    _ACTION_VECTORS.clear()
    _TOTAL_PULLS = 0


def _reward_estimate(action: str) -> float:
    """Mean reward observed for *action* (0 if never pulled)."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n > 0 else 0.0


def _confidence_bound(action: str) -> float:
    """UCB confidence bound for *action*."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    if n == 0:
        return float('inf')
    return math.sqrt(2 * math.log(_TOTAL_PULLS + 1) / n)


def register_action(action_id: str) -> None:
    """Create a random HD representation for a new action."""
    if action_id not in _ACTION_VECTORS:
        vec = _RNG.choice([-1, 1], size=_DIMENSION).astype(np.int8)
        _ACTION_VECTORS[action_id] = vec
        _POLICY.setdefault(action_id, [0.0, 0.0])


def update_policy(update: BanditUpdate, fisher: float) -> None:
    """
    Incorporate a bandit observation together with Fisher confidence.
    The scalar Fisher value modulates the propensity via a sigmoid,
    while its HD encoding influences the similarity term.
    """
    global _TOTAL_PULLS
    _TOTAL_PULLS += 1

    register_action(update.action_id)

    # scalar propensity from Fisher (sigmoid maps to (0,1))
    propensity = 1.0 / (1.0 + math.exp(-fisher))

    # HD similarity between Fisher encoding and the action vector
    fisher_hd = encode_scalar_to_hd(fisher)
    sim = similarity(fisher_hd, _ACTION_VECTORS[update.action_id])

    # UCB components
    cb = _confidence_bound(update.action_id)

    # Composite score (used only for logging here)
    composite_score = propensity + cb + _ALPHA * sim

    # Update the bandit statistics
    total, n = _POLICY[update.action_id]
    _POLICY[update.action_id] = [total + update.reward, n + 1]

    # Debug/trace (could be removed in production)
    print(f"[update] ctx={update.context_id} act={update.action_id} "
          f"reward={update.reward:.3f} fisher={fisher:.3f} prop={propensity:.3f} "
          f"sim={sim:.3f} cb={cb:.3f} score={composite_score:.3f}")


def select_action(context_id: str,
                  candidate_actions: Iterable[str],
                  thetas: Iterable[float],
                  center: float = 0.0,
                  width: float = 1.0) -> BanditAction:
    """
    Given a set of candidate actions and associated angle measurements *thetas*,
    compute Fisher scores, embed them, and pick the action with the highest
    hybrid UCB‑plus‑HD score.
    """
    best_action = None
    best_score = -float('inf')

    for action_id, theta in zip(candidate_actions, thetas):
        register_action(action_id)

        # 1) Fisher information for this observation
        fisher = fisher_score(theta, center=center, width=width)

        # 2) scalar components
        prop = 1.0 / (1.0 + math.exp(-fisher))
        cb = _confidence_bound(action_id)

        # 3) HD similarity component
        fisher_hd = encode_scalar_to_hd(fisher)
        sim = similarity(fisher_hd, _ACTION_VECTORS[action_id])

        # 4) Hybrid score
        score = prop + cb + _ALPHA * sim

        if score > best_score:
            best_score = score
            best_action = BanditAction(
                action_id=action_id,
                propensity=prop,
                expected_reward=_reward_estimate(action_id),
                confidence_bound=cb,
                algorithm="HybridFisherBandit"
            )

    if best_action is None:
        raise RuntimeError("No candidate actions provided")

    print(f"[select] ctx={context_id} chosen={best_action.action_id} score={best_score:.3f}")
    return best_action


# ---------------------------------------------------------------------------
# Demonstration functions
# ---------------------------------------------------------------------------

def demo_fisher_hd_encoding() -> None:
    """Show that encoding a range of Fisher scores yields diverse HD vectors."""
    scores = [fisher_score(theta) for theta in np.linspace(-2, 2, 5)]
    vectors = [encode_scalar_to_hd(s) for s in scores]
    for i, (s, v) in enumerate(zip(scores, vectors)):
        # Hamming distance between successive vectors as a sanity check
        if i:
            hd = np.mean(v != vectors[i - 1])
            print(f"Score {s:.3f} -> HD diff from previous: {hd:.3f}")
        else:
            print(f"Score {s:.3f} -> first HD vector generated")


def demo_bandit_cycle() -> None:
    """Run a tiny bandit loop with synthetic rewards and Fisher cues."""
    reset_policy()
    actions = ["A", "B", "C"]
    for t in range(1, 11):
        # Simulated angle measurement (theta) per round
        theta = random.uniform(-1.5, 1.5)
        # Select action using hybrid scoring
        chosen = select_action(
            context_id=f"step-{t}",
            candidate_actions=actions,
            thetas=[theta] * len(actions),   # same theta for simplicity
            center=0.0,
            width=1.0
        )
        # Synthetic reward: higher for action "B" when theta>0
        reward = 1.0 if (chosen.action_id == "B" and theta > 0) else 0.0
        # Update policy with observed reward and Fisher info
        update_policy(
            BanditUpdate(
                context_id=f"step-{t}",
                action_id=chosen.action_id,
                reward=reward,
                propensity=chosen.propensity
            ),
            fisher=fisher_score(theta)
        )
    # Final policy summary
    print("\nFinal policy statistics:")
    for a, (tot, n) in _POLICY.items():
        print(f"  Action {a}: pulls={int(n)} avg_reward={_reward_estimate(a):.3f}")


def demo_hybrid_prediction() -> None:
    """Predict a timestamp candidate using the hybrid mechanism."""
    # Example: we have three textual timestamp candidates with associated angles
    candidate_actions = ["2023-07-01", "2023-07-15", "2023-08-01"]
    thetas = [-0.8, 0.2, 1.1]  # imagined extracted angles
    best = select_action(
        context_id="doc-42",
        candidate_actions=candidate_actions,
        thetas=thetas,
        center=0.0,
        width=0.5
    )
    print(f"Chosen timestamp: {best.action_id} (propensity={best.propensity:.3f})")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Demo: Fisher → HD encoding ===")
    demo_fisher_hd_encoding()
    print("\n=== Demo: Bandit cycle with Fisher cues ===")
    demo_bandit_cycle()
    print("\n=== Demo: Hybrid timestamp prediction ===")
    demo_hybrid_prediction()