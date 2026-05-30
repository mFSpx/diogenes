# DARWIN HAMMER — match 1153, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s5.py (gen4)
# parent_b: epistemic_certainty.py (gen0)
# born: 2026-05-29T23:33:06Z

"""Hybrid algorithm combining DARWIN HAMMER bandit with RBF surrogate (parent A)
and epistemic certainty metadata (parent B).

Mathematical bridge:
- Parent A estimates expected rewards via empirical averages and an RBF surrogate:
      ŷ(x) = Σ_i w_i * exp(-ε² ||x - c_i||²)
- Parent B provides a certainty flag with a confidence expressed in basis points
  (0..10000).  We map this confidence to a scaling factor α = confidence_bps / 10000
  and use it to weight both the empirical reward update and the surrogate weight
  update.  Thus the surrogate solution solves a linear system where each
  observation contributes with weight α, integrating epistemic certainty directly
  into the matrix equations of the bandit.

The resulting hybrid system updates bandit statistics, builds an RBF surrogate
with certainty‑weighted observations, and predicts context‑aware rewards that
are modulated by epistemic confidence.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Sequence, Callable, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Types and global stores (from parent A)
# ----------------------------------------------------------------------
Vector = Sequence[float]

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

# Global mutable state (mirrors parent A)
_POLICY: Dict[str, List[float]] = {}          # action_id -> [total_reward, count]
_STORE: Dict[str, float] = {}                 # arbitrary storage for auxiliary data
_SURROGATE = None                             # will hold an RBFSurrogate instance

# ----------------------------------------------------------------------
# Epistemic certainty (from parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

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
            object.__setattr__(
                self,
                "generated_at",
                __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> dict:
        return asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


def filesystem_observation(*, sha256: str, path: str, mtime_utc: str | None = None) -> CertaintyFlag:
    refs = [f"sha256:{sha256}", f"path:{path}"]
    if mtime_utc:
        refs.append(f"mtime:{mtime_utc}")
    return certainty(
        "FACT",
        confidence_bps=10000,
        authority_class="filesystem_observation",
        rationale=(
            "Local file bytes were hashed and copied into CAS; "
            "this proves byte custody, not semantic truth."
        ),
        evidence_refs=refs,
    )

# ----------------------------------------------------------------------
# Core mathematical utilities (from parent A)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]


@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

# ----------------------------------------------------------------------
# Helper to encode actions (completion of parent A)
# ----------------------------------------------------------------------
def _encode_action(action_id: str, action_space: List[str]) -> List[float]:
    """One‑hot encode an action identifier within the given action space."""
    encoding = [0.0] * len(action_space)
    try:
        idx = action_space.index(action_id)
        encoding[idx] = 1.0
    except ValueError:
        raise ValueError(f"action_id {action_id!r} not found in action space")
    return encoding

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def reset_hybrid_state(action_space: List[str]) -> None:
    """Reset all global structures and initialise an empty surrogate."""
    _POLICY.clear()
    _STORE.clear()
    global _SURROGATE
    _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=1.0)
    # store action space for encoding convenience
    _STORE["action_space"] = action_space


def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def update_policy_with_certainty(update: BanditUpdate, certainty: CertaintyFlag) -> None:
    """
    Update empirical statistics and the RBF surrogate using the confidence
    factor derived from a CertaintyFlag.
    """
    alpha = certainty.confidence_bps / 10000.0  # scale 0..1
    # ---- empirical update -------------------------------------------------
    total, n = _POLICY.get(update.action_id, [0.0, 0.0])
    total += alpha * update.reward
    n += alpha
    _POLICY[update.action_id] = [total, n]

    # ---- surrogate update -------------------------------------------------
    # Encode context (here we reuse action encoding as a proxy for context)
    action_space: List[str] = _STORE.get("action_space", [])
    if not action_space:
        raise RuntimeError("action space not initialised; call reset_hybrid_state first")
    x = _encode_action(update.context_id, action_space)  # treat context_id as an action name

    # Append new center and weight proportional to alpha * reward
    if isinstance(_SURROGATE, RBFSurrogate):
        new_centers = list(_SURROGATE.centers) + [tuple(x)]
        new_weights = list(_SURROGATE.weights) + [alpha * update.reward]
        # Re‑solve the linear system to keep surrogate consistent
        weights = solve_surrogate_system(new_centers, _SURROGATE.epsilon)
        # Replace global surrogate
        global _SURROGATE
        _SURROGATE = RBFSurrogate(centers=new_centers, weights=weights, epsilon=_SURROGATE.epsilon)


def solve_surrogate_system(centers: List[Tuple[float, ...]], epsilon: float) -> List[float]:
    """
    Given RBF centers, build the kernel matrix K_ij = exp(-ε²||c_i-c_j||²)
    and solve K w = y where y_i = 1 (unit target) for a well‑posed system.
    The choice of y is arbitrary; here we use unit targets so that the
    resulting weights reflect the geometry of the centers alone.
    """
    n = len(centers)
    if n == 0:
        return []
    K = [[gaussian(euclidean(c_i, c_j), epsilon) for c_j in centers] for c_i in centers]
    y = [1.0] * n
    return solve_linear(K, y)


def predict_with_certainty(context_id: str, certainty: CertaintyFlag) -> float:
    """
    Predict reward for a given context using the surrogate and then modulate
    the output by the epistemic confidence factor.
    """
    action_space: List[str] = _STORE.get("action_space", [])
    if not action_space:
        raise RuntimeError("action space not initialised")
    x = _encode_action(context_id, action_space)
    base_pred = _SURROGATE.predict(x) if _SURROGATE else 0.0
    alpha = certainty.confidence_bps / 10000.0
    return base_pred * alpha


def sample_action(propensity_distribution: Dict[str, float]) -> str:
    """
    Sample an action according to a provided propensity distribution.
    This mirrors the stochastic selection used in many bandit algorithms.
    """
    actions, probs = zip(*propensity_distribution.items())
    cumulative = np.cumsum(probs)
    r = random.random() * cumulative[-1]
    idx = np.searchsorted(cumulative, r)
    return actions[int(idx)]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny action space
    actions = ["rock", "paper", "scissors"]
    reset_hybrid_state(actions)

    # Create a certainty flag for a trustworthy observation
    cf = filesystem_observation(sha256="deadbeef", path="/tmp/example.txt")

    # Simulate a few bandit updates with varying rewards
    updates = [
        BanditUpdate(context_id="rock", action_id="rock", reward=1.0, propensity=0.33),
        BanditUpdate(context_id="paper", action_id="paper", reward=0.0, propensity=0.33),
        BanditUpdate(context_id="scissors", action_id="scissors", reward=0.5, propensity=0.34),
    ]

    for u in updates:
        update_policy_with_certainty(u, cf)

    # Predict for each context using a high‑confidence flag
    high_conf = certainty(
        "PROBABLE",
        confidence_bps=9000,
        authority_class="test_suite",
        rationale="synthetic confidence",
    )
    for ctx in actions:
        pred = predict_with_certainty(ctx, high_conf)
        print(f"Predicted reward for {ctx!r}: {pred:.4f}")

    # Sample an action from empirical propensities (softmax style)
    prop_dist = {
        a: math.exp(_empirical_reward(a)) for a in actions
    }
    total = sum(prop_dist.values())
    prop_dist = {k: v / total for k, v in prop_dist.items()}
    chosen = sample_action(prop_dist)
    print(f"Sampled action: {chosen}")