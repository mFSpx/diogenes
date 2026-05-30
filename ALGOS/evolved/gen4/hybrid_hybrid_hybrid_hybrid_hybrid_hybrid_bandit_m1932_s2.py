# DARWIN HAMMER — match 1932, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_ternar_m132_s0.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s1.py (gen3)
# born: 2026-05-29T23:39:50Z

"""
Hybrid Decision-Hygiene, Geometric Context & Bandit‑Router Matrix Fusion
=======================================================================

Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_geomet_hybrid_hybrid_ternar_m132_s0.py`
  Provides a decision‑hygiene feature extractor that yields a grade‑1 multivector
  (here represented as a numeric feature vector) and geometric utilities such as
  Euclidean distance and Voronoi‑like region scoring.

* **Parent B** – `hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s1.py`
  Implements a contextual linear bandit (`LinUCB`‑style) with a mutable policy
  dictionary and matrix‑based resource‑allocation updates.

Mathematical Bridge
-------------------
The feature vector **f** produced by the decision‑hygiene extractor is interpreted as
a *context* in the linear bandit model.  The bandit selects an action **a** by
maximising  


Q_a = θ_a·f + α·√(fᵀA_a⁻¹f)


where `θ_a` is the estimated reward vector for action *a* and `A_a` is the
Gram matrix of observed contexts for *a*.  After receiving a reward **r**, the
algorithm updates both the bandit statistics **(θ, A)** *and* a global resource
allocation matrix **R** using the outer‑product update  


R ← R + η·r·(f fᵀ)


Thus the geometric algebra side contributes the context and distance measures,
while the bandit side contributes the decision policy and matrix‑based learning.
The three core functions below demonstrate this unified system.
"""

import re
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Decision‑hygiene feature extraction (Parent A)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE    = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE  = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no c)\b", re.I)

def extract_features(text: str) -> np.ndarray:
    """
    Count occurrences of the five keyword groups and return a 5‑dimensional
    feature vector (float).  The vector is L2‑normalised to serve as a unit
    context for the linear bandit.
    """
    counts = np.array([
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
        len(SUPPORT_RE.findall(text)),
        len(BOUNDARY_RE.findall(text)),
    ], dtype=float)

    norm = np.linalg.norm(counts)
    return counts / norm if norm > 0 else counts

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Geometric helper from Parent A – plain Euclidean norm."""
    return float(np.linalg.norm(v1 - v2))

# ----------------------------------------------------------------------
# Bandit core (Parent B) extended with contextual linear updates
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
    context: np.ndarray   # feature vector used at decision time
    action_id: str
    reward: float

# Global mutable state (policy statistics)
_POLICY_REWARDS: dict[str, float] = {}          # Σ r
_POLICY_COUNTS: dict[str, int] = {}            # N
_POLICY_A: dict[str, np.ndarray] = {}          # Gram matrix A_a (d×d)
_POLICY_B: dict[str, np.ndarray] = {}          # b_a = Σ r·f

def reset_policy() -> None:
    _POLICY_REWARDS.clear()
    _POLICY_COUNTS.clear()
    _POLICY_A.clear()
    _POLICY_B.clear()

def _init_action_stats(action_id: str, dim: int) -> None:
    """Lazy initialisation of matrices for a new action."""
    if action_id not in _POLICY_A:
        _POLICY_A[action_id] = np.identity(dim)      # A_a ← I_d
        _POLICY_B[action_id] = np.zeros(dim)         # b_a ← 0

def _theta(action_id: str) -> np.ndarray:
    """Estimated reward vector θ_a = A_a⁻¹ b_a (ridge‑regularised by identity)."""
    A = _POLICY_A[action_id]
    b = _POLICY_B[action_id]
    return np.linalg.solve(A, b)

def select_action(context: np.ndarray,
                  actions: list[str],
                  alpha: float = 1.0,
                  algorithm: str = 'linucb',
                  seed: int | str | None = 7) -> BanditAction:
    """
    Contextual Linear‑UCB action selection.
    - `context` is the normalized feature vector from the decision‑hygiene extractor.
    - `alpha` controls the width of the confidence interval.
    - Returns a BanditAction with the computed propensity, expected reward and bound.
    """
    if not actions:
        raise ValueError('At least one action required')
    rng = random.Random(seed)

    # epsilon‑greedy fallback (kept from Parent B)
    if algorithm == 'epsilon_greedy' and rng.random() < 0.1:
        chosen = rng.choice(actions)
        exp_reward = _POLICY_REWARDS.get(chosen, 0.0) / max(1, _POLICY_COUNTS.get(chosen, 0))
        bound = 0.0
        return BanditAction(chosen, 1.0 / len(actions), exp_reward, bound, algorithm)

    # Ensure matrices exist
    dim = context.shape[0]
    for a in actions:
        _init_action_stats(a, dim)

    # LinUCB score for each action
    scores = {}
    for a in actions:
        theta = _theta(a)
        A = _POLICY_A[a]
        # confidence term sqrt(fᵀ A⁻¹ f)
        conf = math.sqrt(float(context @ np.linalg.solve(A, context)))
        q = float(theta @ context) + alpha * conf
        scores[a] = q

    chosen = max(scores, key=scores.get)
    exp_reward = _POLICY_REWARDS.get(chosen, 0.0) / max(1, _POLICY_COUNTS.get(chosen, 0))
    bound = alpha * math.sqrt(float(context @ np.linalg.solve(_POLICY_A[chosen], context)))
    propensity = 1.0 / len(actions)  # uniform prior (could be refined)
    return BanditAction(chosen, propensity, exp_reward, bound, algorithm)

def update_policy(updates: list[BanditUpdate]) -> None:
    """
    Incorporate observed rewards and update both the linear bandit statistics
    (A, b) and the simple cumulative reward bookkeeping.
    """
    for upd in updates:
        a = upd.action_id
        r = upd.reward
        f = upd.context
        dim = f.shape[0]
        _init_action_stats(a, dim)

        # Update cumulative reward stats
        _POLICY_REWARDS[a] = _POLICY_REWARDS.get(a, 0.0) + r
        _POLICY_COUNTS[a] = _POLICY_COUNTS.get(a, 0) + 1

        # Linear bandit matrix updates (ridge‑regularised)
        _POLICY_A[a] = _POLICY_A[a] + np.outer(f, f)
        _POLICY_B[a] = _POLICY_B[a] + r * f

# ----------------------------------------------------------------------
# Resource‑allocation matrix that fuses geometric outer‑product updates
# ----------------------------------------------------------------------
def init_resource_matrix(dim: int) -> np.ndarray:
    """Create a symmetric positive‑definite matrix R (dim×dim)."""
    return np.identity(dim)

def update_resource_matrix(R: np.ndarray,
                           context: np.ndarray,
                           reward: float,
                           eta: float = 0.05) -> np.ndarray:
    """
    Hybrid update:
        R ← R + η·reward·(f fᵀ)
    This mirrors the matrix‑based update described in Parent B while using the
    geometric outer product of the decision‑hygiene context (Parent A).
    The function returns the updated matrix for convenience.
    """
    outer = np.outer(context, context)
    R += eta * reward * outer
    # Keep symmetry (numerical drift)
    R = (R + R.T) / 2.0
    return R

# ----------------------------------------------------------------------
# Example high‑level pipeline exposing the hybrid behaviour
# ----------------------------------------------------------------------
def hybrid_step(text: str,
                actions: list[str],
                resource_matrix: np.ndarray,
                alpha: float = 1.0,
                eta: float = 0.05) -> tuple[BanditAction, np.ndarray]:
    """
    One complete decision‑allocation cycle:
    1. Extract features from `text`.
    2. Select an action using contextual LinUCB.
    3. Simulate a reward (here proportional to the Euclidean norm of the feature vector).
    4. Update the bandit policy and the shared resource matrix.
    Returns the chosen action and the updated resource matrix.
    """
    # 1. Feature extraction (geometric context)
    f = extract_features(text)

    # 2. Action selection (bandit)
    act = select_action(f, actions, alpha=alpha)

    # 3. Simulated reward – for demonstration we use a simple function:
    #    reward = 1 + 0.5 * (distance to a fixed prototype vector)
    prototype = np.ones_like(f) / np.linalg.norm(np.ones_like(f))
    reward = 1.0 + 0.5 * euclidean_distance(f, prototype)

    # 4. Policy and matrix updates
    update_policy([BanditUpdate(context=f, action_id=act.action_id, reward=reward)])
    R_new = update_resource_matrix(resource_matrix, f, reward, eta=eta)

    return act, R_new

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We have verified the source and captured a screenshot. "
        "The next steps include planning the rollout and allocating budget. "
        "If any delay occurs we will wait until tomorrow. "
        "Support from the team is confirmed."
    )
    actions = ["allocate", "defer", "escalate"]
    dim = 5  # matches the feature vector length
    R = init_resource_matrix(dim)

    reset_policy()
    for i in range(3):
        action, R = hybrid_step(sample_text, actions, R)
        print(f"Round {i+1}: chosen='{action.action_id}', expected_reward={action.expected_reward:.3f}, bound={action.confidence_bound:.3f}")
    print("Final resource matrix R:")
    print(R)