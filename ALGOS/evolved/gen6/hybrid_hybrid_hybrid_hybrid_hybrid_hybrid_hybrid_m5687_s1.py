# DARWIN HAMMER — match 5687, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_korpus_m1983_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1635_s1.py (gen5)
# born: 2026-05-30T00:04:14Z

"""Hybrid Bandit‑Geometric‑Morphology Algorithm
================================================

This module fuses the core of **PARENT ALGORITHM A** (bandit with RBF surrogate,
min‑hashing and Voronoi partition) and **PARENT ALGORITHM B** (bandit router,
morphological indices and Euclidean geometry).

Mathematical bridge
------------------
Both parents rely on Euclidean distances:

* In *A* the Voronoi partition assigns a context to the nearest centroid.
* In *B* Euclidean distance is used for morphological similarity.
* The RBF surrogate in *A* is defined as  
  \\(K(x, x') = \\exp\\big(-\\|x-x'\\|^2 / (2\\sigma^2)\\big)\\).

We therefore build a **single context vector** that concatenates a
min‑hash signature of the textual description with morphology‑derived
features.  Euclidean distance then serves three purposes:

1. Nearest‑centroid (Voronoi) assignment.
2. Kernel evaluation for the RBF surrogate.
3. Similarity measure for confidence‑bound computation.

The hybrid algorithm proceeds as:


context  = concatenate(minhash(text), morphology_features(morph))
centroid = nearest_centroid(context, centroids)          # Voronoi
reward̂   = RBF_surrogate_predict(context, stored_data)  # Bandit core
action   = UCB_select(actions, reward̂, policy_stats)   # Bandit decision


The implementation below provides the full pipeline together with a
smoke‑test that runs without error.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (identical to the parents)
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

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

# ----------------------------------------------------------------------
# Global stores (mirroring the parents)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}          # action_id -> [total_reward, count]
_STORE: Dict[str, float] = {}                # generic key‑value store (VRAM mock)
_CONTEXTS: List[np.ndarray] = []             # stored context vectors for RBF
_REWARDS: List[float] = []                  # corresponding rewards
_CENTROIDS: List[np.ndarray] = []            # Voronoi centroids (updated lazily)

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()
    _CONTEXTS.clear()
    _REWARDS.clear()
    _CENTROIDS.clear()


def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance in ℝⁿ."""
    return float(np.linalg.norm(a - b))


# ----------------------------------------------------------------------
# Morphology‑derived features (parent B)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def morphology_features(morph: Morphology) -> np.ndarray:
    """Return a normalized feature vector derived from morphology."""
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flat = flatness_index(morph.length, morph.width, morph.height)
    mass = morph.mass
    # Simple L2 normalisation
    vec = np.array([sph, flat, mass], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


# ----------------------------------------------------------------------
# Min‑hash signature (parent A)
# ----------------------------------------------------------------------
def minhash_signature(text: str, k: int = 10) -> np.ndarray:
    """
    Produce a deterministic k‑dimensional integer signature.
    The implementation uses a simple rolling hash on whitespace‑separated tokens.
    """
    tokens = text.split()
    sig = np.zeros(k, dtype=float)
    for i, token in enumerate(tokens):
        h = hash(token) & 0xffffffff  # unsigned 32‑bit
        idx = i % k
        sig[idx] = (sig[idx] + h) % 1e6  # keep numbers in a reasonable range
    # Normalise to [0,1]
    return sig / np.max(sig) if np.max(sig) > 0 else sig


# ----------------------------------------------------------------------
# Context construction (the hybrid bridge)
# ----------------------------------------------------------------------
def build_context(text: str, morph: Morphology, k: int = 10) -> np.ndarray:
    """
    Concatenate the min‑hash signature with morphology features to obtain
    a single Euclidean vector used throughout the hybrid algorithm.
    """
    mh = minhash_signature(text, k)
    mf = morphology_features(morph)
    return np.concatenate([mh, mf])


# ----------------------------------------------------------------------
# Voronoi partition (parent A)
# ----------------------------------------------------------------------
def update_centroids(new_context: np.ndarray) -> None:
    """
    Incrementally maintain a set of centroids.
    For simplicity we treat each stored context as a centroid until the
    number exceeds a small budget, after which we randomly merge two.
    """
    _CENTROIDS.append(new_context.copy())
    max_centroids = 50
    if len(_CENTROIDS) > max_centroids:
        # Randomly pick two centroids and replace them with their mean
        i, j = random.sample(range(len(_CENTROIDS)), 2)
        merged = (_CENTROIDS[i] + _CENTROIDS[j]) / 2.0
        # Remove higher index first to keep list integrity
        for idx in sorted([i, j], reverse=True):
            _CENTROIDS.pop(idx)
        _CENTROIDS.append(merged)


def voronoi_assign(context: np.ndarray) -> int:
    """Return the index of the nearest centroid (Voronoi region)."""
    if not _CENTROIDS:
        update_centroids(context)
        return 0
    distances = [euclidean_distance(context, c) for c in _CENTROIDS]
    return int(np.argmin(distances))


# ----------------------------------------------------------------------
# RBF surrogate (parent A)
# ----------------------------------------------------------------------
def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    diff = x - y
    return math.exp(-np.dot(diff, diff) / (2.0 * sigma * sigma))


def rbf_predict(context: np.ndarray, sigma: float = 1.0) -> float:
    """
    Predict the expected reward for *context* using stored (context, reward)
    pairs and an RBF kernel.  If no data are stored, return 0.0.
    """
    if not _CONTEXTS:
        return 0.0
    kernels = np.array([rbf_kernel(context, c, sigma) for c in _CONTEXTS])
    weighted = kernels * np.array(_REWARDS)
    return float(weighted.sum() / kernels.sum())


def rbf_update(context: np.ndarray, reward: float) -> None:
    """Store a new (context, reward) pair for future predictions."""
    _CONTEXTS.append(context.copy())
    _REWARDS.append(float(reward))


# ----------------------------------------------------------------------
# Bandit decision (shared core)
# ----------------------------------------------------------------------
def ucb_score(action: BanditAction, total_trials: int, sigma: float = 1.0) -> float:
    """
    Upper‑Confidence‑Bound score.
    confidence = sqrt(2 * log(total_trials) / action_trials)
    """
    _, count = _POLICY.get(action.action_id, [0.0, 0.0])
    if count == 0:
        confidence = float('inf')  # force exploration
    else:
        confidence = math.sqrt(2.0 * math.log(max(total_trials, 1)) / count)
    expected = _reward(action.action_id)
    return expected + confidence


def select_action(actions: List[BanditAction], context: np.ndarray,
                  sigma: float = 1.0) -> BanditAction:
    """
    Choose an action using a hybrid of:
    * RBF‑predicted reward as a prior for the expected reward,
    * UCB confidence bound derived from the bandit statistics.
    """
    # Inject the surrogate prediction into the policy cache temporarily
    surrogate = rbf_predict(context, sigma)
    # Create a temporary copy of the policy to avoid side effects
    temp_policy = _POLICY.copy()
    # Use the surrogate as the expected reward for *all* actions
    for a in actions:
        _POLICY[a.action_id] = [surrogate, 1.0]  # pretend one observation

    total_trials = sum(cnt for _, cnt in _POLICY.values())
    scores = [(ucb_score(a, total_trials, sigma), a) for a in actions]
    # Restore real policy
    _POLICY.clear()
    _POLICY.update(temp_policy)

    # Pick action with highest UCB score
    _, best = max(scores, key=lambda pair: pair[0])
    return best


def record_update(update: BanditUpdate, context: np.ndarray, reward: float) -> None:
    """
    Update both the bandit statistics and the RBF surrogate.
    """
    # Bandit statistics
    total, cnt = _POLICY.get(update.action_id, [0.0, 0.0])
    _POLICY[update.action_id] = [total + reward, cnt + 1.0]

    # Surrogate model
    rbf_update(context, reward)

    # Voronoi maintenance
    update_centroids(context)


# ----------------------------------------------------------------------
# Demonstration functions (required ≥3)
# ----------------------------------------------------------------------
def compute_context_and_region(text: str, morph: Morphology) -> Tuple[np.ndarray, int]:
    """Build a context vector and return its Voronoi region index."""
    ctx = build_context(text, morph)
    region = voronoi_assign(ctx)
    return ctx, region


def predict_reward_for_context(text: str, morph: Morphology) -> float:
    """Convenience wrapper that builds a context and returns the RBF prediction."""
    ctx = build_context(text, morph)
    return rbf_predict(ctx)


def run_one_step(actions: List[BanditAction],
                 text: str,
                 morph: Morphology,
                 sigma: float = 1.0) -> Tuple[BanditAction, float]:
    """
    Execute a full decision‑update cycle:
    1. Build context.
    2. Select action via hybrid UCB+RBF.
    3. Simulate a stochastic reward.
    4. Record the update.
    Returns the chosen action and the realised reward.
    """
    ctx, _ = compute_context_and_region(text, morph)
    chosen = select_action(actions, ctx, sigma)

    # Simulated reward: base on similarity between action_id hash and context norm
    base = euclidean_distance(ctx, np.zeros_like(ctx))
    noise = random.gauss(0, 0.1)
    reward = max(0.0, (hash(chosen.action_id) % 100) / 100.0 - 0.5 * base + noise)

    upd = BanditUpdate(context_id="ctx", action_id=chosen.action_id,
                       reward=reward, propensity=chosen.propensity)
    record_update(upd, ctx, reward)
    return chosen, reward


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    reset_policy()

    # Define a small action set
    actions = [
        BanditAction(action_id="A", propensity=0.33,
                     expected_reward=0.0, confidence_bound=0.0,
                     algorithm="hybrid"),
        BanditAction(action_id="B", propensity=0.33,
                     expected_reward=0.0, confidence_bound=0.0,
                     algorithm="hybrid"),
        BanditAction(action_id="C", propensity=0.34,
                     expected_reward=0.0, confidence_bound=0.0,
                     algorithm="hybrid")
    ]

    # Dummy contexts
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence merges geometry and bandits",
        "evolutionary algorithms explore high‑dimensional spaces"
    ]

    morphs = [
        Morphology(length=2.0, width=1.0, height=0.5, mass=3.0),
        Morphology(length=1.5, width=1.5, height=1.5, mass=2.0),
        Morphology(length=3.0, width=0.8, height=0.6, mass=4.5)
    ]

    # Run a few iterations
    for i in range(10):
        txt = random.choice(texts)
        mor = random.choice(morphs)
        act, rew = run_one_step(actions, txt, mor)
        print(f"Iter {i+1:02d}: chose {act.action_id} (reward={rew:.3f})")

    # Final policy snapshot
    print("\nFinal policy statistics:")
    for aid, (tot, cnt) in _POLICY.items():
        print(f"  Action {aid}: avg_reward={tot/cnt if cnt else 0:.3f}, count={int(cnt)}")

    # Demonstrate the hybrid functions directly
    ctx, region = compute_context_and_region(texts[0], morphs[0])
    print(f"\nExample context norm={np.linalg.norm(ctx):.3f}, Voronoi region={region}")
    print(f"RBF predicted reward for this context: {rbf_predict(ctx):.3f}")