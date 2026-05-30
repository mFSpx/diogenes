# DARWIN HAMMER — match 4673, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2590_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_vorono_m1361_s5.py (gen4)
# born: 2026-05-29T23:57:25Z

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit arm."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Immutable observation coming from the environment."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Simple policy statistics (mean reward, count, total propensity)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])


def reset_policy() -> None:
    """Erase all learned reward statistics."""
    _POLICY.clear()


def _policy_stats(action_id: str) -> Tuple[float, float, float]:
    """Return (total_reward, count, total_propensity) for the given action."""
    return tuple(_POLICY[action_id])


def update_policy(updates: Iterable[BanditUpdate]) -> None:
    """Incrementally incorporate reward observations."""
    for u in updates:
        total, cnt, total_prop = _policy_stats(u.action_id)
        _POLICY[u.action_id] = [
            total + float(u.reward),
            cnt + 1.0,
            total_prop + float(u.propensity),
        ]


def _mean_reward(action_id: str) -> float:
    """Mean reward for an action (0 if never observed)."""
    total, cnt, _ = _policy_stats(action_id)
    return total / cnt if cnt else 0.0


# ----------------------------------------------------------------------
# Count‑Min Sketch implementation
# ----------------------------------------------------------------------
class CountMinSketch:
    """A lightweight Count‑Min Sketch with configurable depth and width."""

    def __init__(self, depth: int = 4, width: int = 1024, seed: int = 0) -> None:
        self.depth = depth
        self.width = width
        self.table = np.zeros((depth, width), dtype=np.float64)
        self.seed = seed
        # Pre‑compute a list of salts for each row to avoid re‑hashing the seed.
        random.seed(seed)
        self._salts = [random.randrange(1 << 30) for _ in range(depth)]

    def _hash(self, item: str) -> List[int]:
        """Row‑wise column indices for a given item."""
        return [
            int(
                hashlib.sha256(f"{salt}:{item}".encode()).hexdigest(),
                16,
            )
            % self.width
            for salt in self._salts
        ]

    def add(self, item: str, weight: float = 1.0) -> None:
        """Add *weight* to the counters associated with *item*."""
        for row, col in enumerate(self._hash(item)):
            self.table[row, col] += weight

    def estimate(self, item: str) -> float:
        """Return the (upper‑bound) frequency estimate for *item*."""
        return min(self.table[row, col] for row, col in enumerate(self._hash(item)))


# Global sketch instance – shared across the hybrid algorithm.
_CMS = CountMinSketch(depth=6, width=2048, seed=42)


# ----------------------------------------------------------------------
# Fisher‑information based utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Unnormalised Gaussian evaluated at *theta*."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian‑like intensity model.

    The intensity is ``I = max(gaussian_beam, eps)`` to avoid division by zero.
    The score is (dI/dθ)² / I.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def _dynamic_fisher_params(rewards: List[float]) -> Tuple[float, float]:
    """
    Compute a data‑driven centre and width for the Fisher score.

    Centre = mean of rewards, width = std‑dev + small epsilon.
    """
    if not rewards:
        return 0.5, 0.1  # sensible defaults for an empty set
    mu = float(np.mean(rewards))
    sigma = float(np.std(rewards))
    return mu, max(sigma, 1e-3)


# ----------------------------------------------------------------------
# Similarity utilities (unchanged but kept for completeness)
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural similarity index (SSIM) for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


def hygiene_score(text: str, reference_text: str,
                  center: float, width: float) -> float:
    """
    Weighted similarity between *text* and *reference_text*.

    The raw SSIM is modulated by a Fisher score that treats the similarity
    itself as a Gaussian‑like observation.
    """
    x = np.fromiter((ord(c) for c in text), dtype=np.float64, count=len(text))
    y = np.fromiter((ord(c) for c in reference_text), dtype=np.float64, count=len(reference_text))
    similarity = ssim(x, y)
    fisher = fisher_score(similarity, center, width)
    return fisher * similarity


# ----------------------------------------------------------------------
# Core hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_select_action(actions: List[BanditAction]) -> BanditAction:
    """
    Choose the arm that maximises a *information‑aware* utility.

    Utility = FisherScore(expected_reward) × (1 / (1 + sketch_estimate)).
    The sketch estimate acts as a proxy for how often the arm has been tried,
    encouraging exploration of less‑visited arms.
    """
    if not actions:
        raise ValueError("action list must be non‑empty")

    # Dynamic Fisher parameters based on the current reward landscape.
    rewards = [a.expected_reward for a in actions]
    centre, width = _dynamic_fisher_params(rewards)

    utilities = []
    for a in actions:
        fisher = fisher_score(a.expected_reward, centre, width)
        sketch_est = _CMS.estimate(a.action_id)
        # Adding 1 prevents division by zero and smooths the influence.
        utility = fisher / (1.0 + sketch_est)
        utilities.append(utility)

    best_idx = int(np.argmax(utilities))
    return actions[best_idx]


def hybrid_update_policy(updates: Iterable[BanditUpdate]) -> None:
    """
    Incorporate observations into both the exact policy statistics and the
    Count‑Min Sketch. The sketch receives the *propensity* as weight, which
    reflects the confidence of the original sampling distribution.
    """
    for u in updates:
        _CMS.add(u.action_id, weight=u.propensity)
    update_policy(updates)


def hybrid_run(actions: List[BanditAction],
               updates: Iterable[BanditUpdate]) -> BanditAction:
    """
    Execute a single round of the hybrid algorithm.

    Returns the selected action so that callers can inspect the decision.
    """
    selected = hybrid_select_action(actions)
    hybrid_update_policy(updates)
    return selected


# ----------------------------------------------------------------------
# Simple smoke test (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset global state for reproducibility
    reset_policy()
    _CMS = CountMinSketch(depth=6, width=2048, seed=42)

    actions = [
        BanditAction("action1", 0.5, 1.0, 0.1, "algorithm1"),
        BanditAction("action2", 0.3, 0.8, 0.2, "algorithm2"),
        BanditAction("action3", 0.2, 0.4, 0.3, "algorithm3"),
    ]

    updates = [
        BanditUpdate("ctx1", "action1", 1.0, 0.5),
        BanditUpdate("ctx2", "action2", 0.5, 0.3),
        BanditUpdate("ctx3", "action3", 0.2, 0.2),
    ]

    chosen = hybrid_run(actions, updates)
    print(f"Chosen action: {chosen.action_id}")
    print("Policy statistics:")
    for aid in _POLICY:
        print(f"  {aid}: total_reward={_POLICY[aid][0]:.2f}, count={_POLICY[aid][1]}, "
              f"total_propensity={_POLICY[aid][2]:.2f}")
    print("Sketch estimates (approx. frequencies):")
    for a in actions:
        print(f"  {a.action_id}: {int(_CMS.estimate(a.action_id))}")