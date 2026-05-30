# DARWIN HAMMER — match 1215, survivor 6
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_infota_m346_s1.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_bandit_router_m56_s1.py (gen2)
# born: 2026-05-29T23:34:25Z

"""Hybrid Fisher‑SSIM‑Bandit algorithm.

Parent A provides a Fisher information score for a directional parameter θ
(gaussian beam model). Parent B provides an SSIM similarity metric between
packet payloads and a multi‑armed bandit framework that selects actions based
on estimated reward and an upper‑confidence bound (UCB).

Mathematical bridge:
    • The Fisher score 𝓕(θ) quantifies how informative a sensing angle is.
    • The SSIM value 𝒮(x, y) quantifies similarity between the current packet
      payload and a reference payload.
    • Their product 𝓕(θ)·𝒮(x, y) is used as a *weight* for the bandit’s expected
      reward. The weighted reward drives the UCB selection, thereby fusing
      information‑theoretic sensing (Fisher) with content similarity (SSIM) in
      the decision‑making (bandit) loop.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks (Fisher information)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Parent B building blocks (SSIM + Bandit)
# ----------------------------------------------------------------------
def compute_ssim(
    x: Sequence[float],
    y: Sequence[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM) for 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit arm."""
    action_id: str
    algorithm: str = "fisher_ssim_ucb"


class BanditPolicy:
    """Simple UCB‑based multi‑armed bandit policy."""
    def __init__(self):
        # action_id -> [cumulative_reward, count]
        self._stats: Dict[str, List[float]] = {}

    def reward(self, action_id: str) -> float:
        total, n = self._stats.get(action_id, [0.0, 0.0])
        return total / n if n > 0 else 0.0

    def count(self, action_id: str) -> float:
        return self._stats.get(action_id, [0.0, 0.0])[1]

    def total_counts(self) -> float:
        return sum(v[1] for v in self._stats.values())

    def update(self, action_id: str, reward: float) -> None:
        stats = self._stats.setdefault(action_id, [0.0, 0.0])
        stats[0] += reward
        stats[1] += 1.0

    def ucb_score(self, action_id: str, confidence: float = 2.0) -> float:
        """Upper‑confidence bound for a given arm."""
        n = self.count(action_id)
        total = self.total_counts()
        if n == 0:
            # encourage exploration of unseen arms
            return float("inf")
        avg = self.reward(action_id)
        exploration = confidence * math.sqrt(math.log(total + 1.0) / n)
        return avg + exploration


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_information_weight(
    theta: float,
    center: float,
    width: float,
    payload: Sequence[float],
    reference_payload: Sequence[float],
) -> float:
    """
    Compute the product of Fisher information and SSIM similarity.
    This scalar serves as a weight for the bandit reward.
    """
    f = fisher_score(theta, center, width)
    s = compute_ssim(payload, reference_payload)
    return f * s


def select_action_ucb(
    policy: BanditPolicy,
    actions: List[BanditAction],
    weight: float,
    confidence: float = 2.0,
) -> BanditAction:
    """
    Choose the action with the highest weighted UCB score.
    The weight (Fisher·SSIM) scales the UCB value, coupling sensing
    quality with the bandit decision.
    """
    best_action = None
    best_score = -float("inf")
    for act in actions:
        base = policy.ucb_score(act.action_id, confidence)
        score = weight * base
        if score > best_score:
            best_score = score
            best_action = act
    return best_action  # type: ignore


def hybrid_step(
    policy: BanditPolicy,
    actions: List[BanditAction],
    packet: Dict[str, Sequence[float]],
    theta: float,
    center: float,
    width: float,
    reference_payload: Sequence[float],
) -> Tuple[BanditAction, float]:
    """
    One iteration of the hybrid algorithm:
        1. Compute Fisher·SSIM weight.
        2. Select an action via weighted UCB.
        3. Simulate a stochastic reward (here: weight plus Gaussian noise).
        4. Update the bandit policy.
    Returns the chosen action and the observed reward.
    """
    payload = packet.get("payload", [])
    if not isinstance(payload, Sequence) or not payload:
        raise ValueError("packet must contain a non‑empty 'payload' sequence")

    weight = hybrid_information_weight(theta, center, width, payload, reference_payload)
    chosen = select_action_ucb(policy, actions, weight)

    # Simulated environment: reward is proportional to the weight with noise
    noise = random.gauss(mu=0.0, sigma=0.01)
    reward = weight + noise

    policy.update(chosen.action_id, reward)
    return chosen, reward


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(0)

    # Define a reference payload (e.g., expected packet shape)
    reference = [math.sin(i / 5.0) for i in range(50)]

    # Create a synthetic packet whose payload is a noisy version of the reference
    packet = {
        "payload": [v + random.gauss(0, 0.05) for v in reference]
    }

    # Bandit actions (e.g., routing decisions)
    action_ids = [f"route_{i}" for i in range(4)]
    actions = [BanditAction(aid) for aid in action_ids]

    # Initialise policy
    policy = BanditPolicy()

    # Parameters for the Fisher beam
    theta = 0.3
    center = 0.0
    width = 1.0

    # Run a few hybrid steps
    for step in range(10):
        act, rew = hybrid_step(
            policy,
            actions,
            packet,
            theta,
            center,
            width,
            reference,
        )
        print(f"Step {step+1:2d}: selected {act.action_id}, reward={rew:.5f}")

    # Final statistics
    print("\nFinal bandit statistics:")
    for aid in action_ids:
        avg = policy.reward(aid)
        cnt = policy.count(aid)
        print(f"  {aid}: count={int(cnt)}, avg_reward={avg:.5f}")

    sys.exit(0)