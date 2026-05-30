# DARWIN HAMMER — match 1215, survivor 7
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_infota_m346_s1.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_bandit_router_m56_s1.py (gen2)
# born: 2026-05-29T23:34:25Z

"""Hybrid Fisher‑SSIM Bandit algorithm.

Parents:
- hybrid_hybrid_fisher_locali_hybrid_hybrid_infota_m346_s1.py (Fisher information for a Gaussian beam and MinHash‑based infotaxis)
- hybrid_hybrid_ternary_route_hybrid_bandit_router_m56_s1.py (SSIM similarity and multi‑armed bandit decision making)

Mathematical bridge:
Both parents expose a *similarity* or *information density* term.
The Fisher score 𝓕(θ) quantifies the information content of a sensing direction,
while the SSIM index 𝒮(x,y) measures payload similarity.
We fuse them by defining a joint information weight

    𝓦(θ, x, y) = 𝓕(θ) · 𝒮(x, y)

which is then used as the reward multiplier inside a classic Upper‑Confidence‑Bound (UCB)
bandit update.  The resulting hybrid metric simultaneously guides the choice of
sensing angle and the selection of the packet‑handling action that maximises expected
information gain.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Sequence

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
# Parent B building blocks (SSIM & Bandit)
# ----------------------------------------------------------------------
def compute_ssim(
    x: Sequence[float],
    y: Sequence[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index Measure between two 1‑D signals."""
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


@dataclass
class BanditAction:
    """Record for a single arm of the bandit."""
    action_id: str
    expected_reward: float = 0.0
    count: int = 0
    total_reward: float = 0.0
    # optional extra data (e.g., reference payload for SSIM)
    reference_payload: List[float] = field(default_factory=list)


# ----------------------------------------------------------------------
# Hybrid core: combine Fisher information, SSIM and UCB bandit logic
# ----------------------------------------------------------------------
def hybrid_information(
    theta: float,
    center: float,
    width: float,
    payload_ref: Sequence[float],
    payload_obs: Sequence[float],
) -> float:
    """
    Joint information weight 𝓦 = Fisher(θ) · SSIM(payload_ref, payload_obs).

    The weight is dimensionless and bounded by the SSIM term ([-1, 1]) while
    Fisher supplies a positive scaling factor that favours directions with
    higher curvature of the Gaussian beam.
    """
    f = fisher_score(theta, center, width)
    s = compute_ssim(list(payload_ref), list(payload_obs))
    return f * s


def ucb_score(action: BanditAction, total_counts: int, alpha: float = 2.0) -> float:
    """
    Classic Upper‑Confidence‑Bound score.
    The exploration term is scaled by `alpha` and by the inverse square root
    of the arm count.
    """
    if action.count == 0:
        # Encourage trying unseen actions
        return float("inf")
    exploitation = action.expected_reward
    exploration = alpha * math.sqrt(math.log(total_counts + 1) / action.count)
    return exploitation + exploration


def hybrid_bandit_step(
    actions: Dict[str, BanditAction],
    theta: float,
    center: float,
    width: float,
    packet: Dict[str, Any],
) -> Tuple[BanditAction, float]:
    """
    Perform one decision‑selection + update cycle.

    1. Compute a hybrid information weight for each action using its stored
       reference payload and the packet's observed payload.
    2. Multiply the weight by a base reward (here taken as 1.0) to obtain the
       stochastic reward for the chosen arm.
    3. Update the arm statistics with the obtained reward.
    4. Return the selected action and the received reward.
    """
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        raise ValueError("packet must contain a numeric 'payload' list")

    total_counts = sum(a.count for a in actions.values()) + 1  # +1 to avoid log(0)

    # Compute a hybrid UCB for each action
    best_action = None
    best_score = -float("inf")
    for act in actions.values():
        weight = hybrid_information(
            theta,
            center,
            width,
            act.reference_payload,
            payload,
        )
        # Base reward is weight (could be scaled further)
        ucb = ucb_score(act, total_counts) * weight
        if ucb > best_score:
            best_score = ucb
            best_action = act

    if best_action is None:
        raise RuntimeError("No actions available for selection")

    # Simulated stochastic reward: we treat the hybrid weight as the deterministic
    # part and add a small Gaussian noise to mimic measurement uncertainty.
    reward = best_score + random.gauss(0, 0.01)

    # Update statistics
    best_action.count += 1
    best_action.total_reward += reward
    best_action.expected_reward = best_action.total_reward / best_action.count

    return best_action, reward


# ----------------------------------------------------------------------
# Helper to create a synthetic MinHash‑like signature (optional bridge)
# ----------------------------------------------------------------------
def minhash_signature(tokens: Sequence[str], k: int = 64) -> List[int]:
    """
    Very lightweight MinHash signature: for each of `k` random seeds we keep the
    minimum 64‑bit hash of all tokens.  This mimics the original parent A
    MinHash component and can be used as an auxiliary similarity measure if
    desired.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    seeds = [random.randint(0, 2 ** 31 - 1) for _ in range(k)]
    signature = [2 ** 64 - 1] * k
    for token in tokens:
        for i, seed in enumerate(seeds):
            h = int.from_bytes(
                hashlib.blake2b(
                    seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore"),
                    digest_size=8,
                ).digest(),
                "big",
            )
            if h < signature[i]:
                signature[i] = h
    return signature


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _demo() -> None:
    # Define a simple angular model
    theta_center = 0.0
    theta_width = 0.5

    # Create three dummy actions each with a different reference payload
    actions: Dict[str, BanditAction] = {
        "A": BanditAction(
            action_id="A",
            reference_payload=[0.1 * math.sin(i) for i in range(20)],
        ),
        "B": BanditAction(
            action_id="B",
            reference_payload=[0.1 * math.cos(i) for i in range(20)],
        ),
        "C": BanditAction(
            action_id="C",
            reference_payload=[0.05 * (i % 3) for i in range(20)],
        ),
    }

    # Simulate a stream of packets with random payloads and random sensing angles
    for step in range(30):
        theta = random.uniform(-1.0, 1.0)  # sensing direction
        payload = [random.random() for _ in range(20)]
        packet = {"payload": payload}
        selected, reward = hybrid_bandit_step(
            actions, theta, theta_center, theta_width, packet
        )
        print(
            f"Step {step:02d}: θ={theta:+.3f}, selected={selected.action_id}, reward={reward:.4f}"
        )

    # Show final estimated rewards
    print("\nFinal estimated rewards:")
    for act in actions.values():
        print(
            f"Action {act.action_id}: count={act.count}, expected_reward={act.expected_reward:.4f}"
        )


if __name__ == "__main__":
    _demo()