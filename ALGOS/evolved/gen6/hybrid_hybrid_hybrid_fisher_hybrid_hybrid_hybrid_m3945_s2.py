# DARWIN HAMMER — match 3945, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_ternary_lens__m71_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_liquid_m2175_s0.py (gen5)
# born: 2026-05-29T23:52:48Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_fisher_locali_hybrid_ternary_lens (Gaussian beam, Fisher score, SSIM)
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_liquid (Bandit policy, MinHash similarity, liquid time constant)

Mathematical Bridge:
The Fisher information of a Gaussian beam quantifies the sensitivity of a continuous
parameter θ. MinHash similarity quantifies the overlap of two discrete token sets.
We fuse them by defining a *hybrid reward* R = I_Fisher(θ) · S_MinHash, i.e. the product
of the Fisher information and the MinHash similarity. This scalar reward drives a
multi‑armed bandit update (UCB selection) and simultaneously modulates a liquid
time‑constant τ via an exponential update τ←τ·exp(γ·(R−½)). Thus the continuous
optics‑style sensitivity and the discrete set‑similarity are mathematically coupled
into a single adaptive decision‑making loop."""
import math
import random
import sys
from pathlib import Path
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """One‑dimensional Structural Similarity Index."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    cov = np.cov(x, y, ddof=0)[0, 1]
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
_MAX64 = (1 << 64) - 1
_POLICY: Dict[str, List[float]] = {}  # action_id -> [total_reward, count]


def reset_policy() -> None:
    """Clear the global reward statistics."""
    _POLICY.clear()


def update_policy(updates: List["BanditUpdate"]) -> None:
    """Accumulate rewards for each action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += u.reward
        stats[1] += 1


@dataclass
class BanditUpdate:
    action_id: str
    reward: float


def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using Blake2b (64‑bit output)."""
    import hashlib
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], num_perm: int) -> np.ndarray:
    """Compute MinHash signature for a set of tokens."""
    sig = np.full(num_perm, _MAX64, dtype=np.uint64)
    for token in tokens:
        for i in range(num_perm):
            sig[i] = min(sig[i], _hash(i, token))
    return sig


def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Compute MinHash similarity between two signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("signatures must have the same length")
    return float(np.mean(sig1 == sig2))


# ----------------------------------------------------------------------
# Hybrid core: reward, action selection, liquid time constant
# ----------------------------------------------------------------------
def hybrid_reward(theta: float, center: float, width: float,
                  tokens_a: List[str], tokens_b: List[str],
                  num_perm: int = 64) -> float:
    """
    Compute the fused reward R = I_Fisher(θ) * S_MinHash.
    The Fisher information provides a continuous‑parameter sensitivity,
    while MinHash similarity measures discrete overlap.
    """
    I = fisher_score(theta, center, width)
    sig_a = minhash_signature(tokens_a, num_perm)
    sig_b = minhash_signature(tokens_b, num_perm)
    S = minhash_similarity(sig_a, sig_b)
    return I * S


def select_action_ucb(action_ids: List[str]) -> str:
    """
    Upper‑Confidence‑Bound (UCB) selection using the global _POLICY.
    If an action has never been tried, it is returned immediately to ensure exploration.
    """
    total_counts = sum(stats[1] for stats in _POLICY.values()) + 1e-9
    ucb_values: Dict[str, float] = {}
    for aid in action_ids:
        total_reward, count = _POLICY.get(aid, [0.0, 0.0])
        if count == 0:
            # Force exploration
            return aid
        avg = total_reward / count
        bonus = math.sqrt(2 * math.log(total_counts) / count)
        ucb_values[aid] = avg + bonus
    # Choose the action with the highest UCB value
    return max(ucb_values, key=ucb_values.get)


def adapt_liquid_time_constant(tau: float, reward: float, gamma: float = 0.1) -> float:
    """
    Exponential adaptation of the liquid time constant τ.
    τ_{new} = τ * exp(γ·(R - 0.5)), where 0.5 is a neutral reward baseline.
    """
    return tau * math.exp(gamma * (reward - 0.5))


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def hybrid_step(theta: float, center: float, width: float,
                tokens_a: List[str], tokens_b: List[str],
                action_ids: List[str],
                tau: float) -> Tuple[str, float, float]:
    """
    Perform a single hybrid iteration:
    1. Compute reward via hybrid_reward.
    2. Select an action with UCB.
    3. Update the bandit policy with the reward.
    4. Adapt the liquid time constant τ.
    Returns (selected_action, updated_tau, reward).
    """
    reward = hybrid_reward(theta, center, width, tokens_a, tokens_b)
    action = select_action_ucb(action_ids)
    update_policy([BanditUpdate(action, reward)])
    new_tau = adapt_liquid_time_constant(tau, reward)
    return action, new_tau, reward


def simulate_hybrid(num_steps: int = 10) -> None:
    """
    Run a short simulation of the hybrid system.
    Demonstrates interaction between continuous optics, discrete similarity,
    bandit learning and liquid time‑constant adaptation.
    """
    reset_policy()
    # Fixed Gaussian parameters (could be made dynamic)
    center = 0.0
    width = 1.0
    # Token pools for two documents / contexts
    vocab = ["evidence", "plan", "pause", "ask", "verify", "checklist",
             "later", "review", "document", "audit", "schedule", "budget"]
    # Pre‑define a set of actions
    actions = [f"arm_{i}" for i in range(5)]
    # Initial liquid time constant
    tau = 1.0

    for step in range(num_steps):
        theta = random.gauss(0, 2)                     # random continuous parameter
        tokens_a = random.sample(vocab, k=5)           # random token subset A
        tokens_b = random.sample(vocab, k=5)           # random token subset B
        act, tau, rew = hybrid_step(theta, center, width,
                                    tokens_a, tokens_b,
                                    actions, tau)
        # Light‑weight diagnostics (no external I/O)
        _ = (act, tau, rew)  # placeholder to avoid lint warnings

    # Final policy snapshot (useful for debugging)
    _final_policy = dict(_POLICY)  # noqa: F841


if __name__ == "__main__":
    # Smoke test: run the simulation without raising exceptions.
    simulate_hybrid(num_steps=20)
    # Verify that SSIM works on two Gaussian beams sampled over a grid.
    grid = np.linspace(-3, 3, 200)
    beam1 = np.vectorize(lambda th: gaussian_beam(th, 0.0, 1.0))(grid)
    beam2 = np.vectorize(lambda th: gaussian_beam(th, 0.5, 1.2))(grid)
    _ = ssim(beam1, beam2)  # noqa: F841
    sys.exit(0)