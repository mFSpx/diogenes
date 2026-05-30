# DARWIN HAMMER — match 3945, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_ternary_lens__m71_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_liquid_m2175_s0.py (gen5)
# born: 2026-05-29T23:52:48Z

import math
import random
import sys
from typing import List, Tuple, Dict
import numpy as np

# ----------------------------------------------------------------------
# Parent-A building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
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
# Parent-B building blocks
# ----------------------------------------------------------------------
_MAX64 = (1 << 64) - 1
_POLICY: Dict[str, List[float]] = {}  # action_id -> [total_reward, count]

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List["BanditUpdate"]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += u.reward
        stats[1] += 1

@dataclass
class BanditUpdate:
    action_id: str
    reward: float

def _hash(seed: int, token: str) -> int:
    import hashlib
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: List[str], num_perm: int) -> np.ndarray:
    sig = np.full(num_perm, _MAX64, dtype=np.uint64)
    for token in tokens:
        for i in range(num_perm):
            sig[i] = min(sig[i], _hash(i, token))
    return sig

def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    if sig1.shape != sig2.shape:
        raise ValueError("signatures must have the same length")
    return float(np.mean(sig1 == sig2))

# ----------------------------------------------------------------------
# Hybrid core: reward, action selection, liquid time constant
# ----------------------------------------------------------------------
def hybrid_reward(theta: float, center: float, width: float, tokens_a: List[str], tokens_b: List[str], num_perm: int = 64) -> float:
    I = fisher_score(theta, center, width)
    sig_a = minhash_signature(tokens_a, num_perm)
    sig_b = minhash_signature(tokens_b, num_perm)
    S = minhash_similarity(sig_a, sig_b)
    return I * S

def select_action_ucb(action_ids: List[str]) -> str:
    total_counts = sum(stats[1] for stats in _POLICY.values()) + 1e-9
    ucb_values: Dict[str, float] = {}
    for aid in action_ids:
        total_reward, count = _POLICY.get(aid, [0.0, 0.0])
        if count == 0:
            return aid
        avg = total_reward / count
        bonus = math.sqrt(2 * math.log(total_counts) / count)
        ucb_values[aid] = avg + bonus
    return max(ucb_values, key=ucb_values.get)

def adapt_liquid_time_constant(tau: float, reward: float, gamma: float = 0.1) -> float:
    return tau * math.exp(gamma * (reward - 0.5))

def hybrid_step(theta: float, center: float, width: float, tokens_a: List[str], tokens_b: List[str], action_ids: List[str], tau: float) -> Tuple[str, float, float]:
    reward = hybrid_reward(theta, center, width, tokens_a, tokens_b)
    action = select_action_ucb(action_ids)
    update_policy([BanditUpdate(action, reward)])
    new_tau = adapt_liquid_time_constant(tau, reward)
    return action, new_tau, reward

def improved_hybrid_step(theta: float, center: float, width: float, tokens_a: List[str], tokens_b: List[str], action_ids: List[str], tau: float) -> Tuple[str, float, float]:
    # Introduce a new parameter to control the sensitivity of the Fisher score
    alpha = 0.5
    I = fisher_score(theta, center, width) ** alpha
    sig_a = minhash_signature(tokens_a, 64)
    sig_b = minhash_signature(tokens_b, 64)
    S = minhash_similarity(sig_a, sig_b)
    reward = I * S
    action = select_action_ucb(action_ids)
    update_policy([BanditUpdate(action, reward)])
    new_tau = adapt_liquid_time_constant(tau, reward)
    return action, new_tau, reward

def simulate_hybrid(num_steps: int = 10) -> None:
    reset_policy()
    center = 0.0
    width = 1.0
    vocab = ["evidence", "plan", "pause", "ask", "verify", "checklist", "later", "review", "document", "audit", "schedule", "budget"]
    actions = [f"arm_{i}" for i in range(5)]
    tau = 1.0
    for step in range(num_steps):
        theta = random.gauss(0, 2)
        tokens_a = random.sample(vocab, k=5)
        tokens_b = random.sample(vocab, k=5)
        act, tau, rew = improved_hybrid_step(theta, center, width, tokens_a, tokens_b, actions, tau)

simulate_hybrid()