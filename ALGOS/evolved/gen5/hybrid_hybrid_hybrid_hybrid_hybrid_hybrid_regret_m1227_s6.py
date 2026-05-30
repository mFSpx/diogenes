# DARWIN HAMMER — match 1227, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1161_s1.py (gen4)
# parent_b: hybrid_hybrid_regret_engine_hybrid_model_vram_sc_m1177_s1.py (gen4)
# born: 2026-05-29T23:34:33Z

"""Hybrid Algorithm integrating Fisher Score weighted regret (Parent A) with
Signature‑based similarity and Bandit decision (Parent B).

Mathematical bridge:
- Parent A provides `fisher_score(theta, center, width)` which evaluates the
  information content of a parameter `theta`.
- Parent B computes a regret‑based weight for each action (`compute_regret_weighted_strategy`).

The hybrid multiplies the regret weight of an action by a Fisher score that
is evaluated on a transformed version of the action’s expected value
(`theta = expected_value`).  This yields a *Fisher‑weighted regret* that
captures both statistical curvature (Fisher) and decision‑theoretic regret.

For similarity, Parent A’s `ssim` (structural similarity) is applied to the
binary signature vectors produced by Parent B’s `signature`.  The resulting
SSIM value is used as a confidence bound for the bandit action.

Thus the three core functions below fuse the governing equations of both
parents into a single unified decision‑making system."""
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

import numpy as np

# ---------- Parent A components ----------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index between two 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

# ---------- Parent B components ----------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # probability of selection
    expected_reward: float
    confidence_bound: float
    algorithm: str

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(
        np.frombuffer(
            np.frombuffer(
                bytearray(hashlib.blake2b(data, digest_size=8).digest()),
                dtype=np.uint8
            ).tobytes(),
            dtype=np.uint64
        )[0],
        'big'
    )

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Min‑hash style signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two integer signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def compute_regret_weighted_strategy(actions: List[MathAction],
                                     counterfactuals: List[MathCounterfactual]) -> Dict[str, float]:
    """Regret = expected_value - counterfactual outcome."""
    if not actions:
        return {}
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    regret_weights: Dict[str, float] = {}
    for a in actions:
        regret = a.expected_value - cf.get(a.id, 0.0)
        regret_weights[a.id] = max(regret, 0.0)
    return regret_weights

# ---------- Hybrid Functions ----------
def fisher_weighted_regret(actions: List[MathAction],
                           counterfactuals: List[MathCounterfactual],
                           theta_center: float,
                           theta_width: float) -> Dict[str, float]:
    """
    Combine Parent B's regret weights with Parent A's Fisher score.
    Each action's expected value is interpreted as a `theta` argument for
    the Fisher information; the resulting score scales the regret weight.
    """
    base_regret = compute_regret_weighted_strategy(actions, counterfactuals)
    weighted: Dict[str, float] = {}
    for a in actions:
        theta = a.expected_value  # map expected value → angle‑like parameter
        f_score = fisher_score(theta, theta_center, theta_width)
        weighted[a.id] = base_regret[a.id] * f_score
    return weighted

def signature_ssim(tokens_a: Iterable[str],
                  tokens_b: Iterable[str],
                  k: int = 128) -> float:
    """
    Produce min‑hash signatures for two token sets (Parent B) and evaluate
    their structural similarity using Parent A's SSIM.
    The signatures are cast to float arrays to satisfy the SSIM routine.
    """
    sig_a = np.array(signature(tokens_a, k), dtype=np.float64)
    sig_b = np.array(signature(tokens_b, k), dtype=np.float64)
    # Normalise to the dynamic range expected by SSIM (0‑255)
    max_val = max(sig_a.max(), sig_b.max(), 1.0)
    sig_a_norm = (sig_a / max_val) * 255.0
    sig_b_norm = (sig_b / max_val) * 255.0
    return ssim(sig_a_norm, sig_b_norm)

def hybrid_bandit_action(actions: List[MathAction],
                         counterfactuals: List[MathCounterfactual],
                         artifact_tokens: Iterable[str],
                         reference_tokens: Iterable[str],
                         theta_center: float,
                         theta_width: float,
                         artifact_id: str) -> BanditAction:
    """
    Produce a BanditAction whose propensity is proportional to the
    Fisher‑weighted regret, and whose confidence bound is derived from the
    SSIM similarity between the artifact's token signature and a reference
    signature.
    """
    # 1. Fisher‑weighted regret
    fw_regret = fisher_weighted_regret(actions, counterfactuals,
                                       theta_center, theta_width)

    # 2. Normalise propensities to a probability distribution
    total = sum(fw_regret.values()) + 1e-12
    propensities = {aid: w / total for aid, w in fw_regret.items()}

    # 3. Expected reward is the raw regret (before Fisher scaling)
    raw_regret = compute_regret_weighted_strategy(actions, counterfactuals)
    expected_reward = raw_regret.get(artifact_id, 0.0)

    # 4. Confidence bound from SSIM of signatures
    confidence = signature_ssim(artifact_tokens, reference_tokens)

    # 5. Assemble result
    return BanditAction(
        action_id=artifact_id,
        propensity=propensities.get(artifact_id, 0.0),
        expected_reward=expected_reward,
        confidence_bound=confidence,
        algorithm="FisherRegret+SSIMBandit"
    )

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Sample actions and counterfactuals
    actions = [
        MathAction(id="A", expected_value=1.2),
        MathAction(id="B", expected_value=0.8),
        MathAction(id="C", expected_value=1.5)
    ]
    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=0.9),
        MathCounterfactual(action_id="B", outcome_value=0.6),
        MathCounterfactual(action_id="C", outcome_value=1.0)
    ]

    # Token sets for signature/SSIM
    artifact_tokens = ["alpha", "beta", "gamma", "delta"]
    reference_tokens = ["alpha", "epsilon", "gamma", "zeta"]

    # Parameters for Fisher score
    theta_center = 1.0
    theta_width = 0.5

    # Choose an artifact/action to evaluate
    chosen_id = "A"

    bandit = hybrid_bandit_action(
        actions,
        counterfactuals,
        artifact_tokens,
        reference_tokens,
        theta_center,
        theta_width,
        chosen_id
    )

    print("Hybrid Bandit Action:")
    print(f"  action_id        : {bandit.action_id}")
    print(f"  propensity       : {bandit.propensity:.4f}")
    print(f"  expected_reward  : {bandit.expected_reward:.4f}")
    print(f"  confidence_bound : {bandit.confidence_bound:.4f}")
    print(f"  algorithm        : {bandit.algorithm}")