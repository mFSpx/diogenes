# DARWIN HAMMER — match 2666, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_regret_engine_hybrid_model_vram_sc_m1177_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_tropical_maxp_m941_s0.py (gen5)
# born: 2026-05-29T23:43:24Z

"""Hybrid Regret‑Bandit + HDC‑Tropical Engine

Parents:
- hybrid_hybrid_regret_engine_hybrid_model_vram_sc_m1177_s0.py (Regret‑Weighted Liquid‑Time‑Constant MinHash + VRAM‑Bandit Scheduler)
- hybrid_hybrid_hybrid_hybrid_hybrid_tropical_maxp_m941_s0.py (Hyperdimensional Computing binding + fractional power + Tropical Max‑Plus)

Mathematical Bridge:
The bridge is the *MinHash signature* produced by the regret‑weighted strategy.  
That signature is interpreted as a high‑dimensional binary (or integer) vector and
*bound* to a Hyperdimensional Computing (HDC) representation of a Morphology
object via the HDC binding operator (element‑wise multiplication).  
The bound vector is then evaluated with a *tropical max‑plus* polynomial
`τ(v) = max_i (v_i + i)`.  The resulting tropical score modulates the
propensity and confidence bound of each BanditAction, while the Jaccard‑like
similarity of two MinHash signatures supplies a regret‑weight that scales the
expected reward.  This yields a single unified update rule that simultaneously
leverages regret‑weighted similarity, high‑dimensional binding, and tropical
decision boundaries.
"""

import hashlib
import math
import random
import re
import sys
import pathlib
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared from both parents)
# ----------------------------------------------------------------------


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
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # probability of being chosen
    expected_reward: float
    confidence_bound: float    # UCB‑style bound
    algorithm: str


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Utility functions (parent A)
# ----------------------------------------------------------------------


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        hashlib.blake2b(data, digest_size=8).digest(), "big"
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash‑style signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    equal = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return equal / len(sig_a)


# ----------------------------------------------------------------------
# Utility functions (parent B)
# ----------------------------------------------------------------------


EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)


def random_vector(dim: int = 10000, seed: str | int | None = None) -> List[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def morphology_vector(m: Morphology, dim: int = 10000) -> List[float]:
    """HDC representation of a Morphology object."""
    seed = int.from_bytes(
        hashlib.sha256(
            f"{m.length}{m.width}{m.height}{m.mass}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    vec = np.array(random_vector(dim, seed))
    scaling = np.array([m.length, m.width, m.height, m.mass])
    scaling = np.pad(
        scaling,
        (0, dim // 4 - len(scaling)),
        mode="constant",
    )
    return (vec * scaling[:dim]).tolist()


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Classic MinHash on 5‑character shingles."""
    cleaned = re.sub(r"\s+", " ", text or "").strip().lower()
    if not cleaned:
        return [((1 << 64) - 1)] * k
    shingles = [cleaned[i : i + 5] for i in range(len(cleaned) - 4)]
    sig = []
    for i in range(k):
        min_val = (1 << 64) - 1
        for sh in shingles:
            h = _hash(i, sh)
            if h < min_val:
                min_val = h
        sig.append(min_val)
    return sig


# ----------------------------------------------------------------------
# Hybrid core operations
# ----------------------------------------------------------------------


def hdc_bind(vec_a: List[float], vec_b: List[float]) -> np.ndarray:
    """HDC binding operator – element‑wise multiplication."""
    if len(vec_a) != len(vec_b):
        raise ValueError("vectors must be of equal length for binding")
    return np.multiply(vec_a, vec_b)


def tropical_maxplus_score(bound_vec: np.ndarray) -> float:
    """
    Tropical max‑plus evaluation.
    τ(v) = max_i (v_i + i)
    The index term `i` acts as the tropical “weight”.
    """
    indices = np.arange(bound_vec.size, dtype=np.float64)
    return float(np.max(bound_vec + indices))


def hybrid_bandit_update(
    actions: List[BanditAction],
    tokens: Iterable[str],
    morphology: Morphology,
    text_context: str,
) -> List[BanditAction]:
    """
    Single‑step hybrid update:
    1. Compute a MinHash signature from the token set (regret engine).
    2. Compute a MinHash from the surrounding text (tropical side).
    3. Derive a similarity‑based regret weight.
    4. Bind the numeric signature to the HDC morphology vector.
    5. Evaluate the bound vector with tropical max‑plus → global score.
    6. Modulate each action's propensity and confidence bound using the
       regret weight and the tropical score.
    """
    # 1. Token signature
    token_sig = signature(tokens, k=128)

    # 2. Text MinHash
    text_sig = minhash_for_text(text_context, k=128)

    # 3. Regret weight (similarity between the two signatures)
    regret_weight = similarity(token_sig, text_sig)  # in [0,1]

    # 4. Convert integer signature to a float vector for binding
    sig_float = np.array(token_sig, dtype=np.float64)
    sig_float = sig_float / np.max(sig_float)  # normalise to [0,1]

    # 5. HDC representation of morphology
    hdc_vec = np.array(morphology_vector(morphology, dim=len(sig_float)), dtype=np.float64)

    # 6. Binding
    bound = hdc_bind(sig_float.tolist(), hdc_vec.tolist())

    # 7. Tropical max‑plus score
    trop_score = tropical_maxplus_score(bound)

    # 8. Scale factor derived from tropical score (soft‑scaled to (0,1))
    trop_factor = sigmoid(np.array([trop_score]))[0]

    # 9. Update each action
    updated = []
    for act in actions:
        # Propensity is pulled towards the regret weight and tropical factor
        new_propensity = act.propensity * (0.5 + 0.5 * regret_weight) * (0.5 + 0.5 * trop_factor)
        new_propensity = float(np.clip(new_propensity, 0.0, 1.0))

        # Confidence bound receives an additive boost proportional to tropical factor
        new_conf = act.confidence_bound + 0.1 * trop_factor

        # Expected reward is adjusted by regret weight (higher similarity → higher reward)
        new_reward = act.expected_reward * (1.0 + 0.5 * regret_weight)

        updated.append(
            BanditAction(
                action_id=act.action_id,
                propensity=new_propensity,
                expected_reward=new_reward,
                confidence_bound=new_conf,
                algorithm=act.algorithm,
            )
        )
    return updated


def select_best_action(actions: List[BanditAction]) -> BanditAction:
    """
    Choose the action with the highest Upper Confidence Bound (UCB):
    UCB_i = expected_reward_i + confidence_bound_i
    """
    if not actions:
        raise ValueError("action list cannot be empty")
    scores = [a.expected_reward + a.confidence_bound for a in actions]
    best_idx = int(np.argmax(scores))
    return actions[best_idx]


def hybrid_decision_pipeline(
    raw_tokens: List[str],
    morphology: Morphology,
    text_context: str,
    candidate_actions: List[BanditAction],
) -> BanditAction:
    """
    End‑to‑end pipeline:
    - Update the bandit actions with hybrid dynamics.
    - Return the best action according to the updated UCB scores.
    """
    updated = hybrid_bandit_update(
        candidate_actions, raw_tokens, morphology, text_context
    )
    return select_best_action(updated)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample tokens and text
    tokens = ["alpha", "beta", "gamma", "delta"]
    text = "The quick brown fox jumps over the lazy dog. Evidence suggests similarity."

    # Sample morphology
    morph = Morphology(length=1.2, width=0.8, height=0.5, mass=2.3)

    # Candidate bandit actions
    actions = [
        BanditAction(
            action_id="A1",
            propensity=0.6,
            expected_reward=1.0,
            confidence_bound=0.2,
            algorithm="HybridRegretBandit",
        ),
        BanditAction(
            action_id="A2",
            propensity=0.4,
            expected_reward=0.8,
            confidence_bound=0.3,
            algorithm="HybridRegretBandit",
        ),
        BanditAction(
            action_id="A3",
            propensity=0.5,
            expected_reward=0.9,
            confidence_bound=0.25,
            algorithm="HybridRegretBandit",
        ),
    ]

    best = hybrid_decision_pipeline(tokens, morph, text, actions)
    print(f"Chosen action: {best.action_id}")
    print(f"Propensity: {best.propensity:.4f}")
    print(f"Expected Reward: {best.expected_reward:.4f}")
    print(f"Confidence Bound: {best.confidence_bound:.4f}")