# DARWIN HAMMER — match 21, survivor 7
# gen: 4
# parent_a: korpus_text.py (gen0)
# parent_b: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s5.py (gen3)
# born: 2026-05-29T23:26:34Z

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants and utilities (derived from Parent A)
# ----------------------------------------------------------------------
INT16_MAX = 2 ** 15 - 1


def _shingles(text: str, width: int = 5) -> List[str]:
    """Return a list of overlapping substrings (shingles) of length *width*."""
    cleaned = " ".join(text.split()).lower()
    return [cleaned[i : i + width] for i in range(len(cleaned) - width + 1)]


def _hash_token(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of *token* mixed with *seed*."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 64) -> List[int]:
    """Compute a MinHash signature of length *k* for the given *tokens*."""
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not token_set:
        return [0] * k
    # generate k independent seeds
    seeds = [random.randrange(0, 2 ** 32) for _ in range(k)]
    signature = []
    for seed in seeds:
        min_hash = min(_hash_token(seed, t) for t in token_set)
        signature.append(min_hash)
    return signature


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Parent‑A style helper: MinHash signature of a raw text string."""
    return minhash_signature(_shingles(text or ""), width=5, k=k)  # type: ignore[arg-type]


def shannon_entropy(chars: List[str]) -> float:
    """Simple Shannon entropy over a list of characters."""
    if not chars:
        return 0.0
    prob = {}
    for c in chars:
        prob[c] = prob.get(c, 0) + 1
    total = len(chars)
    return -sum((count / total) * math.log2(count / total) for count in prob.values())


def entropy_for_text(text: str) -> float:
    """Parent‑A helper: entropy of the first 10 000 characters."""
    return float(shannon_entropy(list((text or "")[:10000]))) if text else 0.0


def hash_quantized_embedding(text: str, dim: int = 128) -> List[int]:
    """Deterministic pseudo‑embedding: hash the text repeatedly to fill *dim* int16 slots."""
    rng = random.Random(hash(text))
    return [rng.randint(-INT16_MAX, INT16_MAX) for _ in range(dim)]


def vector_literal(text: str, dim: int = 128) -> np.ndarray:
    """Parent‑A helper: convert quantized embedding to a float32 numpy vector in [-1,1]."""
    ints = hash_quantized_embedding(text or "", dim=dim)
    return np.array([float(v) / float(INT16_MAX) for v in ints], dtype=np.float32)


# ----------------------------------------------------------------------
# Core data structures (shared by both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """An action with a textual description and scalar attributes."""
    id: str
    description: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass
class StoreState:
    """Honeybee‑style store providing the bounded *dance* signal."""
    dance: float = 1.0  # default neutral scaling

    def update(self, reward: float) -> None:
        """Simple exponential moving average to keep *dance* in [0.5, 1.5]."""
        alpha = 0.1
        self.dance = max(0.5, min(1.5, (1 - alpha) * self.dance + alpha * reward))


# ----------------------------------------------------------------------
# Hybrid mathematics (the bridge)
# ----------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    """Regret‑weighting sigmoid g(x)."""
    return 1.0 / (1.0 + math.exp(-x))


def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b) or not sig_a:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def linucb_confidence(
    A_inv: np.ndarray, x: np.ndarray, alpha: float = 1.0
) -> float:
    """LinUCB confidence bound √(xᵀ A⁻¹ x) scaled by *alpha*."""
    return alpha * math.sqrt(float(x.T @ A_inv @ x))


def hybrid_score(
    action: MathAction,
    counterfactuals: List[MathCounterfactual],
    ref_signature: List[int],
    store: StoreState,
    A_inv: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """
    Compute the hybrid score S_i for *action*.

    Steps:
    1. Raw regret value R_i = EV - cost - risk + Σ counterfactual contribution.
    2. Regret weighting g(R_i) = sigmoid(R_i).
    3. MinHash signature of the action description.
    4. Jaccard similarity with *ref_signature*.
    5. LinUCB confidence bound inflated by the similarity term.
    6. Final score: g(R_i) · (1 + sim) · store.dance · (1 + conf).
    """
    # 1. Raw regret value
    cf_sum = sum(
        cf.outcome_value * cf.probability
        for cf in counterfactuals
        if cf.action_id == action.id
    )
    R_i = action.expected_value - action.cost - action.risk + cf_sum

    # 2. Regret weighting
    g_R = _sigmoid(R_i)

    # 3. Signature of the action description
    act_sig = minhash_signature(_shingles(action.description), k=len(ref_signature))

    # 4. Similarity term
    sim = jaccard_similarity(act_sig, ref_signature)

    # 5. Confidence bound
    x_vec = vector_literal(action.description)  # feature vector
    conf = linucb_confidence(A_inv, x_vec, alpha=alpha)

    # 6. Combine
    score = g_R * (1.0 + sim) * store.dance * (1.0 + conf)
    return score


def softmax(scores: List[float]) -> List[float]:
    """Numerically stable softmax."""
    max_s = max(scores) if scores else 0.0
    exps = [math.exp(s - max_s) for s in scores]
    sum_exps = sum(exps) or 1.0
    return [e / sum_exps for e in exps]


def select_action(actions: List[MathAction],
                  scores: List[float]) -> MathAction:
    """Sample an action proportionally to softmax(scores)."""
    probs = softmax(scores)
    chosen_id = random.choices([a.id for a in actions], weights=probs, k=1)[0]
    return next(a for a in actions if a.id == chosen_id)


def compute_reference_signature(past_actions: List[MathAction]) -> List[int]:
    """
    Build a reference MinHash signature from the concatenation of 
    past action descriptions.
    """
    text = " ".join([a.description for a in past_actions])
    return minhash_for_text(text)


def update_store(store: StoreState, reward: float) -> StoreState:
    store.update(reward)
    return store


def weighted_regret_value(action: MathAction, 
                           counterfactuals: List[MathCounterfactual]) -> float:
    cf_sum = sum(cf.outcome_value * cf.probability 
                 for cf in counterfactuals 
                 if cf.action_id == action.id)
    return action.expected_value - action.cost - action.risk + cf_sum


def action_similarity(action: MathAction, 
                       ref_action: MathAction) -> float:
    act_sig = minhash_signature(_shingles(action.description), 
                                k=len(minhash_signature(_shingles(ref_action.description))))
    ref_sig = minhash_signature(_shingles(ref_action.description))
    return jaccard_similarity(act_sig, ref_sig)


def multi_step_hybrid_score(actions: List[MathAction], 
                           counterfactuals_list: List[List[MathCounterfactual]], 
                           store: StoreState, 
                           A_inv: np.ndarray, 
                           ref_action: MathAction, 
                           alpha: float = 1.0) -> List[float]:
    scores = []
    ref_signature = minhash_for_text(ref_action.description)
    for action, counterfactuals in zip(actions, counterfactuals_list):
        score = hybrid_score(action, counterfactuals, ref_signature, store, A_inv, alpha)
        scores.append(score)
    return scores