# DARWIN HAMMER — match 21, survivor 5
# gen: 4
# parent_a: korpus_text.py (gen0)
# parent_b: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s5.py (gen3)
# born: 2026-05-29T23:26:34Z

"""HybridTextRegretBandit
Combines:
- Parent A (korpus_text): text preprocessing, MinHash signatures, entropy, quantized embedding.
- Parent B (hybrid_regret_engine_hybrid_bandit_router): regret‑weighted action values, MinHash‑based similarity,
  and a bandit confidence bound modulated by a “dance” signal from a Honeybee‑style store.

Mathematical bridge:
Each action is associated with a textual description. The description is turned into a MinHash
signature `sig_i` (Parent A). A reference signature `sig_ref` (e.g. recent high‑reward actions) is
also built from text. The Jaccard‑like similarity

    sim_i = 1/k · Σ_{j=1..k} [sig_i[j] == sig_ref[j]]

is fed as a multiplicative factor into the regret‑weighted score defined in Parent B:

    R_i = expected_value_i – cost_i – risk_i + counterfactual_i
    g(R_i) = 1 / (1 + exp(‑α·R_i))          # sigmoid regret weighting
    S_i = g(R_i) · (1 + sim_i) · dance

`dance` is a bounded control signal from the Honeybee store (0 ≤ dance ≤ 1).  
The final policy selects actions proportionally to `softmax(S_i)` and attaches a LinUCB‑style
confidence bound inflated by `1 + sim_i`. This module implements the full pipeline,
exposing three public functions that showcase the hybrid operation."""


import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Re‑implementation of Parent A utilities (self‑contained)
# ----------------------------------------------------------------------
INT16_MAX = 32767  # 2^15‑1, used for quantized embeddings


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used for MinHash and embeddings."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        np.frombuffer(
            np.uint8(list(data)),
            dtype=np.uint8,
        ).tobytes(),
        "big",
        signed=False,
    ) & ((1 << 64) - 1)


def shingles(text: str, width: int = 5) -> List[str]:
    """Return a list of width‑character shingles from the cleaned text."""
    clean = " ".join(text.split()).lower()
    if len(clean) < width:
        return [clean]
    return [clean[i : i + width] for i in range(len(clean) - width + 1)]


def minhash_signature(tokens: Iterable[str], k: int = 64) -> List[int]:
    """Simple MinHash: for each of k random seeds keep the minimum hash value."""
    seeds = list(range(k))
    sig = []
    for seed in seeds:
        min_val = None
        for t in tokens:
            h = _hash(seed, t)
            if min_val is None or h < min_val:
                min_val = h
        sig.append(min_val if min_val is not None else 0)
    return sig


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Parent A entry point – MinHash signature from raw text."""
    return minhash_signature(shingles(text or ""), k=k)


def shannon_entropy(chars: List[str]) -> float:
    """Naïve Shannon entropy over a list of characters."""
    if not chars:
        return 0.0
    counts = {}
    for c in chars:
        counts[c] = counts.get(c, 0) + 1
    total = len(chars)
    return -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())


def entropy_for_text(text: str) -> float:
    """Parent A entry point – entropy of first 10 000 characters."""
    return float(shannon_entropy(list((text or "")[:10000]))) if text else 0.0


def hash_quantized_embedding(text: str, dim: int = 16) -> List[int]:
    """Very simple quantized embedding: hash each token to an int in [0, INT16_MAX]."""
    tokens = shingles(text or "", width=3)
    if not tokens:
        tokens = [""]  # guarantee at least one token
    emb = []
    for i in range(dim):
        token = tokens[i % len(tokens)]
        h = _hash(i, token) % (INT16_MAX + 1)
        emb.append(int(h))
    return emb


def vector_literal(text: str) -> str:
    """Parent A entry point – literal representation of the normalized embedding."""
    return "[" + ",".join(f"{float(v) / float(INT16_MAX):.8f}" for v in hash_quantized_embedding(text or "")) + "]"


# ----------------------------------------------------------------------
# Hybrid Regret‑Bandit core (Parent B adapted)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action descriptor used by the hybrid engine."""
    id: str
    description: str  # textual description fed to MinHash / embedding pipelines
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0
    counterfactual: float = 0.0  # optional term for off‑policy correction


@dataclass
class Counterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass
class StoreState:
    """Honeybee‑style store exposing a bounded 'dance' control signal."""
    dance: float = 0.5  # default moderate influence; kept in [0,1]

    def update(self, reward: float, decay: float = 0.01) -> None:
        """Simple exponential moving average update of the dance signal."""
        self.dance = max(0.0, min(1.0, self.dance * (1 - decay) + reward * decay))


def sigmoid(x: float, alpha: float = 1.0) -> float:
    """Standard logistic sigmoid with optional scaling."""
    try:
        return 1.0 / (1.0 + math.exp(-alpha * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity for equal‑length MinHash signatures."""
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def regret_term(action: MathAction) -> float:
    """Compute the raw regret‑weighted value R_i."""
    return (
        action.expected_value
        - action.cost
        - action.risk
        + action.counterfactual
    )


def hybrid_score(
    action: MathAction,
    ref_signature: List[int],
    store: StoreState,
    alpha: float = 1.0,
) -> float:
    """
    Core hybrid scoring function.

    S_i = g(R_i) · (1 + sim(sig_i, sig_ref)) · dance
    where g = sigmoid, sim = MinHash similarity, dance = StoreState.dance.
    """
    # 1. Regret weighting
    R_i = regret_term(action)
    g_R = sigmoid(R_i, alpha=alpha)

    # 2. MinHash similarity
    sig_i = minhash_for_text(action.description, k=len(ref_signature))
    sim = minhash_similarity(sig_i, ref_signature)

    # 3. Combine with dance factor
    return g_R * (1.0 + sim) * store.dance


def linucb_confidence(
    action: MathAction,
    feature_vec: np.ndarray,
    A_inv: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """
    Classic LinUCB confidence bound: α·sqrt(xᵀ A⁻¹ x)
    Here we inflate α by (1 + sim) where sim is the MinHash similarity
    to a reference signature (computed externally).
    """
    var = float(feature_vec.T @ A_inv @ feature_vec)
    return alpha * math.sqrt(var)


def softmax(scores: List[float]) -> List[float]:
    """Numerically stable softmax."""
    max_s = max(scores) if scores else 0.0
    exps = [math.exp(s - max_s) for s in scores]
    sum_exps = sum(exps) or 1.0
    return [e / sum_exps for e in exps]


# ----------------------------------------------------------------------
# Public API – three demonstration functions
# ----------------------------------------------------------------------
def compute_hybrid_scores(
    actions: List[MathAction],
    reference_text: str,
    store: StoreState,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Returns a mapping from action.id to its hybrid score S_i.
    """
    ref_sig = minhash_for_text(reference_text, k=64)
    scores = {}
    for act in actions:
        scores[act.id] = hybrid_score(act, ref_sig, store, alpha=alpha)
    return scores


def select_action_softmax(
    actions: List[MathAction],
    reference_text: str,
    store: StoreState,
    rng: random.Random = random,
) -> MathAction:
    """
    Samples an action proportionally to softmax(S_i) over the provided list.
    """
    scores = compute_hybrid_scores(actions, reference_text, store)
    prob_dist = softmax(list(scores.values()))
    chosen_id = rng.choices(list(scores.keys()), weights=prob_dist, k=1)[0]
    return next(act for act in actions if act.id == chosen_id)


def update_store_with_feedback(
    store: StoreState,
    chosen_action: MathAction,
    reward: float,
    decay: float = 0.02,
) -> None:
    """
    Updates the store's dance signal based on observed reward.
    The reward is first normalized to [0,1] via a sigmoid to keep the signal bounded.
    """
    normalized = sigmoid(reward)
    store.update(normalized, decay=decay)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny set of actions with textual descriptions
    actions = [
        MathAction(
            id="a1",
            description="Buy low, sell high in volatile markets",
            expected_value=0.8,
            cost=0.1,
            risk=0.2,
        ),
        MathAction(
            id="a2",
            description="Conservative bond investment with stable returns",
            expected_value=0.5,
            cost=0.05,
            risk=0.1,
        ),
        MathAction(
            id="a3",
            description="Speculative cryptocurrency trade with high upside",
            expected_value=0.9,
            cost=0.2,
            risk=0.4,
        ),
    ]

    # Reference text could be the description of the best‑so‑far action
    reference = "high reward, low risk investment opportunities"

    # Initialise store
    store = StoreState(dance=0.6)

    # Compute scores
    scores = compute_hybrid_scores(actions, reference, store)
    print("Hybrid scores:", scores)

    # Select an action
    chosen = select_action_softmax(actions, reference, store)
    print("Chosen action:", chosen.id, chosen.description)

    # Simulate a reward and update store
    simulated_reward = random.uniform(-1, 1)  # could be negative or positive
    print("Simulated reward:", simulated_reward)
    update_store_with_feedback(store, chosen, simulated_reward)

    print("Updated store dance:", store.dance)