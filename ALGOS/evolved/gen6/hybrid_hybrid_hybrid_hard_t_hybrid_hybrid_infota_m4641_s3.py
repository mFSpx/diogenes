# DARWIN HAMMER — match 4641, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hard_truth_ma_kan_m27_s1.py (gen2)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m1066_s0.py (gen5)
# born: 2026-05-29T23:57:09Z

"""Hybrid Stylometry‑KAN‑Entropic‑MinHash Bandit Scheduler
====================================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_hard_truth_ma_kan_m27_s1.py``  
  Provides stylometric feature extraction (function‑word categories, punctuation)
  and a KAN‑style linear transformation of the resulting feature vector.

* **Parent B** – ``hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m1066_s0.py``  
  Supplies an *entropic MinHash* construction for probability distributions,
  an entropy estimator, and a simple contextual bandit with a knapsack constraint.

Mathematical Bridge
-------------------
The stylometric feature vector **v** is a discrete probability distribution
(over function‑word categories and punctuation frequencies).  
We (i) map **v** through a KAN‑like linear layer **W·v + b** (Parent A) and
(ii) treat the transformed distribution as a *bandit action* whose similarity
to other actions is measured by the *entropic MinHash* signature (Parent B).

During action selection the knapsack constraint enforces a resource budget,
while a diversity term – the average MinHash similarity between chosen actions –
penalises redundant selections.  Thus the hybrid system fuses the matrix
operations of the KAN with the entropy‑based MinHash similarity of the bandit
scheduler."""

import math
import random
import sys
import pathlib
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry / KAN utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Tokenise a string into alphabetic lower‑case words."""
    return [word for word in (text or "").lower().split() if word.isalpha()]


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a *stylometric* frequency vector.
    The vector contains normalized counts for each function‑word category
    and for punctuation characters.
    """
    toks = words(text)
    total_words = len(toks) or 1

    # category frequencies
    cat_counts = {}
    for cat, vocab in FUNCTION_CATS.items():
        cat_counts[cat] = sum(1 for w in toks if w in vocab) / total_words

    # punctuation frequency (as a proportion of total characters)
    punct_count = sum(1 for ch in text if ch in PUNCT)
    total_chars = len(text) or 1
    punct_freq = punct_count / total_chars

    cat_counts["punctuation"] = punct_freq
    return cat_counts


def stylometry_vector(text: str) -> np.ndarray:
    """
    Convert the LSM dictionary into a dense numpy vector and
    ensure it sums to one (probability distribution).
    Ordering: sorted keys of FUNCTION_CATS plus ``punctuation``.
    """
    vec_dict = lsm_vector(text)
    keys = sorted(vec_dict.keys())
    vec = np.array([vec_dict[k] for k in keys], dtype=float)
    # Normalise to a proper distribution (avoid division by zero)
    total = vec.sum()
    if total == 0:
        return np.full_like(vec, 1.0 / len(vec))
    return vec / total


def kan_transform(vec: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    A lightweight KAN‑style transformation:
        y = σ(W·x + b)
    where σ is a piecewise‑linear activation (here ReLU) applied element‑wise.
    """
    linear = weight @ vec + bias
    return np.maximum(0.0, linear)  # ReLU


# ----------------------------------------------------------------------
# Parent B – entropic MinHash & bandit/knapsack utilities
# ----------------------------------------------------------------------
def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps))
                for p in probabilities if p > 0)


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')


def signature(tokens: List[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def entropic_minhash(probabilities: List[float], k: int = 128) -> List[int]:
    """
    Build a MinHash signature from a probability distribution.
    The distribution is first stringified token‑wise.
    """
    tokens = [str(p) for p in probabilities]
    return signature(tokens, k)


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def compute_action_signature(action_vec: np.ndarray, k: int = 128) -> List[int]:
    """
    Convert a (potentially transformed) stylometric distribution into an
    entropic MinHash signature.
    """
    prob_list = action_vec.tolist()
    return entropic_minhash(prob_list, k)


def diversity_penalty(selected_sigs: List[List[int]]) -> float:
    """
    Average pairwise MinHash similarity among the selected signatures.
    Larger values indicate less diversity.
    """
    if len(selected_sigs) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(selected_sigs)):
        for j in range(i + 1, len(selected_sigs)):
            total += similarity(selected_sigs[i], selected_sigs[j])
            count += 1
    return total / count  # mean similarity


def knapsack_bandit_select(
    actions: List[np.ndarray],
    rewards: List[float],
    costs: List[float],
    capacity: float,
    lambda_div: float = 0.5,
    k: int = 128,
) -> Tuple[List[int], float]:
    """
    Approximate 0/1 knapsack with a contextual bandit flavour.
    - ``actions``: stylometric vectors (already KAN‑transformed).
    - ``rewards``: estimated payoff of each action.
    - ``costs``: resource consumption of each action.
    - ``capacity``: total budget.
    - ``lambda_div``: weight of the diversity penalty (0 = ignore diversity).

    The algorithm:
    1. Compute MinHash signatures for all actions.
    2. Greedily pick items by descending (adjusted_reward / cost) ratio,
       where ``adjusted_reward = reward - λ * avg_similarity_to_already_picked``.
    3. Stop when the next item would exceed the budget.
    Returns the indices of the selected actions and the achieved total reward.
    """
    n = len(actions)
    if not (len(rewards) == len(costs) == n):
        raise ValueError('actions, rewards and costs must have equal length')

    sigs = [compute_action_signature(a, k) for a in actions]

    selected = []
    total_reward = 0.0
    used = 0.0

    # Pre‑compute ratios for initial ordering
    order = sorted(range(n), key=lambda i: rewards[i] / max(costs[i], 1e-12), reverse=True)

    for idx in order:
        if used + costs[idx] > capacity:
            continue

        # compute similarity to already selected actions
        if selected:
            cur_sims = [similarity(sigs[idx], sigs[j]) for j in selected]
            avg_sim = sum(cur_sims) / len(cur_sims)
        else:
            avg_sim = 0.0

        adj_reward = rewards[idx] - lambda_div * avg_sim
        # Re‑evaluate the ratio with the adjusted reward
        ratio = adj_reward / max(costs[idx], 1e-12)

        # Accept if the adjusted ratio is still positive
        if ratio <= 0:
            continue

        selected.append(idx)
        used += costs[idx]
        total_reward += rewards[idx]

        if used >= capacity:
            break

    return selected, total_reward


# ----------------------------------------------------------------------
# Demonstration / Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample texts representing different "authors"
    texts = [
        "I think therefore I am. The quick brown fox jumps over the lazy dog!",
        "She sells sea shells on the sea shore. Can you can a can as a canned can?",
        "To be, or not to be, that is the question.",
        "All that glitters is not gold; but all gold glitters.",
    ]

    # 1. Extract stylometric vectors
    raw_vecs = [stylometry_vector(t) for t in texts]

    # 2. Initialise a random KAN weight matrix and bias (compatible dimensions)
    dim_in = raw_vecs[0].shape[0]
    dim_out = dim_in  # keep same size for simplicity
    rng = np.random.default_rng(42)
    W = rng.normal(scale=0.1, size=(dim_out, dim_in))
    b = rng.normal(scale=0.01, size=dim_out)

    # 3. Apply KAN transformation
    transformed = [kan_transform(v, W, b) for v in raw_vecs]

    # 4. Mock rewards and costs for a bandit scenario
    rewards = [rng.uniform(0.5, 1.5) for _ in transformed]
    costs = [rng.uniform(1.0, 3.0) for _ in transformed]
    capacity = 6.0

    # 5. Run hybrid selector
    chosen, total = knapsack_bandit_select(
        actions=transformed,
        rewards=rewards,
        costs=costs,
        capacity=capacity,
        lambda_div=0.3,
        k=64,
    )

    print("Chosen action indices:", chosen)
    print("Total reward achieved:", total)
    print("Remaining capacity:", capacity - sum(costs[i] for i in chosen))
    # Verify that each chosen action's signature can be recomputed without error
    for i in chosen:
        sig = compute_action_signature(transformed[i], k=64)
        assert len(sig) == 64
    print("Smoke test passed.")