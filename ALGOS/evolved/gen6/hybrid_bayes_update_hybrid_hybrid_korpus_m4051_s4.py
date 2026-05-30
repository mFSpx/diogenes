# DARWIN HAMMER — match 4051, survivor 4
# gen: 6
# parent_a: bayes_update.py (gen0)
# parent_b: hybrid_hybrid_korpus_text_h_hybrid_hybrid_hybrid_m2363_s1.py (gen5)
# born: 2026-05-29T23:53:17Z

"""Hybrid Bayesian‑Bandit Text Fusion
===================================

Parent algorithms:
* **bayes_update.py** – provides Bayesian marginalisation and posterior update.
* **hybrid_hybrid_korpus_text_h_hybrid_hybrid_hybrid_m2363_s1.py** – generates text‑derived vectors
  using min‑hash signatures, entropy, and a path‑signature based context vector that feeds a
  contextual bandit.

Mathematical bridge
-------------------
Both parents ultimately manipulate *probabilities* that can be interpreted as
*likelihoods* of a hypothesis given evidence.  The text‑processing side yields a
numeric “expected reward” `r̂(a)` for a chosen bandit action `a`.  This reward can be
treated as a **likelihood** in the Bayesian formula:


posterior = prior * likelihood / marginal
marginal = likelihood * prior + false_positive * (1‑prior)


Thus the hybrid algorithm:
1. Builds a context vector `c` from the text (min‑hash → entropy → path signature).
2. Uses `c` in a simple contextual bandit to obtain an expected reward `r̂`.
3. Feeds `r̂` as the likelihood into the Bayesian update, optionally using an
   entropy‑derived false‑positive rate.

The code below implements this fused pipeline in a compact, testable form.
"""

import math
import random
import re
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Bayesian primitives
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)·P(H) + P(E|¬H)·P(¬H)"""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)·P(H) / P(E)"""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Parent B – Text vector generation & contextual bandit (simplified)
# ----------------------------------------------------------------------
def minhash_signature(text: str, k: int = 64) -> List[int]:
    """
    Generate a min‑hash signature of length `k` for `text`.
    The shingle size is fixed at 5 characters.
    """
    cleaned = re.sub(r"\s+", " ", text or "").strip().lower()
    shingles = [cleaned[i : i + 5] for i in range(len(cleaned) - 4)]
    # Simple deterministic hash for reproducibility
    signature = [min(hash(sh) for sh in shingles) % (2 ** k) for _ in range(k)]
    return signature


def entropy_from_signature(sig: List[int]) -> float:
    """
    Compute Shannon entropy of the min‑hash signature interpreted as a
    discrete distribution over its integer values.
    """
    if not sig:
        return 0.0
    counts = {}
    for v in sig:
        counts[v] = counts.get(v, 0) + 1
    total = len(sig)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy


def path_signature(text: str) -> np.ndarray:
    """
    Very lightweight “path signature”: cumulative sum of Unicode code points
    of the characters, reshaped as a 1‑D float array.
    """
    codes = np.fromiter((ord(ch) for ch in text), dtype=float, count=len(text))
    return np.cumsum(codes)


def flatten_and_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Flatten (already 1‑D) and L2‑normalize the vector.
    """
    if vec.size == 0:
        return vec
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


def build_context_vector(text: str) -> np.ndarray:
    """
    Combine the three primitives:
    * min‑hash → entropy (scalar)
    * path signature → vector
    The scalar entropy is appended to the normalized path signature,
    yielding the final context vector.
    """
    mh = minhash_signature(text, k=32)                     # shorter signature for speed
    ent = entropy_from_signature(mh)
    path_vec = flatten_and_normalize(path_signature(text))
    # Append entropy as an extra dimension
    return np.append(path_vec, ent)


def epsilon_greedy_select(
    context: np.ndarray,
    actions: List[str],
    reward_estimator: Callable[[np.ndarray, str], float],
    epsilon: float = 0.1,
) -> Tuple[str, float]:
    """
    Contextual ε‑greedy bandit.
    Returns the selected action and its estimated reward.
    """
    if not actions:
        raise ValueError("actions list cannot be empty")
    if random.random() < epsilon:
        chosen = random.choice(actions)
        reward = reward_estimator(context, chosen)
        return chosen, reward

    # Exploit: pick action with maximal estimated reward
    rewards = [(a, reward_estimator(context, a)) for a in actions]
    chosen, reward = max(rewards, key=lambda x: x[1])
    return chosen, reward


def simple_reward_estimator(context: np.ndarray, action: str) -> float:
    """
    Dummy reward estimator: dot product between context and a deterministic
    hash‑derived vector for the action, scaled to [0,1].
    """
    random.seed(hash(action))               # deterministic per action
    weight = np.array([random.random() for _ in range(context.size)])
    raw = np.dot(context, weight)
    # Sigmoid to squash into (0,1)
    return 1.0 / (1.0 + math.exp(-raw))


# ----------------------------------------------------------------------
# Hybrid operation – Bayesian fusion of bandit reward
# ----------------------------------------------------------------------
def hybrid_posterior(
    prior: float,
    text: str,
    actions: List[str],
    false_positive_rate: float = 0.01,
    epsilon: float = 0.1,
) -> Tuple[float, str]:
    """
    End‑to‑end hybrid algorithm.

    1. Build a context vector from `text`.
    2. Run a contextual ε‑greedy bandit to obtain an action and its expected reward.
    3. Treat the reward as the likelihood `P(E|H)`.
    4. Compute the marginal using the entropy‑derived false‑positive term.
    5. Return the Bayesian posterior probability of the hypothesis and the chosen action.
    """
    if not (0.0 <= prior <= 1.0):
        raise ValueError("prior must be in [0,1]")

    context = build_context_vector(text)

    # Bandit step – gives us a likelihood‑like quantity
    action, expected_reward = epsilon_greedy_select(
        context, actions, simple_reward_estimator, epsilon=epsilon
    )

    # Clamp reward to a valid probability range
    likelihood = max(0.0, min(1.0, expected_reward))

    # Use entropy to modulate the false‑positive rate (optional refinement)
    entropy = entropy_from_signature(minhash_signature(text, k=32))
    fp = false_positive_rate * (1.0 + entropy / math.log2(2 ** 32))

    marginal = bayes_marginal(prior, likelihood, fp)
    posterior = bayes_update(prior, likelihood, marginal)

    return posterior, action


# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def demo_context_vector():
    """Print a sample context vector for a hard‑coded sentence."""
    txt = "The quick brown fox jumps over the lazy dog."
    vec = build_context_vector(txt)
    print("Context vector (len={}):".format(vec.size))
    print(vec)


def demo_bandit():
    """Run the bandit on a sample text and display the chosen action and reward."""
    txt = "Sample text for bandit testing."
    actions = ["label_A", "label_B", "label_C"]
    context = build_context_vector(txt)
    act, rew = epsilon_greedy_select(context, actions, simple_reward_estimator, epsilon=0.0)
    print(f"Chosen action: {act}, estimated reward: {rew:.4f}")


def demo_hybrid():
    """Full hybrid pipeline demonstration."""
    txt = "Hybrid algorithm merges Bayesian reasoning with contextual bandits."
    prior = 0.6
    actions = ["spam", "ham", "promo"]
    post, act = hybrid_posterior(prior, txt, actions)
    print(f"Prior: {prior:.2f}, Posterior after observing '{act}': {post:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Context Vector ===")
    demo_context_vector()
    print("\n=== Demo: Bandit Selection ===")
    demo_bandit()
    print("\n=== Demo: Hybrid Bayesian‑Bandit Fusion ===")
    demo_hybrid()