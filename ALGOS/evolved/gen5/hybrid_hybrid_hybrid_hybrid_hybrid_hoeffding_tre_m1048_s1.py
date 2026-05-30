# DARWIN HAMMER — match 1048, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_decisi_m176_s1.py (gen4)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s5.py (gen1)
# born: 2026-05-29T23:32:37Z

import numpy as np
import random
import math
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    """Very simple whitespace tokenizer."""
    return text.lower().split()


def _build_vocab(*texts: str) -> Dict[str, int]:
    """Create a deterministic vocabulary mapping each unique token to an index."""
    vocab: Dict[str, int] = {}
    idx = 0
    for txt in texts:
        for token in _tokenize(txt):
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def _text_vector(text: str, vocab: Dict[str, int]) -> np.ndarray:
    """Return a TF (term‑frequency) vector for *text* over *vocab*."""
    vec = np.zeros(len(vocab), dtype=float)
    for token in _tokenize(text):
        if token in vocab:
            vec[vocab[token]] += 1.0
    if vec.sum() > 0:
        vec /= vec.sum()          # normalise to a probability distribution
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two non‑zero vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm != 0 else 0.0


def ssim_like_similarity(payload: str, prototype: str) -> float:
    """
    A lightweight, deterministic analogue of SSIM for textual data.
    It combines cosine similarity of TF vectors with a penalty
    based on length disparity, mimicking the luminance‑contrast‑structure
    decomposition of the original SSIM.
    """
    vocab = _build_vocab(payload, prototype)
    p_vec = _text_vector(payload, vocab)
    q_vec = _text_vector(prototype, vocab)

    cos_sim = cosine_similarity(p_vec, q_vec)

    # length penalty: values in [0,1] where 1 means equal length
    len_penalty = 1.0 - abs(len(payload) - len(prototype)) / max(len(payload), len(prototype), 1)

    # blend the two components – weights sum to 1
    return 0.6 * cos_sim + 0.4 * len_penalty


def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """
    Hoeffding bound ε = sqrt( (R² * ln(1/δ)) / (2n) )
    where R is the range of the random variable (max‑gain – min‑gain).
    """
    if range_ <= 0:
        raise ValueError("range_ must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def gini_impurity(class_counts: Counter) -> float:
    """Standard Gini impurity for a multinomial class distribution."""
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(class_counts.values()), dtype=float) / total
    return 1.0 - np.sum(probs ** 2)


def max_gini_gain(current_gini: float) -> float:
    """
    The theoretical maximum reduction in Gini impurity from a split is the
    current impurity itself (a perfectly pure child node yields Gini = 0).
    """
    return current_gini


def bayesian_split_posterior(prior: float, gain_gap: float, epsilon: float) -> float:
    """
    Simple Bayesian update where the likelihood of a “real” split is modeled
    as a sigmoid of the normalised gain gap.
    """
    # Normalise gain_gap to [0,1] using epsilon as scale
    norm_gain = 1.0 / (1.0 + math.exp(-5 * (gain_gap - epsilon) / max(epsilon, 1e-9)))
    # Posterior ∝ prior * likelihood
    posterior_unnorm = prior * norm_gain
    # Assume complementary likelihood for “no split” is (1‑norm_gain)
    posterior = posterior_unnorm / (posterior_unnorm + (1 - prior) * (1 - norm_gain) + 1e-12)
    return posterior


@dataclass(frozen=True)
class SplitDecision:
    """Result of the hybrid Hoeffding‑Gini‑Bayesian split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    posterior: float
    reason: str


def hybrid_decision(
    payload: str,
    prototype: str,
    class_counts: Counter,
    delta: float,
    n: int,
    prior_split_prob: float = 0.5,
) -> SplitDecision:
    """
    Compute a split decision that fuses:
      * a deterministic SSIM‑like similarity (textual domain),
      * Gini impurity reduction,
      * Hoeffding statistical guarantee,
      * Bayesian posterior over the split hypothesis.
    """
    # 1️⃣ similarity
    similarity = ssim_like_similarity(payload, prototype)

    # 2️⃣ impurity & potential gain
    current_gini = gini_impurity(class_counts)
    max_gain = max_gini_gain(current_gini)

    # 3️⃣ Hoeffding bound on the gain estimate
    epsilon = hoeffding_bound(max_gain, delta, n)

    # 4️⃣ Heuristic gain gap: similarity scales the attainable Gini gain
    gain_gap = similarity * max_gain

    # 5️⃣ Bayesian posterior
    posterior = bayesian_split_posterior(prior_split_prob, gain_gap, epsilon)

    # 6️⃣ Final deterministic decision (threshold = 0.5 on posterior)
    should_split = posterior > 0.5

    reason = (
        f"similarity={similarity:.4f}, "
        f"current_gini={current_gini:.4f}, max_gain={max_gain:.4f}, "
        f"ε={epsilon:.6f}, gain_gap={gain_gap:.6f}, "
        f"posterior={posterior:.4f}"
    )
    return SplitDecision(should_split, epsilon, gain_gap, posterior, reason)


if __name__ == "__main__":
    # Example usage
    payload = "The quick brown fox jumps over the lazy dog."
    prototype = "A fast dark-colored fox leaps above a sleepy canine."
    class_counts = Counter({"spam": 30, "ham": 70})
    delta = 0.01
    n = 200

    decision = hybrid_decision(payload, prototype, class_counts, delta, n)
    print(decision)