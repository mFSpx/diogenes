# DARWIN HAMMER — match 717, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_label_foundry_m21_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_decisi_m260_s1.py (gen3)
# born: 2026-05-29T23:30:43Z

import math
import random
import sys
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Iterable
import numpy as np
import re

# ----------------------------------------------------------------------
# Sketch primitives
# ----------------------------------------------------------------------

class CountMinSketch:
    """Count‑Min sketch for non‑negative integer streams.

    Provides an unbiased estimate of the total count for any key.
    """

    def __init__(self, width: int = 2000, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = random.Random(seed)
        self.seeds = [rng.randint(1, 2 ** 31 - 1) for _ in range(depth)]

    def _hash(self, key: str, i: int) -> int:
        h = hashlib.blake2b(digest_size=8, key=self.seeds[i].to_bytes(4, "little"))
        h.update(key.encode("utf-8"))
        return int.from_bytes(h.digest(), "big") % self.width

    def update(self, key: str, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.tables[i, idx] += increment

    def estimate(self, key: str) -> int:
        """Return the minimum count over all hash rows (the CMS estimate)."""
        mins = [self.tables[i, self._hash(key, i)] for i in range(self.depth)]
        return min(mins)

    def total(self) -> int:
        """Total count of all updates (sum over first row, approximates N)."""
        return int(self.tables.sum())


class HyperLogLog:
    """Very small HyperLogLog implementation for cardinality estimation."""

    def __init__(self, b: int = 10):
        self.b = b  # number of bits for register index
        self.m = 1 << b
        self.registers = np.zeros(self.m, dtype=np.uint8)

    def _rho(self, w: int) -> int:
        """Position of first 1-bit in w (1‑based)."""
        return (w & -w).bit_length()

    def add(self, item: str) -> None:
        h = int(hashlib.sha1(item.encode("utf-8")).hexdigest(), 16)
        idx = h >> (64 - self.b)
        w = (h << self.b) & ((1 << 64) - 1)
        rank = self._rho(w)
        self.registers[idx] = max(self.registers[idx], rank)

    def cardinality(self) -> float:
        """Raw HyperLogLog estimate using the standard harmonic mean."""
        Z = 1.0 / np.sum(2.0 ** -self.registers)
        E = self._alpha_mm() * Z
        # Small range correction
        if E <= (5.0 / 2.0) * self.m:
            V = np.count_nonzero(self.registers == 0)
            if V != 0:
                E = self.m * math.log(self.m / V)
        return E

    def _alpha_mm(self) -> float:
        if self.m == 16:
            return 0.673 * self.m * self.m
        if self.m == 32:
            return 0.697 * self.m * self.m
        if self.m == 64:
            return 0.709 * self.m * self.m
        return (0.7213 / (1 + 1.079 / self.m)) * self.m * self.m


# ----------------------------------------------------------------------
# RLCT regression (log‑loss vs log‑sample‑size)
# ----------------------------------------------------------------------

def estimate_rlct(losses: List[float], samples: List[int]) -> float:
    """Fit log L = λ·log n + c by ordinary least squares and return λ̂."""
    if len(losses) < 2:
        return 0.0
    log_n = np.log(np.array(samples, dtype=float))
    log_L = np.log(np.array(losses, dtype=float) + 1e-12)  # avoid log(0)
    A = np.vstack([log_n, np.ones_like(log_n)]).T
    λ, _ = np.linalg.lstsq(A, log_L, rcond=None)[0]
    return float(λ)


# ----------------------------------------------------------------------
# Parent B – ternary lens & regex feature extraction
# ----------------------------------------------------------------------

# Mapping from classification string to one‑hot ternary vector.
_LENS_MAP = {
    "usable_now": np.array([1, 0, 0], dtype=float),
    "research_only": np.array([0, 1, 0], dtype=float),
    "needs_conversion": np.array([0, 0, 1], dtype=float),
}

def ternary_lens_vector(classification: str) -> np.ndarray:
    """Return a (3,) one‑hot vector for the given classification."""
    return _LENS_MAP.get(classification, np.zeros(3, dtype=float))

# Nine simple regex patterns (place‑holders for real ones).
_REGEX_PATTERNS = [
    r"\berror\b",
    r"\bwarning\b",
    r"\bfailed\b",
    r"\bsuccess\b",
    r"\btimeout\b",
    r"\bretry\b",
    r"\bdeprecated\b",
    r"\bexperimental\b",
    r"\bcritical\b",
]

def extract_feature_counts(text: str) -> np.ndarray:
    """Count occurrences of each regex pattern in *text*; returns a (9,) vector."""
    counts = np.zeros(len(_REGEX_PATTERNS), dtype=float)
    for i, pat in enumerate(_REGEX_PATTERNS):
        counts[i] = len(re.findall(pat, text, flags=re.IGNORECASE))
    return counts


# ----------------------------------------------------------------------
# Hybrid score computation
# ----------------------------------------------------------------------

def hybrid_ucb_score(
    action: str,
    reward_cms: CountMinSketch,
    pull_cms: CountMinSketch,
    hll: HyperLogLog,
    rlct_lambda: float,
    total_pulls: int,
    F: np.ndarray,
    classification: str,
    context_text: str,
    alpha: float = 2.0,
    gamma: float = 0.1,
) -> float:
    """
    Compute the hybrid UCB score for *action*.

    Parameters
    ----------
    action : str
        Identifier of the bandit arm.
    reward_cms : CountMinSketch
        Sketch accumulating summed rewards per action.
    pull_cms : CountMinSketch
        Sketch accumulating pull counts per action.
    hll : HyperLogLog
        Sketch of distinct context identifiers (used for n̂).
    rlct_lambda : float
        Estimated RLCT λ̂.
    total_pulls : int
        Global number of pulls N.
    F : np.ndarray
        Fusion matrix of shape (3, 9).
    classification : str
        Ternary lens classification for the current context.
    context_text : str
        Raw textual context from which regex counts are extracted.
    alpha : float, optional
        Exploration coefficient in the sqrt term.
    gamma : float, optional
        Weighting factor for the lens-score term.

    Returns
    -------
    float
        Hybrid UCB value.
    """
    # Estimate mean reward μ̂_a = total_reward / count
    reward_sum = reward_cms.estimate(action)
    pulls = pull_cms.estimate(action) + 1e-9  # avoid division by zero
    mu_hat = reward_sum / pulls

    # Classic UCB exploration term
    exploration = math.sqrt((alpha * math.log(total_pulls + 1)) / pulls)

    # RLCT term
    rlct_term = rlct_lambda * math.log(hll.cardinality() + 1)

    # Lens-score term
    lens_vector = ternary_lens_vector(classification)
    feature_counts = extract_feature_counts(context_text)
    lens_score = np.dot(lens_vector, np.dot(F, feature_counts))
    lens_term = gamma * lens_score

    # Hybrid UCB score
    ucb_score = mu_hat + exploration + rlct_term + lens_term

    return ucb_score


# Improved version of the hybrid algorithm
class HybridBandit:
    def __init__(self, num_actions: int, num_features: int, alpha: float = 2.0, gamma: float = 0.1):
        self.num_actions = num_actions
        self.num_features = num_features
        self.alpha = alpha
        self.gamma = gamma
        self.reward_cms = CountMinSketch()
        self.pull_cms = CountMinSketch()
        self.hll = HyperLogLog()
        self.rlct_lambda = 0.0
        self.total_pulls = 0
        self.F = np.random.rand(3, num_features)

    def update(self, action: str, reward: float, classification: str, context_text: str):
        self.reward_cms.update(action, reward)
        self.pull_cms.update(action)
        self.hll.add(context_text)
        self.total_pulls += 1
        self.rlct_lambda = estimate_rlct([reward], [self.total_pulls])

    def select_action(self, classification: str, context_text: str) -> str:
        best_score = -float('inf')
        best_action = None
        for action in range(self.num_actions):
            score = hybrid_ucb_score(
                str(action),
                self.reward_cms,
                self.pull_cms,
                self.hll,
                self.rlct_lambda,
                self.total_pulls,
                self.F,
                classification,
                context_text,
                self.alpha,
                self.gamma,
            )
            if score > best_score:
                best_score = score
                best_action = str(action)
        return best_action


# Example usage
if __name__ == "__main__":
    num_actions = 5
    num_features = 9
    bandit = HybridBandit(num_actions, num_features)

    for _ in range(100):
        classification = "usable_now"
        context_text = "This is a sample context text."
        action = bandit.select_action(classification, context_text)
        reward = np.random.rand()
        bandit.update(action, reward, classification, context_text)
        print(f"Selected action: {action}, Reward: {reward}")