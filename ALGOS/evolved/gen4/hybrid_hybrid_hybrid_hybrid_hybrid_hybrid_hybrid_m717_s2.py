# DARWIN HAMMER — match 717, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_label_foundry_m21_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_decisi_m260_s1.py (gen3)
# born: 2026-05-29T23:30:43Z

"""Hybrid Bandit‑Lens‑Sketch Fusion
================================

This module fuses the mathematics of the two parent algorithms:

* **Parent A** – a contextual bandit that sketches per‑action reward
  frequencies (Count‑Min), sketches distinct contexts (HyperLogLog) and
  augments the classic Upper‑Confidence‑Bound (UCB) with a loss‑driven
  Real‑Log‑Canonical‑Threshold (RLCT) term λ·log n.

* **Parent B** – a ternary “lens” classification **L**∈{0,1}³ and a nine‑dimensional
  regex‑derived count vector **c**∈ℕ⁹ that are combined through a learned
  fusion matrix **F**∈ℝ³ˣ⁹ as

      score_lens = L · (F · c) = Σ_i Σ_j L_i c_j F_{ij} .

The **mathematical bridge** is the scalar `score_lens`, which can be added as
an extra bias term to the bandit’s confidence bound.  The resulting hybrid
selection criterion for action *a* is

    UCB_hybrid(a) = μ̂_a
                    + √(α·log N / N_a)
                    + λ̂·log n̂
                    + L_a · (F · c_a)

where

* μ̂_a  – sketch‑estimated mean reward,
* N_a   – sketch‑estimated pull count,
* N     – total number of pulls,
* n̂    – sketch‑estimated number of distinct contexts,
* λ̂    – RLCT estimate from the observed loss sequence,
* L_a   – ternary lens vector for the current context,
* c_a   – regex count vector extracted from the same context.

The code below implements the three sketch primitives, RLCT regression,
lens‑feature extraction, and the hybrid UCB decision rule.  A small smoke
test demonstrates a complete loop of updates and selections."""

import math
import random
import sys
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

import numpy as np

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

    # RLCT term using estimated distinct contexts n̂
    n_hat = hll.cardinality() + 1e-9
    rlct_term = rlct_lambda * math.log(n_hat)

    # Lens‑feature term
    L = ternary_lens_vector(classification)          # (3,)
    c = extract_feature_counts(context_text)         # (9,)
    lens_term = float(L @ (F @ c))                    # scalar

    return mu_hat + exploration + rlct_term + lens_term


def select_action(
    actions: List[str],
    reward_cms: CountMinSketch,
    pull_cms: CountMinSketch,
    hll: HyperLogLog,
    rlct_lambda: float,
    total_pulls: int,
    F: np.ndarray,
    classifications: Dict[str, str],
    contexts: Dict[str, str],
) -> Tuple[str, float]:
    """Return the action with the highest hybrid UCB score and its score."""
    best_action = None
    best_score = -math.inf
    for a in actions:
        score = hybrid_ucb_score(
            action=a,
            reward_cms=reward_cms,
            pull_cms=pull_cms,
            hll=hll,
            rlct_lambda=rlct_lambda,
            total_pulls=total_pulls,
            F=F,
            classification=classifications.get(a, ""),
            context_text=contexts.get(a, ""),
        )
        if score > best_score:
            best_score = score
            best_action = a
    return best_action, best_score


def update_models(
    action: str,
    reward: float,
    reward_cms: CountMinSketch,
    pull_cms: CountMinSketch,
    hll: HyperLogLog,
    loss_history: List[float],
    sample_history: List[int],
) -> Tuple[float, List[float], List[int]]:
    """
    Update all sketches with a new observation and return the updated RLCT λ̂.

    Returns
    -------
    lambda_hat : float
        Updated RLCT estimate.
    loss_history, sample_history : lists
        Updated histories (used for RLCT regression).
    """
    reward_cms.update(action, int(reward))
    pull_cms.update(action, 1)
    hll.add(action)  # using action identifier as a proxy for context id

    # For RLCT we treat negative reward as loss.
    loss = max(0.0, -reward)
    loss_history.append(loss)
    sample_history.append(pull_cms.total())
    lambda_hat = estimate_rlct(loss_history, sample_history)
    return lambda_hat, loss_history, sample_history


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Define three arms.
    actions = ["alpha", "beta", "gamma"]

    # Initialize sketches.
    reward_cms = CountMinSketch(width=1024, depth=4, seed=1)
    pull_cms = CountMinSketch(width=1024, depth=4, seed=2)
    hll = HyperLogLog(b=10)

    # Randomly initialise a fusion matrix F.
    F = np.random.randn(3, 9)

    # Dummy classifications and context texts per action.
    classifications = {
        "alpha": "usable_now",
        "beta": "research_only",
        "gamma": "needs_conversion",
    }
    contexts = {
        "alpha": "The operation succeeded without error.",
        "beta": "Warning: deprecated API used. Possible timeout.",
        "gamma": "Critical failure! Error and retry required.",
    }

    # Histories for RLCT estimation.
    loss_hist: List[float] = []
    sample_hist: List[int] = []

    total_pulls = 0
    lambda_hat = 0.0

    # Simulate 30 rounds.
    for round_idx in range(30):
        # Choose action via hybrid UCB.
        chosen, score = select_action(
            actions,
            reward_cms,
            pull_cms,
            hll,
            lambda_hat,
            total_pulls,
            F,
            classifications,
            contexts,
        )

        # Simulate a stochastic reward (higher for "alpha").
        true_means = {"alpha": 1.0, "beta": 0.5, "gamma": 0.2}
        reward = random.gauss(true_means[chosen], 0.3)

        # Update models.
        lambda_hat, loss_hist, sample_hist = update_models(
            chosen,
            reward,
            reward_cms,
            pull_cms,
            hll,
            loss_hist,
            sample_hist,
        )
        total_pulls += 1

        # Simple progress output.
        print(
            f"Round {round_idx+1:02d}: chosen={chosen:6s} reward={reward: .2f} "
            f"score={score: .3f} λ̂={lambda_hat: .3f}"
        )

    print("\nFinal estimated RLCT λ̂:", lambda_hat)
    print("Estimated distinct contexts (HyperLogLog):", int(hll.cardinality()))
    for a in actions:
        est_reward = reward_cms.estimate(a) / max(1, pull_cms.estimate(a))
        print(f"Action {a}: pulls={pull_cms.estimate(a)} est_reward={est_reward:.3f}")