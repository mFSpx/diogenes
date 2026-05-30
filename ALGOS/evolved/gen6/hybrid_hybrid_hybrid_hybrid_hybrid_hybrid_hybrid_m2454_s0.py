# DARWIN HAMMER — match 2454, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m717_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1085_s0.py (gen5)
# born: 2026-05-29T23:42:25Z

"""Hybrid Bandit‑Lens‑Geometric‑Gini Fusion
=======================================

Parents:
- **Parent A** – contextual bandit with Count‑Min sketch for reward
  frequencies, HyperLogLog for distinct contexts, RL‑based loss term
  λ·log n and a ternary “lens” term `L·(F·c)`.
- **Parent B** – computes a geometric product of two feature vectors,
  derives weekday‑frequency statistics via the Doomsday algorithm,
  evaluates the Gini coefficient on the product distribution and uses
  regex‑derived feature weights for bandit reward scoring.

**Mathematical bridge**  
Both parents expose a *scalar* that can be added to the classic
UCB‑type confidence bound:

* Parent A contributes `L·(F·c)`.
* Parent B contributes `γ·Gini(g)`, where `g` is the geometric product of
  the regex count vector `c` and the weekday‑frequency vector `w`.

The fused selection criterion for an action *a* is therefore


UCB_fused(a) = μ̂_a
               + sqrt(α·log N / N_a)
               + λ̂·log n̂
               + L_a·(F·c_a)                # lens term (A)
               + γ·Gini( g_a )              # geometric‑Gini term (B)


`g_a = geometric_product(c_a, w)` and `w` is obtained from the
input date via the Doomsday weekday calculation.  The term
`γ·Gini(g_a)` injects the distributional information of Parent B into
the confidence bound, completing a true hybrid of the two topologies.

The implementation below provides:
* sketch primitives (`CountMinSketch`, `HyperLogLog`),
* lens extraction (`extract_lens`),
* regex count vector (`regex_counts`),
* geometric product & Gini coefficient,
* RL‑CT estimator (simple linear regression on observed losses),
* the fused UCB computation and an action selector.

A minimal smoke test demonstrates a full loop of updates and selections.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# ----------------------------------------------------------------------
# Sketch primitives
# ----------------------------------------------------------------------


class CountMinSketch:
    """Count‑Min sketch for non‑negative integer streams."""

    def __init__(self, width: int = 2048, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = random.Random(seed)
        self.seeds = [rng.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, key: str, i: int) -> int:
        h = hash((key, self.seeds[i]))
        return h % self.width

    def add(self, key: str, count: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.tables[i, idx] += count

    def estimate(self, key: str) -> int:
        return min(self.tables[i, self._hash(key, i)] for i in range(self.depth))


class HyperLogLog:
    """Very small HyperLogLog approximation using a Python set (exact)."""

    def __init__(self):
        self._set = set()

    def add(self, item: str) -> None:
        self._set.add(item)

    def cardinality(self) -> int:
        return len(self._set)


# ----------------------------------------------------------------------
# Regex‑derived feature set (9 patterns)
# ----------------------------------------------------------------------
REGEX_PATTERNS = [
    re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
               re.I),
    re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|milestone)\b", re.I),
    re.compile(r"\b(?:error|fail|bug|exception|traceback|stacktrace)\b", re.I),
    re.compile(r"\b(?:success|passed|ok|done|finished|completed)\b", re.I),
    re.compile(r"\b(?:user|client|customer|account|profile)\b", re.I),
    re.compile(r"\b(?:cpu|memory|disk|io|latency|throughput|bandwidth)\b", re.I),
    re.compile(r"\b(?:update|upgrade|patch|release|version)\b", re.I),
    re.compile(r"\b(?:security|vulnerability|cve|exploit|attack)\b", re.I),
    re.compile(r"\b(?:train|model|predict|accuracy|loss|epoch)\b", re.I),
]


def regex_counts(text: str) -> np.ndarray:
    """Return a 9‑dim count vector for the predefined regex patterns."""
    counts = np.zeros(len(REGEX_PATTERNS), dtype=np.int64)
    for i, pat in enumerate(REGEX_PATTERNS):
        counts[i] = len(pat.findall(text))
    return counts


# ----------------------------------------------------------------------
# Lens extraction (3‑dim ternary vector)
# ----------------------------------------------------------------------


def extract_lens(text: str) -> np.ndarray:
    """
    Produce a ternary lens vector L ∈ {0,1}³:
    * L[0] – presence of any evidence‑type token,
    * L[1] – presence of any planning‑type token,
    * L[2] – presence of any error‑type token.
    """
    L = np.zeros(3, dtype=np.int64)
    L[0] = int(bool(REGEX_PATTERNS[0].search(text)))
    L[1] = int(bool(REGEX_PATTERNS[1].search(text)))
    L[2] = int(bool(REGEX_PATTERNS[2].search(text)))
    return L


# ----------------------------------------------------------------------
# Doomsday weekday frequencies (7‑dim vector)
# ----------------------------------------------------------------------


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def doomsday_weekday(year: int) -> int:
    """
    Return the weekday (0=Monday … 6=Sunday) of the Doomsday for the given year.
    Classic Conway algorithm.
    """
    anchor = {0: 2, 1: 0, 2: 5, 3: 3}[year % 4]
    yy = year % 100
    dooms = (yy // 12) + (yy % 12) + ((yy % 12) // 4) + anchor
    if is_leap(year):
        dooms += 1
    return dooms % 7


def weekday_frequencies(date: datetime) -> np.ndarray:
    """
    Produce a 7‑dim vector w where w[i] is the count of the weekday i
    in the 30‑day window ending on `date`.  The window is simulated
    deterministically using the Doomsday weekday as a seed.
    """
    seed = doomsday_weekday(date.year)
    rng = random.Random(seed + date.toordinal())
    w = np.zeros(7, dtype=np.int64)
    for offset in range(30):
        wd = (date.weekday() - offset) % 7
        w[wd] += 1
    # add a tiny random perturbation to avoid exact zeros
    w += rng.randint(0, 1)
    return w


# ----------------------------------------------------------------------
# Geometric product & Gini coefficient
# ----------------------------------------------------------------------


def geometric_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Simple geometric (outer) product flattened to a 1‑D array.
    For vectors of length m and n the result has length m·n.
    """
    return np.outer(v1, v2).reshape(-1)


def gini_coefficient(x: np.ndarray) -> float:
    """Compute the Gini coefficient of a non‑negative 1‑D array."""
    if x.size == 0:
        return 0.0
    sorted_x = np.sort(x.astype(np.float64))
    n = x.size
    cumulative = np.cumsum(sorted_x)
    sum_x = cumulative[-1]
    if sum_x == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_x) / n
    return float(gini)


# ----------------------------------------------------------------------
# RLCT estimator (simple linear regression on loss vs log n)
# ----------------------------------------------------------------------


class RLCTEstimator:
    """
    Maintains a running list of (log_n, loss) pairs and fits λ̂ in
    loss ≈ λ̂·log n  (no intercept) via ordinary least squares.
    """

    def __init__(self):
        self.log_ns = []
        self.losses = []

    def observe(self, n: int, loss: float) -> None:
        if n <= 0:
            return
        self.log_ns.append(math.log(n))
        self.losses.append(loss)

    def lambda_hat(self) -> float:
        if not self.log_ns:
            return 0.0
        X = np.array(self.log_ns)
        y = np.array(self.losses)
        # λ̂ = (X·y) / (X·X)
        numerator = np.dot(X, y)
        denominator = np.dot(X, X)
        return float(numerator / denominator) if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid bandit core
# ----------------------------------------------------------------------


class HybridBandit:
    """
    Stores per‑action statistics, sketches, and provides the fused UCB
    selection rule.
    """

    def __init__(self, actions, alpha=1.0, gamma=1.0):
        self.actions = actions
        self.alpha = alpha          # exploration coefficient
        self.gamma = gamma          # weight for Gini term
        self.total_pulls = 0

        # per‑action accumulators
        self.sum_rewards = {a: 0.0 for a in actions}
        self.pull_counts = {a: 0 for a in actions}

        # sketches
        self.reward_sketch = CountMinSketch()
        self.context_sketch = HyperLogLog()

        # RLCT estimator
        self.rlct = RLCTEstimator()

        # lens‑fusion matrix F (3×9)
        rng = np.random.default_rng(42)
        self.F = rng.normal(loc=0.0, scale=0.1, size=(3, 9))

    def update(self, action, reward, context_text, timestamp):
        """Incorporate a new observation."""
        # update raw stats
        self.sum_rewards[action] += reward
        self.pull_counts[action] += 1
        self.total_pulls += 1

        # sketch updates
        self.reward_sketch.add(f"{action}:{context_text}", count=reward)
        self.context_sketch.add(context_text)

        # RLCT observation (using loss = 1 - reward as a placeholder)
        self.rlct.observe(self.total_pulls, loss=1.0 - reward)

    def _estimate_mean(self, action):
        """Mean reward estimate using the sketch."""
        key = f"{action}:*"
        # Since we cannot query a wildcard, fall back to raw average
        count = self.pull_counts[action]
        return self.sum_rewards[action] / count if count > 0 else 0.0

    def _estimate_count(self, action):
        """Pull count estimate using the sketch (fallback to raw)."""
        return self.pull_counts[action]

    def _estimate_distinct_contexts(self):
        return self.context_sketch.cardinality()

    def _lens_term(self, L, c):
        return float(L @ (self.F @ c))

    def _gini_term(self, c, w):
        g = geometric_product(c, w)
        return self.gamma * gini_coefficient(g)

    def ucb(self, action, context_text, timestamp):
        """Compute the fused UCB value for a given action."""
        mu_hat = self._estimate_mean(action)
        N_a = self._estimate_count(action)
        N = self.total_pulls if self.total_pulls > 0 else 1

        # classic UCB part
        confidence = math.sqrt(self.alpha * math.log(N + 1) / (N_a + 1e-9))

        # RLCT term
        lam_hat = self.rlct.lambda_hat()
        n_hat = self._estimate_distinct_contexts()
        rlct_term = lam_hat * math.log(n_hat + 1)

        # lens term
        L = extract_lens(context_text)
        c = regex_counts(context_text)
        lens_term = self._lens_term(L, c)

        # Gini term from geometric product
        w = weekday_frequencies(timestamp)
        gini_term = self._gini_term(c, w)

        return mu_hat + confidence + rlct_term + lens_term + gini_term

    def select_action(self, context_text, timestamp):
        """Select the action with the highest fused UCB."""
        best_action = None
        best_score = -float('inf')
        for a in self.actions:
            score = self.ucb(a, context_text, timestamp)
            if score > best_score:
                best_score = score
                best_action = a
        return best_action


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    actions = ["accept", "reject", "defer"]
    bandit = HybridBandit(actions, alpha=1.5, gamma=0.8)

    # synthetic stream of 200 interactions
    for step in range(200):
        # generate a random context (sentence) and a timestamp
        ctx = random.choice([
            "The evidence was verified and the plan was approved.",
            "An error occurred during the model training epoch.",
            "User reported a latency issue on the server.",
            "Security patch was applied successfully.",
            "The document contains the final results."
        ])
        ts = datetime.now()

        # select action
        act = bandit.select_action(ctx, ts)

        # simulate reward (higher for "accept" on positive contexts)
        base = {"accept": 0.7, "reject": 0.3, "defer": 0.5}[act]
        reward = random.random() < base
        reward = 1.0 if reward else 0.0

        # update bandit
        bandit.update(act, reward, ctx, ts)

    # final report
    print("=== Final statistics ===")
    for a in actions:
        pulls = bandit.pull_counts[a]
        avg = bandit._estimate_mean(a)
        print(f"Action {a!r}: pulls={pulls}, avg_reward={avg:.3f}")
    print(f"Estimated λ̂ (RLCT): {bandit.rlct.lambda_hat():.4f}")
    print(f"Distinct contexts observed: {bandit._estimate_distinct_contexts()}")
    sys.exit(0)