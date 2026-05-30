# DARWIN HAMMER — match 4708, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2169_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m717_s2.py (gen4)
# born: 2026-05-29T23:57:43Z

"""Hybrid Bandit‑Tree‑Lens Algorithm
===================================

This module fuses the two parent algorithms **A** (Hybrid Bandit‑Tree) and **B**
(Hybrid Bandit‑Lens‑Sketch).  The mathematical bridge is the *count‑min sketch*:
both parents rely on a sketch **S** to estimate action frequencies.  Parent A
adds a tropical (max‑plus) bias `max_i (S_{i,h(a)} + μ̂_a)` to the reward, while
parent B adds a lens bias `L_a·(F·c_a)` to an Upper‑Confidence‑Bound (UCB)
formula.  The hybrid score therefore combines these two scalar biases:


HybridScore(a) = max_i (S_{i,h(a)} + μ̂_a)                     # tropical bias
                + √(α·log N / N̂_a)                           # UCB exploration
                + λ̂·log n̂                                    # RLCT regulariser
                + L_a · (F·c_a)                               # lens bias


where  

* `μ̂_a`  – mean reward estimated from the sketch,  
* `N̂_a`  – sketch‑estimated pull count,  
* `N`    – total pulls,  
* `n̂`    – distinct‑context estimate from a HyperLogLog‑like sketch,  
* `λ̂`    – regression coefficient on observed losses,  
* `L_a`  – ternary lens vector for the current context,  
* `c_a`  – nine‑dimensional regex‑derived count vector,  
* `F`    – learned fusion matrix (3×9).

The implementation provides three core functions that demonstrate the hybrid
operation:

1. `tropical_bias` – computes the max‑plus term from the count‑min sketch.
2. `lens_bias` – extracts `L` and `c` from a context and evaluates `L·(F·c)`.
3. `select_hybrid_action` – ε‑greedy selection using the full `HybridScore`.

All required sketch primitives, a simple RLCT estimator, and a smoke test are
included.  The code runs with only the Python standard library and NumPy. 
"""

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Global stores (mirroring parent B)
# ----------------------------------------------------------------------
_POLICY_REWARDS: Dict[str, float] = {}   # cumulative reward per action
_POLICY_COUNTS: Dict[str, int] = {}      # pull count per action

# ----------------------------------------------------------------------
# Count‑Min Sketch (parent B)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Simple Count‑Min sketch for non‑negative integer streams."""
    def __init__(self, width: int = 2000, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = random.Random(seed)
        self.seeds = [rng.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, key: str, i: int) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(key.encode('utf-8'))
        h.update(self.seeds[i].to_bytes(4, 'little'))
        return int.from_bytes(h.digest(), 'little') % self.width

    def add(self, key: str, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.tables[i, idx] += increment

    def estimate(self, key: str) -> int:
        """Return the min‑over‑rows estimate for the key."""
        return min(self.tables[i, self._hash(key, i)] for i in range(self.depth))

    def matrix(self) -> np.ndarray:
        """Expose the full sketch matrix for tropical operations."""
        return self.tables


# ----------------------------------------------------------------------
# HyperLogLog‑like distinct‑context estimator (very lightweight)
# ----------------------------------------------------------------------
class DistinctSketch:
    """Very small distinct counter using a Python set (placeholder for HLL)."""
    def __init__(self):
        self._set = set()

    def add(self, item: str) -> None:
        self._set.add(item)

    def estimate(self) -> int:
        return len(self._set)


# ----------------------------------------------------------------------
# RLCT (Real‑Log‑Canonical‑Threshold) estimator
# ----------------------------------------------------------------------
class RLCTEstimator:
    """Online linear regression of loss vs. log(count) to obtain λ̂."""
    def __init__(self):
        self._xs: List[float] = []   # log n
        self._ys: List[float] = []   # loss
        self._lambda: float = 0.0

    def add_observation(self, loss: float, count: int) -> None:
        if count <= 0:
            return
        self._xs.append(math.log(count))
        self._ys.append(loss)
        self._update()

    def _update(self) -> None:
        if len(self._xs) < 2:
            self._lambda = 0.0
            return
        x = np.array(self._xs)
        y = np.array(self._ys)
        # Simple least‑squares slope (λ̂)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        self._lambda = slope

    @property
    def lambda_hat(self) -> float:
        return self._lambda


# ----------------------------------------------------------------------
# Lens extraction (parent B)
# ----------------------------------------------------------------------
def extract_lens(context: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce a ternary lens vector L∈{0,1}³ and a 9‑dimensional count vector c∈ℕ⁹
    from the raw context string.

    For demonstration we use:
      - L bits: presence of digits, uppercase, punctuation.
      - c counts: frequencies of the characters 'a'..'i' (9 bins).
    """
    has_digit = any(ch.isdigit() for ch in context)
    has_upper = any(ch.isupper() for ch in context)
    has_punct = any(ch in "!?.," for ch in context)
    L = np.array([int(has_digit), int(has_upper), int(has_punct)], dtype=np.float64)

    chars = 'abcdefghi'
    c = np.zeros(9, dtype=np.float64)
    for idx, ch in enumerate(chars):
        c[idx] = context.count(ch)
    return L, c


# ----------------------------------------------------------------------
# Fusion matrix F (learned in parent B; here we initialise randomly)
# ----------------------------------------------------------------------
_FUSION_MATRIX = np.random.randn(3, 9) * 0.1  # small random weights


# ----------------------------------------------------------------------
# Tropical bias (parent A)
# ----------------------------------------------------------------------
def tropical_bias(skm: CountMinSketch, action: str, mu_hat: float) -> float:
    """
    Compute max_i (S_{i, h(action)} + μ̂_action) using the sketch matrix.
    """
    S = skm.matrix()
    # Hash column for the action (same as used in sketch)
    cols = []
    for i in range(skm.depth):
        idx = skm._hash(action, i)
        cols.append(S[i, idx])
    return max(val + mu_hat for val in cols)


# ----------------------------------------------------------------------
# Lens bias (parent B)
# ----------------------------------------------------------------------
def lens_bias(context: str) -> float:
    """
    Compute L·(F·c) for a given context.
    """
    L, c = extract_lens(context)
    return float(L @ (_FUSION_MATRIX @ c))


# ----------------------------------------------------------------------
# Hybrid UCB / selection
# ----------------------------------------------------------------------
def select_hybrid_action(
    actions: List[str],
    context: str,
    skm: CountMinSketch,
    distinct_sketch: DistinctSketch,
    rlct: RLCTEstimator,
    epsilon: float = 0.1,
    alpha: float = 1.0,
) -> str:
    """
    ε‑greedy selection using the combined HybridScore.

    Parameters
    ----------
    actions : list of action identifiers.
    context : current contextual string.
    skm : Count‑Min sketch storing (action) counts.
    distinct_sketch : estimator for number of distinct contexts.
    rlct : RLCT estimator (provides λ̂).
    epsilon : exploration probability.
    alpha : exploration coefficient for the UCB term.
    """
    if random.random() < epsilon:
        return random.choice(actions)

    total_pulls = sum(_POLICY_COUNTS.get(a, 0) for a in actions) + 1
    n_hat = distinct_sketch.estimate() + 1  # avoid log(0)

    lens_term = lens_bias(context)

    best_action = None
    best_score = -math.inf

    for a in actions:
        # Retrieve or initialise statistics
        count = _POLICY_COUNTS.get(a, 0)
        reward_sum = _POLICY_REWARDS.get(a, 0.0)
        mu_hat = reward_sum / count if count > 0 else 0.0

        # Sketch‑based count estimate (CMS)
        n_hat_a = skm.estimate(a) + 1  # smoothing to avoid div‑zero

        # Tropical bias
        t_bias = tropical_bias(skm, a, mu_hat)

        # UCB exploration term
        explore = math.sqrt(alpha * math.log(total_pulls) / n_hat_a)

        # RLCT regulariser
        rlct_term = rlct.lambda_hat * math.log(n_hat)

        # Full hybrid score
        score = t_bias + explore + rlct_term + lens_term

        if score > best_score:
            best_score = score
            best_action = a

    return best_action or random.choice(actions)


# ----------------------------------------------------------------------
# Policy update utilities
# ----------------------------------------------------------------------
def update_policy(action: str, reward: float, skm: CountMinSketch,
                  distinct_sketch: DistinctSketch,
                  rlct: RLCTEstimator) -> None:
    """Record reward, update sketches and RLCT estimator."""
    # Update global policy statistics
    _POLICY_REWARDS[action] = _POLICY_REWARDS.get(action, 0.0) + reward
    _POLICY_COUNTS[action] = _POLICY_COUNTS.get(action, 0) + 1

    # Update count‑min sketch (frequency of action)
    skm.add(action, 1)

    # Update distinct‑context sketch
    distinct_sketch.add(action)  # using action as proxy for context id

    # Update RLCT with observed loss (here loss = 1 - reward)
    loss = 1.0 - reward
    total_counts = sum(_POLICY_COUNTS.values())
    rlct.add_observation(loss, total_counts)


def reset_all() -> None:
    """Reset global stores and sketches."""
    _POLICY_REWARDS.clear()
    _POLICY_COUNTS.clear()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)

    actions = [f"action_{i}" for i in range(5)]
    contexts = [
        "User123 clicked button!",
        "admin ACCESS granted?",
        "error 404 not found.",
        "hello world",
        "DATA packet received."
    ]

    # Initialise components
    cms = CountMinSketch(width=1024, depth=4, seed=7)
    distinct = DistinctSketch()
    rlct_est = RLCTEstimator()

    # Simulate 100 rounds
    for step in range(100):
        ctx = random.choice(contexts)
        chosen = select_hybrid_action(
            actions=actions,
            context=ctx,
            skm=cms,
            distinct_sketch=distinct,
            rlct=rlct_est,
            epsilon=0.1,
            alpha=1.0,
        )
        # Simulated stochastic reward (Bernoulli with bias depending on action)
        prob = 0.2 + 0.15 * (int(chosen.split('_')[1]) % 5)  # varied probabilities
        reward = 1.0 if random.random() < prob else 0.0

        update_policy(chosen, reward, cms, distinct, rlct_est)

    # Print final statistics
    print("Final action counts:", _POLICY_COUNTS)
    print("Final estimated λ̂ (RLCT):", rlct_est.lambda_hat)
    print("Distinct contexts seen:", distinct.estimate())
    print("Sample hybrid scores for each action:")
    for a in actions:
        mu = _POLICY_REWARDS.get(a, 0.0) / max(1, _POLICY_COUNTS.get(a, 0))
        print(f"  {a}: μ̂={mu:.3f}, count_est={cms.estimate(a)}")