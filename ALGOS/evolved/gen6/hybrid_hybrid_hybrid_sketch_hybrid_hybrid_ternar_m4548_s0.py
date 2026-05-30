# DARWIN HAMMER — match 4548, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s2.py (gen5)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hoeffding_tre_m1040_s1.py (gen2)
# born: 2026-05-29T23:56:27Z

"""Hybrid Algorithm Fusion of `hybrid_hybrid_sketches_hybr_hybrid_hdc_hy_m561_s2.py`
and `hybrid_hybrid_ternary_route_hybrid_hoeffding_tre_m1040_s1.py`.

Mathematical Bridge
-------------------
The fusion rests on three shared mathematical concepts:

1. **Probabilistic Frequency Estimation** – The Count‑Min Sketch (CM‑Sketch) from
   Algorithm A provides a lightweight estimate of how often each symbolic item
   appears in a stream.

2. **Similarity & Uncertainty Quantification** – Algorithm B supplies the
   Structural Similarity Index (SSIM) to compare high‑dimensional vectors and
   the Hoeffding bound together with the Gini coefficient to assess confidence
   and inequality of reward distributions.

3. **Vector‑Space Binding & Bundling** – Both parents operate on binary
   hyper‑dimensional vectors, binding (element‑wise multiplication) and
   bundling (element‑wise summation).  

The hybrid algorithm therefore:
* builds a modulation vector by weighting symbol vectors with CM‑Sketch counts,
* evaluates similarity between modulation vectors using SSIM,
* decides whether to incorporate bandit feedback into the policy by applying
  a Hoeffding‑based confidence test modulated by the Gini coefficient of the
  observed rewards.

The resulting system unifies streaming frequency estimation, hyper‑dimensional
binding, similarity measurement, and statistically‑grounded policy updates.
"""

import hashlib
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared from parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Global policy store (parent A)
# ----------------------------------------------------------------------
_POLICY: dict[str, List[float]] = {}


def reset_policy() -> None:
    """Clear the global reward statistics."""
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """Accumulate raw reward sums and counts for each action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def _reward_estimate(action_id: str) -> float:
    """Mean reward for an action, or 0.0 if never seen."""
    total, n = _POLICY.get(action_id, (0.0, 0.0))
    return total / n if n else 0.0


# ----------------------------------------------------------------------
# Core primitives from parent A
# ----------------------------------------------------------------------
def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a CM‑Sketch table for a list of string items."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table


def symbol_vector(symbol: str, dim: int = 10000) -> List[int]:
    """Deterministic bipolar vector (+1 / -1) seeded by the symbol."""
    seed = int.from_bytes(hashlib.sha256(symbol.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def bind(a: List[int], b: List[int]) -> List[int]:
    """Element‑wise multiplication (binding) of two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[List[int]]) -> List[int]:
    """Element‑wise summation (bundling) of a list of vectors."""
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("all vectors must share the same dimension")
    return [sum(comp) for comp in zip(*vectors)]


# ----------------------------------------------------------------------
# Core primitives from parent B
# ----------------------------------------------------------------------
def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Structural Similarity Index (SSIM) between two 1‑D arrays."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return numerator / denominator


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a Bernoulli‑like variable with mean r."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient of a non‑negative value list."""
    xs = sorted(float(v) for v in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = 0.0
    for i, x in enumerate(xs, 1):
        cumulative += i * x
    return (2 * cumulative) / (n * sum(xs)) - (n + 1) / n


# ----------------------------------------------------------------------
# Hybrid Functions (the new fused logic)
# ----------------------------------------------------------------------
def generate_modulation_vector(context_items: List[str], dim: int = 10000) -> np.ndarray:
    """
    Build a high‑dimensional modulation vector from a stream of symbolic items.
    The CM‑Sketch supplies a count for each item; each item’s symbol vector is
    bound to its count (scalar multiplication) and all bound vectors are bundled.
    The result is returned as a NumPy float array ready for SSIM comparison.
    """
    sketch = count_min_sketch(context_items, width=128, depth=5)

    # Helper to retrieve an approximate count for a given item
    def approx_count(item: str) -> int:
        return min(
            sketch[d][int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % 128]
            for d in range(len(sketch))
        )

    bound_vectors: List[List[int]] = []
    for item in set(context_items):
        count = approx_count(item)
        base_vec = symbol_vector(item, dim=dim)
        # Bind by scalar multiplication (count) – equivalent to repeated binding with a unary vector
        scaled_vec = [x * count for x in base_vec]
        bound_vectors.append(scaled_vec)

    if not bound_vectors:
        # No items → return a zero vector
        return np.zeros(dim, dtype=np.float64)

    bundled = bundle(bound_vectors)
    # Normalise to unit variance to keep SSIM numerically stable
    arr = np.array(bundled, dtype=np.float64)
    if np.std(arr) > 0:
        arr = (arr - np.mean(arr)) / np.std(arr)
    return arr


def similarity_with_confidence(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    delta: float = 0.05,
    n_samples: int = 100,
) -> float:
    """
    Compute a confidence‑adjusted similarity between two modulation vectors.
    The raw SSIM score is penalised by a Hoeffding bound that reflects the
    uncertainty due to finite sample size (`n_samples`). Larger `n_samples`
    shrink the bound, increasing trust in the similarity value.
    """
    raw_ssim = compute_ssim(vec_a, vec_b)
    # Use the SSIM as the proxy for the underlying Bernoulli mean r
    bound = hoeffding_bound(r=raw_ssim, delta=delta, n=n_samples)
    # Confidence‑adjusted similarity (clipped to [0,1])
    adj = max(0.0, min(1.0, raw_ssim - bound))
    return adj


def update_policy_with_gini(
    updates: List[BanditUpdate],
    delta: float = 0.05,
    min_samples: int = 5,
) -> None:
    """
    Incorporate bandit feedback only when the reward distribution for an action
    is sufficiently balanced (low Gini) and the Hoeffding confidence bound
    indicates the estimate is reliable.

    For each action:
        * compute mean reward,
        * compute Hoeffding bound using the mean as `r`,
        * compute Gini of the collected rewards,
        * accept the update if Gini < 0.4 and bound < 0.1.
    """
    # Accumulate rewards per action for Gini calculation
    rewards_by_action: dict[str, List[float]] = defaultdict(list)
    for u in updates:
        rewards_by_action[u.action_id].append(u.reward)

    # Apply updates respecting the statistical guards
    for action_id, rewards in rewards_by_action.items():
        n = len(rewards)
        if n < min_samples:
            continue  # not enough data for a reliable bound

        mean_reward = sum(rewards) / n
        bound = hoeffding_bound(r=mean_reward, delta=delta, n=n)
        gini = gini_coefficient(rewards)

        if gini < 0.4 and bound < 0.1:
            # Accept the updates for this action
            accepted = [
                BanditUpdate(context_id="merged", action_id=action_id, reward=r, propensity=1.0)
                for r in rewards
            ]
            update_policy(accepted)
        # else: reject silently – the policy remains unchanged for this batch


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset any previous state
    reset_policy()

    # Create two synthetic contexts
    context1 = ["apple", "banana", "apple", "cherry", "date", "apple"]
    context2 = ["banana", "banana", "fig", "grape", "apple", "fig"]

    # Generate modulation vectors
    vec1 = generate_modulation_vector(context1)
    vec2 = generate_modulation_vector(context2)

    # Compute confidence‑adjusted similarity
    sim = similarity_with_confidence(vec1, vec2, delta=0.05, n_samples=200)
    print(f"Adjusted similarity between contexts: {sim:.4f}")

    # Simulate bandit feedback
    fake_updates = [
        BanditUpdate(context_id="c1", action_id="click_ad", reward=1.0, propensity=0.5),
        BanditUpdate(context_id="c1", action_id="click_ad", reward=0.0, propensity=0.5),
        BanditUpdate(context_id="c2", action_id="click_ad", reward=1.0, propensity=0.6),
        BanditUpdate(context_id="c2", action_id="click_ad", reward=1.0, propensity=0.6),
        BanditUpdate(context_id="c3", action_id="click_ad", reward=0.0, propensity=0.4),
        BanditUpdate(context_id="c3", action_id="click_ad", reward=0.0, propensity=0.4),
    ]

    # Apply the Gini‑guarded policy update
    update_policy_with_gini(fake_updates, delta=0.05, min_samples=3)

    # Show the learned mean reward
    mean = _reward_estimate("click_ad")
    print(f"Estimated mean reward for 'click_ad': {mean:.3f}")

    # Ensure the module runs without error
    sys.exit(0)