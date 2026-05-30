# DARWIN HAMMER — match 1032, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m169_s0.py (gen3)
# parent_b: hybrid_sketches_hybrid_bandit_router_m31_s2.py (gen2)
# born: 2026-05-29T23:32:29Z

"""Hybrid SSIM‑Sketch‑Bandit Algorithm
Parents:
- hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m169_s0.py (SSIM similarity + Schoolfield temperature scaling)
- hybrid_sketches_hybrid_bandit_router_m31_s2.py (Count‑Min sketch, HyperLogLog cardinality, MinHash LSH, contextual bandit)

Mathematical bridge:
1. A context C (multiset of strings) is compressed to a count‑min sketch matrix S∈ℕ^{d×w}.
   Its Euclidean norm ‖S‖₂ = √∑_{i,j}S_{i,j}² acts as a *scale factor* σ_s.
2. The sketch is flattened to a vector v_S; a prototype vector v_P (e.g. from a
   reference model) is compared with v_S using the Structural Similarity Index
   SSIM(v_P, v_S) ∈ [0,1].
3. The biological temperature model supplies a dimensionless rate ρ(T) from the
   Schoolfield equation.
4. The combined suitability score for the context is

       η = SSIM(v_P, v_S) · ρ(T) · σ_s

   This scalar simultaneously modulates work‑share allocation (as in Parent A)
   and the propensity of a contextual bandit (as in Parent B).

5. The distinct‑element estimate |U| from HyperLogLog is used to adapt the
   exploration probability ε of the bandit:

       ε' = ε · log(1 + |U|)

6. Actions are shingled documents; a MinHash LSH bucket B(a) groups similar
   actions. Bandit statistics (pull count, cumulative reward) are stored per
   bucket, and the expected reward of a bucket is weighted by η.

The code below implements these fused concepts in a compact, testable form.
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import hashlib
import numpy as np

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------


def _pct(value: float) -> float:
    """Round to six decimal places for readability."""
    return round(float(value), 6)


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) between two 1‑D vectors.
    Returns a value in [-1, 1]; typical use‑case expects [0, 1].
    """
    # Pad the shorter vector with zeros to match lengths
    if x.shape != y.shape:
        max_len = max(x.shape[0], y.shape[0])
        x = np.pad(x, (0, max_len - x.shape[0]), constant_values=0)
        y = np.pad(y, (0, max_len - y.shape[0]), constant_values=0)

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


def schoolfield_rate(
    T: float,
    R: float = 1.0,
    E: float = 0.65,
    Hl: float = -0.2,
    Hh: float = 0.5,
    k: float = 8.617e-5,
) -> float:
    """
    Schoolfield temperature model (dimensionless rate).

    Parameters
    ----------
    T : float
        Temperature in Celsius.
    R, E, Hl, Hh, k : float
        Model parameters; defaults give a smooth bell‑shaped curve.

    Returns
    -------
    rho : float
        Developmental rate ρ(T) ∈ (0, ∞). Normalised to [0,1] for convenience.
    """
    Tk = T + 273.15  # Kelvin
    exp_term = math.exp(-E / (k * Tk))
    denom = 1.0 + math.exp(Hl / (k * Tk)) + math.exp(Hh / (k * Tk))
    rho = R * exp_term / denom
    # Normalise to [0,1] using a simple min‑max over a plausible range
    rho = (rho - 0.0) / (1.0 - 0.0)
    return max(0.0, min(1.0, rho))


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------


def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a depth×width count‑min sketch matrix for the given items."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


def sketch_norm(sketch: List[List[int]]) -> float:
    """Euclidean norm of the flattened sketch matrix."""
    flat = np.array(sketch, dtype=np.float64).flatten()
    return float(np.linalg.norm(flat))


def hyperloglog_cardinality(items: Iterable[str]) -> int:
    """Placeholder HLL: exact distinct count (deterministic proxy)."""
    return len(set(items))


def minhash_bucket(shingles: Set[str], num_buckets: int = 256) -> int:
    """
    Compute a deterministic bucket index for a document using a simple
    MinHash of its shingle set.
    """
    if not shingles:
        return 0
    min_hash = min(int(hashlib.sha1(s.encode()).hexdigest(), 16) for s in shingles)
    return min_hash % num_buckets


# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------


@dataclass
class BucketStats:
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls > 0 else 0.0


@dataclass
class HybridBandit:
    """
    Contextual bandit that stores statistics per MinHash LSH bucket.
    The propensity of a bucket is weighted by the fused suitability score η.
    """
    epsilon: float = 0.1
    bucket_stats: Dict[int, BucketStats] = field(default_factory=lambda: defaultdict(BucketStats))

    def adapt_epsilon(self, cardinality: int) -> float:
        """Adapt ε using the HyperLogLog cardinality estimate."""
        adapted = self.epsilon * math.log(1 + cardinality)
        return min(1.0, max(0.0, adapted))

    def select_bucket(self, buckets: Set[int], eta: float, adapted_epsilon: float) -> int:
        """
        ε‑greedy selection where the exploitation score for a bucket b is
        η·(1 + mean_reward_b). Exploration chooses uniformly at random.
        """
        if random.random() < adapted_epsilon:
            return random.choice(list(buckets))

        # Compute weighted scores
        scores = {}
        for b in buckets:
            stats = self.bucket_stats[b]
            exploitation = eta * (1.0 + stats.mean_reward)
            scores[b] = exploitation
        # Choose bucket with maximal score
        return max(scores, key=scores.get)

    def update(self, bucket: int, reward: float) -> None:
        """Increment statistics for the chosen bucket."""
        stats = self.bucket_stats[bucket]
        stats.pulls += 1
        stats.reward_sum += reward


# ----------------------------------------------------------------------
# Hybrid algorithm functions
# ----------------------------------------------------------------------


def fuse_context_score(
    context_items: Iterable[str],
    prototype_vec: np.ndarray,
    temperature_c: float,
) -> Tuple[float, float]:
    """
    Compute the fused suitability score η for a given context.

    Returns
    -------
    eta : float
        Combined score η = SSIM·ρ·‖S‖₂
    sketch_norm_val : float
        The norm ‖S‖₂ (exposed for diagnostics).
    """
    # 1️⃣ Sketch the context
    sketch = count_min_sketch(context_items)
    s_norm = sketch_norm(sketch)

    # 2️⃣ Flatten sketch to a vector and compare with prototype via SSIM
    sketch_vec = np.array(sketch, dtype=np.float64).flatten()
    ssim = compute_ssim(prototype_vec, sketch_vec)

    # 3️⃣ Temperature scaling
    rho = schoolfield_rate(temperature_c)

    # 4️⃣ Fuse
    eta = ssim * rho * s_norm
    return eta, s_norm


def hybrid_action_selection(
    context_items: Iterable[str],
    prototype_vec: np.ndarray,
    actions_shingles: Dict[str, Set[str]],
    temperature_c: float,
    bandit: HybridBandit,
) -> Tuple[str, int]:
    """
    Select an action given a context, using the fused η score to weight the
    bandit’s propensity.

    Returns
    -------
    chosen_action : str
        Identifier of the selected action.
    chosen_bucket : int
        The LSH bucket that was selected (useful for updates).
    """
    # Compute η and auxiliary values
    eta, _ = fuse_context_score(context_items, prototype_vec, temperature_c)

    # Cardinality adapts ε
    cardinality = hyperloglog_cardinality(context_items)
    adapted_eps = bandit.adapt_epsilon(cardinality)

    # Map actions to buckets
    action_to_bucket = {
        aid: minhash_bucket(shingles) for aid, shingles in actions_shingles.items()
    }
    buckets = set(action_to_bucket.values())

    # Select a bucket via the hybrid bandit
    chosen_bucket = bandit.select_bucket(buckets, eta, adapted_eps)

    # Resolve to a concrete action (pick first belonging to the bucket)
    for aid, b in action_to_bucket.items():
        if b == chosen_bucket:
            return aid, chosen_bucket

    # Fallback (should not happen)
    return random.choice(list(actions_shingles.keys())), chosen_bucket


def hybrid_update(
    bandit: HybridBandit,
    chosen_bucket: int,
    reward: float,
) -> None:
    """Update the bandit statistics after observing a reward."""
    bandit.update(chosen_bucket, reward)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Synthetic prototype vector (e.g., from a reference model)
    prototype = np.random.rand(256).astype(np.float64)

    # Synthetic context: a multiset of strings
    context = [f"event_{random.randint(0, 100)}" for _ in range(500)]

    # Define a small action space with shingle sets
    actions = {
        f"action_{i}": {f"term_{random.randint(0, 50)}" for _ in range(20)}
        for i in range(10)
    }

    # Instantiate the hybrid bandit
    hb = HybridBandit(epsilon=0.05)

    # Perform a single selection‑update cycle
    chosen_action, bucket = hybrid_action_selection(
        context_items=context,
        prototype_vec=prototype,
        actions_shingles=actions,
        temperature_c=22.0,
        bandit=hb,
    )
    print(f"Chosen action: {chosen_action} (bucket {bucket})")

    # Simulate a reward (e.g., 0‑1 feedback)
    simulated_reward = random.random()
    hybrid_update(hb, bucket, simulated_reward)

    # Show updated stats for the bucket
    stats = hb.bucket_stats[bucket]
    print(f"Updated bucket stats – pulls: {stats.pulls}, mean reward: {_pct(stats.mean_reward)}")