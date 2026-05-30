# DARWIN HAMMER — match 5159, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2321_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m2511_s0.py (gen6)
# born: 2026-05-30T00:00:28Z

import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Sequence, Dict
import numpy as np
import hashlib

class CountMinSketch:
    def __init__(self, width: int = 1000, depth: int = 5):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=int)
        self.seeds = [random.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item, seed):
        return (hash((item, seed)) % self.width)

    def add(self, item):
        for d, seed in enumerate(self.seeds):
            idx = self._hash(item, seed)
            self.table[d, idx] += 1

    def estimate(self, item) -> int:
        return min(self.table[d, self._hash(item, seed)] for d, seed in enumerate(self.seeds))


class MinHashSketch:
    def __init__(self, num_perm: int = 64):
        self.num_perm = num_perm
        self.seeds = [random.randint(1, 2**31 - 1) for _ in range(num_perm)]
        self.minhash = [2**63 - 1] * num_perm

    def _hash(self, item, seed):
        return hash((item, seed))

    def add(self, item):
        for i, seed in enumerate(self.seeds):
            h = self._hash(item, seed)
            if h < self.minhash[i]:
                self.minhash[i] = h

    def signature(self) -> List[int]:
        return self.minhash


def estimate_rlct(losses: Sequence[float], sample_sizes: Sequence[int]) -> float:
    if len(losses) != len(sample_sizes) or len(losses) < 2:
        raise ValueError("Need at least two (loss, n) pairs")
    log_n = np.log(np.array(sample_sizes, dtype=float))
    log_L = np.log(np.array(losses, dtype=float) + 1e-12)  
    A = np.vstack([log_n, np.ones_like(log_n)]).T
    λ, _ = np.linalg.lstsq(A, log_L, rcond=None)[0]
    return λ


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    lead = path[:-1]
    lag = path[1:]
    return np.column_stack((lead, lag))


def compute_path_signature(path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    level1 = np.mean(path, axis=0, keepdims=True)  
    level2 = (path.T @ path) / path.shape[0]       
    return level1, level2


def signature_entropy(level2_sig: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(level2_sig)  
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)  
    probs = eigvals / eigvals.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def minhash_force_series(data: Sequence[float]) -> List[int]:
    return [int(hashlib.sha256(str(x).encode()).hexdigest(), 16) % (2**31 - 1) for x in data]


def integrate_force_series(force_series: Sequence[int]) -> float:
    cumulative = np.cumsum(force_series)
    velocities = np.diff(cumulative, prepend=0)
    peak_vel = float(np.max(np.abs(velocities)))
    return peak_vel


def hybrid_feature_vector(
    path: np.ndarray,
    ternary_labels: Sequence[int],
    text_tokens: Sequence[str],
    force_data: Sequence[float],
    losses: Sequence[float],
    sample_sizes: Sequence[int]
) -> Tuple[np.ndarray, float, float]:
    lvl1, lvl2 = compute_path_signature(path)
    H = signature_entropy(lvl2)

    label_counts = Counter(ternary_labels)
    label_vec = np.array([label_counts.get(-1, 0),
                          label_counts.get(0, 0),
                          label_counts.get(1, 0)], dtype=float)

    token_counts = Counter(text_tokens)
    top_k = 20
    most_common = token_counts.most_common(top_k)
    token_vec = np.array([cnt for _, cnt in most_common], dtype=float)
    if token_vec.size < top_k:
        token_vec = np.pad(token_vec, (0, top_k - token_vec.size), constant_values=0)

    force_series = minhash_force_series(force_data)
    peak_vel = integrate_force_series(force_series)

    Φ = np.concatenate([lvl1.ravel(), lvl2.ravel(), label_vec, token_vec, np.array([peak_vel])])

    λ_hat = estimate_rlct(losses, sample_sizes)
    γ = (1 + λ_hat) * (1 + H)
    return Φ, γ


def hybrid_ucb(
    action: str,
    sketch: CountMinSketch,
    total_counts: Dict[str, int],
    γ: float
) -> float:
    bonus = γ * np.sqrt(2 * np.log(sum(total_counts.values())) / sketch.estimate(action))
    return bonus


class HybridBandit:
    def __init__(self, 
                 width: int = 1000, 
                 depth: int = 5, 
                 num_perm: int = 64):
        self.sketch = CountMinSketch(width, depth)
        self.min_hash_sketch = MinHashSketch(num_perm)
        self.total_counts = defaultdict(int)
        self.losses = []
        self.sample_sizes = []

    def select_action(self, actions: List[str]) -> str:
        ucbs = []
        for action in actions:
            self.sketch.add(action)
            bonus = hybrid_ucb(action, self.sketch, dict(self.total_counts), 1.0)
            ucbs.append(bonus)
        return actions[np.argmax(ucbs)]

    def update(self, action: str, reward: float, n: int):
        self.total_counts[action] += 1
        self.losses.append(reward)
        self.sample_sizes.append(n)

    def get_feature_vector(self, 
                          path: np.ndarray, 
                          ternary_labels: Sequence[int], 
                          text_tokens: Sequence[str], 
                          force_data: Sequence[float]) -> Tuple[np.ndarray, float]:
        return hybrid_feature_vector(path, ternary_labels, text_tokens, force_data, self.losses, self.sample_sizes)


if __name__ == "__main__":
    bandit = HybridBandit()
    path = np.array([1, 2, 3])
    ternary_labels = [-1, 0, 1]
    text_tokens = ["token1", "token2"]
    force_data = [1.0, 2.0, 3.0]
    feature_vector, γ = bandit.get_feature_vector(path, ternary_labels, text_tokens, force_data)
    print(feature_vector)
    print(γ)