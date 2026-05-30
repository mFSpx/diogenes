# DARWIN HAMMER — match 3444, survivor 2
# gen: 5
# parent_a: korpus_text.py (gen0)
# parent_b: hybrid_hybrid_hybrid_cockpi_hybrid_hybrid_bandit_m509_s0.py (gen4)
# born: 2026-05-29T23:50:11Z

import numpy as np
import re
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import date
from collections import defaultdict

INT16_MAX = 32767

def shingles(text: str, width: int = 5) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    return [text[i:i+width] for i in range(len(text)-width+1)]

def murmurhash3(x: str) -> int:
    c = 0xcc9e2d51
    r = 0x1b87637f
    h1 = 0
    h2 = 0
    for b in bytes(x, 'utf-8'):
        k = b
        k *= c
        k &= 0xffffffff
        k ^= k >> r
        k *= c
        k &= 0xffffffff
        h1 ^= k
        h1 &= 0xffffffff
        h1 *= r
        h1 &= 0xffffffff
        h2 ^= k
        h2 &= 0xffffffff
        h2 *= c
        h2 &= 0xffffffff
    h1 ^= (len(x) & 0xffffffff)
    h1 &= 0xffffffff
    h2 ^= (len(x) & 0xffffffff)
    h2 &= 0xffffffff
    h1 ^= h2 & 0xffffffff
    h1 &= 0xffffffff
    h2 ^= h1 & 0xffffffff
    h2 &= 0xffffffff
    h1 ^= h2 & 0xffffffff
    h1 &= 0xffffffff
    h2 ^= h1 & 0xffffffff
    h2 &= 0xffffffff
    return h1 & 0xffffffff

def signature(shingles: List[str], k: int = 64) -> List[int]:
    return [murmurhash3(shingle) % k for shingle in shingles]

def shannon_entropy(text: List[str]) -> float:
    from collections import Counter
    counter = Counter(text)
    total = sum(counter.values())
    return -sum((count/total)*math.log2(count/total) for count in counter.values())

def hash_quantized_embedding(text: str) -> List[int]:
    return [ord(c) for c in text]

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

def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def minhash_for_text(text: str, k: int = 64) -> List[int]:
    return signature(shingles(text or "", width=5), k=k)

def entropy_for_text(text: str) -> float:
    return float(shannon_entropy(list((text or "")[:10000]))) if text else 0.0

def vector_literal(text: str) -> List[float]:
    embedding = hash_quantized_embedding(text or "")
    return [float(v) / float(INT16_MAX) for v in embedding]

def allocate_hybrid(
    total_units: float,
    date: date,
    text: str,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = ()
) -> Dict[str, float]:
    dow = date.weekday() + 1
    weight_vec = weekday_weight_vector(groups, dow % 7)
    minhash = minhash_for_text(text)
    entropy = entropy_for_text(text)
    vector = vector_literal(text)

    # Combine minhash, entropy, and vector into a single score
    score = np.dot(weight_vec, [entropy] + [sum(minhash)] + vector)

    deterministic_units = total_units * deterministic_target_pct / 100.0
    residual_units = total_units - deterministic_units
    residual_allocation = {group: residual_units * w for group, w in zip(groups, weight_vec)}
    allocation = {group: deterministic_units * w + residual_allocation.get(group, 0.0) for group in groups}
    
    # Normalize allocation to prevent negative values
    min_allocation = min(allocation.values())
    if min_allocation < 0:
        offset = -min_allocation
        allocation = {group: value + offset for group, value in allocation.items()}
    
    return allocation

_POLICY: Dict[str, List[float]] = defaultdict(list)

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        _POLICY[u.action_id].append(u.reward)

def get_policy(action_id: str) -> List[float]:
    return _POLICY.get(action_id, [])

def thompson_sampling_policy(action_ids: List[str], epsilon: float = 0.1) -> str:
    sampled_rewards = []
    for action_id in action_ids:
        rewards = get_policy(action_id)
        if rewards:
            sampled_reward = np.random.beta(1 + sum(rewards), 1 + len(rewards) - sum(rewards))
        else:
            sampled_reward = np.random.uniform(0, 1)
        sampled_rewards.append((action_id, sampled_reward))
    sampled_rewards.sort(key=lambda x: x[1], reverse=True)
    best_action_id = sampled_rewards[0][0]
    if np.random.uniform(0, 1) < epsilon:
        return random.choice(action_ids)
    return best_action_id

if __name__ == "__main__":
    text = "This is a sample text."
    date = date.today()
    groups = ("group1", "group2", "group3")
    total_units = 100.0
    allocation = allocate_hybrid(total_units, date, text, groups=groups)
    print(allocation)
    update = BanditUpdate("context1", "action1", 10.0, 0.5)
    update_policy([update])
    policy = get_policy("action1")
    print(policy)
    action_ids = ["action1", "action2", "action3"]
    best_action_id = thompson_sampling_policy(action_ids)
    print(best_action_id)