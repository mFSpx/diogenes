# DARWIN HAMMER — match 5077, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1512_s5.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_geomet_m429_s3.py (gen4)
# born: 2026-05-29T23:59:40Z

"""
Hybrid Algorithm: Fusing DARWIN HAMMER (match 1512, survivor 5) and 
                 DARWIN HAMMER (match 429, survivor 3)

This hybrid algorithm combines the perceptual hashing and MinHash 
signatures from the first parent with the bandit routing and 
geometric operations from the second parent. The mathematical 
bridge between the two parents lies in the use of similarity 
measures (Hamming distance and Jaccard similarity) to inform 
the bandit routing policy.

The hybrid algorithm uses the perceptual hash and MinHash 
signatures to compute node similarities, which are then used 
as rewards in the bandit routing policy. The policy is updated 
based on the rewards, and the best action is selected using 
the LinUCB algorithm.

The governing equations of both parents are integrated through 
the use of the node similarity measure, which is computed 
using the perceptual hash and MinHash signatures. The matrix 
operations from the second parent (e.g., distance, nearest, 
assign) are used to compute the rewards and update the policy.
"""

import numpy as np
import math
import random
import sys
import hashlib
from pathlib import Path
from collections import Counter
from typing import Mapping, Hashable, Set, List, Dict, Tuple

Node = Hashable
Graph = Mapping[Node, Set[Node]]
ValuesMap = Mapping[Node, List[float]]
TokensMap = Mapping[Node, Set[str]]

def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")

def minhash_signature(tokens: Set[str], num_hashes: int = 7) -> List[int]:
    signatures: List[int] = []
    for seed in range(num_hashes):
        def hash_fn(x: str) -> int:
            return int(hashlib.md5((x + str(seed)).encode()).hexdigest(), 16)

        hashed = [hash_fn(tok) for tok in tokens]
        signatures.append(min(hashed) if hashed else 0)
    return signatures

def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)

def node_similarity(
    phash_a: int,
    phash_b: int,
    sig_a: List[int],
    sig_b: List[int],
) -> float:
    ham_sim = 1.0 - hamming_distance(phash_a, phash_b) / 64.0
    mh_sim = jaccard_similarity(sig_a, sig_b)
    return 0.5 * ham_sim + 0.5 * mh_sim

class HybridBanditRouter:
    def __init__(self):
        self._POLICY = {}

    def reset_policy(self):
        self._POLICY.clear()

    def update_policy(self, updates):
        for u in updates:
            s = self._POLICY.setdefault(u.action_id, [0.0, 0.0])
            s[0] += float(u.reward)
            s[1] += 1.0

    def _reward(self, a: str) -> float:
        total, n = self._POLICY.get(a, [0.0, 0.0])
        return total / n if n else 0.0

    def select_action(self, context: dict[str, float], actions: list[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> dict:
        if not actions:
            raise ValueError('actions required')
        rng = random.Random(seed)
        if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
            chosen = rng.choice(actions)
        elif algorithm == 'thompson':
            chosen = max(actions, key=lambda a: rng.betavariate(1 + max(0, self._reward(a)), 1 + max(0, 1 - self._reward(a))))
        else:
            scale = np.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0
            chosen = max(actions, key=lambda a: self._reward(a) + 0.1 * scale / np.sqrt(1 + self._POLICY.get(a, [0, 0])[1]))
        return {'action_id': chosen, 'propensity': 1.0 / len(actions), 'expected_reward': self._reward(chosen), 'confidence_bound': 1.0 / np.sqrt(1 + self._POLICY.get(chosen, [0, 0])[1])}

def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Tuple[float, float], seeds: list[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Tuple[float, float]], seeds: list[Tuple[float, float]]) -> dict[int, list[Tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def hybrid_operation(points: list[Tuple[float, float]], seeds: list[Tuple[float, float]], 
                      values: List[float], tokens: Set[str]) -> dict:
    phash = compute_phash(values)
    sig = minhash_signature(tokens)
    similarities = []
    for seed in seeds:
        point = (seed[0], seed[1])
        dist = distance(point, (0, 0))
        sim = node_similarity(phash, compute_phash([dist]), sig, minhash_signature(tokens))
        similarities.append(sim)
    router = HybridBanditRouter()
    router.update_policy([type('Update', (), {'action_id': i, 'reward': sim}) for i, sim in enumerate(similarities)])
    action = router.select_action({}, list(range(len(seeds))))
    return assign(points, [seeds[i] for i in range(len(seeds)) if i != action['action_id']])

if __name__ == "__main__":
    points = [(1, 2), (3, 4), (5, 6)]
    seeds = [(0, 0), (10, 10), (20, 20)]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    tokens = {"token1", "token2", "token3"}
    result = hybrid_operation(points, seeds, values, tokens)
    print(result)