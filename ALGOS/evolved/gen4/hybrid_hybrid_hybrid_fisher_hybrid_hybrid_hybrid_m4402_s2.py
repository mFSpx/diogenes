# DARWIN HAMMER — match 4402, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_fisher_m375_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_hoeffding_tre_m16_s2.py (gen3)
# born: 2026-05-29T23:55:30Z

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Iterable
import hashlib

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def count_min_sketch(items, width=64, depth=4):
    table = [[0]*width for _ in range(depth)]
    for item in items:
        for d in range(depth): 
            table[d][int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(),16)%width]+=1
    return table

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

class HybridFisherCountMinSketch:
    def __init__(self, width=64, depth=4, center: float = 0.0, width_fisher: float = 1.0):
        self.width = width
        self.depth = depth
        self.center = center
        self.width_fisher = width_fisher

    def fit(self, items):
        self.table = count_min_sketch(items, self.width, self.depth)
        self.fisher_scores = []
        for d in range(self.depth):
            for w in range(self.width):
                theta = self.table[d][w]
                score = fisher_score(theta, self.center, self.width_fisher)
                self.fisher_scores.append(score)

    def get_fisher_scores(self):
        return self.fisher_scores

class HybridHoeffdingTropicalMaxPlus:
    def __init__(self, delta: float = 0.05, n: int = 100):
        self.delta = delta
        self.n = n

    def fit(self, tree):
        self.best_gain = 0.0
        self.second_best_gain = 0.0
        for node in tree:
            gain = node['gain']
            if gain > self.best_gain:
                self.second_best_gain = self.best_gain
                self.best_gain = gain
            elif gain > self.second_best_gain:
                self.second_best_gain = gain
        self.eps = hoeffding_bound(self.best_gain, self.delta, self.n)
        self.gap = self.best_gain - self.second_best_gain
        self.split = self.gap > self.eps
        self.reason = "gap_exceeds_bound" if self.gap > self.eps else "wait"

    def get_split(self):
        return self.split, self.eps, self.gap, self.reason

class HybridSketchFisherSimilarity:
    def __init__(self, width=64, depth=4, center: float = 0.0, width_fisher: float = 1.0):
        self.width = width
        self.depth = depth
        self.center = center
        self.width_fisher = width_fisher

    def fit(self, items1, items2):
        self.table1 = count_min_sketch(items1, self.width, self.depth)
        self.table2 = count_min_sketch(items2, self.width, self.depth)
        self.fisher_scores1 = []
        self.fisher_scores2 = []
        for d in range(self.depth):
            for w in range(self.width):
                theta1 = self.table1[d][w]
                theta2 = self.table2[d][w]
                score1 = fisher_score(theta1, self.center, self.width_fisher)
                score2 = fisher_score(theta2, self.center, self.width_fisher)
                self.fisher_scores1.append(score1)
                self.fisher_scores2.append(score2)
        self.similarity = np.corrcoef(self.fisher_scores1, self.fisher_scores2)[0, 1]

    def get_similarity(self):
        return self.similarity

if __name__ == "__main__":
    items1 = [random.randint(0, 100) for _ in range(100)]
    items2 = [random.randint(0, 100) for _ in range(100)]
    tree = [{'gain': random.random()} for _ in range(10)]

    fisher_sketch = HybridFisherCountMinSketch()
    fisher_sketch.fit(items1)
    fisher_scores = fisher_sketch.get_fisher_scores()

    hoeffding_tree = HybridHoeffdingTropicalMaxPlus()
    hoeffding_tree.fit(tree)
    split, eps, gap, reason = hoeffding_tree.get_split()

    sketch_similarity = HybridSketchFisherSimilarity()
    sketch_similarity.fit(items1, items2)
    similarity = sketch_similarity.get_similarity()

    print(f"Fishers scores: {fisher_scores}")
    print(f"Split: {split}, eps: {eps}, gap: {gap}, reason: {reason}")
    print(f"Similarity: {similarity}")