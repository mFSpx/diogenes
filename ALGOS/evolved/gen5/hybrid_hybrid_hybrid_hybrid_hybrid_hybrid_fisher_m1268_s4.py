# DARWIN HAMMER — match 1268, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m1150_s0.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_caputo_fracti_m1185_s1.py (gen3)
# born: 2026-05-29T23:35:01Z

import math
import random
import sys
from pathlib import Path
import numpy as np

def hoeffding_bound(range_R: float, delta: float, n: int) -> float:
    if n <= 0:
        raise ValueError("sample size n must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt((range_R ** 2) * math.log(1.0 / delta) / (2.0 * n))

def tropical_max_plus(vector: np.ndarray) -> float:
    if vector.size == 0:
        raise ValueError("input vector must not be empty")
    return float(np.max(vector))

def tropical_linear(weights: np.ndarray, features: np.ndarray) -> float:
    if weights.shape != features.shape:
        raise ValueError("weights and features must have the same shape")
    return float(np.max(weights + features))

def fractional_decay(alpha: float, t: float) -> float:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if t < 0:
        raise ValueError("time t must be non-negative")
    return (t ** (alpha - 1)) / math.gamma(alpha)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    if x.shape != y.shape:
        raise ValueError("signals must have the same shape")
    if x.size == 0:
        raise ValueError("signals must not be empty")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)

def fused_edge_cost(weights: np.ndarray,
                    times: np.ndarray,
                    alpha: float,
                    ssim_val: float,
                    lambda_: float,
                    fisher: float,
                    hoeffding_B: float) -> float:
    if weights.shape != times.shape:
        raise ValueError("weights and times must have the same shape")
    decay = np.vectorize(fractional_decay)(alpha, times)
    weighted_decay = np.sum(weights * decay)
    cost = weighted_decay * (1.0 - ssim_val) + lambda_ * fisher * (1.0 + hoeffding_B)
    return cost

class HybridNode:
    def __init__(self,
                 feature_idx: int,
                 split_value: float,
                 left=None,
                 right=None,
                 weight: float = 1.0):
        self.feature_idx = feature_idx
        self.split_value = split_value
        self.left = left
        self.right = right
        self.weight = weight  

    def route(self, sample: np.ndarray) -> 'HybridNode':
        feature = sample[self.feature_idx]
        decision = tropical_max_plus(np.array([0.0, feature - self.split_value]))
        if decision > 0:
            return self.right if self.right else self
        else:
            return self.left if self.left else self

    def compute_cost(self,
                     sample: np.ndarray,
                     times: np.ndarray,
                     alpha: float,
                     lambda_: float,
                     delta: float,
                     range_R: float,
                     fisher_params: tuple) -> float:
        n = len(sample)
        hoeffding_B = hoeffding_bound(range_R, delta, n)
        theta, center, width = fisher_params
        fisher = fisher_score(theta, center, width)
        ssim_val = ssim(sample, np.zeros_like(sample))
        weights = np.ones_like(sample) * self.weight
        cost = fused_edge_cost(weights, times, alpha, ssim_val, lambda_, fisher, hoeffding_B)
        return cost

def generate_random_tree(depth: int, feature_count: int) -> HybridNode:
    if depth == 0:
        return HybridNode(feature_idx=random.randint(0, feature_count - 1), 
                           split_value=random.uniform(-10, 10), 
                           weight=random.uniform(0, 1))
    else:
        node = HybridNode(feature_idx=random.randint(0, feature_count - 1), 
                          split_value=random.uniform(-10, 10), 
                          weight=random.uniform(0, 1))
        node.left = generate_random_tree(depth - 1, feature_count)
        node.right = generate_random_tree(depth - 1, feature_count)
        return node

def traverse_tree(node: HybridNode, sample: np.ndarray) -> float:
    current_node = node
    cost = 0.0
    times = np.array([1.0])
    alpha = 1.5
    lambda_ = 0.1
    delta = 0.01
    range_R = 10.0
    fisher_params = (0.0, 0.0, 1.0)
    while True:
        next_node = current_node.route(sample)
        if next_node == current_node:
            cost += current_node.compute_cost(sample, times, alpha, lambda_, delta, range_R, fisher_params)
            break
        else:
            cost += current_node.compute_cost(sample, times, alpha, lambda_, delta, range_R, fisher_params)
            current_node = next_node
    return cost

# Usage
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    tree = generate_random_tree(5, 10)
    sample = np.random.rand(10)
    cost = traverse_tree(tree, sample)
    print("Cost:", cost)