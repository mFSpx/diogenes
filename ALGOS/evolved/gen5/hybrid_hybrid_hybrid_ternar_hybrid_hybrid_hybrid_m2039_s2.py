# DARWIN HAMMER — match 2039, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hoeffding_tre_m1040_s3.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m277_s0.py (gen4)
# born: 2026-05-29T23:40:34Z

import math
import numpy as np
from collections.abc import Iterable
import random
import sys
import pathlib
from typing import Tuple

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0: raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*sum(xs))

def ssim(x: np.ndarray, y: np.ndarray, C1: float, C2: float) -> float:
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.cov(x, y)[0, 1]
    
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2)
    
    return numerator / denominator

class CountMinSketch:
    def __init__(self, width: int, depth: int):
        self.width = width
        self.depth = depth
        self.table = [[0 for _ in range(width)] for _ in range(depth)]

    def update(self, item: int):
        for i in range(self.depth):
            index = hash(item) % self.width
            self.table[i][index] += 1

    def estimate(self, item: int) -> int:
        estimates = []
        for i in range(self.depth):
            index = hash(item) % self.width
            estimates.append(self.table[i][index])
        return min(estimates)

class HyperLogLog:
    def __init__(self, num_registers: int):
        self.num_registers = num_registers
        self.registers = [0] * num_registers

    def update(self, item: int):
        x = hash(item)
        register_index = x % self.num_registers
        register_value = x // self.num_registers
        self.registers[register_index] = max(self.registers[register_index], register_value.bit_length() - 1)

    def estimate(self) -> int:
        alpha = 0.7213 / (1 + 1.079 / self.num_registers)
        estimate = alpha * self.num_registers ** 2 / sum([2 ** -m for m in self.registers])
        return int(estimate)

def hybrid_hoeffding_ssim(values: Iterable[float], best_gain: float, second_best_gain: float, r: float, delta: float, n: int, 
                          prototype_vector: np.ndarray, tie_threshold: float = 0.05, C1: float = 1e-4, C2: float = 1e-4) -> Tuple[float, float]:
    eps = hoeffding_bound(r, delta, n)
    gini = gini_coefficient(values)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    
    ssim_result = ssim(np.array(list(values)), prototype_vector, C1, C2)
    
    combined_result = (1 if gap > eps else 0) * (1 if ssim_result > 0.5 else 0)
    
    return combined_result, ssim_result

def hybrid_countmin_hyperloglog(values: Iterable[int], width: int, depth: int, num_registers: int) -> Tuple[int, int]:
    countmin = CountMinSketch(width, depth)
    hyperloglog = HyperLogLog(num_registers)
    
    for value in values:
        countmin.update(value)
        hyperloglog.update(value)
    
    countmin_estimate = countmin.estimate(next(iter(values)))
    hyperloglog_estimate = hyperloglog.estimate()
    
    return countmin_estimate, hyperloglog_estimate

def hybrid_hoeffding_hyperloglog(values: Iterable[float], best_gain: float, second_best_gain: float, r: float, delta: float, n: int, 
                                 num_registers: int) -> Tuple[float, int]:
    eps = hoeffding_bound(r, delta, n)
    gini = gini_coefficient(values)
    gap = best_gain - second_best_gain
    
    hyperloglog = HyperLogLog(num_registers)
    for value in map(int, values):
        hyperloglog.update(value)
    
    hyperloglog_estimate = hyperloglog.estimate()
    
    combined_result = 1 if gap > eps and hyperloglog_estimate > 100 else 0
    
    return combined_result, hyperloglog_estimate

if __name__ == "__main__":
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    prototype_vector = np.array([2.5, 2.5])
    best_gain = 10.0
    second_best_gain = 5.0
    r = 0.1
    delta = 0.01
    n = 1000
    tie_threshold = 0.05
    C1 = 1e-4
    C2 = 1e-4
    width = 100
    depth = 10
    num_registers = 100
    
    print(hybrid_hoeffding_ssim(values, best_gain, second_best_gain, r, delta, n, prototype_vector, tie_threshold, C1, C2))
    print(hybrid_countmin_hyperloglog(map(int, values), width, depth, num_registers))
    print(hybrid_hoeffding_hyperloglog(values, best_gain, second_best_gain, r, delta, n, num_registers))