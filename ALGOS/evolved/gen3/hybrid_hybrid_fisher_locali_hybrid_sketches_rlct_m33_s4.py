# DARWIN HAMMER — match 33, survivor 4
# gen: 3
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s5.py (gen2)
# parent_b: hybrid_sketches_rlct_grokking_m5_s0.py (gen1)
# born: 2026-05-29T23:25:21Z

import numpy as np
import math
import random
import sys
import pathlib
import hashlib

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
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

def estimate_rlct_from_losses(train_losses_per_n, n_values):
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if len(losses) != len(ns):
        raise ValueError("train_losses_per_n and n_values must have the same length")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)

def hybrid_fisher_rlct(data, width=64, depth=4):
    sketch = count_min_sketch(data, width, depth)
    flat_sketch = [item for sublist in sketch for item in sublist]
    losses = [item for item in flat_sketch if item > 0]
    n_values = [i+1 for i in range(len(losses))]
    if len(losses) > 1:
        rlct = estimate_rlct_from_losses(losses, n_values)
    else:
        rlct = 0.0

    fisher_info = 0.0
    for theta in np.linspace(-1.0, 1.0, 100):
        fisher_info += fisher_score(theta, 0.0, 0.1)
    return rlct, fisher_info

def weighted_ssim(
    x: list,
    y: list,
    theta: float,
    center: float,
    width: float,
    dynamic_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    weights = [gaussian_beam(theta, center, width) for _ in range(len(x))]
    weights = np.array(weights) / sum(weights)

    mean_x = sum([x[i] * weights[i] for i in range(len(x))])
    mean_y = sum([y[i] * weights[i] for i in range(len(y))])
    var_x = sum([(x[i] - mean_x) ** 2 * weights[i] for i in range(len(x))])
    var_y = sum([(y[i] - mean_y) ** 2 * weights[i] for i in range(len(y))])
    cov_xy = sum([(x[i] - mean_x) * (y[i] - mean_y) * weights[i] for i in range(len(x))])

    if dynamic_range is None:
        dynamic_range = max(max(x), max(y)) - min(min(x), min(y))
    ssim = (2 * cov_xy + k1 * dynamic_range) / (var_x + var_y + k1 * dynamic_range)

    return ssim

def hybrid_sketch_ssim(data1, data2, width=64, depth=4):
    sketch1 = count_min_sketch(data1, width, depth)
    sketch2 = count_min_sketch(data2, width, depth)
    flat_sketch1 = [item for sublist in sketch1 for item in sublist]
    flat_sketch2 = [item for sublist in sketch2 for item in sublist]

    theta = 0.0
    center = 0.0
    width = 0.1
    ssim = weighted_ssim(flat_sketch1, flat_sketch2, theta, center, width)
    return ssim

def improved_hybrid_fisher_rlct_sketch_ssim(data1, data2, width=64, depth=4):
    rlct, fisher_info = hybrid_fisher_rlct(data1, width, depth)
    ssim = hybrid_sketch_ssim(data1, data2, width, depth)
    return rlct, fisher_info, ssim

def improved_count_min_sketch(items, width=64, depth=4):
    table = [[0]*width for _ in range(depth)]
    for item in items:
        for d in range(depth): 
            table[d][int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(),16)%width]+=1
    # Introduce an additional normalization step
    max_val = max(max(row) for row in table)
    if max_val > 0:
        table = [[val / max_val for val in row] for row in table]
    return table

def improved_hybrid_sketch_ssim(data1, data2, width=64, depth=4):
    sketch1 = improved_count_min_sketch(data1, width, depth)
    sketch2 = improved_count_min_sketch(data2, width, depth)
    flat_sketch1 = [item for sublist in sketch1 for item in sublist]
    flat_sketch2 = [item for sublist in sketch2 for item in sublist]

    theta = 0.0
    center = 0.0
    width = 0.1
    ssim = weighted_ssim(flat_sketch1, flat_sketch2, theta, center, width)
    return ssim

def improved_hybrid_fisher_rlct_sketch_ssim(data1, data2, width=64, depth=4):
    sketch1 = improved_count_min_sketch(data1, width, depth)
    flat_sketch1 = [item for sublist in sketch1 for item in sublist]
    losses = [item for item in flat_sketch1 if item > 0]
    n_values = [i+1 for i in range(len(losses))]
    if len(losses) > 1:
        rlct = estimate_rlct_from_losses(losses, n_values)
    else:
        rlct = 0.0

    fisher_info = 0.0
    for theta in np.linspace(-1.0, 1.0, 100):
        fisher_info += fisher_score(theta, 0.0, 0.1)

    sketch2 = improved_count_min_sketch(data2, width, depth)
    flat_sketch2 = [item for sublist in sketch2 for item in sublist]
    theta = 0.0
    center = 0.0
    width = 0.1
    ssim = weighted_ssim(flat_sketch1, flat_sketch2, theta, center, width)
    return rlct, fisher_info, ssim

if __name__ == "__main__":
    data1 = [random.random() for _ in range(100)]
    data2 = [random.random() for _ in range(100)]
    rlct, fisher_info, ssim = improved_hybrid_fisher_rlct_sketch_ssim(data1, data2)
    print("RLCT:", rlct)
    print("Fisher Information:", fisher_info)
    print("Weighted SSIM:", ssim)