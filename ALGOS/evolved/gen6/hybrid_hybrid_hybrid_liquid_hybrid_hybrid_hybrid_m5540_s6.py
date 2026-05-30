# DARWIN HAMMER — match 5540, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_liquid_time_c_hybrid_hybrid_korpus_m2174_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2368_s0.py (gen5)
# born: 2026-05-30T00:02:50Z

import numpy as np
import math
from pathlib import Path
from collections import Counter

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        __import__("hashlib").blake2b(data, digest_size=8).digest(), "big"
    )

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def minhash_signature(arr: list[str], k: int = 64) -> list[int]:
    signature = np.full(k, np.iinfo(np.int64).max, dtype=np.int64)
    for s in arr:
        h = _hash(0, s) % (2 ** 31 - 1)
        idx = h % k
        signature[idx] = min(signature[idx], h)
    return signature.tolist()

def ternary_from_signature(sig: list[int]) -> np.ndarray:
    arr = np.array(sig, dtype=np.int64)
    q1 = np.percentile(arr, 33)
    q2 = np.percentile(arr, 66)
    tern = np.where(arr <= q1, -1, np.where(arr >= q2, 1, 0))
    return tern

def shannon_entropy(ternary_vec: np.ndarray) -> float:
    counter = Counter(ternary_vec.tolist())
    total = sum(counter.values())
    entropy = 0.0
    for cnt in counter.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy

def ssim_index(x: np.ndarray, y: np.ndarray, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> float:
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator if denominator != 0 else 0.0

def voronoi_assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    if not seeds:
        raise ValueError("At least one seed required")
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        dists = [math.hypot(p[0] - s[0], p[1] - s[1]) for s in seeds]
        regions[int(np.argmin(dists))].append(p)
    return regions

def compute_ltc_hidden(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)

def entropy_modulation_factor(hidden: np.ndarray, k: int = 64) -> float:
    tokens = [f"{v:.6f}" for v in hidden]
    sig = minhash_signature(tokens, k=k)
    tern = ternary_from_signature(sig)
    ent = shannon_entropy(tern)
    max_ent = math.log2(3)
    return ent / max_ent if max_ent != 0 else 0.0

def hybrid_ltc_update(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    reference_inputs: list[np.ndarray],
    k: int = 64,
    num_seeds: int = 5,
) -> dict:
    h = compute_ltc_hidden(x, I, W, b)
    tau = entropy_modulation_factor(h, k=k)
    if reference_inputs:
        sims = [ssim_index(I, ref) for ref in reference_inputs]
        sigma = float(np.mean(sims))
    else:
        sigma = 1.0
    output = h * tau * sigma

    if h.size >= 2:
        points = [(float(h[i]), float(h[i + 1])) for i in range(0, h.size - 1, 2)]
    else:
        points = [(float(h[0]), float(h[0]))]

    rng = np.random.default_rng()
    seeds = [(float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))) for _ in range(num_seeds)]
    vor = voronoi_assign(points, seeds)

    return {
        "hidden": h,
        "output": output,
        "tau": tau,
        "sigma": sigma,
        "voronoi": vor,
    }

def generate_random_ltc_parameters(dim_x: int, dim_I: int, hidden_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    W = rng.normal(loc=0.0, scale=1.0, size=(hidden_dim, dim_x + dim_I))
    b = rng.normal(loc=0.0, scale=0.5, size=hidden_dim)
    x = rng.normal(loc=0.0, scale=1.0, size=dim_x)
    I = rng.normal(loc=0.0, scale=1.0, size=dim_I)
    return x, I, W, b

def improved_hybrid_ltc_update(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    reference_inputs: list[np.ndarray],
    k: int = 64,
    num_seeds: int = 5,
    alpha: float = 0.1
) -> dict:
    h = compute_ltc_hidden(x, I, W, b)
    sig_h = minhash_signature([f"{v:.6f}" for v in h], k=k)
    tern_h = ternary_from_signature(sig_h)
    ent_h = shannon_entropy(tern_h)
    max_ent_h = math.log2(3)
    tau_h = ent_h / max_ent_h if max_ent_h != 0 else 0.0

    if reference_inputs:
        sims = [ssim_index(I, ref) for ref in reference_inputs]
        sigma = float(np.mean(sims))
    else:
        sigma = 1.0

    output = h * (tau_h ** alpha) * sigma

    if h.size >= 2:
        points = [(float(h[i]), float(h[i + 1])) for i in range(0, h.size - 1, 2)]
    else:
        points = [(float(h[0]), float(h[0]))]

    rng = np.random.default_rng()
    seeds = [(float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))) for _ in range(num_seeds)]
    vor = voronoi_assign(points, seeds)

    return {
        "hidden": h,
        "output": output,
        "tau": tau_h ** alpha,
        "sigma": sigma,
        "voronoi": vor,
    }

# Test
if __name__ == "__main__":
    dim_x, dim_I, hidden_dim = 10, 10, 20
    x, I, W, b = generate_random_ltc_parameters(dim_x, dim_I, hidden_dim)
    reference_inputs = [np.random.normal(loc=0.0, scale=1.0, size=dim_I) for _ in range(5)]
    result = improved_hybrid_ltc_update(x, I, W, b, reference_inputs)
    print(result)