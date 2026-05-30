# DARWIN HAMMER — match 5606, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m926_s0.py (gen5)
# parent_b: hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s0.py (gen4)
# born: 2026-05-30T00:03:25Z

import sys
import math
import random
import hashlib
import pathlib
from collections import defaultdict
from typing import Iterable, List, Tuple, Dict

import numpy as np

def count_min_sketch(
    items: Iterable[str],
    width: int = 64,
    depth: int = 4,
    seed: int = 42
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        h = hashlib.md5(item.encode("utf-8")).digest()
        base = int.from_bytes(h, byteorder="big")
        for d in range(depth):
            index = (base + d * 0x9e3779b9) % width
            table[d][index] += 1
    return table

def sketch_to_distribution(
    sketch: List[List[int]],
    eps: float = 1e-12,
) -> np.ndarray:
    flat = np.array(sketch, dtype=np.float64).flatten()
    total = flat.sum()
    if total < eps:
        return np.full_like(flat, 1.0 / flat.size)
    prob = flat / total
    prob = np.maximum(prob, eps)
    return prob / prob.sum()

def fisher_information_from_sketch(
    prob_vec: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    inv = 1.0 / np.maximum(prob_vec, eps)
    return np.diag(inv)

def jepa_energy(
    latent: np.ndarray,
    mu: np.ndarray,
    fisher: np.ndarray,
) -> float:
    diff = latent - mu
    return 0.5 * diff.T @ fisher @ diff

def jepa_energy_gradient(
    latent: np.ndarray,
    mu: np.ndarray,
    fisher: np.ndarray,
) -> np.ndarray:
    return fisher @ (latent - mu)

def hybrid_update(
    latent: np.ndarray,
    sketch: List[List[int]],
    new_items: Iterable[str],
    lr: float = 0.1,
    clip_grad: float = 1.0
) -> Tuple[np.ndarray, List[List[int]]]:
    updated_sketch = count_min_sketch(new_items, width=len(sketch[0]), depth=len(sketch))
    for d in range(len(sketch)):
        for w in range(len(sketch[0])):
            sketch[d][w] += updated_sketch[d][w]

    mu = sketch_to_distribution(sketch)
    F = fisher_information_from_sketch(mu)

    grad = jepa_energy_gradient(latent, mu, F)
    grad_norm = np.linalg.norm(grad)
    if grad_norm > clip_grad:
        grad = grad / grad_norm * clip_grad
    latent_new = latent - lr * grad

    return latent_new, sketch

def latent_initialization(dim: int = 768, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim)

def run_hybrid_cycle(
    initial_items: List[str],
    steps: int = 5,
    latent_dim: int = 768,
) -> np.ndarray:
    sketch = count_min_sketch(initial_items)
    latent = latent_initialization(dim=latent_dim)

    for step in range(steps):
        new_items = [f"event_{random.randint(0, 1000)}" for _ in range(20)]
        latent, sketch = hybrid_update(latent, sketch, new_items, lr=0.05)

    return latent

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-15
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    return np.sum(p * np.log(p / q))

def symmetric_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * (kl_divergence(p, q) + kl_divergence(q, p))

def evaluate_sketch_consistency(sketch: List[List[int]], prob_vec: np.ndarray) -> float:
    flat_sketch = np.array(sketch, dtype=np.float64).flatten()
    flat_sketch /= flat_sketch.sum()
    return symmetric_kl_divergence(prob_vec, flat_sketch)

def run_hybrid_cycle_with_evaluation(
    initial_items: List[str],
    steps: int = 5,
    latent_dim: int = 768,
) -> Tuple[np.ndarray, List[float]]:
    sketch = count_min_sketch(initial_items)
    latent = latent_initialization(dim=latent_dim)
    consistencies = []

    for step in range(steps):
        new_items = [f"event_{random.randint(0, 1000)}" for _ in range(20)]
        latent, sketch = hybrid_update(latent, sketch, new_items, lr=0.05)
        prob_vec = sketch_to_distribution(sketch)
        consistencies.append(evaluate_sketch_consistency(sketch, prob_vec))

    return latent, consistencies

if __name__ == "__main__":
    demo_items = [f"init_{i}" for i in range(30)]
    final_latent, consistencies = run_hybrid_cycle_with_evaluation(demo_items, steps=3, latent_dim=128)

    print("Hybrid JEPA latent vector (first 8 components):")
    print(final_latent[:8])
    print("Norm of latent vector:", np.linalg.norm(final_latent))
    print("Sketch consistencies:", consistencies)
    sys.exit(0)