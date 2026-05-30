# DARWIN HAMMER — match 4743, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_krampus_brain_m24_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_liquid_hybrid_path_signatur_m113_s1.py (gen3)
# born: 2026-05-29T23:57:59Z

from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – health & curvature
# ----------------------------------------------------------------------

@dataclass
class Morphology:
    """Simple morphology descriptor."""
    sphericity_index: float  # ≥0
    flatness_index: float    # ≥0


class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1

    def is_open(self) -> bool:
        return self.failures >= self.failure_threshold


def compute_health(endpoint: object, breaker: EndpointCircuitBreaker,
                  failures: int, threshold: int) -> float:
    """
    Health = (1 - failures/threshold) * (1 - recovery_priority)

    For this fusion we treat `recovery_priority` as the breaker open‑state
    (0 if closed, 1 if open).  The function is deliberately simple and
    defensive.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    failure_ratio = min(max(failures / threshold, 0.0), 1.0)
    recovery_priority = 1.0 if breaker.is_open() else 0.0
    health = (1.0 - failure_ratio) * (1.0 - recovery_priority)
    return max(min(health, 1.0), 0.0)


def compute_curvature_score(morph: Morphology, health: float) -> float:
    """
    curvature_score = health * (0.5 + 0.5 * tanh(morph_curvature))

    where morph_curvature = sphericity_index * flatness_index.
    The result lies in [0, 1].
    """
    morph_curvature = morph.sphericity_index * morph.flatness_index
    curv_factor = 0.5 + 0.5 * np.tanh(morph_curvature)  # Use numpy for vectorized operations
    return max(min(health * curv_factor, 1.0), 0.0)


# ----------------------------------------------------------------------
# Parent B – MinHash similarity, diffusion schedule, path signatures
# ----------------------------------------------------------------------

MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of a token with a seed."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def diffusion_amplitude_schedule(curv_score: float, sim: float,
                                 steps: int) -> np.ndarray:
    """
    Cosine‑shaped amplitude schedule modulated by curvature and similarity.

        a[t] = curv_score * (1 - sim) * 0.5 * (1 + cos(π·t/steps))

    Returns an array of length `steps`.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    t = np.arange(steps)
    schedule = curv_score * (1.0 - sim) * 0.5 * (1.0 + np.cos(np.pi * t / steps))  # Use numpy for vectorized operations
    return schedule


def generate_deterministic_path(curv_score: float, steps: int) -> np.ndarray:
    """
    Simple 3‑D helix whose radius is proportional to the curvature score.

    Returns an array of shape (steps, 3).
    """
    t = np.linspace(0, 2 * np.pi, steps)  # Use numpy's pi
    radius = curv_score  # ≤1, keeps the helix small
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = radius * t / (2 * np.pi)  # one turn per unit radius
    return np.stack([x, y, z], axis=1)


def apply_diffusion(path: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    """
    Add Gaussian noise with zero mean and std = amplitude[t] to each coordinate
    at timestep t.
    """
    if path.shape[0] != amplitude.shape[0]:
        raise ValueError("path length and amplitude schedule must match")
    noise = np.random.normal(loc=0.0, scale=amplitude[:, None], size=path.shape)
    return path + noise


def level_one_signature(path: np.ndarray) -> np.ndarray:
    """First level signature: total displacement (integral of velocity)."""
    if path.shape[0] < 2:
        return np.zeros(path.shape[1])
    increments = np.diff(path, axis=0)
    return increments.sum(axis=0)


def level_two_signature(path: np.ndarray) -> np.ndarray:
    """
    Second level signature (approximation):
    Σ_i (Δx_i ⊗ Δx_i) where ⊗ is the outer product.
    Returns a flattened vector of length d*d.
    """
    if path.shape[0] < 2:
        d = path.shape[1]
        return np.zeros(d * d)
    increments = np.diff(path, axis=0)  # (n-1, d)
    outer = np.einsum('ni,nj->ij', increments, increments)
    return outer.ravel()


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------

def hybrid_signature_pipeline(text_a: str, text_b: str,
                              endpoint: object,
                              breaker: EndpointCircuitBreaker,
                              failures: int,
                              threshold: int,
                              morph: Morphology,
                              steps: int = 256) -> Dict[str, np.ndarray]:
    """
    Full hybrid operation:
    1. Compute health and curvature score (Parent A).
    2. Compute MinHash similarity between the two texts (Parent B).
    3. Build a diffusion amplitude schedule using both scalar quantities.
    4. Generate a deterministic 3‑D path, perturb it, and compute level‑1
       and level‑2 signatures.
    Returns a dictionary with diagnostics and signatures.
    """
    # 1. Health & curvature
    health = compute_health(endpoint, breaker, failures, threshold)
    curv_score = compute_curvature_score(morph, health)

    # 2. MinHash similarity
    toks_a = text_a.lower().split()
    toks_b = text_b.lower().split()
    sig_a = minhash_signature(toks_a)
    sig_b = minhash_signature(toks_b)
    sim = similarity(sig_a, sig_b)

    # 3. Diffusion schedule
    amp_sched = diffusion_amplitude_schedule(curv_score, sim, steps)

    # 4. Path generation & diffusion
    base_path = generate_deterministic_path(curv_score, steps)
    noisy_path = apply_diffusion(base_path, amp_sched)

    # 5. Signatures
    sig_1 = level_one_signature(noisy_path)
    sig_2 = level_two_signature(noisy_path)

    return {
        'health': np.array([health]),
        'curvature_score': np.array([curv_score]),
        'similarity': np.array([sim]),
        'level_one_signature': sig_1,
        'level_two_signature': sig_2
    }