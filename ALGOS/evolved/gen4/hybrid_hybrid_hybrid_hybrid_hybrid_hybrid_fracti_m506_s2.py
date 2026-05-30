# DARWIN HAMMER — match 506, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s5.py (gen3)
# parent_b: hybrid_hybrid_fractional_hd_hybrid_korpus_text_h_m125_s0.py (gen3)
# born: 2026-05-29T23:29:27Z

from __future__ import annotations

import sys
import random
import math
import pathlib
import datetime
from dataclasses import dataclass, asdict
from typing import Iterable, List

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")

def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_aggregate(values: Iterable[float],
                 epsilon: float = 1.0,
                 sensitivity: float = 1.0) -> float:
    total = float(np.sum(list(values)))
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return total + noise

# ----------------------------------------------------------------------
# Hypervector primitives
# ----------------------------------------------------------------------

def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice([-1, 1], size=d)
    vec = rng.normal(size=d)
    return vec / np.linalg.norm(vec)

def minhash_for_text(text: str, k: int = 64) -> List[int]:
    clean = text.replace(" ", "").strip().lower()
    shingles = [clean[i:i + 5] for i in range(len(clean) - 4)]
    signature = [2 ** 31 - 1] * k  
    for s in shingles:
        h = hash(s)
        bucket = h % k
        signature[bucket] = min(signature[bucket], h % 1_000_000)
    return signature

def fractional_power(vec: np.ndarray, power: float) -> np.ndarray:
    magnitude = np.abs(vec) ** power
    phase = np.exp(1j * np.angle(vec))
    return magnitude * phase

# ----------------------------------------------------------------------
# Hybrid constructs
# ----------------------------------------------------------------------

def text_to_hypervector(text: str,
                       dim: int = 8192,
                       power: float = 0.5,
                       k: int = 64) -> np.ndarray:
    signature = minhash_for_text(text, k=k)
    seed = sum(signature) & 0xFFFFFFFF
    hv = random_hv(d=dim, kind="complex", seed=seed)
    return fractional_power(hv, power)

def privacy_risk_from_text(text: str, total_records: int, k: int = 64) -> float:
    signature = minhash_for_text(text, k=k)
    unique_ids = len(set(signature))
    return reconstruction_risk_score(unique_ids, total_records)

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    @staticmethod
    def _now_z() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = self._now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = self._now_z()
        if self.failures >= self.failure_threshold:
            self.open = True

    def is_open(self) -> bool:
        return self.open

def dp_aggregate_hypervectors(hv_list: List[np.ndarray],
                              epsilon: float = 1.0,
                              sensitivity: float = 1.0,
                              k: int = 64) -> np.ndarray:
    if not hv_list:
        raise ValueError("hv_list must contain at least one hypervector")

    norms = [np.linalg.norm(hv) for hv in hv_list]
    noisy_sum = dp_aggregate(norms, epsilon=epsilon, sensitivity=sensitivity)

    scale_factor = noisy_sum / np.linalg.norm(hv_list[0])
    return hv_list[0] * scale_factor

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------

def _smoke_test() -> None:
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Data privacy and hyperdimensional computing intersect."
    ]

    hvs = [text_to_hypervector(t, dim=4096, power=0.7, k=64) for t in sample_texts]
    risks = [privacy_risk_from_text(t, total_records=1000, k=64) for t in sample_texts]
    aggregated_hv = dp_aggregate_hypervectors(hvs, epsilon=1.0, sensitivity=1.0, k=64)

    print("Hypervector shapes:", [hv.shape for hv in hvs])
    print("Privacy risks:", risks)
    print("Aggregated hypervector shape:", aggregated_hv.shape)

_smoke_test()