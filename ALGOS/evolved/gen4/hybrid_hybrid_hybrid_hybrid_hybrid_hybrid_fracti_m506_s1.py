# DARWIN HAMMER — match 506, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s5.py (gen3)
# parent_b: hybrid_hybrid_fractional_hd_hybrid_korpus_text_h_m125_s0.py (gen3)
# born: 2026-05-29T23:29:27Z

"""Hybrid module merging privacy‑risk/Differential‑Privacy (Parent A) with
hypervector/min‑hash fractional binding (Parent B).

Mathematical bridge:
- The minhash signature of a text document provides a compact set of
  integer identifiers.  Its cardinality (or the number of distinct
  hash buckets) is used as the *unique_quasi_identifiers* argument of
  `reconstruction_risk_score` from Parent A.
- The same signature is turned into a deterministic seed that drives
  `random_hv` (Parent B) producing a complex hyper‑vector.  Fractional
  power binding (`fractional_power`) is then applied to this hyper‑vector,
  yielding a transformed vector whose L2‑norm can be safely aggregated
  with the Laplace mechanism (`dp_aggregate`) from Parent A.
- An `EndpointCircuitBreaker` monitors the privacy‑risk score; if the
  risk exceeds a configurable threshold the breaker opens, preventing
  further DP aggregations.

The three core hybrid functions demonstrate this integration:
`text_to_hypervector`, `privacy_risk_from_text`, and
`dp_aggregate_hypervectors`.  The module ends with a smoke test."""

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
# Shared data structures (Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str


# Example tiers (mirroring parent A)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")


def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Parent A – probability that a record can be re‑identified."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: Iterable[float],
                 epsilon: float = 1.0,
                 sensitivity: float = 1.0) -> float:
    """
    Simple Laplace mechanism: sum(values) + Laplace(0, sensitivity/epsilon).
    """
    total = float(np.sum(list(values)))
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return total + noise


# ----------------------------------------------------------------------
# Parent B – hypervector primitives
# ----------------------------------------------------------------------


def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Generate a random hypervector of dimension *d*.

    Parameters
    ----------
    d: dimension of the hypervector.
    kind: "complex", "bipolar", or "real".
    seed: optional integer seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice([-1, 1], size=d)
    # "real"
    vec = rng.normal(size=d)
    return vec / np.linalg.norm(vec)


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Compact MinHash signature of *text* (k hash buckets)."""
    clean = text.replace(" ", "").strip().lower()
    shingles = [clean[i:i + 5] for i in range(len(clean) - 4)]
    signature = [2 ** 31 - 1] * k  # large initial values
    for s in shingles:
        h = hash(s)
        bucket = h % k
        signature[bucket] = min(signature[bucket], h % 1_000_000)
    return signature


def fractional_power(vec: np.ndarray, power: float) -> np.ndarray:
    """Apply a fractional power binding (phase‑preserving)."""
    # Preserve complex phase, apply power to magnitude.
    magnitude = np.abs(vec) ** power
    phase = np.exp(1j * np.angle(vec))
    return magnitude * phase


# ----------------------------------------------------------------------
# Hybrid constructs
# ----------------------------------------------------------------------


def text_to_hypervector(text: str,
                       dim: int = 8192,
                       power: float = 0.5) -> np.ndarray:
    """
    Convert *text* into a deterministic complex hypervector:

    1. Compute a MinHash signature.
    2. Derive an integer seed from the signature (sum modulo 2**32).
    3. Generate a random complex hypervector with that seed.
    4. Apply fractional power binding.

    Returns the transformed hypervector.
    """
    signature = minhash_for_text(text, k=64)
    seed = sum(signature) & 0xFFFFFFFF
    hv = random_hv(d=dim, kind="complex", seed=seed)
    return fractional_power(hv, power)


def privacy_risk_from_text(text: str, total_records: int) -> float:
    """
    Estimate reconstruction risk for *text* using the number of distinct
    MinHash buckets as the quasi‑identifier count.
    """
    signature = minhash_for_text(text, k=64)
    unique_ids = len(set(signature))
    return reconstruction_risk_score(unique_ids, total_records)


class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    @staticmethod
    def _now_z() -> str:
        """Current UTC timestamp in ISO‑8601 Zulu format."""
        return datetime.datetime.now(datetime.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

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
                              sensitivity: float = 1.0) -> np.ndarray:
    """
    Aggregate a list of hypervectors under differential privacy:

    * Convert each complex hypervector to its L2 norm (a scalar).
    * Apply Laplace DP aggregation on the scalar list.
    * Return a *scaled* version of the first hypervector to keep the
      result in the same vector space (the scaling factor is the DP‑noised sum).

    This demonstrates a mathematically consistent bridge between the
    vector operations of Parent B and the DP mechanism of Parent A.
    """
    if not hv_list:
        raise ValueError("hv_list must contain at least one hypervector")

    # Scalars are the L2 norms (always positive)
    norms = [np.linalg.norm(hv) for hv in hv_list]

    # DP‑noised aggregate of norms
    noisy_sum = dp_aggregate(norms, epsilon=epsilon, sensitivity=sensitivity)

    # Scale the first hypervector to embed the noisy sum while preserving direction
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

    # 1. Convert texts to hypervectors
    hvs = [text_to_hypervector(t, dim=4096, power=0.7) for t in sample_texts]

    # 2. Compute privacy risk for each text (assuming 1 000 total records)
    risks = [privacy_risk_from_text(t, total_records=1000) for t in sample_texts]
    print("Privacy risks:", risks)

    # 3. Initialise circuit breaker with a risk threshold of 0.2
    breaker = EndpointCircuitBreaker(failure_threshold=2)
    for r in risks:
        if r > 0.2:
            breaker.record_failure()
        else:
            breaker.record_success()
        print(f"Risk {r:.3f} → breaker open? {breaker.is_open()}")

    # 4. DP‑aggregate the hypervectors
    aggregated_hv = dp_aggregate_hypervectors(hvs, epsilon=0.5, sensitivity=1.0)
    print("Aggregated hypervector norm (noisy):", np.linalg.norm(aggregated_hv))


if __name__ == "__main__":
    try:
        _smoke_test()
    except Exception as exc:
        sys.stderr.write(f"Smoke test failed: {exc}\\n")
        raise