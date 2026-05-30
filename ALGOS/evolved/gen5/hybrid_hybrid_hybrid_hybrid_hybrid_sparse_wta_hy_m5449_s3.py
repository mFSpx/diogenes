# DARWIN HAMMER — match 5449, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_ternary_router_m906_s4.py (gen4)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py (gen2)
# born: 2026-05-30T00:01:54Z

"""Hybrid module combining:

* Parent A – `hybrid_hybrid_hybrid_hard_t_ternary_router_m906_s4.py`:
  tokenization, simple routing via `mock_route_command`.
* Parent B – `hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py`:
  hash‑based sparse expansion, top‑k winner‑take‑all, Laplace‑based
  differential‑privacy aggregation and risk scaling.

Mathematical bridge:
1. Text is tokenised and converted to a dense count vector `v ∈ ℝⁿ`.
2. `v` is projected to a high‑dimensional sparse space `e = Expand(v, m)`
   using locality‑sensitive hashing (Parent B).
3. A binary top‑k mask `M` selects the `k` largest absolute entries of `e`
   (Winner‑Take‑All). The masked vector `ê = M ⊙ e` is summed to a scalar
   `s = Σ ê_i`.
4. `s` is perturbed with Laplace noise `η₁ ~ Laplace(0, Δ/ε₁)` to obtain a
   differentially‑private aggregate `ŝ = s + η₁`.
5. A risk score `ρ = (unique_quasi_identifiers / total_records) * |ŝ|`
   scales a second Laplace noise `η₂ ~ Laplace(0, ρ/ε₂)`. The final noisy
   value `z = ŝ + η₂` drives the routing decision.
6. The (potentially) privacy‑aware intent `z` is fed to the deterministic
   router `mock_route_command`, yielding the final channel.

The three public functions below illustrate the full pipeline, the
intermediate DP aggregation, and the final routing decision."""
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Tokenisation & deterministic routing (mocked)
# ----------------------------------------------------------------------
PUNCTUATION = list("!?;:,.—-()[]{}\"'`/\\|@#$%^")


def tokenize(text: str) -> List[str]:
    """Very simple whitespace + punctuation tokenizer."""
    tokens: List[str] = []
    current = []
    for ch in text:
        if ch.isspace() or ch in PUNCTUATION:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch.lower())
    if current:
        tokens.append("".join(current))
    return tokens


def vectorize(tokens: List[str]) -> List[float]:
    """Bag‑of‑words count vector (order of appearance)."""
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    # Preserve deterministic ordering by sorted token name
    ordered = [freq[t] for t in sorted(freq.keys())]
    return [float(v) for v in ordered]


class MockRouteResult:
    """Mimic the object returned by the real route_command."""
    def __init__(self, channel: str, state: str):
        self.channel = channel
        self.state = state

    def to_dict(self) -> Dict[str, Any]:
        return {"engine_channel": self.channel, "outbound_state": self.state}


def mock_route_command(text: str, intent: str, context: Dict[str, Any]) -> MockRouteResult:
    """
    Very lightweight stand‑in for the real FairyFuse routing routine.
    It simply returns a deterministic channel based on the intent hash.
    """
    h = hash(intent) & 0xFFFFFFFF
    if h % 3 == 0:
        channel = "cpu_fairyfuse_ternary"
    elif h % 3 == 1:
        channel = "gpu_fairyfuse_binary"
    else:
        channel = "fallback_cpu"
    return MockRouteResult(channel, "draft_only")


# ----------------------------------------------------------------------
# Parent B – Sparse Winner‑Take‑All & DP utilities
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Binary mask with 1 at indices of the top‑k absolute values."""
    k = max(0, min(k, len(values)))
    # sort by absolute magnitude, break ties by index
    winners = {
        i
        for i, _ in sorted(
            enumerate(values),
            key=lambda x: (-abs(x[1]), x[0])
        )[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]


def laplace_noise(scale: float) -> float:
    """Generate a single Laplace(0, scale) sample using inverse transform."""
    u = random.random() - 0.5
    return scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_expand_and_mask(
    text: str,
    m: int = 256,
    k: int = 10,
    salt: str = "hybrid"
) -> List[float]:
    """
    End‑to‑end: tokenise → vectorise → hash‑expand → top‑k mask → masked vector.
    Returns the masked high‑dimensional vector `ê`.
    """
    tokens = tokenize(text)
    dense = vectorize(tokens)
    expanded = expand(dense, m, salt)
    mask = top_k_mask(expanded, k)
    masked = [v * mask_i for v, mask_i in zip(expanded, mask)]
    return masked


def dp_aggregate_with_risk(
    masked_vector: List[float],
    epsilon1: float = 1.0,
    epsilon2: float = 0.5,
    total_records: int = 1000,
    unique_quasi_identifiers: int = 10
) -> float:
    """
    1. Sum the masked vector.
    2. Add Laplace noise with scale Δ/ε₁ (Δ = 1 for a unit‑sensitivity sum).
    3. Compute risk = (uqid / total) * |noisy_sum|.
    4. Add a second Laplace noise with scale = risk/ε₂.
    Returns the final noisy scalar used for routing.
    """
    if epsilon1 <= 0 or epsilon2 <= 0:
        raise ValueError("epsilons must be positive")
    raw_sum = float(np.sum(masked_vector))
    # First DP step
    eta1 = laplace_noise(scale=1.0 / epsilon1)
    noisy_sum = raw_sum + eta1
    # Risk scaling
    risk = (unique_quasi_identifiers / max(1, total_records)) * abs(noisy_sum)
    eta2 = laplace_noise(scale=risk / epsilon2) if risk > 0 else 0.0
    final = noisy_sum + eta2
    return final


def hybrid_route_decision(
    text: str,
    intent: str,
    context: Dict[str, Any] | None = None,
    *,
    m: int = 256,
    k: int = 10,
    epsilon1: float = 1.0,
    epsilon2: float = 0.5,
    total_records: int = 1000,
    unique_quasi_identifiers: int = 10,
    salt: str = "hybrid"
) -> MockRouteResult:
    """
    Full pipeline:
    * Build masked sparse representation.
    * Compute DP‑noised aggregate with risk scaling.
    * Derive a privacy‑aware intent by appending the noisy scalar.
    * Route via the deterministic mock router.
    """
    if context is None:
        context = {}
    masked = hybrid_expand_and_mask(text, m=m, k=k, salt=salt)
    noisy_scalar = dp_aggregate_with_risk(
        masked,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        total_records=total_records,
        unique_quasi_identifiers=unique_quasi_identifiers,
    )
    # Create a privacy‑aware intent string (purely illustrative)
    privacy_intent = f"{intent}|score={noisy_scalar:.4f}"
    return mock_route_command(text, privacy_intent, context)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog! Does it really?"
    sample_intent = "describe_animal_action"
    result = hybrid_route_decision(sample_text, sample_intent)
    print("Routing result:", result.to_dict())