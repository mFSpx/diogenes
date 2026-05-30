# DARWIN HAMMER — match 5713, survivor 1
# gen: 6
# parent_a: hybrid_tri_algo_conduit_hybrid_cockpit_metri_m68_s2.py (gen2)
# parent_b: hybrid_fractional_hdc_hybrid_hybrid_hybrid_m629_s0.py (gen5)
# born: 2026-05-30T00:04:30Z

"""Hybrid Conduit‑Metric‑HDC Algorithm

Parents:
- **PARENT A** (`hybrid_tri_algo_conduit_hybrid_cockpit_metri_m68_s2.py`): computes signal and noise scores from raw byte data using entropy,
  status‑code, MIME type, keyword hits and structural links.
- **PARENT B** (`hybrid_fractional_hdc_hybrid_hybrid_hybrid_m629_s0.py`): builds random hypervectors, binds them with circular convolution,
  evaluates splits with a Hoeffding bound and a tropical max‑plus gain, and forms a regret term.

Mathematical Bridge:
The scalar *signal* (and *noise*) produced by Parent A is injected into the hyperdimensional space of
Parent B as a uniform hypervector.  Circular convolution (the binding operator of fractional HDC) fuses this
signal‑hypervector with a concept‑hypervector, producing a bound vector whose maximal component serves as a
*tropical gain* G.  A Hoeffding bound ε computed from the data length yields a regret term R = ε – G.
Both the confidence gap (signal – noise) and the regret term are then used to decide the final action,
thereby unifying the two topologies into a single decision engine.
"""

import math
import random
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – signal / noise utilities
# ----------------------------------------------------------------------


def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    """Shannon entropy of the first *sample* bytes, expressed in bytes."""
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(list(chunk)) / 8.0


def shannon_entropy(sequence: List[int]) -> float:
    """Classic Shannon entropy (bits) for a sequence of integer symbols."""
    seq_len = len(sequence)
    if seq_len == 0:
        return 0.0
    freq = {}
    for item in sequence:
        freq[item] = freq.get(item, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / seq_len
        entropy -= p * math.log(p, 2)
    return entropy


def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Tuple[float, float]:
    """Return (signal, noise) scores in [0,1] derived from data characteristics."""
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = (
        0.12
        if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml"))
        else 0.02
    )
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)

    signal = max(
        0.0,
        min(
            1.0,
            0.20
            + status_bonus
            + mime_bonus
            + size_bonus
            + keyword_bonus
            + structure_bonus
            + 0.12 * entropy,
        ),
    )
    noise = max(
        0.0,
        min(
            1.0,
            0.58
            - 0.22 * entropy
            - keyword_bonus
            - structure_bonus
            + (0.12 if size < 64 else 0.0),
        ),
    )
    return signal, noise


# ----------------------------------------------------------------------
# Parent B – hyperdimensional utilities
# ----------------------------------------------------------------------


def random_hv(dim: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Generate a random hypervector.
    *complex* kind returns a unit‑norm complex vector; *binary* returns ±1.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        angles = rng.random(dim) * 2 * math.pi
        hv = np.exp(1j * angles)  # unit complex numbers
        return hv
    elif kind == "binary":
        hv = rng.choice([-1, 1], size=dim)
        return hv.astype(float)
    else:
        raise ValueError(f"Unsupported kind: {kind}")


def circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binding via circular convolution using FFT (fractional HDC style)."""
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape for convolution")
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    return np.fft.ifft(fa * fb)


def hoeffding_epsilon(n: int, delta: float = 0.05) -> float:
    """Hoeffding bound ε = sqrt( ln(1/δ) / (2n) )."""
    if n <= 0:
        return float("inf")
    return math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def tropical_gain(vec: np.ndarray) -> float:
    """Tropical max‑plus gain: max_i (real(vec_i))."""
    return float(np.max(np.real(vec)))


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ConduitDecision:
    action: str
    confidence_gap: float
    epsilon: float
    signal_score: float
    noise_score: float
    dormancy_probability: float
    recovery_priority: float
    reason: str


# ----------------------------------------------------------------------
# Hybrid core functions (minimum three)
# ----------------------------------------------------------------------


def compute_signal_noise(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Tuple[float, float]:
    """Wrapper around Parent A's signal_scores."""
    return signal_scores(data, status_code, mime, keyword_hits, structural_links)


def bind_signal_to_concept(
    signal: float, concept: str, dim: int = 1024, seed: int | None = None
) -> np.ndarray:
    """Create a hypervector for *concept* and bind the scalar *signal* via circular convolution."""
    # Concept hypervector (complex unit‑norm)
    hv_concept = random_hv(dim=dim, kind="complex", seed=seed or hash(concept) & 0xFFFFFFFF)
    # Signal hypervector – uniform complex phase representing the scalar
    hv_signal = np.full(dim, signal, dtype=complex)
    bound = circular_convolution(hv_concept, hv_signal)
    return bound


def hybrid_decision(
    data: bytes,
    concept: str,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> ConduitDecision:
    """Fuse Parent A's metric evaluation with Parent B's HDC binding and regret engine."""
    # 1️⃣ Signal / noise from raw data
    signal, noise = compute_signal_noise(
        data, status_code, mime, keyword_hits, structural_links
    )
    confidence_gap = signal - noise

    # 2️⃣ Bind signal to concept in hyperdimensional space
    bound_vec = bind_signal_to_concept(signal, concept, dim=2048)

    # 3️⃣ Tropical gain from the bound vector
    gain = tropical_gain(bound_vec)

    # 4️⃣ Hoeffding bound based on data length
    epsilon = hoeffding_epsilon(len(data))

    # 5️⃣ Regret term (positive regret = higher dormancy probability)
    regret = max(0.0, epsilon - gain)

    # 6️⃣ Decision logic
    action = "accept" if confidence_gap > 0 else "reject"
    reason = (
        "Signal dominates noise and regret is low"
        if confidence_gap > 0 and regret < epsilon * 0.5
        else "Signal weak or regret high"
    )

    return ConduitDecision(
        action=action,
        confidence_gap=confidence_gap,
        epsilon=epsilon,
        signal_score=signal,
        noise_score=noise,
        dormancy_probability=regret,
        recovery_priority=gain,
        reason=reason,
    )


def evaluate_batch(
    batch: List[bytes],
    concept: str,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> List[ConduitDecision]:
    """Process a list of byte payloads, returning a decision per element."""
    decisions = []
    for data in batch:
        dec = hybrid_decision(
            data,
            concept,
            status_code,
            mime,
            keyword_hits,
            structural_links,
        )
        decisions.append(dec)
    return decisions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic payloads
    payloads = [
        b"<!DOCTYPE html><html><body>Hello World</body></html>",
        b'{"user":"alice","action":"login","status":"success"}',
        b"\x00\x01\x02\x03\x04\x05",
    ]

    # Run hybrid decisions
    results = evaluate_batch(
        payloads,
        concept="web_request",
        status_code=200,
        mime="application/json",
        keyword_hits=2,
        structural_links=1,
    )

    for i, res in enumerate(results, 1):
        print(f"Payload {i}: Action={res.action}, Confidence={res.confidence_gap:.3f}, "
              f"Epsilon={res.epsilon:.5f}, Regret={res.dormancy_probability:.5f}, Reason={res.reason}")