# DARWIN HAMMER — match 20, survivor 4
# gen: 3
# parent_a: hybrid_infotaxis_minhash_m63_s0.py (gen1)
# parent_b: hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s1.py (gen2)
# born: 2026-05-29T23:25:08Z

"""Hybrid Algorithm: Entropic MinHash with Chelydrid Strike Dynamics.

Parents:
- hybrid_infotaxis_minhash_m63_s0.py (Entropic MinHash)
- hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s1.py (Chelydrid strike‑drag model)

Mathematical Bridge:
The MinHash signature of a probability distribution is interpreted as a discrete
force series.  Each integer in the signature is scaled to a force magnitude and
fed to the Chelydrid strike integrator, which solves the drag‑limited equation of
motion.  The resulting peak velocity (a proxy for “selection cost”) modulates the
entropy‑based similarity between two distributions.  Thus the hybrid similarity
combines:
    • Jaccard‑like overlap of MinHash signatures,
    • Hamming‑distance‑derived drag coefficient,
    • Entropy of the probability vectors,
    • Peak velocity from the physics integration.  The final metric is a
      dimension‑less score in [0,1]."""

from __future__ import annotations

import math
import hashlib
import random
import sys
import pathlib
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A (Entropic MinHash)
# ----------------------------------------------------------------------


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability mass function."""
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0.0
    )


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used by MinHash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Standard MinHash signature."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def entropic_minhash(probabilities: List[float], k: int = 128) -> List[int]:
    """Generate a MinHash signature directly from a probability vector."""
    tokens = [f"{p:.12g}" for p in probabilities]
    return signature(tokens, k)


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Entropy after a Bernoulli observation with success probability p_hit."""
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


# ----------------------------------------------------------------------
# Core utilities from Parent B (Chelydrid strike dynamics)
# ----------------------------------------------------------------------


def compute_dhash(values: List[float]) -> int:
    """Bitwise hash encoding monotonic decreasing relationships."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return (a ^ b).bit_count()


def integrate_strike(
    force_series: Iterable[float],
    dt: float,
    m_head: float,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
    v0: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Simple 1‑D drag‑limited integration.
    Returns (final_velocity, total_distance, peak_velocity).
    """
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        x += v * dt
        peak = max(peak, v)
    return v, x, peak


def pulse_force(peak_force: float, steps: int) -> List[float]:
    """Triangular pulse of length `steps`."""
    if steps <= 0:
        return []
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [
        peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)
    ]


# ----------------------------------------------------------------------
# Hybrid layer: mapping MinHash signatures → force series & similarity
# ----------------------------------------------------------------------


def signature_to_force_series(sig: List[int], scale: float = 1e-6) -> List[float]:
    """
    Convert a MinHash signature to a deterministic force series.
    The low 16 bits of each hash entry are taken, scaled, and used as force.
    """
    forces = [(h & 0xFFFF) * scale for h in sig]
    # Normalise to have comparable magnitude across signatures
    max_f = max(forces) if forces else 1.0
    return [f / max_f for f in forces]


def strike_state_from_signature(sig: List[int]) -> Tuple[float, float, float]:
    """
    Run the Chelydrid strike integrator on the force series derived from a signature.
    Returns (final_velocity, distance, peak_velocity).
    """
    forces = signature_to_force_series(sig)
    # Use a modest time step; the number of steps equals the signature length.
    dt = 0.01
    return integrate_strike(forces, dt=dt, m_head=1.0)


def hybrid_similarity(
    probs_a: List[float],
    probs_b: List[float],
    k: int = 128,
    drag_weight: float = 0.5,
) -> float:
    """
    Unified similarity score between two probability distributions.

    Steps:
    1. Compute MinHash signatures (A).
    2. Jaccard‑like overlap `sim_j`.
    3. Compute dhash of each signature and turn Hamming distance into a
       drag coefficient `c_drag = 1 + drag_weight * (hamming / bits)`.
    4. Run the strike integration with that drag coefficient to obtain
       peak velocities `v_a`, `v_b`.
    5. Combine entropy, Jaccard overlap and the physics‑derived cost
       into a final metric in [0,1].
    """
    # 1. signatures
    sig_a = entropic_minhash(probs_a, k)
    sig_b = entropic_minhash(probs_b, k)

    # 2. Jaccard‑like similarity
    sim_j = similarity(sig_a, sig_b)

    # 3. Drag from Hamming distance
    bits = k - 1  # dhash produces k-1 bits
    dh_a = compute_dhash(sig_a)
    dh_b = compute_dhash(sig_b)
    ham = hamming_distance(dh_a, dh_b)
    c_drag = 1.0 + drag_weight * (ham / max(bits, 1))

    # 4. Strike integration with modified drag coefficient
    #   We inject `c_drag` by scaling the global drag coefficient in the physics model.
    v_a, _, peak_a = integrate_strike(
        signature_to_force_series(sig_a),
        dt=0.01,
        m_head=1.0,
        drag_cd=0.3 * c_drag,
    )
    v_b, _, peak_b = integrate_strike(
        signature_to_force_series(sig_b),
        dt=0.01,
        m_head=1.0,
        drag_cd=0.3 * c_drag,
    )
    avg_peak = (peak_a + peak_b) / 2.0

    # 5. Entropy component – treat `sim_j` as observation success probability
    ent = expected_entropy(sim_j, probs_a, probs_b)

    # Combine: high Jaccard, low drag (i.e. low ham), low entropy, low peak velocity → high score
    drag_factor = math.exp(-0.5 * (c_drag - 1.0))          # maps larger drag to smaller factor
    velocity_factor = math.exp(-0.3 * avg_peak)           # penalise large peak velocities
    score = sim_j * drag_factor * velocity_factor * (1.0 - ent)
    return max(0.0, min(1.0, score))


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------


def demo_signature():
    """Print a MinHash signature for a random probability vector."""
    probs = np.random.dirichlet(np.ones(5)).tolist()
    sig = entropic_minhash(probs, k=64)
    print("Probabilities:", probs)
    print("Signature (first 8 values):", sig[:8])


def demo_strike():
    """Show strike dynamics for a synthetic force series."""
    forces = pulse_force(peak_force=5.0, steps=50)
    v, x, peak = integrate_strike(forces, dt=0.02, m_head=0.8)
    print(f"Final velocity={v:.3f}, distance={x:.3f}, peak={peak:.3f}")


def demo_hybrid():
    """Compute hybrid similarity between two random distributions."""
    a = np.random.dirichlet(np.ones(7)).tolist()
    b = np.random.dirichlet(np.ones(7)).tolist()
    score = hybrid_similarity(a, b, k=96)
    print(f"Hybrid similarity score: {score:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: MinHash Signature ===")
    demo_signature()
    print("\n=== Demo: Chelydrid Strike ===")
    demo_strike()
    print("\n=== Demo: Hybrid Similarity ===")
    demo_hybrid()