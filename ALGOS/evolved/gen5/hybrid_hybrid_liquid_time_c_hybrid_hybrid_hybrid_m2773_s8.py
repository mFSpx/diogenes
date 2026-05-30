# DARWIN HAMMER — match 2773, survivor 8
# gen: 5
# parent_a: hybrid_liquid_time_constant_minhash_m10_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m900_s0.py (gen4)
# born: 2026-05-29T23:45:50Z

import hashlib
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


# ----------------------------------------------------------------------
# Utility functions – shared by both parent algorithms
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """
    64‑bit Blake2b hash of ``seed`` concatenated with ``token``.
    The function returns an unsigned integer in the range ``[0, 2**64)``.
    """
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    # Clip to avoid overflow in exp
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------
# Ternary‑router‑style helpers
# ----------------------------------------------------------------------
def compute_phash(values: Sequence[int]) -> int:
    """
    Simple locality‑sensitive hash: for each of the first 64 values we set a bit
    to 1 if the value is greater than or equal to the median of the whole list.
    """
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Return the Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def pulse_force(peak_force: float, steps: int) -> List[float]:
    """
    Symmetric triangular pulse that peaks at ``peak_force``.
    The pulse has ``steps`` samples; the centre of the pulse is at ``(steps‑1)/2``.
    """
    if peak_force < 0 or steps <= 0:
        raise ValueError("peak_force must be non‑negative and steps positive")
    mid = (steps - 1) / 2.0
    if mid == 0:
        return [peak_force]
    # Normalise so that the maximum value equals ``peak_force``.
    return [
        peak_force * (1 - abs(i - mid) / mid) for i in range(steps)
    ]


# ----------------------------------------------------------------------
# Fusion primitives
# ----------------------------------------------------------------------
def burst_admission_model(edge_weights: List[float], burst_scores: List[float]) -> List[float]:
    """
    Additive fusion of raw edge weights with burst‑action scores.
    The two lists must have the same length.
    """
    if len(edge_weights) != len(burst_scores):
        raise ValueError("edge_weights and burst_scores must be of equal length")
    return [w + b for w, b in zip(edge_weights, burst_scores)]


def ternary_router_style(edge_weights: List[float], burst_scores: List[float]) -> List[float]:
    """
    Multiplicative fusion used by the original ternary router.
    The two lists must have the same length.
    """
    if len(edge_weights) != len(burst_scores):
        raise ValueError("edge_weights and burst_scores must be of equal length")
    return [w * b for w, b in zip(edge_weights, burst_scores)]


# ----------------------------------------------------------------------
# Core hybrid forward pass
# ----------------------------------------------------------------------
def hybrid_forward(
    sequence: Iterable[str],
    initial_state: List[float],
    alpha: float,
    *,
    peak_force: float = 1.0,
) -> List[float]:
    """
    Execute the hybrid Liquid‑Time‑Constant / Ternary‑Router pipeline.

    Parameters
    ----------
    sequence:
        Ordered collection of tokens (e.g. words, events).
    initial_state:
        Initial hidden state of the LTC network. Its dimensionality determines the
        dimensionality of the hidden trajectory.
    alpha:
        Coupling coefficient between the similarity signal and the LTC gating.
    peak_force:
        Maximum amplitude of the triangular pulse used as a base edge weight.

    Returns
    -------
    The final hidden state after processing the whole ``sequence``.
    """
    # ------------------------------------------------------------------
    # 1. Build a MinHash‑style signature for each prefix of the sequence.
    # ------------------------------------------------------------------
    prefixes: List[int] = []          # hash of each prefix (cumulative)
    signatures: List[int] = []        # 64‑bit LSH signature per prefix

    for i, token in enumerate(sequence, start=1):
        # Update cumulative hash (XOR works as a cheap order‑insensitive combiner)
        prefixes.append(_hash(0, token) ^ (prefixes[-1] if prefixes else 0))
        # Compute a 64‑bit signature from the set of prefix hashes seen so far
        signatures.append(compute_phash(prefixes))

    # ------------------------------------------------------------------
    # 2. Derive a similarity signal from successive signatures.
    #    We obtain a value in [0, 1] where 1 means identical signatures.
    # ------------------------------------------------------------------
    similarity: List[float] = []
    for a, b in zip(signatures[:-1], signatures[1:]):
        ham = hamming_distance(a, b)
        similarity.append(1.0 - ham / 64.0)   # 64‑bit signatures → max distance 64

    # Pad the similarity list so it aligns with the number of tokens.
    # The first token has no predecessor; we treat its similarity as 0.
    similarity = [0.0] + similarity

    # ------------------------------------------------------------------
    # 3. Initialise edge weights with a triangular pulse and fuse with the
    #    similarity‑derived burst scores using the two fusion stages.
    # ------------------------------------------------------------------
    raw_edge_weights = pulse_force(peak_force, len(sequence))
    # Stage‑1: additive burst admission
    admitted = burst_admission_model(raw_edge_weights, similarity)
    # Stage‑2: multiplicative ternary routing
    fused_weights = ternary_router_style(admitted, similarity)

    # ------------------------------------------------------------------
    # 4. Run the Liquid‑Time‑Constant recurrent dynamics.
    # ------------------------------------------------------------------
    tau_base = 1.0          # base time constant
    dt = 0.01               # integration step
    state = np.array(initial_state, dtype=float)

    # Ensure the hidden dimension matches the length of ``state``.
    # The edge‑weight signal is scalar; we broadcast it to the hidden dimension.
    for i, token in enumerate(sequence):
        # Gating function depends on the current hidden activation and similarity.
        gating = sigmoid(state.sum())
        s_t = similarity[i]

        # Effective time constant shrinks when gating or similarity is high.
        tau_eff = tau_base / (1.0 + tau_base * (gating + alpha * s_t))

        # Input term: fused edge weight (scalar) broadcast to hidden dimension.
        input_term = (gating + alpha * s_t) * fused_weights[i]

        # Continuous‑time update (Euler integration).
        dx = -(1.0 / tau_eff) * state + input_term
        state = state + dx * dt

    return state.tolist()


# ----------------------------------------------------------------------
# Simple sanity‑check when run as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_sequence = ["hello", "world", "python", "programming"]
    demo_initial = [0.0, 0.0, 0.0]   # 3‑dimensional hidden state
    demo_alpha = 0.5

    final_state = hybrid_forward(demo_sequence, demo_initial, demo_alpha)
    print("Final hidden state:", final_state)