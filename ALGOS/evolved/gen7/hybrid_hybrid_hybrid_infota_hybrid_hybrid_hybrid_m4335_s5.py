# DARWIN HAMMER — match 4335, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s6.py (gen6)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_fisher_m2596_s2.py (gen4)
# born: 2026-05-29T23:54:59Z

"""
Hybrid Algorithm: MinHash‑Entropy Pheromone + Probabilistic‑Tropical Fusion

Parents
-------
* **Parent A** – provides a MinHash signature for a token set, computes the
  Shannon entropy of that signature and treats it as a pheromone signal that
  decays exponentially with a half‑life.
* **Parent B** – supplies probabilistic primitives (broadcast/acceptance
  probabilities, simulated annealing temperature) and a tropical algebra
  interface (max as tropical addition, add as tropical multiplication) together
  with a Hoeffding bound.

Mathematical Bridge
-------------------
The bridge is built in three stages:

1. **Entropy → Signal** – The signature entropy `H` becomes the initial pheromone
   strength `s₀`.  Its exponential decay `s(t) = s₀·½^{t/τ}` (τ = half‑life)
   supplies a time‑varying scalar.
2. **Probabilistic Modulation** – The decayed signal is multiplied by the
   broadcast probability `p_b` and the acceptance probability `p_a`
   (temperature‑controlled).  This yields an *expected* information value
   `E = s(t)·p_b·p_a`.
3. **Tropical Fusion with Hoeffding** – A Hoeffding bound `ε` on the token
   sample size is combined with `E` using tropical algebra:
   - tropical addition = `max`
   - tropical multiplication = `+`
   The tropical polynomial `P(ε) = max_i (c_i + i·ε)` (coefficients `c_i`) is
   then added to `E` (tropical multiplication) producing the final hybrid
   score `S = E + P(ε)`.

The resulting system couples information‑theoretic uncertainty, temporal decay,
probabilistic annealing, and a statistically‑rigorous confidence bound within a
single scalar decision metric.

"""

import hashlib
import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Set

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1
DEFAULT_K = 128
DEFAULT_HALF_LIFE = 60.0  # seconds

# ----------------------------------------------------------------------
# Parent A – MinHash & Pheromone
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used by the MinHash signature."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = DEFAULT_K) -> List[int]:
    """MinHash signature of a token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k

    mins = [MAX64] * k
    for seed in range(k):
        for t in toks:
            h = _hash(seed, t)
            if h < mins[seed]:
                mins[seed] = h
    return mins


def signature_entropy(sig: List[int]) -> float:
    """Shannon entropy (bits) of the multiset of signature values."""
    if not sig:
        return 0.0
    cnt = Counter(sig)
    total = len(sig)
    ent = -sum((c / total) * math.log(c / total, 2) for c in cnt.values())
    return ent


@dataclass
class Pheromone:
    """Exponential‑decay pheromone whose strength is an information measure."""
    signal: float
    half_life: float = DEFAULT_HALF_LIFE
    timestamp: float = 0.0

    def decay(self, now: float) -> None:
        """Apply exponential decay up to time ``now``."""
        elapsed = max(0.0, now - self.timestamp)
        factor = 0.5 ** (elapsed / self.half_life)
        self.signal *= factor
        self.timestamp = now


# ----------------------------------------------------------------------
# Parent B – Probabilistic primitives & Tropical algebra
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    if total_phases < 1 or current_phase < 1:
        raise ValueError("phases must be positive")
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0.0 <= alpha <= 1.0):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0.0 < delta < 1.0) or n <= 0:
        raise ValueError("invalid Hoeffding parameters")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def t_add(x, y):
    """Tropical addition = max."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication = ordinary addition."""
    return np.add(x, y)


def tropical_polynomial(coeffs: Iterable[float], x: float) -> float:
    """
    Evaluate a tropical polynomial P(x) = max_i (c_i + i·x).

    ``coeffs`` is an iterable of coefficients c_0, c_1, ..., c_n.
    """
    coeffs = list(coeffs)
    if not coeffs:
        return -np.inf
    values = [c + i * x for i, c in enumerate(coeffs)]
    return max(values)


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def hybrid_expected_entropy(
    tokens: Iterable[str],
    now: float,
    half_life: float,
    total_phases: int,
    current_phase: int,
    temperature: float,
) -> float:
    """
    Compute the expected entropy after applying probabilistic modulation.

    Steps:
    1. Build MinHash signature and its entropy H.
    2. Initialise a pheromone with strength H and decay it to ``now``.
    3. Multiply the decayed signal by broadcast and acceptance probabilities.
    """
    sig = signature(tokens)
    H = signature_entropy(sig)

    pher = Pheromone(signal=H, half_life=half_life, timestamp=now)
    pher.decay(now)  # no elapsed time on first call but keeps API consistent

    p_b = broadcast_probability(total_phases, current_phase)
    # Use the distance to maximal possible entropy as delta_e
    max_entropy = math.log2(MAX64)  # theoretical upper bound for 64‑bit values
    delta_e = max_entropy - H
    p_a = acceptance_probability(delta_e, temperature)

    expected = pher.signal * p_b * p_a
    return expected


def hybrid_tropical_score(
    tokens: Iterable[str],
    now: float,
    half_life: float,
    total_phases: int,
    current_phase: int,
    temperature: float,
    hoeffding_r: float = 1.0,
    hoeffding_delta: float = 0.05,
    poly_coeffs: Iterable[float] = (0.0, 1.0, 2.0),
) -> float:
    """
    Produce the final hybrid scalar score.

    1. Compute the expected entropy ``E`` via ``hybrid_expected_entropy``.
    2. Compute a Hoeffding bound ``ε`` on the sample size |tokens|.
    3. Evaluate a tropical polynomial ``P(ε)``.
    4. Combine ``E`` and ``P(ε)`` with tropical multiplication (ordinary addition).

    The result ``S = E + P(ε)`` is a single number that can be used for ranking,
    decision making, or as a reward signal.
    """
    E = hybrid_expected_entropy(
        tokens,
        now,
        half_life,
        total_phases,
        current_phase,
        temperature,
    )

    n = max(1, len(list(tokens)))  # avoid zero division
    epsilon = hoeffding_bound(hoeffding_r, hoeffding_delta, n)

    P = tropical_polynomial(poly_coeffs, epsilon)

    # Tropical multiplication = ordinary addition
    S = t_mul(E, P)  # returns E + P
    return float(S)


def update_pheromone_with_tokens(
    pher: Pheromone,
    new_tokens: Iterable[str],
    now: float,
    half_life: float,
) -> None:
    """
    Decay the existing pheromone to ``now`` and incorporate additional information
    from ``new_tokens`` by adding the entropy of the new token set to the current
    signal (before decay).  The half‑life may be updated to reflect a changed
    temporal scale.
    """
    pher.decay(now)
    added_entropy = signature_entropy(signature(new_tokens))
    pher.signal += added_entropy
    pher.half_life = half_life
    pher.timestamp = now


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)

    # Example token set
    tokens = [f"token_{i}" for i in range(1, 101)]

    # Temporal parameters
    start_time = 0.0
    later_time = 30.0  # seconds later

    # Hybrid score before any update
    score1 = hybrid_tropical_score(
        tokens=tokens,
        now=start_time,
        half_life=DEFAULT_HALF_LIFE,
        total_phases=5,
        current_phase=2,
        temperature=cooling_temperature(0),
    )
    print(f"Initial hybrid score: {score1:.4f}")

    # Initialise a pheromone from the same tokens
    initial_entropy = signature_entropy(signature(tokens))
    pher = Pheromone(signal=initial_entropy, half_life=DEFAULT_HALF_LIFE, timestamp=start_time)

    # Add new tokens after some time has passed
    new_tokens = [f"new_{i}" for i in range(1, 21)]
    update_pheromone_with_tokens(pher, new_tokens, now=later_time, half_life=DEFAULT_HALF_LIFE)

    # Re‑compute hybrid score with the updated pheromone (using its signal directly)
    # Here we bypass the full pipeline and demonstrate the modularity.
    expected = hybrid_expected_entropy(
        tokens=tokens + new_tokens,
        now=later_time,
        half_life=DEFAULT_HALF_LIFE,
        total_phases=5,
        current_phase=3,
        temperature=cooling_temperature(5),
    )
    print(f"Expected entropy after update: {expected:.4f}")

    # Final hybrid score after update
    score2 = hybrid_tropical_score(
        tokens=tokens + new_tokens,
        now=later_time,
        half_life=DEFAULT_HALF_LIFE,
        total_phases=5,
        current_phase=3,
        temperature=cooling_temperature(5),
    )
    print(f"Hybrid score after update: {score2:.4f}")

    # Verify that the pheromone signal reflects decay + added entropy
    pher.decay(later_time + 10)  # decay further 10 seconds
    print(f"Pheromone signal after extra decay: {pher.signal:.4f}")