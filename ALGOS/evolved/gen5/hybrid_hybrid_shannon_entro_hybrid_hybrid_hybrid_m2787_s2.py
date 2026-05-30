# DARWIN HAMMER — match 2787, survivor 2
# gen: 5
# parent_a: hybrid_shannon_entropy_rsa_cipher_m51_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m2159_s0.py (gen4)
# born: 2026-05-29T23:45:53Z

"""Hybrid module merging Shannon‑entropy‑RSA (parent A) with Pheromone‑NLMS dynamics (parent B).

Mathematical bridge
-------------------
* The pheromone field provides a time‑varying vector **s** = (s₁,…,s_k) of
  signal strengths (real numbers ≥0).
* By scaling with a common factor **Q** we obtain integer masses **mᵢ = ⌊Q·sᵢ⌉**.
* RSA encryption **E(m) = mᵉ mod n** is a permutation of ℤₙ*; therefore the
  multiset {mᵢ} is bijectively mapped to {m'ᵢ = E(mᵢ)} while preserving the
  total mass modulo *n*.
* Normalising the encrypted masses yields a new probability distribution
  **p'**, whose Shannon entropy **H(p')** can be compared with the original
  entropy **H(p)** of the unencrypted pheromone distribution.
* The sequence of encrypted‑entropy values is fed to an NLMS adaptive filter,
  which predicts the next entropy value and updates its weight vector.
* The predictor’s error drives a feedback that adaptively rescales the
  pheromone half‑life, closing the loop between information‑theoretic,
  number‑theoretic and adaptive‑signal‑processing components.

The implementation below realises this fusion with three public functions
that expose the hybrid operation:
`hybrid_entropy_step`, `hybrid_predict_entropy` and `hybrid_update_pheromone`. """

from __future__ import annotations

import math
import random
import sys
import pathlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# RSA utilities (from parent A)
# ----------------------------------------------------------------------
def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm."""
    if a == 0:
        return b, 0, 1
    g, y, x = _egcd(b % a, a)
    return g, x - (b // a) * y, y


def _modinv(a: int, m: int) -> int:
    """Modular inverse of a modulo m."""
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % m


def generate_rsa_keypair(prime_bits: int = 16) -> Tuple[int, int, int]:
    """Generate a tiny RSA keypair (e, d, n) for demonstration purposes."""
    def _rand_prime(bits: int) -> int:
        while True:
            p = random.getrandbits(bits) | 1
            if all(p % q for q in range(3, int(math.isqrt(p)) + 1, 2)):
                return p

    p = _rand_prime(prime_bits)
    q = _rand_prime(prime_bits)
    n = p * q
    phi = (p - 1) * (q - 1)

    e = 65537
    while math.gcd(e, phi) != 1:
        e += 2
    d = _modinv(e, phi)
    return e, d, n


def rsa_encrypt_int(message: int, e: int, n: int) -> int:
    """RSA encryption of a single integer."""
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)


def rsa_decrypt_int(ciphertext: int, d: int, n: int) -> int:
    """RSA decryption of a single integer."""
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)


# ----------------------------------------------------------------------
# Shannon entropy utilities (core of parent A)
# ----------------------------------------------------------------------
def shannon_entropy(probs: List[float]) -> float:
    """Return the base‑2 Shannon entropy of a probability vector."""
    H = 0.0
    for p in probs:
        if p > 0.0:
            H -= p * math.log2(p)
    return H


def normalize(values: List[float]) -> List[float]:
    """Normalise a list of non‑negative numbers to a probability distribution."""
    total = sum(values)
    if total == 0.0:
        raise ValueError("cannot normalize a zero‑vector")
    return [v / total for v in values]


# ----------------------------------------------------------------------
# NLMS predictor (core of parent B)
# ----------------------------------------------------------------------
class NLMSPredictor:
    """A simple Normalised Least‑Mean‑Squares adaptive filter."""
    def __init__(self, order: int, mu: float = 0.5, eps: float = 1e-8):
        self.order = order
        self.mu = mu
        self.eps = eps
        self.w = np.zeros(order)          # weight vector
        self.buffer = np.zeros(order)     # most recent inputs

    def predict(self, new_sample: float) -> float:
        """Predict the next value given a new sample, then update the filter."""
        # shift buffer, insert new sample at position 0
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = new_sample

        # linear prediction
        y = float(np.dot(self.w, self.buffer))

        # error is defined as (desired – prediction). Here we treat the
        # current buffer[0] as the desired signal for the next step.
        d = new_sample
        e = d - y

        # Normalised LMS weight update
        norm_factor = self.eps + np.dot(self.buffer, self.buffer)
        self.w += (self.mu / norm_factor) * e * self.buffer
        return y


# ----------------------------------------------------------------------
# Pheromone system (from parent B, trimmed and made compatible)
# ----------------------------------------------------------------------
class PheromoneField:
    """Manages pheromone signals with exponential decay."""
    def __init__(self):
        self.pheromones: Dict[str, Dict] = {}

    def calculate_signal(self,
                         surface_key: str,
                         signal_kind: str,
                         signal_value: float,
                         half_life_seconds: float) -> float:
        """Create or update a pheromone entry, applying exponential decay."""
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
        else:
            entry = self.pheromones[surface_key]
            elapsed = (now - entry["created_time"]).total_seconds()
            decayed = entry["signal_value"] * math.pow(0.5, elapsed / entry["half_life_seconds"])
            # overwrite with new raw value (the model may decide to combine)
            entry.update({
                "signal_kind": signal_kind,
                "signal_value": signal_value + decayed,   # simple additive model
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            })
        return self.pheromones[surface_key]["signal_value"]

    def current_vector(self) -> List[float]:
        """Return the list of current signal values (order is arbitrary)."""
        return [info["signal_value"] for info in self.pheromones.values()]


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_entropy_step(field: PheromoneField,
                        rsa_pub: Tuple[int, int, int],
                        scaling_factor: int) -> Tuple[float, float]:
    """
    Perform one hybrid entropy step:
    1. Extract the current pheromone vector and normalise it → p.
    2. Compute Shannon entropy H(p).
    3. Convert p to integer masses using *scaling_factor* and RSA‑encrypt each mass.
    4. De‑scale the encrypted masses back to a probability vector p' and compute H(p').
    Returns (H_original, H_encrypted).
    """
    e, _, n = rsa_pub
    raw_vals = field.current_vector()
    if not raw_vals:
        raise ValueError("Pheromone field is empty")
    p = normalize(raw_vals)

    # original entropy
    H_orig = shannon_entropy(p)

    # integer masses (ensure they are < n)
    masses = [int(round(v * scaling_factor)) % n for v in p]

    # RSA encryption (bijection on Z_n*)
    encrypted = [rsa_encrypt_int(m, e, n) for m in masses]

    # map back to real numbers in [0,1) by dividing by scaling_factor
    # (the division by n would also work; we keep scaling_factor for consistency)
    p_enc = normalize([float(c) / scaling_factor for c in encrypted])

    H_enc = shannon_entropy(p_enc)
    return H_orig, H_enc


def hybrid_predict_entropy(past_entropies: List[float],
                           predictor: NLMSPredictor) -> float:
    """
    Feed a sequence of entropy measurements into the NLMS predictor and
    obtain the next predicted entropy value.
    """
    prediction = 0.0
    for val in past_entropies:
        prediction = predictor.predict(val)
    return prediction


def hybrid_update_pheromone(field: PheromoneField,
                           surface_key: str,
                           signal_kind: str,
                           signal_value: float,
                           half_life_seconds: float) -> float:
    """
    Wrapper that updates the pheromone field and returns the new signal value.
    """
    return field.calculate_signal(surface_key, signal_kind, signal_value, half_life_seconds)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialise components
    rsa_key = generate_rsa_keypair(prime_bits=16)      # (e, d, n)
    scaling_Q = rsa_key[2] // 1024                     # ensure masses << n
    pher_field = PheromoneField()
    nlms = NLMSPredictor(order=5, mu=0.3)

    # 2. Populate pheromone field with a few synthetic entries
    for i in range(7):
        key = f"node_{i}"
        kind = "typeA"
        value = random.uniform(0.5, 5.0)
        half_life = random.uniform(30.0, 120.0)
        hybrid_update_pheromone(pher_field, key, kind, value, half_life)

    # 3. Perform several hybrid steps, collecting entropy history
    entropy_history: List[float] = []
    for step in range(10):
        H_orig, H_enc = hybrid_entropy_step(pher_field, rsa_key, scaling_Q)
        entropy_history.append(H_orig)
        # simple feedback: if encrypted entropy is higher, shorten half‑life
        for entry in pher_field.pheromones.values():
            entry["half_life_seconds"] = max(5.0, entry["half_life_seconds"] * (0.9 if H_enc > H_orig else 1.1))

        # 4. Predict next entropy using NLMS
        pred = hybrid_predict_entropy(entropy_history[-5:], nlms) if len(entropy_history) >= 5 else H_orig
        print(f"Step {step+1:02d} | H_orig={H_orig:.4f} | H_enc={H_enc:.4f} | Predicted_next={pred:.4f}")

    print("Smoke test completed without errors.")