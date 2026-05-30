# DARWIN HAMMER — match 777, survivor 2
# gen: 4
# parent_a: hybrid_jepa_energy_hybrid_sparse_wta_hy_m181_s0.py (gen3)
# parent_b: hybrid_shannon_entropy_rsa_cipher_m51_s1.py (gen1)
# born: 2026-05-29T23:30:51Z

"""
Hybrid Module: JEPA‑Darwin‑Hammer + Shannon‑RSA

Parents:
- hybrid_jepa_energy_hybrid_sparse_wta_hy_m181_s0.py (model‑pool free‑energy management)
- hybrid_shannon_entropy_rsa_cipher_m51_s1.py (RSA encryption with Shannon entropy)

Mathematical Bridge:
The bridge is the *information‑theoretic free energy*:
    F = E_model  –  λ·H(cipher)
where E_model is the energetic cost/penalty from the JEPA‑Darwin model pool,
and H(cipher) is the Shannon entropy of the RSA‑encrypted identifier of a model.
High entropy (more uncertainty) reduces the free energy, reflecting a privacy
benefit, while low entropy increases it.  This unifies the variational free‑energy
principle of JEPA with the entropy‑based security measure of RSA.
"""

from __future__ import annotations
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from collections import Counter
from collections.abc import Hashable, Iterable
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Model pool with energetic bookkeeping
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g., "T1", "T2", "T3"


class ModelPool:
    """Manages a pool of models under a RAM ceiling while tracking free energy."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: dict[str, ModelTier] = {}
        self._energy: float = 0.0  # accumulates energetic penalties/rewards

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def add_model(self, model: ModelTier) -> None:
        # Tier conflict penalty
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10
        # RAM overflow penalty
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy += 1e6
        self.loaded[model.name] = model

    def load(self, model: ModelTier) -> None:
        """Load without eviction – reward for successful load."""
        self._energy -= 1e4
        self.add_model(model)

    def load_with_eviction(self, model: ModelTier) -> None:
        """Evict oldest models until there is space, then load."""
        self._energy -= 1e3
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # pop the first inserted (dict preserves insertion order in Py3.7+)
            oldest_name = next(iter(self.loaded))
            self.loaded.pop(oldest_name)
        self.load(model)

    def free_energy(self) -> float:
        """Current free energy of the pool (lower is better)."""
        return self._energy

# ----------------------------------------------------------------------
# Parent B – RSA encryption + Shannon entropy
# ----------------------------------------------------------------------
def rsa_encrypt(message: int, e: int, n: int) -> int:
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)


def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)


def shannon_entropy(
    observations: Iterable[Hashable | float], is_distribution: bool = False
) -> float:
    xs = list(observations)
    if not xs:
        return 0.0
    if is_distribution:
        probs = [float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        c = Counter(xs)
        total = sum(c.values())
        probs = [v / total for v in c.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def calculate_entropy(cipher_int: int, n: int) -> float:
    """Entropy of the binary representation of a ciphertext padded to n.bit_length()."""
    bits = np.array([int(b) for b in bin(cipher_int)[2:].zfill(n.bit_length())])
    return shannon_entropy(bits.tolist(), is_distribution=False)


# ----------------------------------------------------------------------
# Hybrid Functions – marrying free energy with encrypted‑identifier entropy
# ----------------------------------------------------------------------
def encrypt_identifier(name: str, e: int, n: int) -> int:
    """
    Convert a model name to an integer (via deterministic hash), reduce modulo n,
    then RSA‑encrypt it.
    """
    # Deterministic positive integer from string
    raw = abs(hash(name)) % n
    return rsa_encrypt(raw, e, n)


def identifier_entropy(name: str, e: int, n: int) -> float:
    """Entropy of the RSA‑encrypted identifier of a model."""
    cipher = encrypt_identifier(name, e, n)
    return calculate_entropy(cipher, n)


def hybrid_load_model(pool: ModelPool, model: ModelTier, e: int, n: int, λ: float = 0.5) -> None:
    """
    Load a model while adjusting the pool's free energy using the entropy of the
    model's encrypted identifier.

    Free energy update:
        ΔF = -reward_load  +  λ·(1 - H/ H_max)

    where H is the Shannon entropy of the ciphertext and H_max = n.bit_length().
    """
    # Base loading (rewards/penalties from ModelPool)
    pool.load(model)

    # Entropy‑based privacy term
    H = identifier_entropy(model.name, e, n)
    H_max = n.bit_length()
    privacy_penalty = λ * (1.0 - H / H_max) * 1e3  # scale to be comparable with other terms
    pool._energy += privacy_penalty  # higher entropy → smaller penalty


def find_optimal_exponent(models: Iterable[ModelTier], n: int, λ: float = 0.5) -> int:
    """
    Search for the RSA public exponent e (2 ≤ e < n) that minimizes the total
    free‑energy contribution from the entropy term across all model identifiers.
    """
    best_e = 2
    best_score = float("inf")
    for e in range(2, n):
        if math.gcd(e, (n - 1)) != 1:
            continue
        total_penalty = 0.0
        for m in models:
            H = identifier_entropy(m.name, e, n)
            H_max = n.bit_length()
            total_penalty += λ * (1.0 - H / H_max) * 1e3
        if total_penalty < best_score:
            best_score = total_penalty
            best_e = e
    return best_e


def pool_free_energy_with_entropy(pool: ModelPool, e: int, n: int, λ: float = 0.5) -> float:
    """
    Compute the pool's free energy augmented by the entropy term of *all*
    currently loaded models.
    """
    base = pool.free_energy()
    entropy_term = 0.0
    for model in pool.loaded.values():
        H = identifier_entropy(model.name, e, n)
        H_max = n.bit_length()
        entropy_term += λ * (1.0 - H / H_max) * 1e3
    return base + entropy_term


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small RSA modulus for demonstration (product of two primes)
    p, q = 61, 53
    n = p * q                # 3233
    phi = (p - 1) * (q - 1)  # 3120
    e = 17                   # common public exponent (coprime with phi)
    d = pow(e, -1, phi)      # private exponent

    # Create a pool and some dummy models
    pool = ModelPool(ram_ceiling_mb=5000)
    models = [
        ModelTier(name="vision_encoder", ram_mb=1200, tier="T1"),
        ModelTier(name="language_decoder", ram_mb=1500, tier="T2"),
        ModelTier(name="privacy_head", ram_mb=800, tier="T3"),
    ]

    # Load models using the hybrid loader
    for m in models:
        hybrid_load_model(pool, m, e, n, λ=0.7)

    print("Loaded models:", list(pool.loaded.keys()))
    print("Base free energy:", pool.free_energy())
    print("Augmented free energy (entropy included):",
          pool_free_energy_with_entropy(pool, e, n, λ=0.7))

    # Find an exponent that would further reduce the entropy penalty
    optimal_e = find_optimal_exponent(models, n, λ=0.7)
    print("Optimal RSA exponent for minimal entropy penalty:", optimal_e)

    # Verify RSA round‑trip on a sample identifier
    sample_name = models[0].name
    cipher = encrypt_identifier(sample_name, optimal_e, n)
    # Decrypt using the private key (requires the same exponent, not optimal_e)
    # Demonstrate that decryption works with the original e/d pair
    decrypted = rsa_decrypt(cipher, d, n)
    # The decrypted integer should equal the hashed identifier modulo n
    expected = abs(hash(sample_name)) % n
    assert decrypted == expected, "RSA round‑trip failed"
    print("RSA round‑trip successful for identifier:", sample_name)