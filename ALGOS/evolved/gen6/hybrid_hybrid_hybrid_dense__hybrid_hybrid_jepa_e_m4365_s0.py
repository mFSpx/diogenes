# DARWIN HAMMER — match 4365, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_dense_associa_hybrid_hybrid_hybrid_m1726_s2.py (gen5)
# parent_b: hybrid_hybrid_jepa_energy_h_hybrid_shannon_entro_m777_s2.py (gen4)
# born: 2026-05-29T23:55:11Z

"""
Hybrid Module: Dense Associative Memory + Model‑Pool Free Energy + RSA‑Shannon Entropy

Parents
-------
- **Parent A** – `hybrid_hybrid_dense_associa_hybrid_hybrid_hybrid_m1726_s2.py`
  Implements a modern Hopfield (Dense Associative) network whose energy
  `E(x) = - (1/β) log Σ_i exp(β M_i·x) + ½‖x‖²` and a pheromone decay model.

- **Parent B** – `hybrid_hybrid_jepa_energy_h_hybrid_shannon_entro_m777_s2.py`
  Manages a pool of models with an accumulated energetic bookkeeping term and
  evaluates the Shannon entropy of an RSA‑encrypted identifier.

Mathematical Bridge
-------------------
Both parents expose *energy‑like* scalars that can be combined via an
information‑theoretic free‑energy expression

    F = E_dense  +  E_pool  – λ·H(cipher)

where  

* `E_dense`  – energy of a query vector in the Dense Associative Memory,  
* `E_pool`   – accumulated energy of the `ModelPool` (including pheromone‑
  weighted contributions),  
* `H(cipher)` – Shannon entropy of the RSA‑encrypted identifier, and  
* `λ`        – a tunable trade‑off coefficient.

The hybrid algorithm below implements this bridge, allowing the pheromone
signal to modulate the contribution of each loaded model before the free‑energy
is formed.



The module provides three public functions that demonstrate the hybrid
operation:

1. `dense_associative_energy(xi, M, beta=1.0)` – computes the Hopfield‑style
   energy of a query vector.
2. `model_pool_energy(pool, pheromone_half_life=3600.0)` – returns the pool’s
   internal energy after applying pheromone decay to each model’s identifier.
3. `hybrid_free_energy(xi, M, pool, identifier, beta=1.0, lam=0.1, beta_ph=1.0)`
   – evaluates the full free‑energy `F` described above.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1‑D array."""
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def _lse(z: np.ndarray) -> float:
    """Log‑sum‑exp of a 1‑D array (numerically stable)."""
    m = np.max(z)
    return m + math.log(np.exp(z - m).sum())


def calculate_pheromone_signal(
    surface_key: str,
    signal_kind: str,
    signal_value: float,
    half_life_seconds: float,
) -> float:
    """Exponential decay of a pheromone signal."""
    now = datetime.now(timezone.utc)
    elapsed = now.timestamp()
    return signal_value * math.exp(-elapsed / half_life_seconds)


# ----------------------------------------------------------------------
# Dense Associative Memory (Parent A core)
# ----------------------------------------------------------------------
def dense_associative_energy(xi: np.ndarray, M: np.ndarray, beta: float = 1.0) -> float:
    """
    Compute the Dense Associative Memory energy:

        E(xi) = - (1/β) * log Σ_i exp(β M_i·xi) + ½‖xi‖²

    Parameters
    ----------
    xi : (d,) array
        Query / current state vector.
    M : (N, d) array
        Memory matrix – each row stores a pattern.
    beta : float, optional
        Inverse temperature; larger values sharpen attractors.

    Returns
    -------
    float
        Energy scalar (lower = more stable).
    """
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)          # shape (N,)
    lse_term = _lse(scores) / beta
    quadratic = 0.5 * np.dot(xi, xi)
    return -lse_term + quadratic


# ----------------------------------------------------------------------
# Model Pool and RSA‑Shannon components (Parent B core)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Immutable description of a model."""
    name: str
    ram_mb: int
    tier: str                     # e.g., "T1", "T2", "T3"


@dataclass
class ModelPool:
    """Manages a collection of models under a RAM ceiling and tracks energetic penalties."""
    ram_ceiling_mb: int = 6000
    loaded: Dict[str, ModelTier] = field(default_factory=dict)
    _energy: float = 0.0          # internal energy accumulator

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def add_model(self, model: ModelTier) -> None:
        # Tier‑conflict penalty
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10
        # RAM overflow penalty
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy += 1e6
        self.loaded[model.name] = model

    def load(self, model: ModelTier) -> None:
        """Load a model without eviction – reward for successful load."""
        self._energy -= 1e4
        self.add_model(model)

    @property
    def energy(self) -> float:
        """Current accumulated energy of the pool."""
        return self._energy


# Simple RSA utilities (public key only, deterministic for the demo)
_RSA_E = 65537
_RSA_N = 0xD3C21BCECCEDA100F4A9B9B5E5C5E6D7  # a 128‑bit semi‑random modulus


def _rsa_encrypt(identifier: int) -> int:
    """Encrypt an integer identifier with a fixed public RSA key."""
    return pow(identifier, _RSA_E, _RSA_N)


def shannon_entropy_of_int(value: int) -> float:
    """
    Compute the Shannon entropy of the binary representation of ``value``.
    Treat each bit as a Bernoulli trial; entropy = -p·log₂p - (1-p)·log₂(1-p)
    where p is the proportion of 1‑bits.
    """
    bits = bin(value)[2:]                     # strip '0b'
    n = len(bits)
    if n == 0:
        return 0.0
    ones = bits.count('1')
    p = ones / n
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def model_pool_energy(pool: ModelPool, pheromone_half_life: float = 3600.0) -> float:
    """
    Compute the pool’s energy after applying a pheromone‑decay weighting to each
    loaded model. The pheromone signal is derived from the model name hash.

    Parameters
    ----------
    pool : ModelPool
        The pool whose internal energy is to be reported.
    pheromone_half_life : float, optional
        Half‑life (seconds) for the pheromone decay; defaults to one hour.

    Returns
    -------
    float
        Energy = internal energy – Σ_i w_i, where w_i is a pheromone‑derived
        weight for model *i*.
    """
    total_weight = 0.0
    for name, model in pool.loaded.items():
        # Use hash of the name as a surrogate “signal value”.
        raw_signal = abs(hash(name)) % 1_000_000
        weight = calculate_pheromone_signal(
            surface_key=name,
            signal_kind="model_load",
            signal_value=raw_signal,
            half_life_seconds=pheromone_half_life,
        )
        total_weight += weight
    # The pheromone contribution reduces the effective energy (more pheromone → lower free energy)
    return pool.energy - total_weight


# ----------------------------------------------------------------------
# Hybrid Free‑Energy Computation (the mathematical bridge)
# ----------------------------------------------------------------------
def hybrid_free_energy(
    xi: np.ndarray,
    M: np.ndarray,
    pool: ModelPool,
    identifier: int,
    beta: float = 1.0,
    lam: float = 0.1,
    beta_ph: float = 1.0,
    pheromone_half_life: float = 3600.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the unified free energy

        F = E_dense + E_pool – λ·H(cipher)

    where

    * ``E_dense``  – Dense Associative Memory energy of ``xi``.
    * ``E_pool``   – Model‑pool energy after pheromone weighting.
    * ``H(cipher)`` – Shannon entropy of the RSA‑encrypted ``identifier``.
    * ``λ``        – trade‑off coefficient (default 0.1).

    Returns
    -------
    F : float
        The total free‑energy scalar.
    details : dict
        Intermediate quantities for introspection.
    """
    # 1. Dense Associative energy
    e_dense = dense_associative_energy(xi, M, beta=beta)

    # 2. Pheromone‑modulated pool energy
    e_pool = model_pool_energy(pool, pheromone_half_life=pheromone_half_life)

    # 3. RSA encryption + Shannon entropy
    cipher = _rsa_encrypt(identifier)
    h_cipher = shannon_entropy_of_int(cipher)

    # 4. Combine
    free_energy = e_dense + e_pool - lam * h_cipher

    details = {
        "E_dense": e_dense,
        "E_pool": e_pool,
        "H_cipher": h_cipher,
        "lambda": lam,
    }
    return free_energy, details


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Random seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # 1. Create a small memory matrix (5 patterns, 8‑dimensional)
    M = np.random.randn(5, 8)

    # 2. Query vector
    xi = np.random.randn(8)

    # 3. Initialise a ModelPool and load a few dummy models
    pool = ModelPool(ram_ceiling_mb=2000)
    pool.load(ModelTier(name="alpha", ram_mb=400, tier="T1"))
    pool.load(ModelTier(name="beta", ram_mb=600, tier="T2"))
    pool.load(ModelTier(name="gamma", ram_mb=300, tier="T3"))  # triggers tier conflict penalty

    # 4. Identifier to be encrypted (e.g., a user‑id)
    identifier = 123456789

    # 5. Compute hybrid free energy
    F, info = hybrid_free_energy(
        xi=xi,
        M=M,
        pool=pool,
        identifier=identifier,
        beta=1.5,
        lam=0.2,
        pheromone_half_life=1800.0,
    )

    print(f"Hybrid free energy: {F:.6f}")
    for k, v in info.items():
        print(f"  {k}: {v:.6f}")

    # Verify that the code runs without raising exceptions
    sys.exit(0)