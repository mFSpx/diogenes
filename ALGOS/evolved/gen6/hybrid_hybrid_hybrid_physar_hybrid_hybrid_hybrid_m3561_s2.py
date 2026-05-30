# DARWIN HAMMER — match 3561, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_infota_m875_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hoeffd_hybrid_hdc_serpentin_m2478_s0.py (gen5)
# born: 2026-05-29T23:50:41Z

"""
Hybrid Physarum‑Tropical Hyperdimensional Algorithm
===================================================

This module fuses the **Physarum Network** dynamics (flux‑based conductance
updates, pressure, information density) with the **tropical max‑plus /
hyperdimensional** machinery (vectorised polynomial representation,
binding and similarity).  

Mathematical bridge
------------------
* In the Physarum model the pressure difference Δp on an edge drives a
  flux q = G/ℓ·Δp.  The absolute flux |q| is used to reinforce the
  conductance G.
* In tropical algebra a coefficient c of a polynomial can be interpreted as
  a “weight”.  By representing each weight as a high‑dimensional random
  vector and binding two such vectors we obtain a similarity measure that
  respects the max‑plus semiring.

The bridge is built by **mapping the pressure (or derived information
density) to a tropical coefficient**.  The pressure‑derived coefficient
scales a random hyper‑dimensional vector; the bound of two such vectors
is then combined with the Physarum conductance update.  Consequently the
network’s adaptation is driven simultaneously by physical flux and by
tropical‑hyperdimensional similarity.

The implementation provides three core hybrid operations:
1. ``hybrid_flux`` – Physarum flux computation.
2. ``tropical_vector_from_pressure`` – conversion of pressure to a
   hyper‑dimensional tropical vector.
3. ``hybrid_update`` – joint conductance update together with a bound
   vector, followed by an entropy‑aware similarity measure.
"""

import math
import random
import sys
import pathlib
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ----------------------------------------------------------------------
# Parent A – Physarum utilities
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float,
         pressure_b: float, eps: float = 1e-12) -> float:
    """Flux through an edge given conductance, length and endpoint pressures."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0,
                       decay: float = 0.05) -> float:
    """Physarum conductance adaptation."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def calculate_pressure(conductance: float, edge_length: float, q: float) -> float:
    """Pressure needed to sustain a given flux."""
    return conductance * q / edge_length


def calculate_information_density(pressure: float) -> float:
    """Information density used by Infotaxis (log‑scaled pressure)."""
    return math.log(pressure + 1)


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """Min‑hash style signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [max(0, (1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Parent B – Tropical / Hyperdimensional utilities
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class TropicalPolynomial:
    coeffs: List[float]


def tropical_vector_from_coeffs(coeffs: List[float], dim: int = 1024) -> np.ndarray:
    """Map a list of coefficients to a hyper‑dimensional vector."""
    if not coeffs:
        raise ValueError("coeffs must be non‑empty")
    seed = sum(int(c * 10000) for c in coeffs) & 0xffffffff
    rng = random.Random(seed)
    base = np.array([rng.random() for _ in range(dim)], dtype=float)
    # repeat coefficients to match dimension and multiply element‑wise
    reps = (dim // len(coeffs)) + 1
    coeff_vec = np.array((coeffs * reps)[:dim], dtype=float)
    return base * coeff_vec


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Component‑wise binding (multiplication) of two hyper‑dimensional vectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    return a * b


def tropical_polynomial_similarity(p1: TropicalPolynomial,
                                   p2: TropicalPolynomial,
                                   dim: int = 1024) -> float:
    """Similarity of two tropical polynomials via bound‑vector averaging."""
    v1 = tropical_vector_from_coeffs(p1.coeffs, dim)
    v2 = tropical_vector_from_coeffs(p2.coeffs, dim)
    bound_vec = bind(v1, v2)
    return float(bound_vec.mean())


def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (plus)."""
    return np.add(x, y)


def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tropical matrix product: (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj})"""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcasting to compute all pairwise sums, then max over k
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_flux(conductance: float, edge_length: float,
                pressure_a: float, pressure_b: float) -> float:
    """
    Physarum flux (parent A) combined with a tropical scaling factor.

    The pressure difference Δp drives the flux; the magnitude of Δp is also
    interpreted as a tropical coefficient that will later be turned into a
    hyper‑dimensional vector.
    """
    q = flux(conductance, edge_length, pressure_a, pressure_b)
    return q


def tropical_vector_from_pressure(pressure: float, dim: int = 1024) -> np.ndarray:
    """
    Bridge function: treat the *information density* derived from pressure
    as a single‑element tropical polynomial and embed it into a
    hyper‑dimensional vector.

    The coefficient list contains only the pressure‑based value, ensuring
    a smooth mapping from the Physarum domain to the tropical‑HD domain.
    """
    info_density = calculate_information_density(pressure)
    # Use the density as the sole coefficient of a tropical polynomial
    poly = TropicalPolynomial([info_density])
    return tropical_vector_from_coeffs(poly.coeffs, dim)


def hybrid_update(conductance: float, edge_length: float,
                  pressure_a: float, pressure_b: float,
                  dt: float = 1.0, gain: float = 1.0,
                  decay: float = 0.05,
                  dim: int = 1024) -> Tuple[float, np.ndarray]:
    """
    Joint update step:

    1. Compute Physarum flux and update conductance (parent A).
    2. Convert the resulting pressure (at node *b*) to a hyper‑dimensional
       tropical vector (bridge).
    3. Return the new conductance together with the vector, which can be
       used for downstream tropical similarity or binding operations.
    """
    # 1. Physarum part
    q = hybrid_flux(conductance, edge_length, pressure_a, pressure_b)
    new_conductance = update_conductance(conductance, q, dt, gain, decay)

    # 2. Pressure at node b after flux adaptation
    new_pressure_b = calculate_pressure(new_conductance, edge_length, q)

    # 3. Tropical‑HD embedding
    vec = tropical_vector_from_pressure(new_pressure_b, dim)

    return new_conductance, vec


def hybrid_entropy_similarity(sig_a: List[int], sig_b: List[int],
                              pressure: float, dim: int = 1024) -> float:
    """
    Combine the signature similarity (parent A) with a tropical‑HD
    similarity that is weighted by the information density of the given
    pressure.

    The final score lies in [0, 1] and favours configurations where both
    classical similarity and tropical similarity are high, modulated by
    the entropy‑related pressure.
    """
    # Classical Jaccard‑like similarity
    sim_sig = similarity(sig_a, sig_b)

    # Tropical‑HD similarity based on pressure
    vec = tropical_vector_from_pressure(pressure, dim)
    # For a self‑similarity we bind the vector with itself; the mean of the
    # bound vector is a proxy for its “energy”.
    tropical_sim = float(bind(vec, vec).mean())

    # Weight by information density (higher pressure ⇒ more informative)
    info_weight = calculate_information_density(pressure) / (calculate_information_density(pressure) + 1)

    # Fuse the three components
    combined = (sim_sig * 0.5 + tropical_sim * 0.5) * info_weight
    return max(0.0, min(1.0, combined))


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create a tiny network with two nodes and one edge
    G = 0.8                     # initial conductance
    L = 1.5                     # edge length
    p0 = 2.0                    # pressure at node A
    p1 = 0.5                    # pressure at node B

    # Hybrid update
    new_G, hd_vec = hybrid_update(G, L, p0, p1, dt=0.5, gain=1.2, decay=0.03, dim=512)
    print(f"Updated conductance: {new_G:.4f}")
    print(f"Hyper‑dimensional vector norm: {np.linalg.norm(hd_vec):.4f}")

    # Signatures for two token sets
    tokens1 = ["alpha", "beta", "gamma"]
    tokens2 = ["beta", "delta", "epsilon"]
    sig1 = signature(tokens1, k=64)
    sig2 = signature(tokens2, k=64)

    # Hybrid entropy‑aware similarity
    pressure_example = 1.7
    score = hybrid_entropy_similarity(sig1, sig2, pressure_example, dim=512)
    print(f"Hybrid entropy similarity score: {score:.4f}")

    # Demonstrate tropical matrix multiplication on a tiny matrix
    A = np.array([[0.0, 2.0],
                  [1.5, 0.0]])
    B = np.array([[0.0, 1.0],
                  [3.0, 0.0]])
    C = t_matmul(A, B)
    print("Tropical matrix product C = A ⊗ B:")
    print(C)