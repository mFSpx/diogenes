# DARWIN HAMMER — match 2637, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s2.py (gen2)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_infotaxis_min_m967_s2.py (gen5)
# born: 2026-05-29T23:43:13Z

"""Hybrid Fisher‑Physarum‑Infotaxis Algorithm

Parents:
- hybrid_fisher_locali_hybrid_minimum_cost__m29_s2.py (Gaussian beam, Fisher information, Bayesian scoring)
- hybrid_hybrid_physarum_netw_hybrid_infotaxis_min_m967_s2.py (Physarum network conductance dynamics, MinHash‑modulated updates, entropy‑based certainty)

Mathematical bridge:
Both parents operate on probabilistic scores.  The Fisher information
`I(θ)= (∂μ/∂θ)²/μ` (μ = Gaussian beam intensity) is an information‑theoretic
measure that can be normalised into a probability distribution over a
parameter grid.  The Physarum‑Infotaxis side uses an entropy
`H(p)=−∑p log p` of a token‑distribution together with a MinHash
signature that quantifies set similarity.  The hybrid therefore
* (i) builds a Fisher‑derived probability vector,
* (ii) computes its entropy,
* (iii) feeds the entropy and the MinHash weight into the conductance
    update of a Physarum network.

The resulting system simultaneously smooths chronological data with a
Gaussian beam, extracts Fisher information, and lets that information
drive adaptive flow in a Physarum‑style network, modulated by
information‑theoretic uncertainty captured via MinHash and entropy.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity centred at *center* with standard deviation *width*."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    I(θ) = (∂μ/∂θ)² / μ   where μ = Gaussian intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def gaussian_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a 1‑D Gaussian kernel (using the same functional form as gaussian_beam)."""
    return np.array([gaussian_beam(x, 0.0, sigma) for x in data])


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.utcnow().isoformat() + "Z")


def minhash_signature(tokens: Iterable[str], num_perm: int = 128) -> int:
    """
    Very lightweight MinHash: hash each token, keep the minimum integer
    value over *num_perm* independent hash seeds, and combine them into a
    single 64‑bit integer.
    """
    mins = [2**64 - 1] * num_perm
    for token in tokens:
        token_bytes = token.encode("utf-8")
        for i in range(num_perm):
            hasher = hashlib.blake2b(token_bytes, digest_size=8, person=bytes([i]))
            hv = int.from_bytes(hasher.digest(), "little")
            if hv < mins[i]:
                mins[i] = hv
    # XOR‑fold the vector into one integer
    signature = 0
    for m in mins:
        signature ^= m
    return signature


def entropy_from_distribution(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector."""
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return -float(np.sum(p * np.log(p)))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def fisher_information_grid(
    thetas: np.ndarray, center: float, width: float
) -> np.ndarray:
    """
    Vectorised Fisher information over a grid of θ values.
    Returns a 1‑D array of I(θ) values.
    """
    vec_gauss = np.vectorize(gaussian_beam)
    vec_fisher = np.vectorize(fisher_score)

    intensity = vec_gauss(thetas, center, width)
    # Avoid division by zero inside fisher_score
    intensity = np.maximum(intensity, 1e-12)
    derivative = intensity * (-(thetas - center) / (width * width))
    return (derivative * derivative) / intensity


def physarum_conductance_update(
    conductance: np.ndarray,
    flux: np.ndarray,
    tokens: List[str],
    alpha: float = 0.01,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Update a symmetric conductance matrix C using:
        C_new = C + α * (flux - β * w * L)
    where:
        - flux is a matrix of edge fluxes,
        - w = normalized MinHash weight ∈ [0,1],
        - L is the graph Laplacian (C's degree matrix minus C).
    The MinHash weight injects the information‑theoretic similarity of the
    token set into the physical flow dynamics.
    """
    if conductance.shape != flux.shape:
        raise ValueError("conductance and flux must have the same shape")
    n = conductance.shape[0]

    # 1. MinHash weight
    raw_sig = minhash_signature(tokens)
    # Normalise to [0,1] by dividing by max 64‑bit value
    w = raw_sig / (2**64 - 1)

    # 2. Laplacian of current conductance
    degree = np.sum(conductance, axis=1)
    laplacian = np.diag(degree) - conductance

    # 3. Conductance update
    delta = alpha * (flux - beta * w * laplacian)
    new_conductance = conductance + delta

    # Ensure symmetry and non‑negativity
    new_conductance = (new_conductance + new_conductance.T) / 2.0
    np.clip(new_conductance, 0.0, None, out=new_conductance)
    return new_conductance


def hybrid_fisher_physarum_step(
    thetas: np.ndarray,
    center: float,
    width: float,
    conductance: np.ndarray,
    flux: np.ndarray,
    tokens: List[str],
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Single hybrid iteration:
      1. Compute Fisher information over *thetas*.
      2. Normalise it to a probability distribution and compute its entropy.
      3. Use the entropy as a scaling factor for the Physarum conductance update,
         together with the MinHash‑derived weight.
    Returns:
        - Fisher information vector,
        - Entropy value,
        - Updated conductance matrix.
    """
    # Step 1: Fisher information
    fisher_vec = fisher_information_grid(thetas, center, width)

    # Step 2: Entropy of the normalised Fisher vector
    prob_vec = fisher_vec / np.maximum(fisher_vec.sum(), 1e-12)
    entropy = entropy_from_distribution(prob_vec)

    # Step 3: Modulate α by (1 + entropy) to let higher uncertainty drive
    # stronger adaptation in the network.
    alpha_mod = 0.01 * (1.0 + entropy)
    updated_conductance = physarum_conductance_update(
        conductance, flux, tokens, alpha=alpha_mod, beta=0.5
    )
    return fisher_vec, entropy, updated_conductance


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Parameter grid for Fisher information
    theta_grid = np.linspace(-5.0, 5.0, 101)
    centre = 0.0
    sigma = 1.0

    # 2. Dummy Physarum network (5 nodes)
    N = 5
    rng = np.random.default_rng(42)
    C0 = rng.random((N, N))
    C0 = (C0 + C0.T) / 2.0  # make symmetric
    np.fill_diagonal(C0, 0.0)  # no self‑loops

    # Random flux matrix (must be symmetric, zero diagonal)
    F0 = rng.random((N, N))
    F0 = (F0 + F0.T) / 2.0
    np.fill_diagonal(F0, 0.0)

    # Token set representing some observation vocabulary
    tokens_example = ["alpha", "beta", "gamma", "delta", "epsilon"]

    # Run a single hybrid step
    fisher_vec, ent, C1 = hybrid_fisher_physarum_step(
        thetas=theta_grid,
        center=centre,
        width=sigma,
        conductance=C0,
        flux=F0,
        tokens=tokens_example,
    )

    # Simple sanity prints
    print(f"Fisher info (first 5): {fisher_vec[:5]}")
    print(f"Entropy of Fisher distribution: {ent:.4f}")
    print(f"Conductance matrix change norm: {np.linalg.norm(C1 - C0):.6f}")

    # Verify CertaintyFlag can be instantiated
    flag = CertaintyFlag(
        label="PROBABLE",
        confidence_bps=7500,
        authority_class="research",
        rationale="demo of hybrid system",
    )
    print(f"Created certainty flag: {flag}")