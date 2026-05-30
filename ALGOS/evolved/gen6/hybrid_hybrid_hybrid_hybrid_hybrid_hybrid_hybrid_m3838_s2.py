# DARWIN HAMMER — match 3838, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2011_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_ternar_m333_s2.py (gen3)
# born: 2026-05-29T23:51:56Z

"""
Hybrid Allocation‑Sheaf, NLMS, Liquid‑Time‑Constant & Variational Free‑Energy Fusion.

Parents:
- **Parent A** – deterministic weekday‑dependent allocation weight vector, Schoolfield
  developmental rate ρ(T), NLMS adaptive filter with epistemic confidence flags.
- **Parent B** – weekday‑dependent weight vector w(d) used for (i) Liquid‑Time‑Constant
  (LTC) gating via a MinHash similarity vector s⃗ and (ii) weighting the KL‑term of a
  variational free‑energy (VFE) objective.

Mathematical Bridge:
Both parents expose the same weekday‑dependent stochastic row‑vector **w(d) ∈ ℝ⁴**.
We therefore use **w(d)** as the single topological backbone.  The developmental
rate ρ(T) from Parent A scales every temperature‑sensitive quantity in Parent B:
the NLMS step‑size, the LTC gating sigmoid, and the VFE KL‑weight.  The resulting
system simultaneously (a) adapts filter coefficients under epistemic confidence,
(b) modulates continuous‑time dynamics via a temperature‑aware τ, and (c) evaluates
a probabilistic routing cost with a temperature‑scaled KL penalty.

The code below implements the fused dynamics and provides three exemplar
functions demonstrating the hybrid operation.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, List, Set

import numpy as np

# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
BASE_TAU: float = 1.0          # baseline liquid time constant
ALPHA: float = 5.0             # LTC gating steepness
LAMBDA: float = 0.7            # VFE weighting factor
MINHASH_K: int = 64            # number of hash functions for MinHash
MAX64: int = (1 << 64) - 1     # mask for 64‑bit hashing
_EPISTEMIC_CONFIDENCE = {
    "FACT": 1.0,
    "PROBABLE": 0.9,
    "POSSIBLE": 0.8,
    "BULLSHIT": 0.1,
    "SURE_MAYBE": 0.5,
}
# ----------------------------------------------------------------------
# Parent A – Developmental rate (Schoolfield) and NLMS update
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12000.0      # J mol⁻¹
    t_low: float = 283.15                     # K
    t_high: float = 307.15                    # K
    delta_h_low: float = -45000.0            # J mol⁻¹
    delta_h_high: float = 65000.0            # J mol⁻¹
    r_cal: float = 1.987                     # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield‑Rollinson poikilotherm developmental rate ρ(T)."""
    if temp_k <= 0:
        raise ValueError("Temperature must be > 0 K")
    # Convert cal to J for consistency
    R = params.r_cal * 4.184  # J mol⁻¹ K⁻¹
    # Arrhenius terms
    num = math.exp(-params.delta_h_activation / (R * temp_k))
    den_low = 1.0 + math.exp(params.delta_h_low / R * (1.0 / params.t_low - 1.0 / temp_k))
    den_high = 1.0 + math.exp(params.delta_h_high / R * (1.0 / temp_k - 1.0 / params.t_high))
    rho = params.rho_25 * num / (den_low * den_high)
    return rho


def nlms_update(
    x: np.ndarray,
    d: float,
    w_prev: np.ndarray,
    mu_base: float,
    flag: str,
    temp_k: float,
) -> np.ndarray:
    """
    Normalised Least‑Mean‑Squares (NLMS) coefficient update.

    Parameters
    ----------
    x : np.ndarray
        Input vector (shape = (n,)).
    d : float
        Desired scalar output.
    w_prev : np.ndarray
        Previous coefficient vector (shape = (n,)).
    mu_base : float
        Base step‑size (0 < mu_base ≤ 1).
    flag : str
        Epistemic confidence flag; must be a key of ``_EPISTEMIC_CONFIDENCE``.
    temp_k : float
        Current temperature in Kelvin.

    Returns
    -------
    np.ndarray
        Updated coefficient vector.
    """
    if flag not in _EPISTEMIC_CONFIDENCE:
        raise ValueError(f"Unknown epistemic flag: {flag}")

    # Temperature‑scaled learning rate
    rho = developmental_rate(temp_k)
    mu = mu_base * rho * _EPISTEMIC_CONFIDENCE[flag]

    # NLMS core
    e = d - np.dot(w_prev, x)
    norm = np.dot(x, x) + 1e-12  # avoid division by zero
    w_new = w_prev + (mu * e / norm) * x
    return w_new


# ----------------------------------------------------------------------
# Parent B – Weekday weight vector, MinHash similarity, LTC & VFE
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Produce a normalised weekday‑dependent weight vector w(d).

    The vector follows a sinusoidal pattern whose phase is shifted per group,
    ensuring that the sum of components equals 1.
    """
    if not (0 <= dow <= 6):
        raise ValueError("dow must be in [0, 6] (0=Sun … 6=Sat)")

    raw = []
    for idx, _ in enumerate(groups):
        phase = (dow + idx) / 7.0  # shift each group by its index
        weight = math.sin(2 * math.pi * phase) + 1.0  # range [0, 2]
        raw.append(weight)
    raw_arr = np.array(raw, dtype=float)
    # Normalise to a probability vector (row‑stochastic)
    w = raw_arr / raw_arr.sum()
    return w


def _hash64(value: int, seed: int) -> int:
    """Simple 64‑bit mix function."""
    v = (value ^ seed) & MAX64
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9
    v = (v ^ (v >> 27)) * 0x94d049bb133111eb
    v = v ^ (v >> 31)
    return v & MAX64


def minhash_signature(item_set: Set[int], k: int = MINHASH_K) -> np.ndarray:
    """
    Compute a MinHash signature of length ``k`` for a set of integer tokens.
    """
    sig = np.full(k, MAX64, dtype=np.uint64)
    for token in item_set:
        for i in range(k):
            h = _hash64(token, i)
            if h < sig[i]:
                sig[i] = h
    return sig


def minhash_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """
    Estimate Jaccard similarity from two MinHash signatures.
    """
    if sig_a.shape != sig_b.shape:
        raise ValueError("Signatures must have identical length")
    equal = np.sum(sig_a == sig_b)
    return float(equal) / sig_a.shape[0]


def compute_ltc_tau(
    w: np.ndarray,
    s_vec: np.ndarray,
    temp_k: float,
) -> float:
    """
    Liquid‑Time‑Constant τ = τ₀ / g, where g = σ(α·ρ(T)·w·s⃗).

    Parameters
    ----------
    w : np.ndarray
        Weekday weight vector (shape = (n,)).
    s_vec : np.ndarray
        MinHash similarity vector per group (shape = (n,)).
    temp_k : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Effective time constant τ.
    """
    rho = developmental_rate(temp_k)
    inner = np.dot(w, s_vec)                     # scalar
    g = 1.0 / (1.0 + math.exp(-ALPHA * rho * inner))  # sigmoid
    tau = BASE_TAU / (g + 1e-12)                 # protect against g≈0
    return tau


def compute_free_energy(
    w: np.ndarray,
    kl_vec: np.ndarray,
    expected_log_likelihood: float,
    temp_k: float,
) -> float:
    """
    Variational free‑energy F = E_q[log p] – λ·ρ(T)·(w·KL).

    Parameters
    ----------
    w : np.ndarray
        Weekday weight vector (shape = (n,)).
    kl_vec : np.ndarray
        KL‑divergence per group (shape = (n,)).
    expected_log_likelihood : float
        Expected log‑likelihood term 𝔼_q[log p(x|z)].
    temp_k : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Free‑energy value.
    """
    rho = developmental_rate(temp_k)
    kl_weighted = np.dot(w, kl_vec)
    F = expected_log_likelihood - LAMBDA * rho * kl_weighted
    return F


# ----------------------------------------------------------------------
# Demonstration functions tying all pieces together
# ----------------------------------------------------------------------
def hybrid_step(
    x: np.ndarray,
    d: float,
    w_nlms: np.ndarray,
    mu_base: float,
    flag: str,
    dow: int,
    temp_k: float,
    set_a: Set[int],
    set_b: Set[int],
    kl_vec: np.ndarray,
    expected_log_likelihood: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Execute one hybrid iteration:
    1. Update NLMS coefficients using temperature‑scaled learning rate.
    2. Compute LTC time constant τ from MinHash similarity.
    3. Evaluate variational free‑energy.

    Returns
    -------
    w_updated : np.ndarray
        Updated NLMS coefficient vector.
    tau : float
        Effective liquid‑time constant.
    free_energy : float
        Variational free‑energy value.
    """
    # 1. NLMS update
    w_updated = nlms_update(x, d, w_nlms, mu_base, flag, temp_k)

    # 2. MinHash similarity per group
    # For illustration we treat each group as a distinct token set;
    # here we simply reuse the same two sets and assign the similarity
    # uniformly across groups.
    sig_a = minhash_signature(set_a)
    sig_b = minhash_signature(set_b)
    sim = minhash_similarity(sig_a, sig_b)          # scalar Jaccard estimate
    s_vec = np.full(len(GROUPS), sim)                # broadcast to each group

    # Weekday weight vector
    w_day = weekday_weight_vector(GROUPS, dow)

    # LTC τ
    tau = compute_ltc_tau(w_day, s_vec, temp_k)

    # 3. Free‑energy
    free_energy = compute_free_energy(w_day, kl_vec, expected_log_likelihood, temp_k)

    return w_updated, tau, free_energy


def simulate_hybrid_process(
    steps: int = 10,
    seed: int = 42,
) -> None:
    """
    Run a short deterministic simulation of the hybrid algorithm and print
    intermediate quantities.  All randomness is seeded for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    n = len(GROUPS)
    # Initialise NLMS coefficients to zeros
    w_nlms = np.zeros(n, dtype=float)
    mu_base = 0.5
    dow = datetime.now(timezone.utc).weekday()  # 0=Mon … 6=Sun; we map to 0=Sun
    dow = (dow + 1) % 7  # shift so 0=Sun

    # Dummy temperature trajectory (°C → K)
    temps_c = np.linspace(15, 35, steps)
    temps_k = temps_c + 273.15

    # Dummy epistemic flag sequence
    flags = ["FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE"]
    flag_cycle = [flags[i % len(flags)] for i in range(steps)]

    # Dummy input/output streams
    x_stream = np.random.randn(steps, n)
    d_stream = np.random.randn(steps)

    # Dummy token sets for MinHash
    set_a = {random.randint(0, 1_000_000) for _ in range(200)}
    set_b = {random.randint(0, 1_000_000) for _ in range(200)}

    # Dummy KL vector (positive values) and expected log‑likelihood
    kl_vec = np.abs(np.random.randn(n))
    expected_log_likelihood = -np.random.rand() * 10.0

    for i in range(steps):
        w_nlms, tau, F = hybrid_step(
            x=x_stream[i],
            d=d_stream[i],
            w_nlms=w_nlms,
            mu_base=mu_base,
            flag=flag_cycle[i],
            dow=dow,
            temp_k=temps_k[i],
            set_a=set_a,
            set_b=set_b,
            kl_vec=kl_vec,
            expected_log_likelihood=expected_log_likelihood,
        )
        print(
            f"Step {i+1:2d} | Temp {temps_c[i]:5.1f}°C | Flag {flag_cycle[i]:10s} | "
            f"τ={tau:6.3f} | F={F:8.3f} | NLMS‖w‖={np.linalg.norm(w_nlms):6.3f}"
        )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    simulate_hybrid_process(steps=12)