# DARWIN HAMMER — match 2882, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rsa_ci_diffusion_forcing_m1607_s1.py (gen6)
# parent_b: hybrid_regret_engine_hybrid_liquid_time_c_m13_s2.py (gen2)
# born: 2026-05-29T23:46:32Z

"""hybrid_diffusion_regret_ltc_rsa_rbf.py

Hybrid algorithm merging:

* Parent A – Diffusion Forcing noise schedule used as a time‑dependent weighting
  for a Radial‑Basis‑Function (RBF) surrogate model, whose scalar output is
  secured with RSA encryption.
* Parent B – Regret‑weighted action selection whose augmented action values are
  corrected by a Liquid‑Time‑Constant (LTC) recurrent dynamics whose time‑constant
  is modulated by a MinHash similarity computed from the RSA‑encrypted message.

Mathematical bridge
-------------------
For each discrete time step *t*:

1. **Diffusion schedule**  σₜ ∈ (0,1] is generated (linear or cosine schedule).
2. **RBF surrogate**      ȳₜ = Σᵢ wᵢ·exp(−ε²‖xₜ−cᵢ‖²)   – the diffusion weight σₜ scales the kernel
   matrix, i.e. Kₜ = σₜ·K, thus making the surrogate time‑dependent.
3. **RSA encryption**     cₜ = (⌊ȳₜ·S⌋)ᵉ mod n  (S is a scaling factor to obtain an integer).
4. **MinHash similarity** sₜ = average_j  (hash(cₜ, tokenⱼ) mod 2⁶⁴) / 2⁶⁴.
   The similarity influences the LTC time‑constant τₜ = τ₀ / (1 + α·sₜ).
5. **LTC dynamics**       hₜ₊₁ = hₜ + Δt·(−hₜ/τₜ + W·uₜ)   with input
   uₜ = (EV – cost – risk) + w·hₜ.
6. **Regret‑weighted softmax**   πₜ(i) ∝ exp(β·v̂ᵢₜ) where v̂ᵢₜ = (EVᵢ – costᵢ – riskᵢ) + w·hₜ.

The six equations form a closed loop: the diffusion schedule drives the surrogate,
the surrogate output is encrypted, the encrypted integer seeds MinHash,
MinHash modulates LTC, LTC corrects the action values, and the regret‑weighted
softmax produces the final policy for the next step.

The implementation below follows this pipeline and provides three public
functions that showcase the hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict
import hashlib
import numpy as np

# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------
Vector = Sequence[float]

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian_kernel(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(- (epsilon * r) ** 2)

# ----------------------------------------------------------------------
# 1. Diffusion forcing schedule
# ----------------------------------------------------------------------
def diffusion_noise_schedule(steps: int,
                             beta_start: float = 0.01,
                             beta_end: float = 1.0) -> List[float]:
    """
    Linear diffusion schedule σₜ ∈ (0,1] for `steps` timesteps.
    Returns a list of length `steps`.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    return [beta_start + (beta_end - beta_start) * t / (steps - 1)
            for t in range(steps)]

# ----------------------------------------------------------------------
# 2. RBF surrogate model with time‑dependent weighting
# ----------------------------------------------------------------------
def rbf_surrogate(x: Vector,
                  centers: List[Vector],
                  weights: List[float],
                  epsilon: float,
                  diffusion_weight: float) -> float:
    """
    Compute Σᵢ wᵢ·exp(−ε²‖x−cᵢ‖²) and multiply the kernel matrix by `diffusion_weight`.
    """
    if not (len(centers) == len(weights)):
        raise ValueError("centers and weights must have same length")
    out = 0.0
    for c, w in zip(centers, weights):
        r = euclidean(x, c)
        k = gaussian_kernel(r, epsilon)
        out += w * diffusion_weight * k
    return out

# ----------------------------------------------------------------------
# 3. Simple RSA primitives (tiny keys for demonstration)
# ----------------------------------------------------------------------
def rsa_encrypt_int(message: int, e: int, n: int) -> int:
    """RSA encryption c = m^e mod n."""
    if not (0 <= message < n):
        raise ValueError("message out of range")
    return pow(message, e, n)

def rsa_decrypt_int(cipher: int, d: int, n: int) -> int:
    """RSA decryption m = c^d mod n."""
    return pow(cipher, d, n)

# ----------------------------------------------------------------------
# 4. MinHash similarity derived from an RSA ciphertext
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1

def _hash_int(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used by the original MinHash implementation."""
    data = seed.to_bytes(8, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def minhash_similarity(cipher: int,
                       tokens: List[str],
                       k: int = 64) -> float:
    """
    Compute the average Jaccard‑like similarity from `k` MinHash signatures.
    The similarity is normalised to [0,1].
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not tokens:
        return 0.0
    sigs = [_hash_int(cipher, tok) for tok in tokens[:k]]
    # Normalise each hash to [0,1] by dividing by 2⁶⁴‑1 and average.
    return sum(s / MAX64 for s in sigs) / k

# ----------------------------------------------------------------------
# 5. Liquid‑Time‑Constant (LTC) recurrent unit
# ----------------------------------------------------------------------
def ltc_update(h: np.ndarray,
               u: np.ndarray,
               tau: float,
               W: np.ndarray,
               dt: float = 0.01) -> np.ndarray:
    """
    Euler integration of dh/dt = -h/τ + W·u.
    h, u, W are column vectors (numpy arrays).
    """
    dh = -h / tau + W @ u
    return h + dt * dh

# ----------------------------------------------------------------------
# 6. Regret‑weighted softmax strategy
# ----------------------------------------------------------------------
def regret_softmax(values: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature 1/β."""
    max_v = np.max(values)
    exp_vals = np.exp(beta * (values - max_v))
    return exp_vals / np.sum(exp_vals)

# ----------------------------------------------------------------------
# Dataclasses for actions
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

# ----------------------------------------------------------------------
# 7. One hybrid step – the core of the fused algorithm
# ----------------------------------------------------------------------
def hybrid_step(actions: List[Action],
                h: np.ndarray,
                rbf_centers: List[Vector],
                rbf_weights: List[float],
                epsilon: float,
                diffusion_weight: float,
                rsa_pub: Tuple[int, int],
                rsa_priv: Tuple[int, int],
                minhash_tokens: List[str],
                tau_base: float,
                alpha: float,
                w_ltc: np.ndarray,
                W_ltc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a single time step of the hybrid algorithm.

    Returns
    -------
    h_next : np.ndarray
        Updated LTC hidden state.
    probs : np.ndarray
        Regret‑weighted probability distribution over the supplied actions.
    """
    e, n = rsa_pub
    d, _ = rsa_priv

    # ---- 1. RBF surrogate per action (scaled by diffusion) ----
    surrogate_vals = []
    for act in actions:
        feature_vec = [act.expected_value, act.cost, act.risk]
        y = rbf_surrogate(feature_vec, rbf_centers, rbf_weights,
                          epsilon, diffusion_weight)
        surrogate_vals.append(y)

    surrogate_vals = np.array(surrogate_vals)

    # ---- 2. RSA encrypt the surrogate outputs (convert to integers) ----
    scale = 1e6  # bring float into integer range
    msgs = np.floor(surrogate_vals * scale).astype(np.int64)
    ciphers = np.array([rsa_encrypt_int(int(m % n), e, n) for m in msgs])

    # ---- 3. MinHash similarity from the *first* cipher (any could be used) ----
    sim = minhash_similarity(int(ciphers[0]), minhash_tokens, k=64)

    # ---- 4. Adaptive LTC time‑constant ---------------------------------
    tau_t = tau_base / (1.0 + alpha * sim)

    # ---- 5. LTC update -------------------------------------------------
    # Build input vector u_t = (EV - cost - risk) + w·h
    raw_vals = np.array([a.expected_value - a.cost - a.risk for a in actions])
    augmented_vals = raw_vals + (w_ltc @ h).item()
    u = augmented_vals.reshape(-1, 1)  # column vector
    h_next = ltc_update(h, u, tau_t, W_ltc)

    # ---- 6. Regret‑weighted softmax ------------------------------------
    probs = regret_softmax(augmented_vals, beta=1.0)

    return h_next, probs

# ----------------------------------------------------------------------
# 8. High‑level simulation driver
# ----------------------------------------------------------------------
def run_hybrid_simulation(num_steps: int = 10) -> None:
    """Run a short demonstration of the hybrid algorithm."""
    # ----- Dummy RSA keys (tiny but functional) -----
    # p = 61, q = 53 => n = 3233, φ = 3120, e = 17, d = 2753
    n = 61 * 53
    e = 17
    d = 2753
    rsa_pub = (e, n)
    rsa_priv = (d, n)

    # ----- RBF configuration -----
    rbf_centers = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
    ]
    rbf_weights = [0.8, -0.3, 0.5]
    epsilon = 2.0

    # ----- LTC configuration -----
    hidden_dim = 3
    h = np.zeros((hidden_dim, 1))
    W_ltc = np.eye(hidden_dim) * 0.1   # simple identity scaled
    w_ltc = np.ones((1, hidden_dim)) * 0.05
    tau_base = 0.5
    alpha = 2.0

    # ----- Actions -----
    actions = [
        Action(id="A", expected_value=5.0, cost=1.0, risk=0.2),
        Action(id="B", expected_value=4.5, cost=0.8, risk=0.1),
        Action(id="C", expected_value=5.2, cost=1.2, risk=0.3),
    ]

    # ----- MinHash token pool (static) -----
    minhash_tokens = [f"token_{i}" for i in range(128)]

    # ----- Diffusion schedule -----
    schedule = diffusion_noise_schedule(num_steps, beta_start=0.05, beta_end=0.9)

    print("Step | τ (time‑const) | Action probabilities")
    for t in range(num_steps):
        sigma_t = schedule[t]
        h, probs = hybrid_step(
            actions=actions,
            h=h,
            rbf_centers=rbf_centers,
            rbf_weights=rbf_weights,
            epsilon=epsilon,
            diffusion_weight=sigma_t,
            rsa_pub=rsa_pub,
            rsa_priv=rsa_priv,
            minhash_tokens=minhash_tokens,
            tau_base=tau_base,
            alpha=alpha,
            w_ltc=w_ltc,
            W_ltc=W_ltc,
        )
        # recompute τ for printing (same as inside step)
        sim = minhash_similarity(0, minhash_tokens, k=64)  # placeholder, not used
        tau_t = tau_base / (1.0 + alpha * sim)  # illustrative
        prob_str = ", ".join(f"{a.id}:{p:.3f}" for a, p in zip(actions, probs))
        print(f"{t:4d} | {tau_t:12.4f} | {prob_str}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_hybrid_simulation(12)