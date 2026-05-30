# DARWIN HAMMER — match 4289, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_liquid_time_c_hybrid_hybrid_model__m158_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_bayes_claim_k_m1228_s1.py (gen4)
# born: 2026-05-29T23:54:41Z

"""Hybrid Fusion of MinHash‑based Liquid Time Constant & Bayesian Fold‑Change Detection

Parents:
- hybrid_hybrid_liquid_time_c_hybrid_hybrid_model__m158_s0.py (MinHash similarity → adaptive τ)
- hybrid_hybrid_hybrid_hybrid_hybrid_bayes_claim_k_m1228_s1.py (Regex‑filtered tokens → Bayesian hypothesis → diffusion‑driven τ)

Mathematical bridge:
The MinHash similarity `s` is treated as an evidence signal that updates a
Bayesian hypothesis `p`.  The posterior `p̂` then modulates the effective
liquid‑time constant `τ_eff` used both in the fold‑change detection Euler
integration and in the diffusion‑forcing term.  Thus the two topologies are
merged through the chain  

    tokens ──► MinHash similarity ──► Bayesian update ──► τ_eff
                                          │
                                          ▼
                     Euler‑integrated state (x,y) with diffusion noise

The module provides three core functions that demonstrate this fused dynamics.
"""

import hashlib
import math
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# MinHash utilities (Parent A)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using Blake2b (64‑bit output)."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], num_perm: int = 64) -> np.ndarray:
    """Compute MinHash signature for a set of tokens."""
    sig = np.full(num_perm, MAX64, dtype=np.uint64)
    for token in tokens:
        for i in range(num_perm):
            sig[i] = min(sig[i], _hash(i, token))
    return sig


def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    return float(np.mean(sig1 == sig2))


# ----------------------------------------------------------------------
# Regex‑based token filter (Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def filter_tokens(tokens: List[str]) -> List[str]:
    """Keep tokens that match any of the decision‑hygiene regexes."""
    patterns = (EVIDENCE_RE, PLANNING_RE)
    return [t for t in tokens if any(p.search(t) for p in patterns)]


# ----------------------------------------------------------------------
# Simple Bayesian update (Parent B)
# ----------------------------------------------------------------------
def gaussian_likelihood(x: float, mu: float, sigma: float) -> float:
    """Likelihood of observing x under N(mu, sigma²)."""
    if sigma <= 0:
        return 0.0
    coeff = 1.0 / (math.sqrt(2 * math.pi) * sigma)
    return coeff * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def bayesian_update(prior: float, measurement: float, noise_std: float) -> float:
    """
    One‑step Bayesian update with a Gaussian likelihood.
    Prior and posterior are treated as probabilities (0‑1).
    """
    # Prior is a probability; treat it as weight on hypothesis H=1 vs H=0.
    # Likelihood for H=1 is Gaussian centred at 1, for H=0 at 0.
    lik1 = gaussian_likelihood(measurement, 1.0, noise_std)
    lik0 = gaussian_likelihood(measurement, 0.0, noise_std)
    unnorm_post = prior * lik1
    unnorm_neg = (1 - prior) * lik0
    normalizer = unnorm_post + unnorm_neg + 1e-12
    return unnorm_post / normalizer


# ----------------------------------------------------------------------
# Diffusion schedule (simplified)
# ----------------------------------------------------------------------
def cosine_alpha_schedule(T: int = 1000) -> np.ndarray:
    """Cosine schedule for ᾱ_t used in diffusion forcing."""
    timesteps = np.arange(T)
    alphas_cumprod = np.cos(((timesteps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod


ALPHA_BAR = cosine_alpha_schedule(500)  # pre‑computed schedule


# ----------------------------------------------------------------------
# Core hybrid dynamics
# ----------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def compute_tau(posterior: float, base_tau: float = 1.0, alpha: float = 1.0) -> float:
    """
    Effective liquid‑time constant modulated by posterior probability.
    Larger posterior → smaller τ (faster response).
    """
    return base_tau / (1.0 + alpha * posterior)


def hybrid_state_step(
    x: float,
    y: float,
    tokens_a: List[str],
    tokens_b: List[str],
    dt: float,
    base_tau: float,
    alpha_tau: float,
    weight_matrix: np.ndarray,
    bias: np.ndarray,
) -> Tuple[float, float, float]:
    """
    One integration step that fuses:
      • MinHash similarity → Bayesian posterior → τ_eff
      • Fold‑change detection via Euler integration
      • Diffusion forcing using ᾱ schedule
    Returns updated (x, y, posterior).
    """
    # 1️⃣ Token preparation and MinHash similarity
    filtered_a = filter_tokens(tokens_a)
    filtered_b = filter_tokens(tokens_b)
    sig_a = minhash_signature(filtered_a, num_perm=64)
    sig_b = minhash_signature(filtered_b, num_perm=64)
    s = minhash_similarity(sig_a, sig_b)  # similarity ∈ [0,1]

    # 2️⃣ Bayesian update – treat similarity as noisy measurement
    measurement = s  # observed similarity
    posterior = bayesian_update(prior=0.5, measurement=measurement, noise_std=0.1)

    # 3️⃣ Effective time constant
    tau_eff = compute_tau(posterior, base_tau=base_tau, alpha=alpha_tau)

    # 4️⃣ Fold‑change detection term f = σ(W·[x; I; s] + b)
    #    I is an external input; we reuse the similarity as a proxy.
    I = s
    vec = np.array([x, I, s])
    f = float(sigmoid(weight_matrix @ vec + bias))

    # 5️⃣ Euler integration of the coupled ODEs
    dxdt = -(1.0 / tau_eff + f) * x + f * 1.0  # A is set to 1 for simplicity
    dydt = (1.0 / tau_eff) * y  # mirrors the sign pattern of the original hybrid_step
    x_new = x + dxdt * dt
    y_new = y + dydt * dt

    # 6️⃣ Diffusion forcing (adds stochastic noise)
    t_idx = min(int((1.0 - posterior) * (len(ALPHA_BAR) - 1)), len(ALPHA_BAR) - 1)
    alpha_t = ALPHA_BAR[t_idx]
    eps = random.gauss(0.0, 1.0)
    x_noisy = math.sqrt(alpha_t) * I + math.sqrt(1.0 - alpha_t) * eps
    # Mix diffusion noise into the state (simple additive blend)
    x_new = 0.9 * x_new + 0.1 * x_noisy

    return x_new, y_new, posterior


# ----------------------------------------------------------------------
# Convenience wrapper exposing the three core operations
# ----------------------------------------------------------------------
def compute_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Public helper: filter → MinHash → similarity."""
    return minhash_similarity(
        minhash_signature(filter_tokens(tokens_a)),
        minhash_signature(filter_tokens(tokens_b)),
    )


def update_hypothesis(prior: float, similarity: float) -> float:
    """Public helper: Bayesian update using similarity as observation."""
    return bayesian_update(prior, similarity, noise_std=0.1)


def run_hybrid_simulation(
    steps: int = 20,
    dt: float = 0.05,
    base_tau: float = 1.0,
    alpha_tau: float = 2.0,
) -> Dict[str, List[float]]:
    """Run a short deterministic‑stochastic simulation and return trajectories."""
    # Random token pools for demonstration
    vocab = [
        "evidence", "verify", "plan", "schedule", "random", "noise",
        "hash", "document", "test", "smoke", "unknown", "data"
    ]
    state_x, state_y = 0.0, 0.0
    posterior = 0.5

    # Fixed weight matrix and bias (2‑dimensional for simplicity)
    W = np.array([0.3, -0.2, 0.5])  # shape (3,)
    b = np.array([-0.1])

    traj = {"x": [], "y": [], "posterior": []}

    for _ in range(steps):
        # Randomly sample two token lists
        tokens_a = random.sample(vocab, k=random.randint(3, 6))
        tokens_b = random.sample(vocab, k=random.randint(3, 6))

        state_x, state_y, posterior = hybrid_state_step(
            state_x,
            state_y,
            tokens_a,
            tokens_b,
            dt,
            base_tau,
            alpha_tau,
            weight_matrix=W,
            bias=b,
        )
        traj["x"].append(state_x)
        traj["y"].append(state_y)
        traj["posterior"].append(posterior)

    return traj


if __name__ == "__main__":
    # Smoke test: run a tiny simulation and print final values
    result = run_hybrid_simulation(steps=10)
    print("Final x:", result["x"][-1])
    print("Final y:", result["y"][-1])
    print("Final posterior:", result["posterior"][-1])