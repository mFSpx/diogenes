# DARWIN HAMMER — match 3802, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_minimu_m1728_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2563_s1.py (gen4)
# born: 2026-05-29T23:51:42Z

"""
Hybrid Regret‑Entropy LTC with Ternary Lens Filtering
====================================================

Parent A: ``hybrid_hybrid_hybrid_regret_hybrid_hybrid_minimu_m1728_s0.py`` – provides
MinHash signatures, a Jaccard‑like similarity, and a regret‑matching strategy that
uses the Shannon entropy of the signature as a temperature.

Parent B: ``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2563_s1.py`` – defines a
Liquid‑Time‑Constant (LTC) recurrent cell whose input‑dependent similarity term
derives from MinHash signatures and whose diffusion (noise) forcing is modulated
by a ternary lens audit.

**Mathematical bridge** – The MinHash signature is interpreted both as a discrete
probability distribution (for entropy) *and* as a feature vector (for similarity).
The similarity between the current and a reference signature feeds the ternary
lens, which produces a scalar ``lens_factor`` ∈ {‑1, 0, 1}.  This factor scales the
diffusion coefficient of the LTC cell, while the entropy scales the temperature
of the regret‑matching softmax.  The hybrid step therefore couples the regret
strategy and the LTC dynamics through a shared MinHash‑based representation.

The module implements:
* ``minhash_signature`` – MinHash of a token set.
* ``signature_entropy`` – Shannon entropy of the normalized signature histogram.
* ``regret_strategy`` – Regret‑matching mixed strategy with entropy‑scaled temperature.
* ``ternary_lens`` – Maps similarity to {‑1, 0, 1}.
* ``ltc_update`` – Discrete LTC state update with diffusion scaled by the lens factor.
* ``hybrid_step`` – One iteration that updates regrets, computes a strategy, and
  advances the LTC state using the shared MinHash information.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np

__all__ = [
    "minhash_signature",
    "signature_entropy",
    "regret_strategy",
    "ternary_lens",
    "ltc_update",
    "hybrid_step",
]

# ----------------------------------------------------------------------
# MinHash utilities (from Parent A)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        # maximal hash value for empty set
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Entropy of a MinHash signature (bridge to Parent A)
# ----------------------------------------------------------------------
def signature_entropy(sig: List[int]) -> float:
    """
    Treat the signature as a discrete distribution over hash buckets.
    Entropy H = - Σ p_i log p_i, where p_i is the relative frequency of each bucket value.
    """
    if not sig:
        return 0.0
    # Count occurrences of each distinct hash value
    counts: Dict[int, int] = {}
    for h in sig:
        counts[h] = counts.get(h, 0) + 1
    total = len(sig)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p + 1e-12)  # small epsilon for numerical stability
    return entropy


# ----------------------------------------------------------------------
# Regret‑matching strategy (from Parent A) with entropy‑scaled temperature
# ----------------------------------------------------------------------
def regret_strategy(
    cumulative_regrets: np.ndarray,
    entropy: float,
    base_temperature: float = 1.0,
) -> np.ndarray:
    """
    Convert cumulative regrets into a mixed strategy using a softmax.
    The temperature is increased proportionally to the signature entropy,
    encouraging more exploration when the token set is uncertain.
    """
    if cumulative_regrets.ndim != 1:
        raise ValueError("cumulative_regrets must be a 1‑D array")
    temperature = base_temperature * (1.0 + entropy)
    # Prevent division by zero
    if temperature <= 0.0:
        temperature = 1e-6
    # Standard regret‑matching uses positive regrets only
    positive = np.maximum(cumulative_regrets, 0.0)
    if positive.sum() == 0:
        # Uniform strategy if no positive regrets
        return np.full_like(positive, 1.0 / positive.size)
    # Softmax with temperature
    scaled = positive / temperature
    max_val = np.max(scaled)  # for numerical stability
    exp_vals = np.exp(scaled - max_val)
    probs = exp_vals / exp_vals.sum()
    return probs


# ----------------------------------------------------------------------
# Ternary lens audit (from Parent B)
# ----------------------------------------------------------------------
def ternary_lens(similarity: float, low: float = 0.3, high: float = 0.7) -> int:
    """
    Map similarity to a ternary decision:
    -1 : reject (similarity < low)
     0 : neutral (low ≤ similarity ≤ high)
    +1 : accept (similarity > high)
    """
    if similarity < low:
        return -1
    if similarity > high:
        return 1
    return 0


# ----------------------------------------------------------------------
# Liquid‑Time‑Constant (LTC) recurrent cell (from Parent B)
# ----------------------------------------------------------------------
def ltc_update(
    state: np.ndarray,
    input_vec: np.ndarray,
    weight_in: np.ndarray,
    weight_rec: np.ndarray,
    bias: np.ndarray,
    dt: float = 0.01,
    diffusion_coeff: float = 0.01,
    lens_factor: int = 0,
) -> np.ndarray:
    """
    Discrete LTC update:
        dh = (-h + f(W_in·x + W_rec·h + b)) * dt + σ * η
    where:
        f is a smooth non‑linearity (tanh),
        σ = diffusion_coeff * (1 + lens_factor)  (lens_factor ∈ {‑1,0,1})
        η ~ N(0, I)  (Gaussian noise)
    """
    if state.shape != weight_rec.shape[0:1]:
        raise ValueError("state dimension mismatch")
    # Linear combination
    lin = weight_in @ input_vec + weight_rec @ state + bias
    # Non‑linear activation
    activated = np.tanh(lin)
    # Core deterministic update
    dh = (-state + activated) * dt
    # Diffusion term (noise)
    sigma = diffusion_coeff * (1.0 + lens_factor)  # lens_factor can reduce or increase noise
    if sigma > 0:
        noise = sigma * np.random.randn(*state.shape)
        dh += noise
    return state + dh


# ----------------------------------------------------------------------
# Hybrid step that intertwines regret matching and LTC dynamics
# ----------------------------------------------------------------------
def hybrid_step(
    cumulative_regrets: np.ndarray,
    token_set: Iterable[str],
    prev_state: np.ndarray,
    ref_signature: List[int],
    params: dict,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Perform one hybrid iteration.

    Returns:
        new_regrets   – updated cumulative regrets (same shape as input)
        new_state     – updated LTC state
        cur_signature – MinHash signature of the current token set (used as reference for next step)
    """
    # 1️⃣ Compute current MinHash signature and its entropy
    cur_signature = minhash_signature(token_set, k=params.get("minhash_k", 128))
    ent = signature_entropy(cur_signature)

    # 2️⃣ Regret‑matching strategy (used only to update regrets here)
    strategy = regret_strategy(cumulative_regrets, entropy=ent, base_temperature=params.get("base_temp", 1.0))
    # Simple stochastic regret update: increase regret for actions not taken proportionally to utility
    # For demonstration we treat the strategy as a probability distribution over actions and sample an action.
    chosen_idx = np.random.choice(len(strategy), p=strategy)
    # Mock utility: reward = 1 if chosen action index matches a random “optimal” index
    optimal_idx = params.get("optimal_idx", 0)
    reward = 1.0 if chosen_idx == optimal_idx else 0.0
    # Regret update: r_i ← r_i + (u_i - ū), where u_i is reward for action i, ū is expected reward
    expected_reward = (strategy * reward).sum()
    instantaneous_regret = np.full_like(cumulative_regrets, -expected_reward)
    instantaneous_regret[chosen_idx] += reward
    new_regrets = cumulative_regrets + instantaneous_regret

    # 3️⃣ Compute similarity to reference signature and obtain lens factor
    sim = minhash_similarity(cur_signature, ref_signature)
    lens = ternary_lens(sim, low=params.get("lens_low", 0.3), high=params.get("lens_high", 0.7))

    # 4️⃣ Prepare LTC input vector (use entropy and similarity as features)
    input_vec = np.array([ent, sim], dtype=np.float64)

    # 5️⃣ LTC state update
    new_state = ltc_update(
        state=prev_state,
        input_vec=input_vec,
        weight_in=params["weight_in"],
        weight_rec=params["weight_rec"],
        bias=params["bias"],
        dt=params.get("dt", 0.01),
        diffusion_coeff=params.get("diffusion_coeff", 0.01),
        lens_factor=lens,
    )

    return new_regrets, new_state, cur_signature


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Hyper‑parameters
    ACTIONS = 5
    STATE_DIM = 4

    # Initialize cumulative regrets to zeros
    regrets = np.zeros(ACTIONS, dtype=np.float64)

    # Initial LTC state
    state = np.zeros(STATE_DIM, dtype=np.float64)

    # Random LTC parameters
    weight_in = np.random.randn(STATE_DIM, 2) * 0.1
    weight_rec = np.random.randn(STATE_DIM, STATE_DIM) * 0.1
    bias = np.random.randn(STATE_DIM) * 0.01

    # Reference signature (empty token set -> maximal hash)
    ref_sig = minhash_signature([], k=128)

    # Parameter bundle
    params = {
        "weight_in": weight_in,
        "weight_rec": weight_rec,
        "bias": bias,
        "minhash_k": 128,
        "base_temp": 1.0,
        "optimal_idx": 2,
        "lens_low": 0.3,
        "lens_high": 0.7,
        "dt": 0.01,
        "diffusion_coeff": 0.02,
    }

    # Simulated token streams
    token_streams = [
        ["alpha", "beta", "gamma"],
        ["delta", "epsilon"],
        ["zeta", "eta", "theta", "iota"],
        ["kappa"],
    ]

    print("Starting hybrid simulation...")
    for step, tokens in enumerate(token_streams, 1):
        regrets, state, cur_sig = hybrid_step(regrets, tokens, state, ref_sig, params)
        # Update reference signature for next iteration
        ref_sig = cur_sig
        print(f"Step {step:02d} | Regrets: {regrets.round(3)} | State: {state.round(3)}")

    print("Simulation completed without errors.")