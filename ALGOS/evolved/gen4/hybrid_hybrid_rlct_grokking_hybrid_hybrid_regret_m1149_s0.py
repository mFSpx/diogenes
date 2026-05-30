# DARWIN HAMMER — match 1149, survivor 0
# gen: 4
# parent_a: hybrid_rlct_grokking_hybrid_nlms_omni_cha_m118_s0.py (gen2)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s6.py (gen3)
# born: 2026-05-29T23:33:13Z

"""Hybrid Algorithm: rlct_nlms_omni_chaotic_sprint + regret‑weighted MinHash‑ternary lens

Parents
-------
* **Parent A** – `hybrid_rlct_grokking_hybrid_nlms_omni_cha_m118_s0.py`  
  Provides Real Log‑Canonical‑Threshold (RLCT) estimation from a loss
  sequence and a Normalized Least‑Mean‑Squares (NLMS) adaptive filter.

* **Parent B** – `hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s6.py`  
  Emits a MinHash signature `σ` and a ternary vector `τ`.  From the
  similarity of `σ` to a reference signature a ternary token `τ_s` is
  derived, concatenated with `τ` to form a hybrid state `h`.  The Shannon
  entropy of `h` modulates a regret‑weighted exploration factor.

Mathematical Bridge
-------------------
The bridge is a *temperature* that simultaneously controls:
1. The NLMS step size `μ` – scaled by the inverse RLCT estimate (larger
   RLCT ⇒ flatter loss landscape ⇒ smaller effective step).
2. The regret‑weighted exploration factor – scaled by the Shannon
   entropy of the hybrid discrete state `h`.

Thus the NLMS adaptation and the discrete MinHash‑ternary dynamics are
coupled through a common scalar that reflects both geometric (RLCT) and
informational (entropy) characteristics of the learning process.

The implementation below fuses the core equations of both parents into a
single, self‑contained module."""
import math
import random
import sys
from collections import deque, Counter
from pathlib import Path
import hashlib
import numpy as np

NodeId = str
Edge = tuple  # (src, dst, impedance)


def bayesian_information_criterion(log_likelihood: float,
                                   n_params: int,
                                   n_samples: int) -> float:
    """Standard BIC = -2*logL + n_params*log(n)."""
    return -2.0 * log_likelihood + n_params * math.log(n_samples)


# ----------------------------------------------------------------------
# Parent A core: NLMS and RLCT estimation
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(weights: np.ndarray,
                x: np.ndarray,
                target: float,
                mu: float = 0.5,
                eps: float = 1e-9) -> tuple[np.ndarray, float]:
    """
    Normalized Least‑Mean‑Squares weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector.
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Base step size (0 < mu < 2).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Prediction error (target - y).
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")
    y = nlms_predict(weights, x)
    error = target - y
    power = float(np.dot(x, x)) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error


def _rlct_slope(log_losses: np.ndarray, log_ns: np.ndarray) -> float:
    """
    Simple linear regression slope of log(loss) vs log(sample index).
    The negative slope approximates the RLCT.
    """
    if len(log_losses) < 2:
        return 0.0
    A = np.vstack([log_ns, np.ones_like(log_ns)]).T
    slope, _ = np.linalg.lstsq(A, log_losses, rcond=None)[0]
    return -slope  # RLCT is defined as -slope


def estimate_rlct_from_losses(losses: list[float]) -> float:
    """
    Estimate the Real Log‑Canonical‑Threshold from a history of losses.
    """
    if not losses:
        return 0.0
    ns = np.arange(1, len(losses) + 1, dtype=float)
    log_ns = np.log(ns)
    log_losses = np.log(np.maximum(losses, 1e-12))
    return _rlct_slope(log_losses, log_ns)


# ----------------------------------------------------------------------
# Parent B core: MinHash signature, ternary vector, entropy
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash based on Blake2b."""
    data = seed.to_bytes(8, "big", signed=False) + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: list[str], seed: int, k: int = 128) -> list[int]:
    """
    Compute a MinHash signature of length *k* for *tokens* using *seed*.
    The seed can be any integer (e.g., derived from NLMS weights).
    """
    signature = [2**63 - 1] * k  # initialise with max 64‑bit value
    for token in tokens:
        token_hash = _hash(seed, token)
        for i in range(k):
            # Simple linear family of hash functions: h_i(x) = hash(x) ^ i
            combined = token_hash ^ i
            if combined < signature[i]:
                signature[i] = combined
    return signature


def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    """
    Jaccard‑like similarity for MinHash signatures:
    fraction of equal components.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have the same length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def ternary_token_from_similarity(sim: float) -> int:
    """
    Map similarity to a ternary token:
        sim > 2/3   → +1
        sim < 1/3   → -1
        otherwise  →  0
    """
    if sim > 2.0 / 3.0:
        return +1
    if sim < 1.0 / 3.0:
        return -1
    return 0


def shannon_entropy(state: list[int]) -> float:
    """
    Empirical Shannon entropy of a discrete state vector.
    """
    if not state:
        return 0.0
    counts = Counter(state)
    total = len(state)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        entropy -= p * math.log(p, 2)
    return entropy


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_step(weights: np.ndarray,
                x: np.ndarray,
                target: float,
                tokens: list[str],
                tau: list[int],
                loss_history: deque[float],
                ref_signature: list[int],
                mu_base: float = 0.5,
                eps: float = 1e-9) -> tuple[np.ndarray, float, float]:
    """
    Perform one hybrid iteration that couples NLMS adaptation with the
    MinHash‑ternary state.

    Returns
    -------
    new_weights : np.ndarray
        Updated NLMS weight vector.
    error : float
        Prediction error after the NLMS step.
    mu_eff : float
        Effective step size after RLCT‑ and entropy‑based modulation.
    """
    # ---- 1. NLMS core update (Parent A) ---------------------------------
    new_weights, error = nlms_update(weights, x, target, mu=mu_base, eps=eps)

    # ---- 2. Update loss history and estimate RLCT -----------------------
    loss = (error ** 2) / 2.0  # quadratic loss
    loss_history.append(loss)
    if len(loss_history) > 200:
        loss_history.popleft()
    rlct_est = estimate_rlct_from_losses(list(loss_history))

    # ---- 3. Derive a seed from the continuous weights -------------------
    # Use the integer part of the mean weight scaled to a large integer.
    weight_seed = int(np.round(np.mean(new_weights) * 1e6)) & ((1 << 63) - 1)

    # ---- 4. MinHash signature (Parent B) ---------------------------------
    sigma = minhash_signature(tokens, seed=weight_seed, k=len(ref_signature))

    # ---- 5. Similarity → ternary token τ_s --------------------------------
    sim = similarity(sigma, ref_signature)
    tau_s = ternary_token_from_similarity(sim)

    # ---- 6. Build hybrid discrete state h = [τ_s] ⊕ τ --------------------
    h = [tau_s] + tau

    # ---- 7. Entropy‑driven exploration factor ----------------------------
    ent = shannon_entropy(h)
    # Normalise entropy to [0,1] by dividing by log2(|unique symbols|) max.
    max_entropy = math.log2(len(set(h))) if len(set(h)) > 1 else 1.0
    norm_ent = ent / max_entropy
    exploration_factor = 1.0 + norm_ent  # >1 encourages larger step when high entropy

    # ---- 8. RLCT‑driven temperature --------------------------------------
    # Larger RLCT → flatter landscape → smaller effective step.
    rlct_factor = 1.0 / (1.0 + rlct_est)

    # ---- 9. Effective step size -----------------------------------------
    mu_eff = mu_base * rlct_factor * exploration_factor
    mu_eff = max(0.01, min(mu_eff, 1.9))  # keep within (0,2)

    # ---- 10. Apply the modulated step size to the weights (second pass) ---
    # A second lightweight NLMS pass with the new mu_eff refines the update.
    new_weights, _ = nlms_update(weights, x, target, mu=mu_eff, eps=eps)

    return new_weights, error, mu_eff


def initialize_hybrid(dim: int,
                      token_pool: list[str],
                      tau_len: int = 8,
                      seed: int = 42) -> tuple[np.ndarray,
                                               deque[float],
                                               list[int],
                                               list[int]]:
    """
    Initialise the hybrid system.

    Returns
    -------
    weights : np.ndarray
        Random normalised weight vector of dimension *dim*.
    loss_history : deque
        Empty loss history buffer.
    ref_signature : list[int]
        Reference MinHash signature (fixed for the lifetime of the run).
    tau : list[int]
        Random ternary vector of length *tau_len*.
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=dim).astype(float)

    loss_history: deque[float] = deque(maxlen=500)

    # Fixed reference signature using a deterministic seed
    ref_signature = minhash_signature(token_pool, seed=123456789, k=64)

    # Random ternary vector τ ∈ {‑1,0,+1}
    tau = rng.integers(-1, 2, size=tau_len).tolist()

    return weights, loss_history, ref_signature, tau


def demo_hybrid(iterations: int = 20) -> None:
    """
    Simple smoke‑test that runs the hybrid algorithm on synthetic data.
    """
    # Synthetic problem: learn a linear mapping y = w_true·x
    dim = 10
    rng = np.random.default_rng(0)
    w_true = rng.normal(size=dim)

    # Token pool for MinHash (could be any strings)
    tokens = [f"token_{i}" for i in range(200)]

    # Initialise hybrid state
    weights, loss_hist, ref_sig, tau = initialize_hybrid(dim, tokens, tau_len=5)

    for it in range(iterations):
        x = rng.normal(size=dim)
        target = float(np.dot(w_true, x))  # noiseless target
        weights, err, mu_eff = hybrid_step(weights,
                                           x,
                                           target,
                                           tokens,
                                           tau,
                                           loss_hist,
                                           ref_sig,
                                           mu_base=0.5)

        if (it + 1) % 5 == 0:
            mse = err ** 2
            print(f"Iter {it+1:02d} | Error {err:.4f} | MSE {mse:.4e} | μ_eff {mu_eff:.3f}")

    # Final diagnostics
    final_error = np.linalg.norm(w_true - weights)
    print(f"\nFinal weight Euclidean distance to true: {final_error:.4e}")


if __name__ == "__main__":
    demo_hybrid()