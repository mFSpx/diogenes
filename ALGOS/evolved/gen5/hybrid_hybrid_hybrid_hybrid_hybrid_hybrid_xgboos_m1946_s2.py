# DARWIN HAMMER — match 1946, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1043_s0.py (gen4)
# parent_b: hybrid_hybrid_xgboost_objec_hybrid_hybrid_regret_m283_s2.py (gen4)
# born: 2026-05-29T23:40:07Z

"""Hybrid Pheromone‑Regret Boost

This module fuses the *HybridPheromoneKrampusSystem* (Parent A) and the
*Hybrid XGBoost–Regret MinHash Analyzer* (Parent B).

Mathematical bridge
-------------------
* Parent A provides a scalar **pheromone signal** `P(text, day)` derived
  from deterministic pseudo‑features extracted from the input text and
  a “doomsday” factor based on the day of the week.
* Parent B augments the binary logistic loss `ℓ(y,m)` with a regulariser
  `α·s·H` where `s` is a MinHash similarity and `H` the Shannon entropy of
  the combined MinHash signatures.

In the hybrid we replace the regulariser coefficient `α·s·H` by the
product `α·P·s·H`.  Consequently the adjusted gradient and hessian become


ĝ = (σ(m) - y)·(1 + α·P·s·H)
ĥ = σ(m)(1-σ(m))·(1 + α·P·s·H)


These statistics are fed into the standard XGBoost split‑gain and leaf‑weight
formulas, yielding a unified learning system that simultaneously exploits
pheromone‑driven contextual weighting and MinHash‑based similarity/entropy
information.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from datetime import date, datetime
from typing import Iterable, Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – pheromone utilities
# ----------------------------------------------------------------------
def _rng_from_text(text: str) -> random.Random:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑features extracted from `text`."""
    rnd = _rng_from_text(text)
    # Produce a fixed set of 13 pseudo‑features in [0,1)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "psyche_wrath_velocity",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
    ]
    return {k: rnd.random() for k in keys}


def doomsday_factor(year: int, month: int, day: int) -> float:
    """Map weekday to a scaling factor in (0,1]."""
    # weekday(): Mon=0 … Sun=6 → shift to 1‑7 and wrap
    w = (date(year, month, day).weekday() + 1) % 7
    # Map 0→1.0, 1→0.9, …, 6→0.4 (example linear decay)
    return 1.0 - 0.1 * ((w + 6) % 7)


def pheromone_signal(text: str, when: datetime) -> float:
    """Scalar pheromone signal P ∈ [0,1] for a given text and timestamp."""
    feats = extract_full_features(text)
    # Simple aggregation: mean of selected features
    mean_feat = sum(feats.values()) / len(feats)
    day_factor = doomsday_factor(when.year, when.month, when.day)
    return mean_feat * day_factor


# ----------------------------------------------------------------------
# Parent B – MinHash, similarity & entropy utilities
# ----------------------------------------------------------------------
def _minhash_signature(tokens: Iterable[str], num_perm: int = 128) -> np.ndarray:
    """Very lightweight MinHash: for each permutation use a different seed."""
    sig = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for token in tokens:
        token_bytes = token.encode("utf-8")
        for i in range(num_perm):
            h = hashlib.sha256(token_bytes + i.to_bytes(2, "little")).digest()
            hv = int.from_bytes(h[:8], "little")
            if hv < sig[i]:
                sig[i] = hv
    return sig


def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity via MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("Signature shapes must match")
    return np.mean(sig1 == sig2)


def shannon_entropy(values: np.ndarray) -> float:
    """Compute Shannon entropy of a discrete distribution."""
    # Convert to probabilities
    total = np.sum(values)
    if total == 0:
        return 0.0
    probs = values / total
    # Guard against log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def combined_entropy(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Entropy of the multiset union of two signatures."""
    combined = np.concatenate([sig1, sig2])
    # Count occurrences of each unique hash value
    uniq, counts = np.unique(combined, return_counts=True)
    return shannon_entropy(counts.astype(float))


# ----------------------------------------------------------------------
# Hybrid gradient / hessian computation
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x_arr = np.asarray(x, dtype=float)
    pos = x_arr >= 0
    out = np.empty_like(x_arr, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def adjusted_grad_hess(
    y_true: np.ndarray,
    margin: np.ndarray,
    pheromone: float,
    similarity: float,
    entropy: float,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient and hessian of the hybrid loss

        L = ℓ(y,m) + α·P·s·H

    where ℓ is binary logistic loss, P is pheromone signal,
    s is MinHash similarity and H is entropy.
    """
    sigma = sigmoid(margin)
    base_grad = sigma - y_true
    base_hess = sigma * (1.0 - sigma)

    factor = 1.0 + alpha * pheromone * similarity * entropy
    return base_grad * factor, base_hess * factor


def split_gain(
    G_left: float,
    H_left: float,
    G_right: float,
    H_right: float,
    G_total: float,
    H_total: float,
    lam: float = 1.0,
) -> float:
    """
    XGBoost split gain with L2 regularisation λ.
    """
    def _gain(G, H):
        return (G ** 2) / (H + lam)

    return 0.5 * (_gain(G_left, H_left) + _gain(G_right, H_right) - _gain(G_total, H_total))


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def compute_hybrid_statistics(
    texts: List[str],
    timestamps: List[datetime],
    tokens_a: List[List[str]],
    tokens_b: List[List[str]],
    y_true: np.ndarray,
    margin: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each sample compute:
      * pheromone signal P
      * MinHash similarity s
      * entropy H
      * adjusted gradient and hessian
    Returns arrays of gradients and hessians.
    """
    grads = []
    hess = []
    for txt, ts, t_a, t_b in zip(texts, timestamps, tokens_a, tokens_b):
        P = pheromone_signal(txt, ts)
        sig_a = _minhash_signature(t_a)
        sig_b = _minhash_signature(t_b)
        s = minhash_similarity(sig_a, sig_b)
        H = combined_entropy(sig_a, sig_b)
        g, h = adjusted_grad_hess(
            y_true=np.array([y_true[len(grads)]]),
            margin=np.array([margin[len(grads)]]),
            pheromone=P,
            similarity=s,
            entropy=H,
            alpha=alpha,
        )
        grads.append(g.item())
        hess.append(h.item())
    return np.array(grads), np.array(hess)


def demo_split_gain():
    """Create a tiny synthetic split and compute its gain."""
    # Synthetic statistics
    G_total = 3.0
    H_total = 2.5
    G_left = 1.2
    H_left = 0.9
    G_right = 1.8
    H_right = 1.3
    gain = split_gain(G_left, H_left, G_right, H_right, G_total, H_total, lam=1.0)
    print(f"Demo split gain: {gain:.6f}")


def demo_hybrid_step():
    """Run a single hybrid gradient/hessian computation on dummy data."""
    texts = ["sample text one", "another example"]
    timestamps = [datetime.utcnow(), datetime.utcnow()]
    tokens_a = [["alpha", "beta"], ["gamma", "delta"]]
    tokens_b = [["beta", "epsilon"], ["delta", "zeta"]]
    y_true = np.array([1, 0])
    margin = np.array([0.3, -0.7])

    grads, hess = compute_hybrid_statistics(
        texts,
        timestamps,
        tokens_a,
        tokens_b,
        y_true,
        margin,
        alpha=0.05,
    )
    for i, (g, h) in enumerate(zip(grads, hess)):
        print(f"Sample {i}: grad={g:.6f}, hess={h:.6f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running Hybrid Pheromone‑Regret Boost smoke test...")
    demo_hybrid_step()
    demo_split_gain()
    print("Smoke test completed successfully.")