# DARWIN HAMMER — match 4159, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s4.py (gen3)
# parent_b: hybrid_nlms_omni_chaotic_sprint_m59_s1.py (gen1)
# born: 2026-05-29T23:53:49Z

"""
Hybrid Regret‑Weighted NLMS‑MinHash Engine
Parents:
- hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s4.py (Regret‑Weighted strategy + MinHash similarity)
- hybrid_nlms_omni_chaotic_sprint_m59_s1.py (NLMS adaptive update + chaotic omni‑front core)

Mathematical Bridge:
The Regret‑Weighted strategy produces a decision vector **d**∈ℝⁿ.
The Ternary‑Decision Hygiene Analyzer yields a ternary vector **t**∈{−1,0,1}ⁿ.
Parent A concatenates a MinHash signature **σ(d)** with **t** and computes Shannon entropy.

Parent B adapts a weight vector **w** via the Normalized Least Mean Squares (NLMS) rule:
 w←w+μ·e·x / (‖x‖²+ε)

The fusion treats the similarity s = similarity(σ(d), σ_ref) (Parent A) as a *dynamic learning‑rate*
μ′ = μ·s in the NLMS update, thereby letting the quality of the current decision (measured by
MinHash similarity to a reference set) directly modulate the adaptation speed of the
chaotic omni‑front weights. The updated **w** is then used to scale the regret‑weighted
decision vector before entropy evaluation, yielding a single unified information‑theoretic
measure that reflects both decision quality and learned dynamics.
"""

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity derived from MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def shannon_entropy(vec: np.ndarray) -> float:
    """Entropy of a probability‑like vector (non‑negative, sums to 1)."""
    if np.any(vec < 0):
        raise ValueError("entropy requires non‑negative entries")
    total = vec.sum()
    if total == 0:
        return 0.0
    p = vec / total
    # avoid log(0)
    p = p[p > 0]
    return -float(np.sum(p * np.log2(p)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ----------------------------------------------------------------------
# Parent‑B building blocks (NLMS core)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Normalized LMS update:
        e = target - w·x
        w←w + μ·e·x / (‖x‖² + ε)
    """
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    return weights + (mu * error / power) * x


# ----------------------------------------------------------------------
# Hybrid Engine
# ----------------------------------------------------------------------
@dataclass
class HybridEngine:
    """
    Combines regret‑weighted decisions with NLMS‑adapted weights.
    - weights: trainable vector used to scale the decision vector.
    - ref_signature: MinHash signature of a reference token set (static).
    - mu_base: base learning rate for NLMS; actual μ = mu_base * similarity.
    """
    weights: np.ndarray
    ref_signature: List[int]
    mu_base: float = 0.5
    eps: float = 1e-9

    def _decision_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Produce a regret‑weighted decision vector from tokens.
        For illustration we map each token to a random float in [0,1]
        and apply a sigmoid to obtain values in (0,1).
        """
        rng = np.random.default_rng(seed=hash("".join(tokens)) & ((1 << 32) - 1))
        raw = rng.random(len(self.weights))
        return sigmoid(raw)

    def _ternary_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Produce a ternary hygiene vector: -1, 0, 1 based on token hash parity.
        """
        vec = np.empty(len(self.weights), dtype=int)
        for i, token in enumerate(tokens):
            h = _hash(i, token)
            vec[i] = -1 if h % 3 == 0 else (1 if h % 3 == 1 else 0)
        # If fewer tokens than dimension, pad with zeros
        if len(tokens) < len(self.weights):
            vec[len(tokens) :] = 0
        return vec.astype(float)

    def hybrid_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Concatenate the weighted decision vector with the ternary vector,
        then scale by the adaptive NLMS weights.
        """
        d = self._decision_vector(tokens)          # regret‑weighted part
        t = self._ternary_vector(tokens)           # hygiene part
        combined = np.concatenate([d, t])
        # Ensure weights match combined length (duplicate if necessary)
        if self.weights.size != combined.size:
            # simple tiling to match size
            repeats = int(np.ceil(combined.size / self.weights.size))
            w = np.tile(self.weights, repeats)[: combined.size]
        else:
            w = self.weights
        return combined * w

    def step(self, tokens: List[str], target: float) -> float:
        """
        Perform one hybrid iteration:
        1. Compute MinHash signature of current tokens.
        2. Derive similarity s to the reference signature.
        3. Adjust learning rate μ = mu_base * s.
        4. Build hybrid vector x.
        5. NLMS‑update the internal weights using x and target.
        6. Return entropy of the scaled hybrid vector.
        """
        # 1‑2: similarity‑driven learning rate
        cur_sig = signature(tokens, k=len(self.ref_signature))
        s = similarity(cur_sig, self.ref_signature)
        mu_eff = self.mu_base * s

        # 3‑4: construct feature vector
        x = self.hybrid_vector(tokens)

        # 5: NLMS update (weights adapt to the *current* x)
        self.weights = nlms_update(self.weights, x, target, mu=mu_eff, eps=self.eps)

        # 6: entropy of the (absolute) scaled vector
        entropy = shannon_entropy(np.abs(x))
        return entropy


# ----------------------------------------------------------------------
# Public API functions
# ----------------------------------------------------------------------
def create_engine(
    dimension: int,
    reference_tokens: List[str],
    mu_base: float = 0.5,
) -> HybridEngine:
    """Factory that builds a HybridEngine with random initial weights."""
    rng = np.random.default_rng()
    init_weights = rng.random(dimension)
    ref_sig = signature(reference_tokens, k=128)
    return HybridEngine(weights=init_weights, ref_signature=ref_sig, mu_base=mu_base)


def run_hybrid_sequence(
    engine: HybridEngine,
    token_sequences: List[List[str]],
    targets: List[float],
) -> List[float]:
    """
    Executes a series of hybrid steps.
    Returns the list of entropies produced at each step.
    """
    if len(token_sequences) != len(targets):
        raise ValueError("token_sequences and targets must have the same length")
    entropies = []
    for toks, tgt in zip(token_sequences, targets):
        ent = engine.step(toks, tgt)
        entropies.append(ent)
    return entropies


def evaluate_similarity_to_reference(
    engine: HybridEngine,
    tokens: List[str],
) -> float:
    """Utility exposing the pure MinHash similarity used inside the engine."""
    cur_sig = signature(tokens, k=len(engine.ref_signature))
    return similarity(cur_sig, engine.ref_signature)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a modest dimensionality
    dim = 16

    # Reference token set (static knowledge base)
    reference = ["alpha", "beta", "gamma", "delta", "epsilon"]

    # Instantiate engine
    eng = create_engine(dimension=dim, reference_tokens=reference, mu_base=0.7)

    # Generate synthetic data
    token_seq = [
        ["alpha", "zeta"],
        ["beta", "theta", "iota"],
        ["lambda", "mu", "nu", "xi"],
        ["omicron", "pi"],
    ]
    # Arbitrary targets (e.g., desired scalar output)
    targets = [0.3, 0.6, 0.1, 0.8]

    # Run hybrid sequence
    entropies = run_hybrid_sequence(eng, token_seq, targets)

    # Print results
    for i, (toks, ent) in enumerate(zip(token_seq, entropies), 1):
        sim = evaluate_similarity_to_reference(eng, toks)
        print(
            f"Step {i}: tokens={toks}, similarity={sim:.3f}, entropy={ent:.4f}"
        )
    print("Final weights (first 5 entries):", eng.weights[:5])