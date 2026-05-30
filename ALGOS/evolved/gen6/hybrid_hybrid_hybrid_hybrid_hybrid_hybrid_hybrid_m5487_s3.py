# DARWIN HAMMER — match 5487, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1954_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_liquid_hybrid_pheromone_hyb_m1131_s1.py (gen5)
# born: 2026-05-30T00:02:11Z

"""Hybrid Koopman‑Regret‑MinHash‑LTC Algorithm.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1954_s0.py (Koopman + Regret Engine)
- hybrid_hybrid_hybrid_liquid_hybrid_pheromone_hyb_m1131_s1.py (MinHash signatures + Liquid‑Time‑Constant linear model)

Mathematical bridge:
1. The Koopman observable lifting maps raw audit findings 𝑥∈ℝⁿ to a lifted state 𝑧 = K·x
   (K is the Koopman matrix from the “ternary lens audit”).
2. The lifted state is turned into a token set (via a simple stringification) and fed to the
   MinHash pipeline of the second parent, yielding a signature σ.
3. The signature similarity 𝑠 = Sim(σ, σ_ref) is injected as the scalar “interaction term” I·𝑠
   inside the Liquid‑Time‑Constant (LTC) affine map  f(z) = z·W + I·𝑠 + b .
4. The LTC output is interpreted as a regret‑adjusted utility.  Regret weighting
   (exp(−cost)·utility) selects the final action.

The resulting system fuses the linear dynamics of the Koopman operator, the
probabilistic similarity of MinHash, and the regret‑aware decision rule of the
Regret Engine into a single unified pipeline.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Parent‑A data structures (trimmed)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Atomic action with expected value and cost."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


# ----------------------------------------------------------------------
# Parent‑B utilities (MinHash & LTC)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Sequence[str], k: int = 128) -> List[int]:
    """MinHash signature of a token collection."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2 ** 64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def ltc_f(
    x: np.ndarray,
    I: float,
    W: np.ndarray,
    b: np.ndarray,
    sig: Sequence[int],
    sig_ref: Sequence[int],
) -> np.ndarray:
    """Liquid‑Time‑Constant affine map with MinHash interaction."""
    sim = similarity(sig, sig_ref)
    return x @ W + I * sim + b


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def lift_findings(findings: np.ndarray, koopman: np.ndarray) -> np.ndarray:
    """
    Observable lifting via a Koopman matrix K:
        z = K · findings
    """
    if koopman.shape[1] != findings.shape[0]:
        raise ValueError("Koopman matrix column count must match findings dimension")
    return koopman @ findings


def tokenise_lifted_state(z: np.ndarray, precision: int = 4) -> List[str]:
    """
    Convert each component of the lifted vector into a string token.
    Rounding reduces sensitivity to floating‑point noise.
    """
    return [f"dim{i}:{round(v, precision)}" for i, v in enumerate(z)]


def regret_weighted_selection(
    actions: Sequence[MathAction], utility: np.ndarray, similarity_score: float
) -> MathAction:
    """
    Combine LTC utility with a regret weight.
    Regret weight = exp(−cost); final score = (expected_value + utility) * weight * similarity.
    The action with the highest final score is chosen.
    """
    best = None
    best_score = -math.inf
    for act, u in zip(actions, utility):
        weight = math.exp(-act.cost)
        score = (act.expected_value + float(u)) * weight * similarity_score
        if score > best_score:
            best_score = score
            best = act
    if best is None:
        raise RuntimeError("No actions provided")
    return best


def hybrid_decision(
    findings: np.ndarray,
    tokens: Sequence[str],
    actions: Sequence[MathAction],
    koopman: np.ndarray,
    I: float,
    W: np.ndarray,
    b: np.ndarray,
    sig_ref: Sequence[int],
) -> MathAction:
    """
    Full hybrid pipeline:
    1. Lift raw findings with Koopman operator.
    2. Tokenise lifted state and compute MinHash signature.
    3. Feed signature into LTC affine map (produces a utility vector).
    4. Apply regret‑weighted selection to obtain the final action.
    """
    # 1. Koopman lifting
    z = lift_findings(findings, koopman)

    # 2. Tokenisation + MinHash
    lifted_tokens = tokenise_lifted_state(z)
    sig = signature(lifted_tokens)

    # 3. LTC evaluation (utility per action dimension)
    # We assume W has shape (len(z), len(actions)) and b matches actions count.
    utility = ltc_f(z, I, W, b, sig, sig_ref)

    # 4. Regret‑weighted decision
    sim_score = similarity(sig, sig_ref)
    chosen = regret_weighted_selection(actions, utility, sim_score)
    return chosen


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data
    dim_findings = 6
    n_actions = 4
    np.random.seed(0)
    random.seed(0)

    findings = np.random.randn(dim_findings)

    # Random Koopman matrix (square for simplicity)
    koopman = np.random.randn(dim_findings, dim_findings)

    # LTC parameters
    I = 0.75
    W = np.random.randn(dim_findings, n_actions)
    b = np.random.randn(n_actions)

    # Reference signature (from a fixed reference token set)
    ref_tokens = [f"ref{i}" for i in range(10)]
    sig_ref = signature(ref_tokens)

    # Define a few actions
    actions = [
        MathAction(id="A", expected_value=1.2, cost=0.1),
        MathAction(id="B", expected_value=0.8, cost=0.05),
        MathAction(id="C", expected_value=1.5, cost=0.3),
        MathAction(id="D", expected_value=0.4, cost=0.0),
    ]

    # Run the hybrid decision
    chosen_action = hybrid_decision(
        findings=findings,
        tokens=[],  # not used directly; tokens are derived from lifted state
        actions=actions,
        koopman=koopman,
        I=I,
        W=W,
        b=b,
        sig_ref=sig_ref,
    )

    print(f"Chosen action: {chosen_action.id} (expected_value={chosen_action.expected_value}, cost={chosen_action.cost})")