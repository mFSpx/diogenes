# DARWIN HAMMER — match 1441, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m737_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m69_s0.py (gen4)
# born: 2026-05-29T23:36:21Z

"""Hybrid Fusion of Algorithm A and Algorithm B

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m737_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m69_s0.py (Algorithm B)

Mathematical Bridge:
Algorithm A encodes decision‑hygiene features as high‑dimensional multivectors,
weights them with Shannon entropy and feeds them to a Krampus‑Ollivier‑Ricci
curvature computation. Algorithm B provides a confidence‑scaled similarity
`S = w·(0.5·J+0.5·C)·exp(-0.1·E)` and a rotor update
`R(θ) = S·cosθ + 1·sinθ`.

The fusion proceeds as follows:

1.  Multivectors `a` and `b` are built from feature‑count vectors.
2.  Their geometric product is taken as a scalar similarity `G = a·b`.
3.  The entropy of each feature vector supplies `E_A` and `E_B`; their average
    becomes the `E` term used in the weight‑scaled similarity of Algorithm B.
4.  Confidence (from epistemic flags) supplies `w`.
5.  `S` from Algorithm B modulates the rotor update, but we also inject the
    curvature `K` computed from `a` and `b`. The rotor now rotates the pair
    `(S, K)` instead of `(S, 1)`.
6.  A time‑dependent pruning probability `p(t)=exp(-γ·t)` interpolates between
    a pure similarity contribution (`p=1`) and a curvature‑entropy contribution
    (`p=0`).

The final hybrid decision score is

    score = p(t)·R_similarity + (1‑p(t))·R_curvature

where `R_similarity` and `R_curvature` are the rotor outputs with
`(S,1)` and `(S,K)` respectively.

This module implements the fused mathematics with three public functions
demonstrating the hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Global patterns (from Algorithm A)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    r.I,
)

# ----------------------------------------------------------------------
# Epistemic flag definition (from Algorithm B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # 0 .. 10000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")


# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def geometric_product_mv(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar geometric product (inner product) of two multivectors."""
    return float(np.dot(a, b))


def shannon_entropy(vec: np.ndarray) -> float:
    """Shannon entropy of a non‑negative vector (treated as a probability distribution)."""
    prob = vec.astype(float)
    total = prob.sum()
    if total == 0:
        return 0.0
    prob /= total
    prob = prob[prob > 0]  # avoid log(0)
    return -float(np.sum(prob * np.log2(prob)))


def weight_scaled_similarity(J: float, C: float, E: float, confidence_bps: int) -> float:
    """Algorithm B similarity, confidence‑scaled."""
    w = confidence_bps / 10000.0
    return w * (0.5 * J + 0.5 * C) * math.exp(-0.1 * E)


def rotor_update(a: float, b: float, theta: float) -> float:
    """Standard rotor update from Algorithm B."""
    return a * math.cos(theta) + b * math.sin(theta)


def pruning_probability(gamma: float, t: float) -> float:
    """Time‑dependent pruning probability p(t)=exp(-γ·t)."""
    return math.exp(-gamma * t)


# ----------------------------------------------------------------------
# Hybrid building blocks
# ----------------------------------------------------------------------
def text_to_feature_vector(text: str, vocab: List[str]) -> np.ndarray:
    """
    Very simple bag‑of‑words count vector based on a supplied vocabulary.
    The vocabulary can be the union of evidence‑related tokens and any other
    domain‑specific terms.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = np.zeros(len(vocab), dtype=int)
    token_index = {tok: i for i, tok in enumerate(vocab)}
    for tok in tokens:
        idx = token_index.get(tok)
        if idx is not None:
            counts[idx] += 1
    return counts


def compute_ollivier_ricci_curvature(a: np.ndarray, b: np.ndarray) -> float:
    """
    Proxy for Krampus‑Ollivier‑Ricci curvature.
    For two points a, b we use:
        K = 1 - ||a-b|| / (||a|| + ||b|| + ε)
    This yields values in (-∞,1] and captures how close the points are
    relative to their magnitudes.
    """
    eps = 1e-9
    norm_a = np.linalg.norm(a) + eps
    norm_b = np.linalg.norm(b) + eps
    dist = np.linalg.norm(a - b)
    return 1.0 - dist / (norm_a + norm_b)


def hybrid_decision_score(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    flag: CertaintyFlag,
    theta: float,
    gamma: float,
    t: float,
) -> float:
    """
    Full hybrid score:
      1. Build multivectors (normalized count vectors).
      2. Compute geometric product G.
      3. Compute entropies EA, EB and combine as E = (EA+EB)/2.
      4. Compute similarity S via Algorithm B.
      5. Compute curvature K via the proxy.
      6. Rotor update with (S,1) → R_sim and (S,K) → R_curv.
      7. Blend with pruning probability p(t).
    Returns the blended scalar score.
    """
    # 1. Normalization to multivectors
    a_mv = vec_a.astype(float)
    b_mv = vec_b.astype(float)

    # 2. Geometric product
    G = geometric_product_mv(a_mv, b_mv)

    # 3. Entropy term
    EA = shannon_entropy(a_mv)
    EB = shannon_entropy(b_mv)
    E = (EA + EB) / 2.0

    # 4. Weight‑scaled similarity (Algorithm B)
    S = weight_scaled_similarity(J=G, C=G, E=E, confidence_bps=flag.confidence_bps)

    # 5. Curvature term
    K = compute_ollivier_ricci_curvature(a_mv, b_mv)

    # 6. Rotor updates
    R_sim = rotor_update(S, 1.0, theta)        # similarity‑only rotor
    R_curv = rotor_update(S, K, theta)       # curvature‑augmented rotor

    # 7. Time‑dependent blending
    p = pruning_probability(gamma, t)
    return p * R_sim + (1.0 - p) * R_curv


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def demo_hybrid_score():
    """Run a quick demo with synthetic texts and a certainty flag."""
    vocab = ["evidence", "verify", "plan", "step", "risk", "mitigate"]
    txt1 = "Evidence of risk was verified and documented."
    txt2 = "Plan the steps to mitigate the risk and verify evidence."
    vec1 = text_to_feature_vector(txt1, vocab)
    vec2 = text_to_feature_vector(txt2, vocab)

    flag = CertaintyFlag(
        label="FACT",
        confidence_bps=8500,
        authority_class="internal_audit",
        rationale="automated assessment",
        evidence_refs=("ref1", "ref2"),
        generated_at="2026-05-29T00:00:00Z",
    )

    score = hybrid_decision_score(
        vec_a=vec1,
        vec_b=vec2,
        flag=flag,
        theta=math.pi / 6,
        gamma=0.05,
        t=10.0,
    )
    print(f"Hybrid decision score: {score:.6f}")


def rotor_trajectory(flag: CertaintyFlag, steps: int = 8) -> List[float]:
    """
    Produce a list of rotor outputs over a range of angles.
    Demonstrates how confidence influences the amplitude of the trajectory.
    """
    # Use a fixed similarity base (e.g., G=0.7) and zero curvature for trajectory
    G = 0.7
    E = 0.5  # arbitrary entropy proxy
    S = weight_scaled_similarity(J=G, C=G, E=E, confidence_bps=flag.confidence_bps)

    outputs = []
    for i in range(steps):
        theta = (i / steps) * 2 * math.pi
        outputs.append(rotor_update(S, 1.0, theta))
    return outputs


def curvature_vs_entropy_demo():
    """
    Show the interaction between entropy and curvature on the hybrid score.
    Generates random feature vectors with varying sparsity.
    """
    rng = np.random.default_rng(42)
    flag = CertaintyFlag(
        label="PROBABLE",
        confidence_bps=6000,
        authority_class="review_board",
        rationale="sample demo",
    )
    scores = []
    for sparsity in np.linspace(0.1, 0.9, 9):
        # Create two vectors with given sparsity (fraction of zeros)
        size = 12
        mask = rng.random(size) > sparsity
        vec_a = rng.integers(0, 5, size) * mask
        vec_b = rng.integers(0, 5, size) * mask

        score = hybrid_decision_score(
            vec_a=vec_a,
            vec_b=vec_b,
            flag=flag,
            theta=math.pi / 3,
            gamma=0.02,
            t=5.0,
        )
        scores.append((sparsity, score))
    for sparsity, sc in scores:
        print(f"Sparsity {sparsity:.2f} → hybrid score {sc:.6f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Hybrid Decision Score ===")
    demo_hybrid_score()
    print("\n=== Demo: Rotor Trajectory ===")
    flag_demo = CertaintyFlag(
        label="FACT",
        confidence_bps=9000,
        authority_class="system",
        rationale="trajectory demo",
    )
    traj = rotor_trajectory(flag_demo, steps=12)
    print("Rotor outputs:", ["{:.4f}".format(v) for v in traj])
    print("\n=== Demo: Curvature vs Entropy ===")
    curvature_vs_entropy_demo()