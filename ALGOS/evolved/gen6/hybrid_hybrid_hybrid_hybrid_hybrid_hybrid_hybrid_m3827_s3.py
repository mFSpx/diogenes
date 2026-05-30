# DARWIN HAMMER — match 3827, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_temporal_moti_m1701_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m521_s1.py (gen5)
# born: 2026-05-29T23:51:50Z

"""Hybrid Allocation‑Similarity‑Motif‑SSM Engine
Parents:
- hybrid_hybrid_hybrid_ternar_hybrid_temporal_moti_m1701_s2.py (joint motif scoring &
  deterministic + stochastic resource allocation)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m521_s1.py (state‑space model,
  entropy‑weighted semiseparable causal matrix, morphology health weighting,
  work‑share lane allocation)

Mathematical Bridge
------------------
Both parents treat a *scalar weight* that modulates a linear transform:

* Parent A* builds a joint score `J_i = support_i·(1+z_i)·(1+SSIM_i)` and uses it to
  distribute a total resource `U` among groups.

* Parent B* builds an entropy scalar `E = p·H(hit)+(1‑p)·H(miss)` and multiplies a
  semiseparable causal matrix `C` by `E`; the resulting matrix is then fed
  through a simple state‑space model (SSM) whose output is finally weighted by
  a morphology‑derived health score.

The hybrid therefore:
1. Encodes motifs → binary vectors.
2. Computes a *joint weight* `J_i` exactly as Parent A.
3. Normalises `J_i` to a probability `p_i` and derives an *expected entropy*
   scalar `E_i` per motif (the bridge to Parent B).
4. Builds a semiseparable causal matrix `C_i` for each motif and scales it by
   `E_i`.
5. Propagates `C_i` through a lightweight SSM, obtaining a projection vector
   `y_i`.
6. Multiplies `y_i` by a morphology health factor `H_i`.
7. Allocates a total resource `U`:
   - deterministic fraction `α·U` split equally,
   - stochastic fraction `(1‑α)·U` split proportionally to the *final scores*
     `S_i = H_i·‖y_i‖₂`.
8. Finally distributes each group’s stochastic share across work‑share lanes
   using the lane‑specific `llm_share_pct` and `proof_required` flags (Parent B).

The implementation below follows this pipeline and provides three public
functions that demonstrate the hybrid operation."""
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
DETERMINISTIC_PCT = 90.0  # percent of total units allocated deterministically
SSIM_EPS = 1e-12  # avoid division by zero in SSIM


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class Lane:
    name: str
    llm_share_pct: float  # percentage of lane's stochastic share
    proof_required: bool   # if True, lane gets a small bonus


# ----------------------------------------------------------------------
# Utility functions (Parent A)
# ----------------------------------------------------------------------
def ssim_binary(v1: np.ndarray, v2: np.ndarray) -> float:
    """Structural similarity for binary vectors – Jaccard‑like formulation."""
    inter = np.logical_and(v1, v2).sum()
    union = np.logical_or(v1, v2).sum()
    return (2.0 * inter + SSIM_EPS) / (v1.sum() + v2.sum() + SSIM_EPS)


def encode_patterns(patterns: Iterable[str]) -> List[np.ndarray]:
    """
    Encode each textual pattern as a binary feature vector.
    Simple character‑presence encoding over the ASCII range 32‑126.
    """
    vectors = []
    for pat in patterns:
        vec = np.zeros(95, dtype=np.int8)  # printable ASCII chars
        for ch in pat:
            idx = ord(ch) - 32
            if 0 <= idx < 95:
                vec[idx] = 1
        vectors.append(vec)
    return vectors


def compute_joint_scores(
    supports: List[float],
    vectors: List[np.ndarray],
    proto: np.ndarray,
) -> List[float]:
    """
    Implements Parent A's joint score:
        J_i = support_i * (1 + z_i) * (1 + SSIM_i)
    where z_i is the z‑score of the support distribution.
    """
    if len(supports) != len(vectors):
        raise ValueError("supports and vectors must have the same length")
    # SSIM component
    ssim_vals = [ssim_binary(v, proto) for v in vectors]
    # z‑score component
    mu = np.mean(supports)
    sigma = np.std(supports) + 1e-12
    z_vals = [(s - mu) / sigma for s in supports]
    # Joint score
    joint = [
        s * (1.0 + z) * (1.0 + ssim)
        for s, z, ssim in zip(supports, z_vals, ssim_vals)
    ]
    return joint


def possum_filter(vectors: List[np.ndarray], radius: int = 2) -> List[int]:
    """
    Very light spatial filter: keep the first vector of any cluster whose
    Hamming distance ≤ radius. Returns the indices of kept vectors.
    """
    kept = []
    for i, vi in enumerate(vectors):
        duplicate = False
        for j in kept:
            if np.sum(np.bitwise_xor(vi, vectors[j])) <= radius:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return kept


# ----------------------------------------------------------------------
# Utility functions (Parent B)
# ----------------------------------------------------------------------
def expected_entropy(p: float, h_hit: float = 1.0, h_miss: float = 0.5) -> float:
    """
    Entropy used as scalar weight.
    H = p·H(hit) + (1‑p)·H(miss)
    """
    return p * h_hit + (1.0 - p) * h_miss


def semiseparable_causal_matrix(size: int, decay: float = 0.7) -> np.ndarray:
    """
    Build a simple Toeplitz‑like semiseparable matrix:
        C[i, j] = decay^{|i‑j|}
    """
    idx = np.arange(size)
    diff = np.abs(idx[:, None] - idx[None, :])
    return decay ** diff


def ssm_propagate(
    C: np.ndarray,
    init_state: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """
    One‑step linear state‑space propagation for each token.
    x_{t+1} = A x_t + B u_t,   y_t = C x_t
    Returns the concatenated output vector y (shape = (size,)).
    """
    x = init_state.copy()
    outputs = []
    for u in C:  # treat each row of C as an input token vector
        x = A @ x + B @ u
        y = C @ x
        outputs.append(y)
    return np.concatenate(outputs)


def sphericity_index(m: Morphology) -> float:
    """Parent B helper – shape factor."""
    if min(m.length, m.width, m.height) <= 0:
        raise ValueError("dimensions must be positive")
    return (m.length * m.width * m.height) ** (1.0 / 3.0) / m.length


def flatness_index(m: Morphology) -> float:
    """Parent B helper – shape factor."""
    if min(m.length, m.width, m.height) <= 0:
        raise ValueError("dimensions must be positive")
    return (m.length + m.width) / (2.0 * m.height)


def health_score(m: Morphology) -> float:
    """
    Combine sphericity and flatness into a scalar health factor.
    Normalised to [0, 1] by a simple logistic map.
    """
    sph = sphericity_index(m)
    flat = flatness_index(m)
    raw = sph * 0.6 + (1.0 / (1.0 + flat)) * 0.4
    return 1.0 / (1.0 + math.exp(-10 * (raw - 0.5)))  # steep sigmoid


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def build_hybrid_projections(
    supports: List[float],
    patterns: List[str],
    proto_pattern: str,
    morphology: Morphology,
) -> Tuple[List[float], List[np.ndarray]]:
    """
    End‑to‑end hybrid pipeline returning final scores S_i and the
    corresponding SSM output vectors (for possible downstream use).

    Steps:
    1. Encode patterns, compute joint scores J_i (Parent A).
    2. Normalise J_i → probabilities p_i, compute entropy weight E_i (Parent B).
    3. Build a semiseparable causal matrix C_i for each motif and scale by E_i.
    4. Propagate C_i through a tiny SSM.
    5. Weight the SSM output norm by the morphology health factor H.
    """
    # 1. Encode and joint score
    vectors = encode_patterns(patterns)
    proto_vec = encode_patterns([proto_pattern])[0]
    joint = compute_joint_scores(supports, vectors, proto_vec)

    # 2. Probabilities & entropy
    total_joint = sum(joint) + 1e-12
    probs = [j / total_joint for j in joint]
    entropy_weights = [expected_entropy(p) for p in probs]

    # 3. SSM preparation (same A, B for all motifs – small 4‑dim system)
    dim_state = 4
    A = np.eye(dim_state) * 0.9  # mild decay
    B = np.eye(dim_state) * 0.1

    health = health_score(morphology)

    final_scores = []
    projections = []

    for idx, (E, vec) in enumerate(zip(entropy_weights, vectors)):
        # matrix size = length of binary vector (95)
        C = semiseparable_causal_matrix(len(vec), decay=0.8) * E
        init_state = np.zeros(dim_state)
        out_vec = ssm_propagate(C, init_state, A, B)
        # norm of output as raw magnitude
        magnitude = np.linalg.norm(out_vec)
        # final hybrid score
        S = health * magnitude
        final_scores.append(S)
        projections.append(out_vec)

    return final_scores, projections


def allocate_resources(
    final_scores: List[float],
    total_units: float,
    deterministic_pct: float = DETERMINISTIC_PCT,
    lanes: List[Lane] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Hybrid allocation:
    * Deterministic part (α·U) split equally among GROUPS.
    * Stochastic part ((1‑α)·U) split proportionally to final_scores per group.
    * Within each group, the stochastic share is further divided among work‑share
      lanes using `llm_share_pct` and a small bonus if `proof_required` is True.
    Returns a nested dict: {group: {lane_name: allocated_units}}.
    """
    if lanes is None:
        lanes = [
            Lane("llm", 70.0, False),
            Lane("proof", 20.0, True),
            Lane("fallback", 10.0, False),
        ]

    α = deterministic_pct / 100.0
    det_share = α * total_units
    sto_share = (1.0 - α) * total_units

    # deterministic allocation – equal split
    det_per_group = det_share / len(GROUPS)

    # stochastic allocation – proportional to scores
    score_sum = sum(final_scores) + 1e-12
    group_scores = {g: 0.0 for g in GROUPS}
    # simple round‑robin assignment of scores to groups
    for i, sc in enumerate(final_scores):
        grp = GROUPS[i % len(GROUPS)]
        group_scores[grp] += sc

    # compute stochastic share per group
    sto_per_group = {
        g: (group_scores[g] / score_sum) * sto_share for g in GROUPS
    }

    # final nested allocation
    allocation: Dict[str, Dict[str, float]] = {}
    for g in GROUPS:
        group_total = det_per_group + sto_per_group[g]
        lane_alloc: Dict[str, float] = {}
        for lane in lanes:
            base = group_total * (lane.llm_share_pct / 100.0)
            # proof_required gets a 5 % bump of the lane's base amount
            bonus = base * 0.05 if lane.proof_required else 0.0
            lane_alloc[lane.name] = base + bonus
        allocation[g] = lane_alloc
    return allocation


def possum_filter_hybrid(
    patterns: List[str],
    radius: int = 2,
) -> List[str]:
    """
    Apply the possum spatial filter on the binary encodings of patterns.
    Returns the filtered list of patterns (preserves order of first occurrence).
    """
    vectors = encode_patterns(patterns)
    keep_idx = possum_filter(vectors, radius=radius)
    return [patterns[i] for i in keep_idx]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic inputs
    patterns = [
        "alpha", "beta", "gamma", "delta", "epsilon",
        "zeta", "eta", "theta", "iota", "kappa"
    ]
    supports = [random.uniform(0.5, 5.0) for _ in patterns]
    proto = "prototype"

    morph = Morphology(length=1.2, width=0.8, height=0.5, mass=3.0)

    # 1. filter duplicates
    filtered_patterns = possum_filter_hybrid(patterns, radius=1)

    # 2. hybrid projection & scoring
    scores, proj = build_hybrid_projections(
        supports[: len(filtered_patterns)],
        filtered_patterns,
        proto,
        morph,
    )

    # 3. allocate a total of 1000 resource units
    allocation = allocate_resources(scores, total_units=1000.0)

    # 4. print a concise summary
    print("=== Hybrid Allocation Summary ===")
    for grp, lanes in allocation.items():
        total_grp = sum(lanes.values())
        print(f"Group {grp}: total {total_grp:.2f} units")
        for ln, amt in lanes.items():
            print(f"  Lane {ln}: {amt:.2f}")
    print("\nDone.")