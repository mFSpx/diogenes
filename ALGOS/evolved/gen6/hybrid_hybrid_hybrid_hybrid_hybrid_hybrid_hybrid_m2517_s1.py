# DARWIN HAMMER — match 2517, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m530_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m211_s0.py (gen4)
# born: 2026-05-29T23:42:38Z

"""Hybrid Regret‑Weighted Gaussian‑Fisher Bandit (HRW‑GF‑Bandit)

Parent algorithms:
- **hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m530_s0.py** – provides
  regret‑weighted ternary lens utilities (signatures, similarity, sigmoid) and
  dataclasses ``MathAction`` / ``MathCounterfactual``.
- **hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m211_s0.py** – provides a
  Gaussian‑beam / Fisher‑information model together with a classic multi‑armed
  bandit policy and a minimum‑cost tree cost function.

Mathematical bridge:
The ternary vector derived from the regret‑weighted lens is turned into a
signature (a high‑dimensional hash fingerprint).  The *pairwise similarity* of
signatures is interpreted as a “θ” angle for the Gaussian beam.  The Gaussian
beam yields an intensity that is fed to the Fisher‑information formula,
producing a confidence weight.  This confidence weight modulates the bandit
propensity update, effectively coupling the regret‑weighted decision space with
the probabilistic smoothing of the Fisher‑localisation model.  The resulting
weights are finally projected onto the original ternary space via a least‑squares
fit, completing a closed hybrid loop.

The module implements three core hybrid functions:
1. ``hybrid_ternary_signature`` – builds ternary vectors, signatures and
   similarity matrices.
2. ``gaussian_fisher_confidence`` – maps similarity to a Gaussian intensity and
   Fisher confidence.
3. ``bandit_update_with_confidence`` – updates the bandit policy using the
   confidence‑weighted regret signal and projects the updated policy back onto
   the ternary space.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regret‑weighted ternary lens utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        np.frombuffer(
            np.array(list(data), dtype=np.uint8).tobytes(),
            dtype=np.uint8,
        ).view(np.uint64)[:1],
        "big",
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Parent B – Gaussian / Fisher / Bandit utilities
# ----------------------------------------------------------------------


_POLICY: Dict[str, Tuple[float, int]] = {}  # action_id → (total_reward, count)
_STORE: float = 0.0  # scalar store influencing confidence
_MEAN_HISTORY: List[np.ndarray] = []  # list of μ vectors over time
_W: np.ndarray = np.array([])  # linear weight matrix (A×A)
_ETA: float = 1.0  # exploration scaling
_ALPHA: float = 0.5  # mixing factor for hybrid index
_NODES: Dict[str, Tuple[float, float]] = {}  # node_id → (x, y)
_EDGES: List[Tuple[str, str]] = []  # edges for minimum cost tree
_ROOT: str = ""  # root node for minimum cost tree


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(nodes: Dict[str, Tuple[float, float]],
              edges: List[Tuple[str, str]],
              root: str,
              path_weight: float = 0.2) -> float:
    """Simple total Euclidean length weighted by ``path_weight``."""
    if root not in nodes:
        raise ValueError("root must be a node in the graph")
    total = 0.0
    for a, b in edges:
        if a not in nodes or b not in nodes:
            continue
        total += path_weight * length(nodes[a], nodes[b])
    return total


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def hybrid_ternary_signature(actions: List[MathAction],
                             k: int = 128) -> Tuple[np.ndarray, List[int], List[List[float]]]:
    """
    Build the ternary vector matrix ``T`` (shape: N×3), compute a signature
    for each action, and return the pairwise similarity matrix ``S``.
    """
    # 1. Ternary vector: [expected_value - cost, risk, sigmoid(expected_value - cost)]
    ternary_rows = []
    for a in actions:
        diff = a.expected_value - a.cost
        ternary_rows.append([diff, a.risk, sigmoid(np.array([diff]))[0]])
    T = np.array(ternary_rows, dtype=float)

    # 2. Signature per action (hash of its id)
    sigs = [signature([a.id], k=k) for a in actions]

    # 3. Pairwise similarity matrix
    n = len(actions)
    S = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sim = similarity(sigs[i], sigs[j])
            S[i][j] = S[j][i] = sim

    return T, sigs, S


def gaussian_fisher_confidence(similarity_matrix: List[List[float]],
                               center: float = 0.5,
                               width: float = 0.2) -> np.ndarray:
    """
    Convert a similarity matrix into a confidence matrix using a Gaussian beam
    and Fisher information.  The result ``C`` has the same shape as the input
    and contains values in ``[0, 1]`` that can be used as weighting factors.
    """
    n = len(similarity_matrix)
    C = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            theta = similarity_matrix[i][j]
            intensity = gaussian_beam(theta, center, width)
            conf = fisher_score(theta, center, width)
            # Normalise confidence to [0,1] by a simple sigmoid
            C[i, j] = sigmoid(np.array([conf]))[0] * intensity
    return C


def bandit_update_with_confidence(action_id: str,
                                  reward: float,
                                  confidence: float) -> None:
    """
    Perform a classic empirical‑mean bandit update, but scale the incremental
    reward by ``confidence`` (derived from the Gaussian‑Fisher bridge).  The
    global policy ``_POLICY`` is mutated in‑place.
    """
    total, cnt = _POLICY.get(action_id, (0.0, 0))
    # Confidence‑weighted incremental update
    total += confidence * reward
    cnt += 1
    _POLICY[action_id] = (total, cnt)


def project_policy_to_ternary(T: np.ndarray,
                              policy: Dict[str, Tuple[float, int]],
                              actions: List[MathAction]) -> np.ndarray:
    """
    Build a target vector ``y`` from the current bandit policy (average reward)
    and solve the least‑squares problem ``T w ≈ y`` for the weight vector ``w``.
    The projection aligns the regret‑weighted ternary space with the bandit
    expectations.
    """
    # Assemble target vector y (size N)
    y = np.zeros(T.shape[0], dtype=float)
    id_to_idx = {a.id: i for i, a in enumerate(actions)}
    for aid, (total, cnt) in policy.items():
        if aid not in id_to_idx or cnt == 0:
            continue
        idx = id_to_idx[aid]
        y[idx] = total / cnt  # empirical mean reward

    # Least‑squares solve (with regularisation to avoid singularity)
    if T.size == 0:
        raise ValueError("Ternary matrix T is empty")
    w, residuals, rank, s = np.linalg.lstsq(T, y, rcond=None)
    return w


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a small set of actions
    actions = [
        MathAction(id="A", expected_value=1.2, cost=0.3, risk=0.1),
        MathAction(id="B", expected_value=0.8, cost=0.2, risk=0.2),
        MathAction(id="C", expected_value=1.5, cost=0.5, risk=0.05),
    ]

    # 1. Build ternary matrix, signatures and similarity matrix
    T, sigs, S = hybrid_ternary_signature(actions, k=64)

    # 2. Convert similarity to confidence using Gaussian‑Fisher bridge
    C = gaussian_fisher_confidence(S, center=0.5, width=0.15)

    # 3. Simulate a round of bandit interaction using confidence as weight
    for i, a in enumerate(actions):
        # Fake reward sampled from a normal distribution centred at expected_value
        reward = random.gauss(a.expected_value, 0.1)
        # Use the average confidence of this action with all others as its weight
        conf_weight = float(np.mean(C[i]))
        bandit_update_with_confidence(a.id, reward, conf_weight)

    # 4. Project the updated policy back onto the ternary space
    w = project_policy_to_ternary(T, _POLICY, actions)

    # Output results for visual inspection (not required but harmless)
    print("Ternary matrix T:\n", T)
    print("Similarity matrix S:\n", np.array(S))
    print("Confidence matrix C:\n", C)
    print("Bandit policy (_POLICY):\n", _POLICY)
    print("Projected weights w (least‑squares):\n", w)

    # Verify that the tree cost function runs without error (uses dummy data)
    _NODES = {"root": (0.0, 0.0), "n1": (1.0, 0.0), "n2": (0.0, 1.0)}
    _EDGES = [("root", "n1"), ("root", "n2")]
    _ROOT = "root"
    cost = tree_cost(_NODES, _EDGES, _ROOT)
    print("Minimum‑cost tree total cost:", cost)