# DARWIN HAMMER — match 1813, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s0.py (gen4)
# parent_b: hybrid_rbf_surrogate_tri_algo_conduit_m8_s0.py (gen1)
# born: 2026-05-29T23:38:57Z

"""HybridCertaintyRBF
Combines the epistemic certainty framework of *hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s0.py*
with the radial‑basis‑function surrogate model of *hybrid_rbf_surrogate_tri_algo_conduit_m8_s0.py*.

Mathematical bridge
-------------------
Each `CertaintyFlag` encodes a discrete label, a confidence (basis points) and an
authority class.  We embed these categorical attributes into a real‑valued
feature vector `φ(flag) ∈ ℝ^d`.  The vector serves as a data point for the RBF
surrogate:


y_i ≈ f( φ(flag_i) ) = Σ_j w_j·exp( -ε·‖φ(flag_i)-c_j‖² )


where `{c_j}` are the training points (the same as the inputs) and `w_j` are
learned by solving the linear system `K w = y` (`K_ij = exp(-ε·‖c_i-c_j‖²)+λδ_ij`).

The surrogate predicts an expected *reward* (e.g. the RLCT estimate from the
original bandit) for any new `CertaintyFlag`.  A lightweight bandit then selects
the flag with the highest predicted reward, optionally adding an exploration term.
Thus the certainty information guides the surrogate, and the surrogate guides the
bandit – a closed mathematical loop.

The module provides:
* `encode_certainty(flag) → np.ndarray` – deterministic embedding.
* `fit_certainty_surrogate(flags, rewards, ε, λ) → RBFSurrogate` – trains the RBF.
* `select_best_flag(surrogate, candidates, explore) → CertaintyFlag` – bandit‑style selection.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – epistemic certainty structures
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
AUTHORITY_CLASSES = ("EXPERT", "PEER", "CROWD", "UNKNOWN")

@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable representation of an epistemic certainty claim."""
    label: str
    confidence_bps: int               # basis points, 0 .. 10_000
    authority_class: str
    rationale: str = ""
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if self.authority_class not in AUTHORITY_CLASSES:
            raise ValueError(f"unknown authority class: {self.authority_class!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }

# ----------------------------------------------------------------------
# Parent B – radial‑basis‑function surrogate
# ----------------------------------------------------------------------
Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b with naive Gauss‑Jordan elimination (dense, small systems)."""
    n = len(b)
    # Build augmented matrix
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # Pivot on largest absolute value for numerical stability
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        # Swap rows
        m[col], m[pivot] = m[pivot], m[col]
        # Normalize pivot row
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        # Eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

@dataclass(frozen=True)
class RBFSurrogate:
    """Trained RBF surrogate model."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Predict scalar output for input vector x."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def fit_rbf(points: Iterable[Vector],
            values: Iterable[float],
            epsilon: float = 1.0,
            ridge: float = 1e-9) -> RBFSurrogate:
    """Fit an RBF surrogate to (points, values)."""
    centers = [tuple(map(float, p)) for p in points]
    y = [float(v) for v in values]
    if not centers or len(centers) != len(y):
        raise ValueError("points and values must be non‑empty and same length")
    n = len(centers)
    # Build kernel matrix with optional ridge regularisation
    K = [
        [
            gaussian(euclidean(a, b), epsilon) + (ridge if i == j else 0.0)
            for j, b in enumerate(centers)
        ]
        for i, a in enumerate(centers)
    ]
    w = solve_linear(K, y)
    return RBFSurrogate(centers, w, epsilon)

# ----------------------------------------------------------------------
# Hybrid layer – embedding + surrogate + bandit selection
# ----------------------------------------------------------------------
_label_to_numeric = {lbl: i for i, lbl in enumerate(EPISTEMIC_FLAGS)}
_authority_to_numeric = {auth: i for i, auth in enumerate(AUTHORITY_CLASSES)}

def encode_certainty(flag: CertaintyFlag) -> np.ndarray:
    """
    Deterministic embedding of a CertaintyFlag into ℝ^d.
    Features:
        - one‑hot label (len(EPISTEMIC_FLAGS))
        - normalized confidence (0..1)
        - one‑hot authority class (len(AUTHORITY_CLASSES))
        - rationale length (log‑scaled)
    Returns a 1‑D NumPy array.
    """
    label_vec = np.zeros(len(EPISTEMIC_FLAGS))
    label_vec[_label_to_numeric[flag.label]] = 1.0

    authority_vec = np.zeros(len(AUTHORITY_CLASSES))
    authority_vec[_authority_to_numeric[flag.authority_class]] = 1.0

    confidence_norm = flag.confidence_bps / 10_000.0

    rationale_len = math.log1p(len(flag.rationale))

    # Concatenate all parts
    return np.concatenate([label_vec,
                           np.array([confidence_norm]),
                           authority_vec,
                           np.array([rationale_len])])

def fit_certainty_surrogate(flags: List[CertaintyFlag],
                            rewards: List[float],
                            epsilon: float = 1.0,
                            ridge: float = 1e-9) -> RBFSurrogate:
    """
    Train an RBF surrogate where each training point is the embedding of a
    CertaintyFlag and the target is the observed reward (e.g. RLCT estimate).
    """
    if len(flags) != len(rewards):
        raise ValueError("flags and rewards must have the same length")
    points = [encode_certainty(f).tolist() for f in flags]
    return fit_rbf(points, rewards, epsilon=epsilon, ridge=ridge)

def select_best_flag(surrogate: RBFSurrogate,
                     candidates: List[CertaintyFlag],
                     explore: float = 0.1) -> CertaintyFlag:
    """
    Bandit‑style action selection.
    For each candidate we predict its reward with the surrogate, add an
    ε‑greedy exploration term, and return the flag with the highest perturbed
    score.
    """
    best_score = -math.inf
    best_flag = None
    for flag in candidates:
        x = encode_certainty(flag).tolist()
        pred = surrogate.predict(x)
        # ε‑greedy exploration: with probability `explore` replace prediction by random
        if random.random() < explore:
            pred = random.uniform(0.0, 1.0) * max(1.0, pred)
        if pred > best_score:
            best_score = pred
            best_flag = flag
    if best_flag is None:
        raise RuntimeError("no candidates provided")
    return best_flag

def update_rewards(flags: List[CertaintyFlag],
                  base_rewards: List[float],
                  surrogate: RBFSurrogate,
                  learning_rate: float = 0.5) -> List[float]:
    """
    Simple online update: blend observed reward with surrogate prediction.
    Returns a new list of updated rewards.
    """
    updated = []
    for f, r_obs in zip(flags, base_rewards):
        x = encode_certainty(f).tolist()
        r_pred = surrogate.predict(x)
        r_new = (1 - learning_rate) * r_obs + learning_rate * r_pred
        updated.append(r_new)
    return updated

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    rng = random.Random(42)
    flags_train = [
        CertaintyFlag(label="FACT", confidence_bps=9000, authority_class="EXPERT", rationale="well‑studied"),
        CertaintyFlag(label="PROBABLE", confidence_bps=6000, authority_class="PEER", rationale="some evidence"),
        CertaintyFlag(label="POSSIBLE", confidence_bps=3000, authority_class="CROWD", rationale="speculation"),
        CertaintyFlag(label="BULLSHIT", confidence_bps=1000, authority_class="UNKNOWN", rationale="no basis"),
    ]

    # Synthetic rewards (e.g., RLCT estimates) – higher for more reliable flags
    rewards_train = [0.95, 0.75, 0.45, 0.10]

    # Train surrogate
    surrogate = fit_certainty_surrogate(flags_train, rewards_train, epsilon=1.2, ridge=1e-6)

    # New candidate flags for selection
    candidates = [
        CertaintyFlag(label="FACT", confidence_bps=8500, authority_class="EXPERT", rationale="replication"),
        CertaintyFlag(label="PROBABLE", confidence_bps=5500, authority_class="PEER", rationale="partial data"),
        CertaintyFlag(label="SURE_MAYBE", confidence_bps=4000, authority_class="CROWD", rationale="mixed opinions"),
    ]

    chosen = select_best_flag(surrogate, candidates, explore=0.05)
    print("Chosen flag:", chosen)

    # Demonstrate online update
    observed_rewards = [0.92, 0.70, 0.40]  # pretend we observed these after actions
    updated_rewards = update_rewards(candidates, observed_rewards, surrogate)
    print("Updated rewards:", updated_rewards)