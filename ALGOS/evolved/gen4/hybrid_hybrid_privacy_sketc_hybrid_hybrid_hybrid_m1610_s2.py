# DARWIN HAMMER — match 1610, survivor 2
# gen: 4
# parent_a: hybrid_privacy_sketches_m15_s3.py (gen1)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hoeffding_tre_m933_s1.py (gen3)
# born: 2026-05-29T23:37:53Z

"""Hybrid module combining differential‑privacy sketching (parent A) with
RBF surrogate modelling and Hoeffding‑bound‑driven split decisions (parent B).

Mathematical bridge
-------------------
*Parent A* provides a Count‑Min sketch matrix **C ∈ ℤ^{d×w}** that stores
hashed quasi‑identifier frequencies.  After adding Laplace noise
(**Ĉ = C + Lap(Δ/ε_priv)**) we obtain a noisy frequency estimate per column

f_j = min_{i=1..d} Ĉ[i, j]   ,   j = 0..w‑1

which is a **vector f ∈ ℝ^{w}** representing a (noisy) histogram of distinct
quasi‑identifiers.

*Parent B* defines an RBF surrogate

ŷ(x) = Σ_k w_k · exp(‑(‖x‑c_k‖·ε_rbf)²)

where **c_k** are centres and **w_k** are learned weights (solution of a
linear system).  The surrogate can predict any scalar function of a
feature vector – we use it to predict the reconstruction‑risk score

risk̂ = ŷ(f)

directly from the sketch‑derived vector.

The Hoeffding bound from parent B is employed to decide whether the
current number of processed records **n** is sufficient for a reliable
risk estimate:

Δ = sqrt( (R²·ln(1/δ)) / (2·n) )

If the change in successive risk̂ is ≤ Δ we stop updating the sketch.

The three public functions below illustrate this integration:
1. `privacy_risk_via_rbf` – builds a noisy sketch, extracts f, and predicts
   risk with an RBF surrogate.
2. `incremental_sketch_with_hoeffding` – updates the sketch record‑by‑record
   and stops when the Hoeffding bound guarantees stability.
3. `gini_split_decision_on_sketch` – treats each column bucket as a class
   count and decides whether to split the sketch based on Gini gain,
   re‑using the DP‑noise‑aware counts.

All code is self‑contained and uses only the allowed standard libraries."""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – privacy / sketch utilities (adapted)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                             total_records: int) -> float:
    """Deterministic risk ratio clipped to [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_laplace_noise(scale: float) -> float:
    """Draw a single Laplace(0, scale) sample using the inverse‑CDF method."""
    u = np.random.uniform(-0.5, 0.5)
    return scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


def hash_to_indices(item: str, d: int, w: int) -> Tuple[int, int]:
    """Map a string to a (row, column) pair in the Count‑Min sketch."""
    h = int.from_bytes(item.encode('utf-8'), 'little')
    row = h % d
    col = (h // d) % w
    return row, col


def build_noisy_sketch(records: Iterable[Dict[str, Any]],
                      d: int,
                      w: int,
                      epsilon_privacy: float,
                      sensitivity: float = 1.0) -> np.ndarray:
    """
    Build a Count‑Min sketch from `records`, add Laplace noise, and return
    the noisy matrix Ĉ (shape d×w).
    """
    C = np.zeros((d, w), dtype=float)
    for rec in records:
        # Concatenate all quasi‑identifier values into a single string
        qi = '|'.join(str(v) for k, v in rec.items())
        i, j = hash_to_indices(qi, d, w)
        C[i, j] += 1.0

    scale = sensitivity / epsilon_privacy
    noise = np.vectorize(lambda _: dp_laplace_noise(scale))
    C_noisy = C + noise(C.shape)
    return C_noisy


def min_per_column(sketch: np.ndarray) -> np.ndarray:
    """Return the vector f where f[j] = min_i sketch[i, j]."""
    return np.min(sketch, axis=0)


# ----------------------------------------------------------------------
# Parent B – RBF surrogate, Hoeffding bound, Gini utilities (adapted)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting (returns solution vector)."""
    n = len(b)
    M = [row[:] + [rhs] for row, rhs in zip(A, b)]

    for col in range(n):
        # Pivot
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        M[col], M[pivot] = M[pivot], M[col]

        # Normalize pivot row
        div = M[col][col]
        M[col] = [v / div for v in M[col]]

        # Eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = M[row][col]
            M[row] = [v - factor * p for v, p in zip(M[row], M[col])]

    return [row[-1] for row in M]


@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Sequence[float]) -> float:
        """RBF surrogate prediction for input vector x."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """Hoeffding bound for a bounded random variable."""
    if range_ <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("invalid arguments")
    return math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def gini_impurity_from_counts(counts: Dict[Any, int]) -> float:
    """Standard Gini impurity."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.fromiter((c / total for c in counts.values()), dtype=float)
    return 1.0 - np.sum(probs ** 2)


def gini_gain(parent_counts: Dict[Any, int],
              left_counts: Dict[Any, int],
              right_counts: Dict[Any, int]) -> float:
    """Weighted Gini gain from splitting parent into left/right."""
    n_parent = sum(parent_counts.values())
    if n_parent == 0:
        return 0.0
    parent_imp = gini_impurity_from_counts(parent_counts)
    left_imp = gini_impurity_from_counts(left_counts)
    right_imp = gini_impurity_from_counts(right_counts)

    n_left = sum(left_counts.values())
    n_right = sum(right_counts.values())
    weighted_child_imp = (n_left / n_parent) * left_imp + (n_right / n_parent) * right_imp
    return parent_imp - weighted_child_imp


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def train_rbf_surrogate(features: List[np.ndarray],
                       targets: List[float],
                       epsilon_rbf: float = 1.0) -> RBFSurrogate:
    """
    Train an RBF surrogate on (feature, target) pairs.
    Centres are the provided feature vectors.
    We solve for weights w in K·w = y where K_ij = gaussian(||c_i-c_j||, ε).
    """
    if len(features) != len(targets):
        raise ValueError("features and targets length mismatch")
    n = len(features)
    # Build kernel matrix K
    K = [[gaussian(euclidean(features[i], features[j]), epsilon_rbf) for j in range(n)]
         for i in range(n)]
    # Solve linear system
    w = solve_linear(K, list(targets))
    centers = [tuple(vec.tolist()) for vec in features]
    return RBFSurrogate(centers=centers, weights=w, epsilon=epsilon_rbf)


def privacy_risk_via_rbf(records: Iterable[Dict[str, Any]],
                         d: int,
                         w: int,
                         epsilon_privacy: float,
                         epsilon_rbf: float = 1.0) -> float:
    """
    End‑to‑end hybrid:
    1. Build noisy Count‑Min sketch from `records`.
    2. Extract the min‑per‑column vector f.
    3. Train a tiny RBF surrogate on a synthetic dataset (here we use a
       single training point consisting of f itself and the deterministic
       risk computed from the raw records).
    4. Return the surrogate’s prediction as the DP‑protected risk estimate.
    """
    # Step 1 – noisy sketch
    C_noisy = build_noisy_sketch(records, d, w, epsilon_privacy)

    # Step 2 – feature vector
    f = min_per_column(C_noisy)

    # Deterministic risk for supervision (non‑private, used only for training)
    # In practice this would be replaced by a labelled dataset.
    unique_qi = int(np.sum(f > 0))  # rough count of distinct identifiers
    total = sum(1 for _ in records)
    true_risk = reconstruction_risk_score(unique_qi, total)

    # Step 3 – train surrogate on the single (f, true_risk) pair
    surrogate = train_rbf_surrogate([f], [true_risk], epsilon_rbf=epsilon_rbf)

    # Step 4 – predict
    return surrogate.predict(f)


def incremental_sketch_with_hoeffding(records: Iterable[Dict[str, Any]],
                                     d: int,
                                     w: int,
                                     epsilon_privacy: float,
                                     delta: float = 0.05,
                                     range_: float = 1.0) -> Tuple[np.ndarray, int]:
    """
    Incrementally update a Count‑Min sketch record‑by‑record.
    After each update we compute a DP‑noisy risk estimate (using the
    deterministic formula on the noisy sketch) and stop when the change
    between successive estimates is within the Hoeffding bound.
    Returns the final noisy sketch and the number of processed records.
    """
    C = np.zeros((d, w), dtype=float)
    prev_est = None
    n_processed = 0
    scale = 1.0 / epsilon_privacy  # sensitivity =1

    for rec in records:
        n_processed += 1
        qi = '|'.join(str(v) for k, v in rec.items())
        i, j = hash_to_indices(qi, d, w)
        C[i, j] += 1.0

        # Add Laplace noise only once per iteration (fresh noise)
        C_noisy = C + np.vectorize(lambda _: dp_laplace_noise(scale))(C.shape)
        f = min_per_column(C_noisy)
        est_risk = reconstruction_risk_score(int(np.sum(f > 0)), n_processed)

        if prev_est is not None:
            bound = hoeffding_bound(range_, delta, n_processed)
            if abs(est_risk - prev_est) <= bound:
                break
        prev_est = est_risk

    # Final noisy sketch (with fresh noise for output consistency)
    final_noisy = C + np.vectorize(lambda _: dp_laplace_noise(scale))(C.shape)
    return final_noisy, n_processed


def gini_split_decision_on_sketch(C_noisy: np.ndarray,
                                  threshold: float = 0.0) -> bool:
    """
    Use Gini impurity to decide whether to split the sketch into two
    sub‑sketches based on a simple column‑wise threshold.
    Columns with min‑value > threshold go to the left child, others to the right.
    Returns True if the split yields a positive Gini gain.
    """
    f = min_per_column(C_noisy)
    # Discretise each column into two “classes”: present (1) or absent (0)
    parent_counts = {0: int(np.sum(f <= threshold)), 1: int(np.sum(f > threshold))}
    left_counts = {0: int(np.sum((f <= threshold) & (f <= threshold))),
                   1: int(np.sum((f > threshold) & (f <= threshold)))}
    right_counts = {0: int(np.sum((f <= threshold) & (f > threshold))),
                    1: int(np.sum((f > threshold) & (f > threshold)))}
    # The above logic simplifies to pure counts; we compute them directly:
    left_counts = {0: int(np.sum(f <= threshold)), 1: 0}
    right_counts = {0: 0, 1: int(np.sum(f > threshold))}

    gain = gini_gain(parent_counts, left_counts, right_counts)
    return gain > 0.0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a tiny synthetic dataset
    random.seed(42)
    synthetic = [
        {"name": f"user{i}", "email": f"user{i}@example.com", "age": random.randint(18, 70)}
        for i in range(30)
    ]

    d, w = 5, 50
    eps_priv = 1.0
    eps_rbf = 0.8

    # 1. Hybrid risk via RBF surrogate
    risk_est = privacy_risk_via_rbf(synthetic, d, w, eps_priv, eps_rbf)
    print(f"Hybrid DP risk estimate (RBF surrogate): {risk_est:.4f}")

    # 2. Incremental sketch with Hoeffding early stop
    sketch, n_used = incremental_sketch_with_hoeffding(synthetic, d, w, eps_priv)
    print(f"Incremental sketch stopped after {n_used} records; sketch shape {sketch.shape}")

    # 3. Gini‑based split decision on the final noisy sketch
    split = gini_split_decision_on_sketch(sketch, threshold=0.5)
    print(f"Gini split decision on sketch: {'split' if split else 'no split'}")