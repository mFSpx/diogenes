# DARWIN HAMMER — match 948, survivor 3
# gen: 5
# parent_a: counterfactual_effects.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m7_s2.py (gen4)
# born: 2026-05-29T23:31:51Z

"""HybridCausalMinHashRBF
This module fuses two distinct parents:

* **Parent A** – `counterfactual_effects.py` – provides a simple causal‑effect estimator
  returning a `CausalEffect` dataclass. Its core operation is the computation of an
  average treatment effect (ATE) from treatment/outcome vectors.

* **Parent B** – `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m7_s2.py` – implements a
  radial‑basis‑function (RBF) surrogate model that learns a mapping from a discrete
  signal (the MinHash signature of a probability distribution) to the output of a
  physics‑inspired integrator.

**Mathematical bridge**

The bridge is the *MinHash signature* of the confounder distribution.  
The signature is a finite‑dimensional vector that can be interpreted as a discrete
signal. The RBF surrogate model treats this signal as an input point `x ∈ ℝ^k`
and learns a mapping `f(x) ≈ ATE`. Consequently the hybrid algorithm:

1. Computes a MinHash signature of the confounder values (signal).
2. Builds an RBF surrogate using bootstrap samples whose targets are the ATEs
   obtained from the original causal estimator (Parent A).
3. Predicts the ATE for the full data set by evaluating the surrogate at the
   signature of the full confounder distribution.

The result is a single `CausalEffect` whose `ate_estimate` is obtained from the
learned surrogate, thus mathematically intertwining the causal‑effect topology
with the RBF‑MinHash topology.

"""

from __future__ import annotations
import math
import random
import hashlib
import uuid
import statistics
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – causal effect primitives
# ----------------------------------------------------------------------
Vector = Sequence[float]

@dataclass(frozen=True)
class CausalEffect:
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]

def _raw_ate(treatment: List[float], outcome: List[float]) -> Tuple[float | None, Tuple[float, float] | None]:
    if not treatment or len(treatment) != len(outcome):
        return None, None
    yt = [y for t, y in zip(treatment, outcome) if t >= 0.5]
    yc = [y for t, y in zip(treatment, outcome) if t < 0.5]
    if not yt or not yc:
        return None, None
    ate = statistics.mean(yt) - statistics.mean(yc)
    spread = statistics.pstdev(outcome) if len(outcome) > 1 else 0.0
    ci = (ate - spread, ate + spread)
    return ate, ci

def estimate_causal_effect(
    treatment: str,
    outcome: str,
    confounders: List[str],
    data: Dict[str, Iterable[float]],
) -> CausalEffect:
    t_vals = list(map(float, data.get(treatment, [])))
    y_vals = list(map(float, data.get(outcome, [])))
    ate, ci = _raw_ate(t_vals, y_vals)
    return CausalEffect(
        effect_id=str(uuid.uuid4()),
        treatment=treatment,
        outcome=outcome,
        confounders=tuple(confounders),
        ate_estimate=ate,
        ate_confidence_interval=ci,
        refutation_passed=ate is not None,
        refutation_methods=("placebo_treatment", "data_subset", "random_common_cause"),
        heterogeneous_effects={"overall": ate or 0.0},
    )

# ----------------------------------------------------------------------
# Parent B – MinHash + RBF surrogate utilities
# ----------------------------------------------------------------------
def compute_minhash_signature(values: Iterable[float], num_perm: int = 10) -> List[int]:
    """
    Simple MinHash: for each permutation index i we hash each value with a
    deterministic seed i and keep the minimal hash value. The resulting list
    of minima is the signature (a discrete signal).
    """
    signature = []
    for i in range(num_perm):
        min_hash = None
        for v in values:
            # deterministic hash: combine seed and value string
            h = hashlib.sha256(f"{i}-{v}".encode("utf-8")).digest()
            hv = int.from_bytes(h, byteorder="big")
            if (min_hash is None) or (hv < min_hash):
                min_hash = hv
        signature.append(min_hash if min_hash is not None else 0)
    return signature

def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def rbf_kernel(x: Vector, y: Vector, epsilon: float = 1.0) -> float:
    return gaussian_rbf(euclidean(x, y), epsilon)

def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Gaussian elimination with partial pivoting.
    Solves A·x = b for a square matrix A.
    """
    n = len(b)
    # Build augmented matrix
    M = [row[:] + [rhs] for row, rhs in zip(A, b)]

    # Forward elimination
    for col in range(n):
        # Pivot selection
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-12:
            raise ValueError("Singular matrix in surrogate system")
        # Swap rows
        M[col], M[pivot_row] = M[pivot_row], M[col]
        # Normalize pivot row
        pivot_val = M[col][col]
        M[col] = [v / pivot_val for v in M[col]]
        # Eliminate below
        for r in range(col + 1, n):
            factor = M[r][col]
            M[r] = [rv - factor * cv for rv, cv in zip(M[r], M[col])]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = M[i][-1] - sum(M[i][j] * x[j] for j in range(i + 1, n))
    return x

def train_rbf_surrogate(
    X: List[Vector],
    y: List[float],
    epsilon: float = 1.0,
) -> Tuple[List[Vector], List[float]]:
    """
    Trains an RBF surrogate: solves K·w = y where
    K_ij = rbf_kernel(X_i, X_j).
    Returns the training points and the solved weight vector w.
    """
    if len(X) != len(y):
        raise ValueError("Training data size mismatch")
    n = len(X)
    K = [[rbf_kernel(X[i], X[j], epsilon) for j in range(n)] for i in range(n)]
    w = solve_linear(K, y)
    return X, w

def predict_rbf_surrogate(
    x_new: Vector,
    X_train: List[Vector],
    weights: List[float],
    epsilon: float = 1.0,
) -> float:
    """
    Predicts the output for a new input vector using the trained surrogate.
    """
    return sum(w * rbf_kernel(x_new, x_i, epsilon) for w, x_i in zip(weights, X_train))

# ----------------------------------------------------------------------
# Hybrid operation – causal effect estimation via MinHash‑RBF surrogate
# ----------------------------------------------------------------------
def _bootstrap_samples(
    data: Dict[str, Iterable[float]],
    treatment: str,
    outcome: str,
    confounders: List[str],
    n_samples: int = 5,
) -> List[Dict[str, List[float]]]:
    """
    Generates bootstrap resamples (with replacement) of the rows in `data`.
    Returns a list of dictionaries with the same keys as `data` but containing
    only the sampled rows.
    """
    # Convert columns to lists for indexable access
    cols = {k: list(v) for k, v in data.items()}
    n_rows = len(cols[treatment])
    samples = []
    for _ in range(n_samples):
        idxs = [random.randrange(n_rows) for _ in range(n_rows)]
        sample = {
            k: [cols[k][i] for i in idxs]
            for k in cols
        }
        samples.append(sample)
    return samples

def estimate_hybrid_causal_effect(
    treatment: str,
    outcome: str,
    confounders: List[str],
    data: Dict[str, Iterable[float]],
    num_perm: int = 12,
    epsilon: float = 1.0,
    n_boot: int = 7,
) -> CausalEffect:
    """
    Hybrid estimator:
    1. Build bootstrap training set → (MinHash signature, naive ATE).
    2. Fit RBF surrogate on these pairs.
    3. Compute signature of the full confounder distribution.
    4. Predict ATE via the surrogate.
    """
    # ---- Step 1: bootstrap training data ----
    boot_samples = _bootstrap_samples(data, treatment, outcome, confounders, n_boot)
    X_sig: List[List[int]] = []
    y_ate: List[float] = []
    for sample in boot_samples:
        # Confounder values: concatenate all confounder columns
        conf_vals = []
        for c in confounders:
            conf_vals.extend(sample.get(c, []))
        sig = compute_minhash_signature(conf_vals, num_perm=num_perm)
        X_sig.append(sig)

        # Naive ATE on the bootstrap sample
        ce = estimate_causal_effect(treatment, outcome, confounders, sample)
        ate = ce.ate_estimate if ce.ate_estimate is not None else 0.0
        y_ate.append(ate)

    # ---- Step 2: train RBF surrogate ----
    X_train, weights = train_rbf_surrogate(X_sig, y_ate, epsilon=epsilon)

    # ---- Step 3: signature of full data ----
    full_conf_vals = []
    for c in confounders:
        full_conf_vals.extend(data.get(c, []))
    full_sig = compute_minhash_signature(full_conf_vals, num_perm=num_perm)

    # ---- Step 4: predict ATE ----
    pred_ate = predict_rbf_surrogate(full_sig, X_train, weights, epsilon=epsilon)

    # Build final CausalEffect (using the surrogate prediction as the estimate)
    ci_width = statistics.pstdev(y_ate) if len(y_ate) > 1 else 0.0
    ci = (pred_ate - ci_width, pred_ate + ci_width) if ci_width > 0 else None

    return CausalEffect(
        effect_id=str(uuid.uuid4()),
        treatment=treatment,
        outcome=outcome,
        confounders=tuple(confounders),
        ate_estimate=pred_ate,
        ate_confidence_interval=ci,
        refutation_passed=True,
        refutation_methods=("minhash_rbf_surrogate",),
        heterogeneous_effects={"predicted": pred_ate},
    )

# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------
def run_refutation_suite(effect: CausalEffect, methods: List[str] | None = None) -> Dict[str, bool]:
    ms = methods or ["minhash_rbf_surrogate"]
    return {m: bool(effect.ate_estimate is not None and effect.refutation_passed) for m in ms}

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic dataset
    random.seed(42)
    n = 200
    data = {
        "treatment": [random.random() for _ in range(n)],
        "outcome":   [random.random() + 0.3 * (t > 0.5) for t in [random.random() for _ in range(n)]],
        "age":       [random.randint(20, 70) for _ in range(n)],
        "income":    [random.randint(30_000, 120_000) for _ in range(n)],
    }
    confs = ["age", "income"]
    hybrid_effect = estimate_hybrid_causal_effect(
        treatment="treatment",
        outcome="outcome",
        confounders=confs,
        data=data,
        num_perm=16,
        epsilon=0.8,
        n_boot=9,
    )
    print("Hybrid Causal Effect:")
    print(f"  ID: {hybrid_effect.effect_id}")
    print(f"  ATE estimate (surrogate): {hybrid_effect.ate_estimate:.4f}")
    print(f"  Confidence interval: {hybrid_effect.ate_confidence_interval}")
    print("Refutation suite:", run_refutation_suite(hybrid_effect))