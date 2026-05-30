# DARWIN HAMMER — match 5049, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m417_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m2198_s0.py (gen5)
# born: 2026-05-29T23:59:30Z

"""Hybrid Path–Geometric Algebra Algorithm
Parent A: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m417_s0.py
Parent B: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m2198_s0.py

Mathematical bridge:
- The level‑1 and level‑2 iterated integrals of a lead‑lag transformed path form the
  components of a path signature.  In geometric algebra each basis blade (e_i,
  e_i∧e_j, …) can store a scalar.  We therefore embed the signature into a
  `Multivector` where level‑1 integrals map to 1‑vectors (e_i) and level‑2
  integrals map to bivectors (e_i∧e_j).
- Textual features extracted from the same data are turned into a second
  `Multivector` (feature multivector) whose components weight the corresponding
  blades.
- A regret‑weighted scalar, derived from the feature vector, modulates the
  geometric product of the two multivectors, yielding a single hybrid object
  that simultaneously carries path‑signature information, feature relevance,
  and regret‑based scaling.
"""

import sys
import math
import random
import pathlib
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, FrozenSet

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Lead‑lag transform: interleave (lead, lag) channels for causality encoding.

    Parameters
    ----------
    path : np.ndarray of shape (T, d)

    Returns
    -------
    np.ndarray of shape (2*T-1, 2*d)
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])          # (lead, lag) both at t
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])  # lead advances, lag holds
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def compute_signature(path: np.ndarray, level: int = 2) -> Dict[Tuple[int, ...], float]:
    """Very coarse iterated‑integral signature up to `level` (only 1 and 2).

    Returns a dict mapping index tuples to scalar integrals.
    """
    T, d = path.shape
    # level‑1: simple trapezoidal sums of each coordinate
    sig = {}
    dt = 1.0 / (T - 1) if T > 1 else 1.0
    for i in range(d):
        sig[(i,)] = np.trapz(path[:, i], dx=dt)

    if level >= 2:
        # level‑2: pairwise products integrated (approx by outer product of increments)
        for i in range(d):
            for j in range(i, d):
                # approximate ∫ X_i dX_j
                increments = np.diff(path[:, j])
                mids = (path[:-1, i] + path[1:, i]) / 2.0
                sig[(i, j)] = np.sum(mids * increments)
    return sig


# Simplified feature extraction – returns a deterministic vector for reproducibility
_FEATURE_KEYS = [
    "operator_visceral_ratio", "operator_tech_ratio",
    "operator_legal_osint_ratio", "operator_ledger_density",
    "operator_recursion_score", "operator_directive_ratio",
    "operator_target_density", "psyche_forensic_shield_ratio",
    "psyche_poetic_entropy", "psyche_dissociative_index",
    "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
    "resilience_resource_exhaustion_metric", "resilience_swarm_orchestration_density",
    "resilience_lo"
]  # truncated list; we will reuse the first N keys as needed


def extract_features(text: str) -> np.ndarray:
    """Deterministic pseudo‑feature extractor based on hash of the text."""
    rng = random.Random(hash(text))
    feats = np.array([rng.random() for _ in _FEATURE_KEYS], dtype=float)
    # Normalise to unit L2 norm
    norm = np.linalg.norm(feats) + 1e-12
    return feats / norm


def regret_weighted_factor(features: np.ndarray) -> float:
    """Compute a scalar regret weight from the feature vector.

    Here we use a simple softmax‑like scaling of the first three features.
    """
    if features.size < 3:
        return 1.0
    raw = features[:3]
    exp_raw = np.exp(raw - np.max(raw))
    return float(np.sum(exp_raw) / 3.0)  # average softmax value


# ----------------------------------------------------------------------
# Parent B utilities – geometric algebra core
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Sort a list of basis indices, returning the sorted list and the sign
    of the permutation.  Duplicate indices cancel (grade‑reduction)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate basis vectors
                del lst[j : j + 2]
                n -= 2
                i = -1
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades represented as frozensets of indices."""
    combined = list(blade_a) + list(blade_b)
    sorted_indices, sign = _blade_sign(combined)
    return frozenset(sorted_indices), sign


class Multivector:
    """Sparse multivector where keys are frozensets of basis indices."""

    def __init__(self, components: Dict[FrozenSet[int], float] = None):
        self.components: Dict[FrozenSet[int], float] = dict(components or {})

    def __add__(self, other: "Multivector") -> "Multivector":
        res = self.components.copy()
        for k, v in other.components.items():
            res[k] = res.get(k, 0.0) + v
            if abs(res[k]) < 1e-15:
                del res[k]
        return Multivector(res)

    def __neg__(self) -> "Multivector":
        return Multivector({k: -v for k, v in self.components.items()})

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result: Dict[FrozenSet[int], float] = defaultdict(float)
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                blade, sign = _multiply_blades(ba, bb)
                result[blade] += sign * ca * cb
        # prune near‑zero entries
        result = {k: v for k, v in result.items() if abs(v) > 1e-15}
        return Multivector(result)

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) component, or 0.0 if absent."""
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coeff in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if not blade:
                name = "1"
            else:
                name = "∧".join(f"e{idx}" for idx in sorted(blade))
            terms.append(f"{coeff:.3g}*{name}")
        return " + ".join(terms)


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def signature_to_multivector(sig: Dict[Tuple[int, ...], float]) -> Multivector:
    """Map a level‑1/2 signature dictionary to a geometric multivector.

    - (i,)   -> basis e_i (grade‑1)
    - (i,j) -> basis e_i∧e_j (grade‑2), ordered i≤j
    """
    comps: Dict[FrozenSet[int], float] = {}
    for idx_tuple, value in sig.items():
        if len(idx_tuple) == 1:
            blade = frozenset({idx_tuple[0]})
        elif len(idx_tuple) == 2:
            i, j = idx_tuple
            if i == j:
                # e_i∧e_i = 0, skip
                continue
            blade = frozenset({i, j})
        else:
            continue  # higher grades not handled
        comps[blade] = comps.get(blade, 0.0) + value
    return Multivector(comps)


def features_to_multivector(features: np.ndarray) -> Multivector:
    """Assign each feature to a distinct basis vector.

    Feature k → e_k .  Excess features beyond the dimensionality of the path are ignored.
    """
    comps: Dict[FrozenSet[int], float] = {}
    for k, val in enumerate(features):
        comps[frozenset({k})] = val
    return Multivector(comps)


def hybrid_regret_geometric_product(path: np.ndarray, text: str) -> Multivector:
    """Full hybrid pipeline:

    1. Lead‑lag transform the path.
    2. Compute a level‑2 signature.
    3. Embed the signature into a multivector.
    4. Extract textual features and embed them into another multivector.
    5. Compute a regret‑weighted scalar.
    6. Return regret_factor * (signature_mv * feature_mv).
    """
    # Step 1
    ll_path = lead_lag_transform(path)

    # Step 2
    sig = compute_signature(ll_path, level=2)

    # Step 3
    sig_mv = signature_to_multivector(sig)

    # Step 4
    feats = extract_features(text)
    feat_mv = features_to_multivector(feats)

    # Step 5
    regret_factor = regret_weighted_factor(feats)

    # Step 6
    hybrid_mv = (sig_mv * feat_mv)
    # Scale all components by regret_factor
    scaled_components = {blade: coeff * regret_factor for blade, coeff in hybrid_mv.components.items()}
    return Multivector(scaled_components)


# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def entropy_of_multivector(mv: Multivector) -> float:
    """Shannon entropy of the absolute squared coefficients (treated as a probability distribution)."""
    coeffs = np.array([abs(v) for v in mv.components.values()], dtype=float)
    if coeffs.size == 0:
        return 0.0
    probs = coeffs / coeffs.sum()
    return -float(np.sum(probs * np.log(probs + 1e-12)))


def geometric_regret_score(path: np.ndarray, text: str) -> Tuple[Multivector, float]:
    """Convenience wrapper returning the hybrid multivector and its entropy."""
    hybrid_mv = hybrid_regret_geometric_product(path, text)
    ent = entropy_of_multivector(hybrid_mv)
    return hybrid_mv, ent


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a random 2‑dimensional path with 10 time steps
    rng = np.random.default_rng(42)
    path = rng.normal(size=(10, 2))

    # Sample text
    sample_text = "The quick brown fox jumps over the lazy dog."

    # Run the hybrid pipeline
    hybrid_mv, ent = geometric_regret_score(path, sample_text)

    print("Hybrid multivector:")
    print(hybrid_mv)
    print("\nEntropy of coefficient distribution:", ent)