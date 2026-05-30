# DARWIN HAMMER — match 96, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_geomet_koopman_operator_m59_s0.py (gen3)
# parent_b: hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s3.py (gen2)
# born: 2026-05-29T23:28:10Z

"""
Hybrid module combining:
- Parent A: geometric algebra (Multivector) with Koopman operator linearisation.
- Parent B: pheromone‑based infotaxis and decision‑hygiene scoring with Shannon entropy.

Mathematical bridge:
Decision‑hygiene scores are embedded as coefficients of a multivector (geometric
algebra).  The Koopman operator is estimated from successive score snapshots
(X, X′) and applied to the multivector, producing a linear prediction of future
scores.  The predicted score distribution is fed to a Shannon‑entropy routine;
the resulting entropy modulates pheromone strengths, biasing the stochastic
selection of actions.  Thus the high‑dimensional linearised dynamics (Koopman)
drive the probabilistic pheromone update (entropy‑weighted infotaxis).
"""

import math
import random
import sys
import pathlib
import numpy as np
import re
from collections import Counter

# ----------------------------------------------------------------------
# Geometric Algebra core (Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: list) -> tuple:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
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
                del lst[j : j + 2]
                n -= 2
                sign *= 1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> tuple:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: dict, n: int):
        # components: {frozenset(indices): coefficient}
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        if self.n != other.n:
            raise ValueError("Algebra dimensions must match")
        new_comp = self.components.copy()
        for b, v in other.components.items():
            new_comp[b] = new_comp.get(b, 0.0) + v
        return Multivector(new_comp, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -v for b, v in self.components.items()}, self.n)

    def __mul__(self, other):
        """Geometric product with scalar or another Multivector."""
        if isinstance(other, (int, float)):
            return Multivector({b: v * other for b, v in self.components.items()}, self.n)
        if isinstance(other, Multivector):
            if self.n != other.n:
                raise ValueError("Algebra dimensions must match")
            result = {}
            for blade_a, coef_a in self.components.items():
                for blade_b, coef_b in other.components.items():
                    blade_res, sign = _multiply_blades(blade_a, blade_b)
                    result[blade_res] = result.get(blade_res, 0.0) + sign * coef_a * coef_b
            return Multivector(result, self.n)
        raise TypeError("Unsupported multiplication")

    def __repr__(self):
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), tuple(sorted(x[0])))):
            if blade:
                basis = "e" + "".join(str(i) for i in sorted(blade))
            else:
                basis = "1"
            terms.append(f"{coef:.3g}{basis}")
        return " + ".join(terms) if terms else "0"


# ----------------------------------------------------------------------
# Helper utilities for multivector ↔ vector conversion
# ----------------------------------------------------------------------
def _all_blades(n: int) -> list:
    """Return a sorted list of all possible basis blades for Cl(n,0)."""
    blades = [frozenset()]  # scalar
    for r in range(1, n + 1):
        # generate combinations of indices 1..n of size r
        from itertools import combinations

        for combo in combinations(range(1, n + 1), r):
            blades.append(frozenset(combo))
    # sort by grade then lexicographically
    blades.sort(key=lambda b: (len(b), tuple(sorted(b))))
    return blades


def multivector_to_vector(mv: Multivector) -> np.ndarray:
    """Flatten a multivector into a column vector following the canonical blade order."""
    blades = _all_blades(mv.n)
    vec = np.array([mv.components.get(b, 0.0) for b in blades], dtype=float)
    return vec.reshape(-1, 1)  # column vector


def vector_to_multivector(vec: np.ndarray, n: int) -> Multivector:
    """Reconstruct a Multivector from a column vector using the canonical blade order."""
    blades = _all_blades(n)
    if vec.shape[0] != len(blades):
        raise ValueError("Vector size does not match number of blades")
    comp = {b: float(v) for b, v in zip(blades, vec.ravel()) if abs(v) > 1e-15}
    return Multivector(comp, n)


# ----------------------------------------------------------------------
# Koopman operator estimation (Parent A)
# ----------------------------------------------------------------------
def estimate_koopman_operator(X: np.ndarray, X_prime: np.ndarray) -> np.ndarray:
    """
    Least‑squares estimate of the finite‑dimensional Koopman matrix K
    such that X_prime ≈ K @ X.
    X, X_prime: (m, N) where each column is a snapshot of the state vector.
    Returns K of shape (m, m).
    """
    # Pseudo‑inverse of X
    X_pinv = np.linalg.pinv(X)
    K = X_prime @ X_pinv
    return K


def apply_koopman(multivector: Multivector, K: np.ndarray) -> Multivector:
    """
    Propagate a multivector forward one Koopman step using matrix K.
    """
    vec = multivector_to_vector(multivector)  # (m,1)
    next_vec = K @ vec
    return vector_to_multivector(next_vec, multivector.n)


# ----------------------------------------------------------------------
# Decision hygiene & entropy (Parent B)
# ----------------------------------------------------------------------
def shannon_entropy(scores: np.ndarray) -> float:
    """
    Compute Shannon entropy of a non‑negative score vector.
    Scores are first normalised to a probability distribution.
    """
    if np.any(scores < 0):
        raise ValueError("Scores must be non‑negative")
    total = scores.sum()
    if total == 0:
        return 0.0
    probs = scores / total
    # avoid log(0) by masking zeros
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log2(probs[mask])))


def update_pheromones(
    pheromones: dict,
    action_scores: dict,
    entropy: float,
    decay: float = 0.9,
    entropy_weight: float = 0.5,
) -> dict:
    """
    Update pheromone strengths based on action scores and the current entropy.
    - decay: global evaporation factor.
    - entropy_weight: how strongly entropy influences reinforcement.
    """
    new_pheromones = {}
    max_score = max(action_scores.values()) if action_scores else 1.0
    for act, score in action_scores.items():
        # Reinforcement proportional to normalized score and entropy
        reinforcement = (score / max_score) * (1.0 + entropy_weight * entropy)
        new_val = decay * pheromones.get(act, 1.0) + reinforcement
        new_pheromones[act] = new_val
    return new_pheromones


def select_action(pheromones: dict) -> str:
    """
    Stochastic selection of an action using pheromone probabilities.
    """
    actions = list(pheromones.keys())
    weights = np.array([pheromones[a] for a in actions], dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    probs = weights / weights.sum()
    return random.choices(actions, weights=probs, k=1)[0]


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def build_multivector_from_scores(score_dict: dict, n: int) -> Multivector:
    """
    Encode a dictionary of scalar scores into a multivector.
    The mapping is simple: each key is interpreted as an integer basis index
    (1‑based). The corresponding score becomes the coefficient of that 1‑blade.
    The scalar part holds the sum of all scores.
    """
    components = {}
    scalar_sum = 0.0
    for key, val in score_dict.items():
        try:
            idx = int(key)
        except Exception:
            continue  # ignore non‑numeric keys
        if 1 <= idx <= n:
            components[frozenset([idx])] = float(val)
            scalar_sum += float(val)
    components[frozenset()] = scalar_sum
    return Multivector(components, n)


def hybrid_step(
    prev_mv: Multivector,
    prev_pheromones: dict,
    action_score_dict: dict,
    snapshot_buffer: list,
    n: int,
    decay: float = 0.9,
) -> tuple:
    """
    Perform one hybrid iteration:
    1. Encode current action scores into a multivector.
    2. Append to snapshot buffer and, if enough snapshots, estimate K.
    3. Propagate previous multivector via K (or keep unchanged if K unavailable).
    4. Extract scalar coefficients as a score vector and compute entropy.
    5. Update pheromones using entropy‑weighted reinforcement.
    6. Sample next action.
    Returns (next_multivector, updated_pheromones, selected_action, updated_buffer)
    """
    # 1. Current multivector from scores
    cur_mv = build_multivector_from_scores(action_score_dict, n)

    # 2. Buffer management for Koopman estimation
    snapshot_buffer.append(multivector_to_vector(cur_mv).ravel())
    if len(snapshot_buffer) > 3:
        # keep only the latest three snapshots (X_{t-2}, X_{t-1}, X_t)
        snapshot_buffer = snapshot_buffer[-3:]

    # 3. Estimate and apply Koopman if we have at least two columns
    if len(snapshot_buffer) >= 2:
        X = np.column_stack(snapshot_buffer[:-1])  # past
        X_prime = np.column_stack(snapshot_buffer[1:])  # future
        K = estimate_koopman_operator(X, X_prime)
        pred_mv = apply_koopman(prev_mv, K)
    else:
        pred_mv = prev_mv  # no prediction possible yet

    # 4. Derive a probability vector from the predicted multivector's scalar part
    #    and the coefficients of 1‑blades.
    scores = np.array(
        [pred_mv.components.get(frozenset([i]), 0.0) for i in range(1, n + 1)], dtype=float
    )
    entropy = shannon_entropy(scores)

    # 5. Update pheromones
    updated_pheromones = update_pheromones(prev_pheromones, action_score_dict, entropy, decay)

    # 6. Sample action
    selected_action = select_action(updated_pheromones)

    return pred_mv, updated_pheromones, selected_action, snapshot_buffer


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dimensionality of the geometric algebra (choose small for test)
    N = 4

    # Initial multivector (all zeros except scalar)
    mv0 = Multivector({frozenset(): 0.0}, N)

    # Initial pheromone table for four hypothetical actions
    pheromones = {str(i): 1.0 for i in range(1, N + 1)}

    # Mock decision‑hygiene scores (could come from any upstream model)
    scores = {str(i): random.uniform(0, 10) for i in range(1, N + 1)}

    # Snapshot buffer for Koopman estimation
    buffer = []

    # Run a few hybrid steps
    mv = mv0
    for step in range(5):
        mv, pheromones, action, buffer = hybrid_step(
            prev_mv=mv,
            prev_pheromones=pheromones,
            action_score_dict=scores,
            snapshot_buffer=buffer,
            n=N,
            decay=0.85,
        )
        print(f"Step {step+1}:")
        print(f"  Predicted multivector: {mv}")
        print(f"  Entropy of scores: {shannon_entropy(np.array([mv.components.get(frozenset([i]),0.0) for i in range(1,N+1)])):.4f}")
        print(f"  Updated pheromones: {pheromones}")
        print(f"  Selected action: {action}")
        # Simulate new scores for next iteration
        scores = {k: random.uniform(0, 10) for k in scores}
        print("-" * 40)