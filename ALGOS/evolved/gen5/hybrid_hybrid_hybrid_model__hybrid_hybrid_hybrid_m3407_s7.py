# DARWIN HAMMER — match 3407, survivor 7
# gen: 5
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_hard_truth_ma_m434_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_cockpit_metri_m102_s0.py (gen4)
# born: 2026-05-29T23:50:00Z

import numpy as np
import hashlib
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def deterministic_hash(token: str, bits: int = 64) -> int:
    """
    Produce a deterministic integer hash for a token using SHA‑256.
    The result is truncated to the requested number of bits.
    """
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    # Take the first `bits // 8` bytes and interpret as big‑endian integer
    return int.from_bytes(digest[: bits // 8], "big", signed=False)


def token_vector(tokens: Tuple[str, ...], dim: int) -> np.ndarray:
    """
    Convert a tuple of tokens into a fixed‑dimensional numeric vector.
    Each token contributes a deterministic hash that is reduced modulo `dim`
    and added to the corresponding entry.
    """
    vec = np.zeros(dim, dtype=np.float64)
    for token in tokens:
        idx = deterministic_hash(token) % dim
        vec[idx] += 1.0
    return vec


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Definition of an action used by the regret‑weighted component."""
    id: str
    tokens: Tuple[str, ...]          # token set for deterministic hashing
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


# ----------------------------------------------------------------------
# Fusion logic
# ----------------------------------------------------------------------
class RegretWeightedFusion:
    """
    Implements a mathematically coherent fusion of vector‑matrix updates
    (from the first algorithm) with regret‑weighted value computation
    (from the second algorithm).

    The class maintains a weight matrix `W` of shape (d, d) where `d`
    is the embedding dimension.  Actions are projected into the same
    space via a deterministic token vector.  The regret‑weighted value
    is a scalar obtained by a bilinear form:

        v = e - c - r + (tᵀ W t)

    where `t` is the token vector.  The matrix is updated with a simple
    rank‑1 gradient step that respects the curvature of the bilinear term:

        W ← W + η * (t tᵀ) * (v - v_prev)

    This keeps the dimensions consistent and provides a deeper integration
    of the two mathematical systems.
    """

    def __init__(self, dim: int = 64, learning_rate: float = 0.01):
        """
        Parameters
        ----------
        dim : int
            Dimensionality of the token embedding space.
        learning_rate : float
            Step size for matrix updates.
        """
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        self.dim = dim
        self.eta = learning_rate
        # Initialise weight matrix close to identity for stability
        self.W = np.eye(dim, dtype=np.float64) * 0.1

        # Store the last regret‑weighted value for each action to compute deltas
        self._last_value: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_regret_weighted_value(self, action: MathAction) -> float:
        """
        Compute the regret‑weighted scalar value for a given action.
        """
        t = token_vector(action.tokens, self.dim)               # (d,)
        bilinear = float(t @ self.W @ t)                        # scalar = tᵀ W t
        value = action.expected_value - action.cost - action.risk + bilinear
        return value

    def update_weight_matrix(self, action: MathAction, current_value: float) -> None:
        """
        Update the internal weight matrix using a rank‑1 outer‑product
        derived from the token vector and the change in regret‑weighted value.
        """
        t = token_vector(action.tokens, self.dim)               # (d,)

        # Retrieve previous value; if none, assume no change
        prev = self._last_value.get(action.id, current_value)
        delta = current_value - prev

        if delta != 0.0:
            # Rank‑1 update: η * delta * (t tᵀ)
            outer = np.outer(t, t)                              # (d, d)
            self.W += self.eta * delta * outer

        # Cache the current value for the next iteration
        self._last_value[action.id] = current_value

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------
    def evaluate_and_learn(self, action: MathAction) -> float:
        """
        Compute the regret‑weighted value and immediately apply the matrix update.
        Returns the computed value.
        """
        value = self.compute_regret_weighted_value(action)
        self.update_weight_matrix(action, value)
        return value

    def get_weight_matrix(self) -> np.ndarray:
        """Return a copy of the current weight matrix."""
        return self.W.copy()


# ----------------------------------------------------------------------
# Auxiliary metric (unchanged but made robust)
# ----------------------------------------------------------------------
def compute_cockpit_honesty(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """
    Compute the cockpit honesty metric as the proportion of claims
    supported by evidence.  Returns a value in [0, 1].
    """
    if total_claims_emitted <= 0:
        return 0.0
    ratio = claims_with_evidence / total_claims_emitted
    return max(0.0, min(1.0, ratio))


# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise fusion engine
    fusion = RegretWeightedFusion(dim=32, learning_rate=0.005)

    # Sample actions
    actions = [
        MathAction("a1", ("token1", "token2", "token3"), expected_value=12.0, cost=1.5, risk=0.3),
        MathAction("a2", ("alpha", "beta"), expected_value=8.0, cost=0.8, risk=0.2),
        MathAction("a1", ("token1", "token2", "token3"), expected_value=13.0, cost=1.5, risk=0.3),  # repeated id
    ]

    for act in actions:
        val = fusion.evaluate_and_learn(act)
        print(f"Action {act.id!r} → regret‑weighted value: {val:.4f}")

    print("\nFinal weight matrix (excerpt):")
    print(fusion.get_weight_matrix()[:5, :5])  # show a small block for brevity

    # Cockpit honesty demo
    print("\nCockpit honesty:", compute_cockpit_honesty(10, 20))