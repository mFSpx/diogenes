# DARWIN HAMMER — match 3157, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m2239_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_regret_m2442_s1.py (gen6)
# born: 2026-05-29T23:48:20Z

"""Hybrid Krampus‑Geometric Regret Engine

Parents:
- hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m2239_s1.py (Krampus feature extractor + curvature‑modulated LinUCB)
- hybrid_hybrid_hybrid_geomet_hybrid_hybrid_regret_m2442_s1.py (Shannon‑entropy‑driven geometric rotor + regret‑weighted strategy)

Mathematical bridge
-------------------
Both parents expose a *scalar uncertainty modifier* that multiplies the LinUCB
exploration term:

* Parent A supplies a MinHash‑derived curvature `c ∈ (0,1)`.
* Parent B supplies a Shannon‑entropy `e ≥ 0` that is turned into a geometric
  rotor `R = exp(i·θ)` with `θ = π·e / e_max`.  The rotor acts as a
  similarity‑preserving linear map `x ↦ R·x·R⁻¹`, which in our scalar
  implementation reduces to a uniform scaling factor `ρ(e) = exp(e)`.

The hybrid engine first rotates (scales) the Krampus feature vector `x` by
`ρ(e)`, then feeds the transformed vector `x̃` to the LinUCB bound while
modulating the exploration bonus with the *product* `c·ρ(e)·wₐ`.  Thus the
final upper‑confidence bound is

    UCBₐ(x) = θₐ·x̃ + α·c·ρ(e)·wₐ·√(x̃ᵀ Aₐ⁻¹ x̃)

where `θₐ = Aₐ⁻¹ bₐ` and `wₐ` is the regret‑weight updated after each
interaction.

The code below implements this unified system with three public functions:
`extract_features`, `compute_ucb`, and `update_bandit`.  A lightweight test
runs a few random rounds to demonstrate error‑free execution.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities shared by both parents
# ----------------------------------------------------------------------


def extract_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑feature extractor (Krampus stub).

    The function hashes each word, aggregates the hashes and produces a
    fixed‑size dictionary of 12 synthetic features.  Empty input yields an
    empty dict.
    """
    if not text.strip():
        return {}
    words = text.split()
    # Simple deterministic aggregate
    base = sum(hash(w) for w in words) % 1000
    feats = {
        "operator_visceral_ratio": (base % 10) / 10.0,
        "operator_tech_ratio": ((base // 10) % 10) / 10.0,
        "operator_legal_osint_ratio": ((base // 100) % 10) / 10.0,
        "operator_ledger_density": ((base // 1000) % 10) / 10.0,
        "operator_signal_strength": ((base // 2) % 10) / 10.0,
        "operator_noise_level": ((base // 3) % 10) / 10.0,
        "operator_latency": ((base // 5) % 10) / 10.0,
        "operator_bandwidth": ((base // 7) % 10) / 10.0,
        "operator_error_rate": ((base // 11) % 10) / 10.0,
        "operator_uptime": ((base // 13) % 10) / 10.0,
        "operator_load_factor": ((base // 17) % 10) / 10.0,
        "operator_security_score": ((base // 19) % 10) / 10.0,
    }
    return feats


def dict_to_vector(feat_dict: Dict[str, float]) -> np.ndarray:
    """Convert the feature dict to a deterministic ordered NumPy vector."""
    if not feat_dict:
        return np.zeros(12)
    # Fixed ordering based on sorted keys guarantees reproducibility
    keys = sorted(feat_dict.keys())
    vec = np.array([feat_dict[k] for k in keys], dtype=float)
    return vec


def minhash_curvature(text: str) -> float:
    """Compute a MinHash‑derived curvature in (0,1) via Blake2b and sigmoid."""
    h = hashlib.blake2b(text.encode(), digest_size=8).digest()
    # Convert first 8 bytes to unsigned int
    i = int.from_bytes(h, byteorder="big", signed=False)
    # Map to a moderate range before sigmoid
    raw = (i % 1000) / 100.0  # roughly [-10,10] after centering
    raw = raw - 5.0  # center around 0
    return 1.0 / (1.0 + math.exp(-raw))


def shannon_entropy(text: str) -> float:
    """Shannon entropy of token frequencies in the input text."""
    tokens = text.split()
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy


def rotor_scale_from_entropy(entropy: float, max_entropy: float = 5.0) -> float:
    """Map entropy to a positive scaling factor ρ(e) = exp(e / max_entropy)."""
    # Clamp to avoid overflow
    e = min(entropy, max_entropy)
    return math.exp(e / max_entropy)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------------------------------------------------
# Hybrid bandit structures
# ----------------------------------------------------------------------


class HybridLinUCB:
    """Hybrid LinUCB that incorporates curvature, entropy‑driven rotor scaling,
    and regret‑weighted exploration."""

    def __init__(
        self,
        actions: List[str],
        dim: int = 12,
        alpha: float = 1.0,
    ):
        self.actions = actions
        self.dim = dim
        self.alpha = alpha
        # Per‑action matrices
        self.A: Dict[str, np.ndarray] = {a: np.identity(dim) for a in actions}
        self.b: Dict[str, np.ndarray] = {a: np.zeros(dim) for a in actions}
        # Regret‑weight wₐ, initialised to 1
        self.w: Dict[str, float] = {a: 1.0 for a in actions}
        # Track latest reward per action for regret computation
        self.last_reward: Dict[str, float] = {a: 0.0 for a in actions}

    def _theta(self, a: str) -> np.ndarray:
        """θₐ = Aₐ⁻¹ bₐ (regularised least‑squares estimate)."""
        return np.linalg.solve(self.A[a], self.b[a])

    def compute_ucb(self, a: str, x: np.ndarray, curvature: float, rotor_scale: float) -> float:
        """Calculate the hybrid UCB bound for action *a*.

        Parameters
        ----------
        a : str
            Action identifier.
        x : np.ndarray
            Original Krampus feature vector (dim,).
        curvature : float
            MinHash‑derived curvature `c`.
        rotor_scale : float
            Scaling factor `ρ(e)` derived from entropy.

        Returns
        -------
        float
            Upper‑confidence bound.
        """
        # Apply rotor scaling (geometric transformation)
        x_tilde = rotor_scale * x

        theta = self._theta(a)
        A_inv = np.linalg.inv(self.A[a])
        mean = float(theta.dot(x_tilde))
        uncertainty = math.sqrt(float(x_tilde.T.dot(A_inv).dot(x_tilde)))
        bonus = self.alpha * curvature * rotor_scale * self.w[a] * uncertainty
        return mean + bonus

    def select_action(self, text: str) -> Tuple[str, float, float]:
        """Select the action with the highest hybrid UCB for the given context."""
        feats = extract_features(text)
        x = dict_to_vector(feats)
        curvature = minhash_curvature(text)
        entropy = shannon_entropy(text)
        rotor_scale = rotor_scale_from_entropy(entropy)

        best_a = None
        best_val = -float("inf")
        for a in self.actions:
            val = self.compute_ucb(a, x, curvature, rotor_scale)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a, curvature, rotor_scale

    def update(self, a: str, text: str, reward: float) -> None:
        """Update the internal statistics after taking action *a* with observed
        *reward*.

        The regret weight `wₐ` is multiplied by `1 + regret`,
        where `regret = max_reward_sofar - reward`.
        """
        feats = extract_features(text)
        x = dict_to_vector(feats)
        curvature = minhash_curvature(text)
        entropy = shannon_entropy(text)
        rotor_scale = rotor_scale_from_entropy(entropy)

        x_tilde = rotor_scale * x

        # Standard LinUCB updates on the rotated vector
        self.A[a] += np.outer(x_tilde, x_tilde)
        self.b[a] += reward * x_tilde

        # Regret‑weight update
        max_reward = max(self.last_reward.values())
        regret = max_reward - reward
        self.w[a] *= 1.0 + regret
        self.last_reward[a] = reward


# ----------------------------------------------------------------------
# Public API – three demonstrative functions
# ----------------------------------------------------------------------


def hybrid_ucb_for_text(
    bandit: HybridLinUCB,
    text: str,
) -> Tuple[str, float]:
    """Return the action chosen for *text* together with its hybrid UCB value."""
    action, curvature, rotor_scale = bandit.select_action(text)
    feats = extract_features(text)
    x = dict_to_vector(feats)
    ucb_val = bandit.compute_ucb(action, x, curvature, rotor_scale)
    return action, ucb_val


def hybrid_update(
    bandit: HybridLinUCB,
    action: str,
    text: str,
    reward: float,
) -> None:
    """Convenience wrapper around `HybridLinUCB.update`."""
    bandit.update(action, text, reward)


def simulate_hybrid_bandit(
    actions: List[str],
    iterations: int = 10,
    seed: int = 42,
) -> None:
    """Run a lightweight simulation to demonstrate the hybrid engine."""
    random.seed(seed)
    np.random.seed(seed)
    bandit = HybridLinUCB(actions)

    for i in range(iterations):
        # Generate a random pseudo‑sentence
        words = [random.choice(["alpha", "beta", "gamma", "delta", "epsilon"]) for _ in range(5)]
        text = " ".join(words)

        # Choose action
        chosen, ucb = hybrid_ucb_for_text(bandit, text)

        # Simulated stochastic reward (higher when word "alpha" appears)
        reward = 1.0 if "alpha" in words else 0.0
        reward += random.random() * 0.1  # small noise

        hybrid_update(bandit, chosen, text, reward)

        print(f"Iter {i+1:02d}: text='{text}' | action={chosen} | ucb={ucb:.3f} | reward={reward:.3f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    actions_list = ["click", "scroll", "type", "hover"]
    simulate_hybrid_bandit(actions_list, iterations=15)