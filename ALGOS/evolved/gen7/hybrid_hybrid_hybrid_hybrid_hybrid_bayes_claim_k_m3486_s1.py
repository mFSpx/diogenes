# DARWIN HAMMER — match 3486, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1712_s0.py (gen5)
# parent_b: hybrid_bayes_claim_kernel_hybrid_hybrid_hybrid_m1718_s1.py (gen6)
# born: 2026-05-29T23:50:22Z

"""HybridRiskPheromone
Integrates:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1712_s0.py (pheromone‑based candidate weighting)
- Parent B: hybrid_bayes_claim_kernel_hybrid_hybrid_hybrid_m1718_s1.py (Bayesian update & KL‑divergence)

Mathematical bridge:
Both parents manipulate probability‑like weights.  The bridge is formed by using the
Bayesian posterior (Parent B) to modulate the pheromone signal strength (Parent A)
and then applying a Kullback‑Leibler (KL) divergence penalty (Parent B) to the
resulting pheromone distribution.  This yields a unified risk‑assessment update:
    p_new = BayesianUpdate(p_old, λ)               # λ = likelihood ratio
    p_norm = p_new / Σ p_new                        # normalize to a distribution
    KL = Σ p_norm·log(p_norm / q)                  # q = uniform prior
    decay = exp(‑KL)                               # stronger divergence → stronger decay
    pheromone.value = p_norm·decay
The system therefore fuses Bayesian inference, information‑theoretic regularisation,
and the pheromone‑driven candidate selection of the original algorithms.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np

# ----------------------------------------------------------------------
# Data structures (derived from both parents)
# ----------------------------------------------------------------------
Vector = list[float]


@dataclass
class Pheromone:
    """Core element from Parent A."""
    surface_key: str
    signal_kind: str
    signal_value: float = 1.0
    half_life_seconds: float = 3600.0  # default 1 h


@dataclass(frozen=True)
class MathEvidence:
    """Evidence element from Parent B."""
    id: str
    likelihood_ratio: float  # λ ≥ 0


# ----------------------------------------------------------------------
# Core mathematical primitives (Parent B)
# ----------------------------------------------------------------------
def bayesian_update(prior: float, likelihood_ratio: float) -> float:
    """Return the Bayesian posterior given a prior probability and a likelihood ratio."""
    if prior < 0 or likelihood_ratio < 0:
        raise ValueError("Prior and likelihood ratio must be non‑negative")
    numerator = prior * likelihood_ratio
    denominator = numerator + (1.0 - prior)
    return numerator / denominator if denominator != 0 else 0.0


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(p‖q) where p and q are probability vectors (sum to 1)."""
    eps = np.finfo(float).eps
    p_safe = np.clip(p, eps, 1)
    q_safe = np.clip(q, eps, 1)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def lead_lag_transform(X: np.ndarray) -> np.ndarray:
    """
    Feature expansion used in Parent B.
    Returns concatenated linear and quadratic aggregates of rows in X.
    """
    linear = np.sum(X, axis=1, keepdims=True)
    quadratic = np.sum(X ** 2, axis=1, keepdims=True)
    return np.hstack((linear, quadratic))


# ----------------------------------------------------------------------
# Hybrid system (fusion of both parents)
# ----------------------------------------------------------------------
class HybridPheromoneSystem:
    """Manages pheromones, updates them with Bayesian evidence, and selects candidates."""

    def __init__(self):
        self.pheromones: dict[str, Pheromone] = {}
        self._uniform_prior: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Candidate management (Parent A)
    # ------------------------------------------------------------------
    def add_candidate(self, surface_key: str, signal_kind: str = "risk", init_value: float = 1.0) -> None:
        if surface_key in self.pheromones:
            raise ValueError(f"Candidate {surface_key} already exists")
        self.pheromones[surface_key] = Pheromone(surface_key, signal_kind, init_value)

    def get_values_vector(self) -> np.ndarray:
        return np.array([p.signal_value for p in self.pheromones.values()], dtype=float)

    # ------------------------------------------------------------------
    # Hybrid update (mathematical bridge)
    # ------------------------------------------------------------------
    def update_with_evidence(self, evidence_list: list[MathEvidence]) -> None:
        """
        For each pheromone entry, apply a Bayesian update using the
        corresponding evidence likelihood ratio (if any).  Then compute a KL‑divergence
        penalty against a uniform prior and decay the pheromone values accordingly.
        """
        if not self.pheromones:
            return

        # 1️⃣ Bayesian modulation
        for ev in evidence_list:
            # Simple mapping: evidence.id must match a surface_key; ignore otherwise
            if ev.id in self.pheromones:
                p = self.pheromones[ev.id]
                posterior = bayesian_update(p.signal_value, ev.likelihood_ratio)
                p.signal_value = posterior

        # 2️⃣ Normalise to a probability distribution
        values = self.get_values_vector()
        total = np.sum(values)
        if total == 0:
            # avoid division by zero – reset to uniform
            uniform = np.full_like(values, 1.0 / len(values))
            for idx, key in enumerate(self.pheromones):
                self.pheromones[key].signal_value = uniform[idx]
            return

        p_norm = values / total

        # 3️⃣ KL‑divergence against uniform prior
        N = len(p_norm)
        uniform_prior = np.full(N, 1.0 / N)
        kl = kl_divergence(p_norm, uniform_prior)

        # 4️⃣ Decay factor (the stronger the divergence, the stronger the decay)
        decay = math.exp(-kl)

        # 5️⃣ Apply decay and write back
        for idx, key in enumerate(self.pheromones):
            self.pheromones[key].signal_value = p_norm[idx] * decay

    # ------------------------------------------------------------------
    # Candidate selection (Parent A)
    # ------------------------------------------------------------------
    def select_top(self, k: int = 3) -> list[tuple[str, float]]:
        """
        Return the top‑k candidates sorted by current pheromone strength.
        Each entry is (surface_key, signal_value).
        """
        sorted_items = sorted(
            self.pheromones.items(),
            key=lambda item: item[1].signal_value,
            reverse=True,
        )
        return [(p.surface_key, p.signal_value) for p in (v for _, v in sorted_items[:k])]

    # ------------------------------------------------------------------
    # Utility: feature extraction using lead‑lag (Parent B)
    # ------------------------------------------------------------------
    def embed_candidates(self) -> np.ndarray:
        """
        Produce a 2‑column feature matrix for the current pheromone values
        using the lead‑lag transform.
        """
        values = self.get_values_vector().reshape(-1, 1)  # column vector
        return lead_lag_transform(values)


# ----------------------------------------------------------------------
# Demonstration functions (required ≥3)
# ----------------------------------------------------------------------
def demo_bayesian_kl_update():
    """Show a single Bayesian‑KL update on a tiny system."""
    system = HybridPheromoneSystem()
    for key in ["A", "B", "C"]:
        system.add_candidate(key, init_value=0.5)

    evidence = [
        MathEvidence(id="A", likelihood_ratio=3.0),  # strong support for A
        MathEvidence(id="C", likelihood_ratio=0.2),  # weak support for C
    ]
    system.update_with_evidence(evidence)
    return system.select_top(k=2)


def demo_embedding():
    """Generate the lead‑lag feature matrix for a random pheromone configuration."""
    system = HybridPheromoneSystem()
    for i in range(5):
        system.add_candidate(f"node_{i}", init_value=random.random() + 0.1)
    return system.embed_candidates()


def demo_full_cycle():
    """Run a full cycle: add candidates → update with evidence → select."""
    system = HybridPheromoneSystem()
    # Add 6 candidates with diverse initial strengths
    for i, val in enumerate([0.2, 0.8, 0.5, 0.3, 0.9, 0.4]):
        system.add_candidate(f"cand_{i}", init_value=val)

    # Simulated evidence (some IDs may not match any candidate)
    evidence = [
        MathEvidence(id="cand_1", likelihood_ratio=2.5),
        MathEvidence(id="cand_4", likelihood_ratio=0.7),
        MathEvidence(id="unknown", likelihood_ratio=5.0),  # ignored
    ]

    system.update_with_evidence(evidence)
    top = system.select_top(k=3)
    embed = system.embed_candidates()
    return top, embed


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1️⃣ Demo Bayesian‑KL update
    top_two = demo_bayesian_kl_update()
    print("Top 2 after Bayesian‑KL update:", top_two)

    # 2️⃣ Demo embedding matrix
    embed_matrix = demo_embedding()
    print("Lead‑lag embedding shape:", embed_matrix.shape)
    print(embed_matrix)

    # 3️⃣ Demo full cycle
    top_three, features = demo_full_cycle()
    print("Top 3 after full cycle:", top_three)
    print("Feature matrix (first 3 rows):")
    print(features[:3])