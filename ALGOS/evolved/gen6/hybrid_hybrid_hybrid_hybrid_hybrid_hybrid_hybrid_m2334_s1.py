# DARWIN HAMMER — match 2334, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_ternar_m1199_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_fisher_locali_m420_s1.py (gen5)
# born: 2026-05-29T23:43:17Z

"""Hybrid algorithm merging:
- Parent A: structural similarity (SSIM) based group comparison.
- Parent B: information‑theoretic measures (KL divergence, Fisher score) and pheromone decay logic.

Mathematical bridge:
The algorithm treats each group prototype as a probability‑like vector.
For a global prototype `x` and a group prototype `g` we compute:
    * SSIM(x, g) – captures structural alignment (Parent A).
    * KL(x‖g) – quantifies information loss when `g` approximates `x` (Parent B).
    * FisherScore(x, g) – a symmetric information metric derived from Fisher
      information (Parent B).
These three scalars are fused into a single hybrid similarity score
    H = w₁·SSIM + w₂·exp(‑KL) + w₃·exp(‑FisherScore)
with weights `w₁,w₂,w₃` summing to 1.  The hybrid score drives updates of
pheromone entries that decay over time, providing a unified feedback loop
between structural and probabilistic assessments.
"""

import numpy as np
import math
import random
import sys
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Configuration & small utilities
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
EPS = 1e-12                     # safeguard against division‑by‑zero
C1 = 0.01 ** 2                  # SSIM constants
C2 = 0.03 ** 2

W_SSIM = 0.4
W_KL   = 0.3
W_FISH = 0.3

def _pct(value: float) -> float:
    """Round a float to six decimal places for human‑readable percentages."""
    return round(float(value), 6)

def _safe_std(arr: np.ndarray) -> float:
    """Standard deviation with a tiny epsilon to avoid zero‑variance pitfalls."""
    return max(np.std(arr), EPS)

# ----------------------------------------------------------------------
# Core mathematical building blocks
# ----------------------------------------------------------------------
def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Structural Similarity Index for 1‑D vectors (Parent A)."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x = _safe_std(x)
    sigma_y = _safe_std(y)

    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)

    return float(numerator / denominator)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Discrete Kullback‑Leibler divergence KL(p‖q).
    Both `p` and `q` are assumed to be non‑negative and are normalised
    internally to sum to 1.
    """
    p_norm = p / (p.sum() + EPS)
    q_norm = q / (q.sum() + EPS)

    # Clip to avoid log(0)
    p_clip = np.clip(p_norm, EPS, None)
    q_clip = np.clip(q_norm, EPS, None)

    return float(np.sum(p_clip * np.log(p_clip / q_clip)))


def fisher_score(p: np.ndarray, q: np.ndarray) -> float:
    """
    Symmetric Fisher‑information‑based score.
    For two discrete distributions this reduces to Σ (p‑q)² / q,
    mirroring the Fisher information metric for small perturbations.
    """
    q_clip = np.clip(q, EPS, None)
    diff = p - q
    return float(np.sum(diff ** 2 / q_clip))


def hybrid_group_score(
    prototype: np.ndarray,
    group_vec: np.ndarray,
    weights: Tuple[float, float, float] = (W_SSIM, W_KL, W_FISH)
) -> float:
    """
    Fuse SSIM, KL divergence and Fisher score into a single similarity
    measure for one group.
    """
    ssim = compute_ssim(prototype, group_vec)
    kl   = kl_divergence(prototype, group_vec)
    fish = fisher_score(prototype, group_vec)

    w_ssim, w_kl, w_fish = weights
    # Transform KL/Fisher to similarity‑like quantities via exponential decay
    hybrid = (
        w_ssim * ssim +
        w_kl   * math.exp(-kl) +
        w_fish * math.exp(-fish)
    )
    return hybrid


def compute_hybrid_group_similarities(
    prototype: np.ndarray,
    group_prototypes: Dict[str, np.ndarray],
    weights: Tuple[float, float, float] = (W_SSIM, W_KL, W_FISH)
) -> Dict[str, float]:
    """
    Compute hybrid scores for all groups and normalise them to sum to 1.
    """
    raw = {
        name: hybrid_group_score(prototype, vec, weights)
        for name, vec in group_prototypes.items()
    }
    total = sum(raw.values()) + EPS
    return {name: score / total for name, score in raw.items()}


# ----------------------------------------------------------------------
# Pheromone infrastructure (adapted from Parent B)
# ----------------------------------------------------------------------
class PheromoneEntry:
    """
    Represents a decaying signal attached to a textual surface.
    The `signal_value` is modulated by hybrid similarity scores.
    """
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid1())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = max(1, half_life_seconds)  # avoid division by zero
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay since last_decay based on half‑life."""
        age = self.age_seconds()
        # decay = 0.5 ** (age / half_life)
        return 0.5 ** (age / self.half_life_seconds)

    def apply_decay(self) -> None:
        """Apply decay to `signal_value` and reset the decay timestamp."""
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


def update_pheromones(
    entries: List[PheromoneEntry],
    hybrid_scores: Dict[str, float],
    boost: float = 1.0
) -> None:
    """
    For each pheromone entry, apply decay and then boost its signal
    proportionally to the hybrid similarity of its associated group.
    `boost` scales the reinforcement magnitude.
    """
    for entry in entries:
        entry.apply_decay()
        # The surface_key is expected to contain the group name as a prefix,
        # e.g. "codex:some_id".  We extract the group name safely.
        group = entry.surface_key.split(":")[0]
        score = hybrid_scores.get(group, 0.0)
        entry.signal_value += boost * score


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def generate_random_prototype(dim: int = 128) -> np.ndarray:
    """Create a random non‑negative vector and normalise it to unit L2 norm."""
    vec = np.abs(np.random.randn(dim))
    return vec / (np.linalg.norm(vec) + EPS)


def demo_hybrid_pipeline() -> Tuple[Dict[str, float], List[PheromoneEntry]]:
    """
    End‑to‑end demonstration:
    1. Create a global prototype and per‑group prototypes.
    2. Compute hybrid group similarities.
    3. Initialise pheromone entries for each group.
    4. Update pheromones using the hybrid scores.
    Returns the similarity dict and the list of updated entries.
    """
    # 1. Prototypes
    prototype = generate_random_prototype()
    group_prototypes = {
        name: generate_random_prototype()
        for name in GROUPS
    }

    # 2. Hybrid similarities
    hybrid_scores = compute_hybrid_group_similarities(prototype, group_prototypes)

    # 3. Pheromone entries (one per group)
    entries = [
        PheromoneEntry(
            surface_key=f"{name}:example",
            signal_kind="similarity",
            signal_value=0.5,
            half_life_seconds=30
        )
        for name in GROUPS
    ]

    # 4. Update
    update_pheromones(entries, hybrid_scores, boost=0.2)

    return hybrid_scores, entries


if __name__ == "__main__":
    # Smoke test: run the demo and print concise results.
    scores, pheromones = demo_hybrid_pipeline()
    print("Hybrid group scores (normalised):")
    for g, s in scores.items():
        print(f"  {g}: {_pct(s)}")
    print("\nPheromone states after update:")
    for p in pheromones:
        print(f"  {p.surface_key} -> value: {_pct(p.signal_value)} (decayed factor applied)")
    sys.exit(0)