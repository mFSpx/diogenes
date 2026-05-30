# DARWIN HAMMER — match 1265, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m647_s1.py (gen5)
# parent_b: ssim.py (gen0)
# born: 2026-05-29T23:34:58Z

import math
import random
from dataclasses import dataclass, asdict
from typing import List, Sequence, Tuple

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a specimen."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Feedback supplied to the bandit after an action."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """A minimal stochastic bandit state (Beta‑Bernoulli)."""
    alpha: float = 1.0   # successes + prior
    beta: float = 1.0    # failures + prior
    limit: float = 1.0   # upper bound for probability estimates

    def sample(self) -> float:
        """Draw a probability from the posterior."""
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        """Bayesian update with a binary reward (0/1)."""
        if reward not in (0, 1):
            raise ValueError("Bandit reward must be 0 or 1 for Beta‑Bernoulli update")
        self.alpha += reward
        self.beta += 1 - reward

    @property
    def mean(self) -> float:
        """Posterior mean (clipped to [0, limit])."""
        return max(0.0, min(self.limit, self.alpha / (self.alpha + self.beta)))


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------


def sphericity_index(length: float, width: float, height: float) -> float:
    """Return a dimensionless sphericity measure in (0, 1]."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be strictly positive")
    # Geometric mean divided by the longest side
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def calculate_health_score(morphology: Morphology) -> float:
    """Health is directly proportional to sphericity."""
    return sphericity_index(morphology.length, morphology.width, morphology.height)


# ----------------------------------------------------------------------
# SSIM implementation for generic numeric vectors
# ----------------------------------------------------------------------


def _dynamic_range(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute the data range used for the SSIM stabilising constants."""
    combined = list(x) + list(y)
    mx, mn = max(combined), min(combined)
    dr = mx - mn
    # Guard against a zero range (constant vectors)
    return dr if dr > 0 else 1.0


def ssim(
    x: Sequence[float],
    y: Sequence[float],
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Structural Similarity Index for 1‑D numeric sequences.

    The classic SSIM formula is preserved but adapted to generic vectors.
    Returns a value in [0, 1] where 1 means perfect similarity.
    """
    if len(x) != len(y):
        raise ValueError("Sequences must have equal length")
    if not x:
        raise ValueError("Sequences must not be empty")

    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n

    # Variances (population)
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n

    # Covariance (population)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n

    dr = _dynamic_range(x, y)
    c1 = (k1 * dr) ** 2
    c2 = (k2 * dr) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)

    # Numerical safety – denominator can be zero only for pathological inputs
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return max(0.0, min(1.0, numerator / denominator))


# ----------------------------------------------------------------------
# Fusion logic
# ----------------------------------------------------------------------


def _blend_scores(
    ssim_score: float,
    health1: float,
    health2: float,
    weight_ssim: float = 0.6,
) -> float:
    """
    Produce a single similarity proxy from SSIM and the two health scores.

    The health scores are first normalised to the same [0, 1] range as SSIM.
    """
    # Normalise health scores (they already lie in (0,1] by construction)
    health_avg = (health1 + health2) / 2.0
    return weight_ssim * ssim_score + (1.0 - weight_ssim) * health_avg


def modulate_target_percentage(
    blended_score: float,
    target_percentage: float,
    min_pct: float = 0.0,
    max_pct: float = 1.0,
) -> float:
    """
    Adjust the raw target percentage using the blended similarity score.

    The result is clipped to the user‑specified bounds.
    """
    if not (0.0 <= target_percentage <= 1.0):
        raise ValueError("target_percentage must be within [0, 1]")
    adjusted = target_percentage * blended_score
    return max(min_pct, min(max_pct, adjusted))


def hybrid_algorithm(
    morphology1: Morphology,
    morphology2: Morphology,
    target_percentage: float,
    *,
    weight_ssim: float = 0.6,
) -> float:
    """
    Core hybrid routine.

    1. Compute health (sphericity) for each specimen.
    2. Compute SSIM on the raw feature vectors.
    3. Blend the two similarity measures.
    4. Modulate the supplied target percentage.
    """
    # 1. Health scores
    health1 = calculate_health_score(morphology1)
    health2 = calculate_health_score(morphology2)

    # 2. Feature vectors (including mass for richer signal)
    features1 = [morphology1.length, morphology1.width, morphology1.height, morphology1.mass]
    features2 = [morphology2.length, morphology2.width, morphology2.height, morphology2.mass]

    # 3. SSIM
    ssim_score = ssim(features1, features2)

    # 4. Blend
    blended = _blend_scores(ssim_score, health1, health2, weight_ssim=weight_ssim)

    # 5. Modulate
    return modulate_target_percentage(blended, target_percentage)


# ----------------------------------------------------------------------
# Simple Thompson‑sampling bandit wrapper (optional)
# ----------------------------------------------------------------------


def select_action(
    actions: List[BanditAction],
    state: StoreState,
) -> BanditAction:
    """
    Choose an action using Thompson sampling on the stored beta posterior.
    The action with the highest sampled probability is returned.
    """
    sampled_probs = {a.action_id: state.sample() for a in actions}
    best_id = max(sampled_probs, key=sampled_probs.get)
    return next(a for a in actions if a.action_id == best_id)


def update_bandit(state: StoreState, update: BanditUpdate) -> None:
    """
    Apply a binary reward to the bandit posterior.
    """
    # Convert reward to binary (0/1) – for demonstration we treat any
    # positive reward as success.
    binary_reward = 1 if update.reward > 0 else 0
    state.update(binary_reward)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


def smoke_test() -> None:
    m1 = Morphology(1.0, 2.0, 3.0, 4.0)
    m2 = Morphology(1.1, 2.1, 3.1, 4.1)
    target = 0.5

    adjusted = hybrid_algorithm(m1, m2, target)
    print(f"Adjusted target percentage: {adjusted:.4f}")

    # Demonstrate the tiny bandit wrapper
    actions = [
        BanditAction("A", 0.5, 0.0, 0.0, "thompson"),
        BanditAction("B", 0.5, 0.0, 0.0, "thompson"),
    ]
    state = StoreState()
    chosen = select_action(actions, state)
    print(f"Chosen action: {chosen.action_id}")

    # Fake feedback
    feedback = BanditUpdate("ctx1", chosen.action_id, reward=adjusted, propensity=chosen.propensity)
    update_bandit(state, feedback)
    print(f"Bandit posterior mean after update: {state.mean:.4f}")


if __name__ == "__main__":
    smoke_test()