# DARWIN HAMMER — match 3194, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_semant_hybrid_hard_truth_ma_m1271_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_model__hybrid_hybrid_ssim_h_m1828_s0.py (gen6)
# born: 2026-05-29T23:48:27Z

import numpy as np
from dataclasses import dataclass
from math import log, sqrt
from typing import List, Dict, Any


# ----------------------------------------------------------------------
# Domain data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of an object."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class Document:
    """Document identifier together with its embedding vector."""
    id: str
    vector: List[float]


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision for a single action."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str
    marginal_probability: float  # SSIM‑derived probability
    semantic_priority: float      # probability weighted by morphology


@dataclass(frozen=True)
class BanditUpdate:
    """Container returned by the update routine."""
    context_id: str
    action: BanditAction
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def _positive_dimensions(*dims: float) -> None:
    """Validate that all supplied dimensions are strictly positive."""
    if any(d <= 0 for d in dims):
        raise ValueError("All geometric dimensions must be > 0")


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of the three dimensions to the longest side."""
    _positive_dimensions(length, width, height)
    gm = (length * width * height) ** (1.0 / 3.0)
    longest = max(length, width, height)
    return gm / longest


def flatness_index(length: float, width: float, height: float) -> float:
    """Simple flatness metric used in some downstream analyses."""
    _positive_dimensions(length, width, height)
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    """Physical‑style index that mixes mass, height and a few tunable constants."""
    _positive_dimensions(m.length, m.width, m.height)
    term = b / (k * m.mass * m.height ** 2)
    expo = exp(-k * m.mass / (b * m.height))
    return term * (1 - expo)


# ----------------------------------------------------------------------
# SSIM utilities (scalar version)
# ----------------------------------------------------------------------
def ssim_scalar(x: List[float],
                y: List[float],
                C1: float = 0.01,
                C2: float = 0.03) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two 1‑D signals.
    The implementation follows the classic formulation:

        l   = (2 μx μy + C1) / (μx² + μy² + C1)
        c   = (2 σx σy + C2) / (σx² + σy² + C2)
        s   = (σxy + C3) / (σx σy + C3)   with C3 = C2/2

    The final SSIM is the product l·c·s.
    """
    if len(x) != len(y):
        raise ValueError("Input vectors must have the same length")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mu_x = x_arr.mean()
    mu_y = y_arr.mean()
    sigma_x = x_arr.std(ddof=1)
    sigma_y = y_arr.std(ddof=1)
    sigma_xy = np.cov(x_arr, y_arr, ddof=1)[0, 1]

    C3 = C2 / 2.0

    l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    c = (2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)
    s = (sigma_xy + C3) / (sigma_x * sigma_y + C3)

    return float(l * c * s)


# ----------------------------------------------------------------------
# Bandit logic
# ----------------------------------------------------------------------
def _ucb_confidence(propensity: float,
                    n_pulls: int = 10,
                    delta: float = 0.05) -> float:
    """
    Upper‑Confidence Bound (UCB) term derived from Hoeffding’s inequality.
    The `propensity` plays the role of an importance weight; a smaller
    propensity yields a larger confidence interval.
    """
    if propensity <= 0 or propensity > 1:
        raise ValueError("Propensity must lie in (0, 1]")
    # The factor 2 comes from the two‑sided Hoeffding bound.
    return sqrt(2 * log(2 / propensity) / max(1, n_pulls))


def bandit_update(context_id: str,
                  action_id: str,
                  reward: float,
                  propensity: float,
                  document: Document,
                  morphology: Morphology,
                  C1: float = 0.01,
                  C2: float = 0.03) -> BanditUpdate:
    """
    Perform a single bandit update, fusing SSIM‑derived marginal probability
    with a morphology‑aware semantic priority.

    Returns a `BanditUpdate` that already contains the fully‑populated
    `BanditAction`.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Compute SSIM between the document embedding and a synthetic
    #     vector representing the observed reward.
    # ------------------------------------------------------------------
    reward_vec = [reward] * len(document.vector)  # broadcast reward
    marginal_prob = ssim_scalar(document.vector, reward_vec, C1, C2)

    # ------------------------------------------------------------------
    # 2️⃣  Geometry‑aware semantic priority.
    # ------------------------------------------------------------------
    sph = sphericity_index(morphology.length,
                           morphology.width,
                           morphology.height)
    semantic_priority = marginal_prob * sph

    # ------------------------------------------------------------------
    # 3️⃣  Confidence bound (UCB) – deeper integration: the bound is
    #     scaled by the semantic priority, encouraging exploration on
    #     geometrically “interesting’’ objects.
    # ------------------------------------------------------------------
    base_conf = _ucb_confidence(propensity)
    confidence_bound = base_conf * (1 + semantic_priority)

    # ------------------------------------------------------------------
    # 4️⃣  Assemble the immutable action and the surrounding update.
    # ------------------------------------------------------------------
    action = BanditAction(
        action_id=action_id,
        propensity=propensity,
        expected_reward=reward,
        confidence_bound=confidence_bound,
        algorithm="hybrid_semantic_ssim",
        marginal_probability=marginal_prob,
        semantic_priority=semantic_priority,
    )

    return BanditUpdate(
        context_id=context_id,
        action=action,
        reward=reward,
        propensity=propensity,
    )


def hybrid_algorithm(context_id: str,
                     action_id: str,
                     reward: float,
                     propensity: float,
                     document: Document,
                     morphology: Morphology,
                     C1: float = 0.01,
                     C2: float = 0.03) -> Dict[str, Any]:
    """
    Public façade that runs a single hybrid step and returns a plain‑dict
    suitable for logging or downstream pipelines.
    """
    upd = bandit_update(
        context_id=context_id,
        action_id=action_id,
        reward=reward,
        propensity=propensity,
        document=document,
        morphology=morphology,
        C1=C1,
        C2=C2,
    )

    # Flatten the dataclass for JSON‑friendly output
    return {
        "context_id": upd.context_id,
        "action_id": upd.action.action_id,
        "reward": upd.reward,
        "propensity": upd.propensity,
        "expected_reward": upd.action.expected_reward,
        "confidence_bound": upd.action.confidence_bound,
        "marginal_probability": upd.action.marginal_probability,
        "semantic_priority": upd.action.semantic_priority,
        "algorithm": upd.action.algorithm,
    }


# ----------------------------------------------------------------------
# Simple sanity‑check when executed as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    doc = Document(id="doc1", vector=[0.2, 0.5, 0.9, 0.1])
    morph = Morphology(length=1.2, width=0.8, height=0.6, mass=2.5)

    out = hybrid_algorithm(
        context_id="ctx-001",
        action_id="act-A",
        reward=0.73,
        propensity=0.45,
        document=doc,
        morphology=morph,
        C1=0.01,
        C2=0.03,
    )
    print(out)